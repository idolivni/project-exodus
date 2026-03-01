"""
NEOWISE Time-Series Infrared Ingestion for Project EXODUS.

Queries the NEOWISE Reactivation Single Exposure (L1b) Source Table via the
IRSA TAP service to retrieve ALL individual W1 and W2 measurements for a given
target over time (2013-present).  Each star typically has 20-40 visits per year
across 10+ years, yielding 200-400+ data points — a dense infrared light curve.

This is the foundation of the cross-band temporal analysis: by comparing optical
dimming (Gaia/Kepler/TESS) with infrared brightening (NEOWISE), we can detect
the absorption + re-emission signature of a megastructure (Dyson sphere/swarm).

Public API
----------
query_neowise_timeseries(ra, dec, radius_arcsec=3.6)
    Retrieve all W1/W2 single-exposure measurements for a target.

get_neowise_lightcurve(ra, dec)
    Return cleaned, outlier-rejected W1/W2 light curves as numpy arrays.

batch_query(targets)
    Query multiple targets and return a dict of results.
"""

from __future__ import annotations

import hashlib
import signal
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from astroquery.ipac.irsa import Irsa
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, get_config, load_cache, save_cache, PROJECT_ROOT

log = get_logger("ingestion.neowise_timeseries")

# ── Constants ────────────────────────────────────────────────────────
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
NEOWISE_TABLE = "neowiser_p1bs_psd"

# Default search radius in arcsec — WISE PSF FWHM is ~6" in W1,
# so 3.6" is a conservative match radius
DEFAULT_RADIUS_ARCSEC = 3.6

# Quality cuts (hardened per review)
MIN_QUAL_FRAME = 5   # minimum qual_frame value; >=5 selects best-quality frames
MAX_W1_ERR = 0.2     # maximum acceptable W1 uncertainty (mag)
MAX_W2_ERR = 0.3     # maximum acceptable W2 uncertainty (mag)

# ── NEOWISE calibration constants (from Perplexity research brief) ────
# W2 bad epoch: MJD 57000-57071 had incorrect zero-point correction
# (Meisner & Finkbeiner 2014). Data in this range shows 34 mmag seasonal
# systematics in W2 that are NOT real astrophysics.
W2_BAD_MJD_START = 57000
W2_BAD_MJD_END = 57071

# Pre-NEOWISE reactivation: data before MJD 56700 (mid-Feb 2014) has
# ±1.6% flat-field residuals from the early reactivation period
NEOWISE_STABLE_START_MJD = 56700

# Photometric noise floors (survey-wide, per-epoch single-exposure)
# W1: 2.6 mmag floor for bright sources
# W2: 6.1 mmag floor, plus 34 mmag seasonal variation
W1_NOISE_FLOOR_MMAG = 2.6
W2_NOISE_FLOOR_MMAG = 6.1


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class NEOWISEEpoch:
    """A single NEOWISE observation epoch."""
    mjd: float
    w1_mag: float
    w1_err: float
    w2_mag: float
    w2_err: float
    ra: float
    dec: float
    qual_frame: int = 0


@dataclass
class NEOWISETimeSeries:
    """Full NEOWISE time-series for a single target."""
    target_ra: float
    target_dec: float
    n_epochs: int
    epochs: List[NEOWISEEpoch] = field(default_factory=list)
    mjd: np.ndarray = field(default_factory=lambda: np.array([]))
    w1_mag: np.ndarray = field(default_factory=lambda: np.array([]))
    w1_err: np.ndarray = field(default_factory=lambda: np.array([]))
    w2_mag: np.ndarray = field(default_factory=lambda: np.array([]))
    w2_err: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_w1: float = 0.0
    mean_w2: float = 0.0
    std_w1: float = 0.0
    std_w2: float = 0.0
    time_baseline_years: float = 0.0
    data_source: str = "none"  # "real", "simulated", "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_ra": self.target_ra,
            "target_dec": self.target_dec,
            "n_epochs": self.n_epochs,
            "mean_w1": self.mean_w1,
            "mean_w2": self.mean_w2,
            "std_w1": self.std_w1,
            "std_w2": self.std_w2,
            "time_baseline_years": self.time_baseline_years,
            "data_source": self.data_source,
            "mjd": self.mjd.tolist() if len(self.mjd) > 0 else [],
            "w1_mag": self.w1_mag.tolist() if len(self.w1_mag) > 0 else [],
            "w1_err": self.w1_err.tolist() if len(self.w1_err) > 0 else [],
            "w2_mag": self.w2_mag.tolist() if len(self.w2_mag) > 0 else [],
            "w2_err": self.w2_err.tolist() if len(self.w2_err) > 0 else [],
        }


# =====================================================================
#  ADQL query builder
# =====================================================================

def _build_query(ra: float, dec: float, radius_arcsec: float = DEFAULT_RADIUS_ARCSEC) -> str:
    """Build the ADQL query for NEOWISE single-exposure source table."""
    radius_deg = radius_arcsec / 3600.0
    query = (
        f"SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, ra, dec, qual_frame "
        f"FROM {NEOWISE_TABLE} "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.8f})) = 1 "
        f"AND qual_frame >= {MIN_QUAL_FRAME} "
        f"AND w1sigmpro IS NOT NULL "
        f"ORDER BY mjd"
    )
    return query


# =====================================================================
#  Public API
# =====================================================================

def query_neowise_timeseries(
    ra: float,
    dec: float,
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
    use_cache: bool = True,
) -> NEOWISETimeSeries:
    """Retrieve all W1/W2 single-exposure measurements for a target.

    Parameters
    ----------
    ra : float
        Right ascension in degrees (J2000/ICRS).
    dec : float
        Declination in degrees (J2000/ICRS).
    radius_arcsec : float
        Search radius in arcseconds.
    use_cache : bool
        If True, check local cache before querying IRSA.

    Returns
    -------
    NEOWISETimeSeries
        The time-series data with all epochs.
    """
    cache_key = f"neowise_ts_{ra:.6f}_{dec:.6f}_{radius_arcsec:.1f}"

    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            log.info("NEOWISE time-series loaded from cache for (%.4f, %.4f)", ra, dec)
            return _dict_to_timeseries(cached, ra, dec)

    if not HAS_ASTROQUERY:
        log.warning(
            "astroquery not available; returning simulated NEOWISE data for (%.4f, %.4f)",
            ra, dec,
        )
        result = _simulate_neowise(ra, dec)
        # Do NOT cache simulated data — allow retry when IRSA recovers
        return result

    # Build and execute TAP query (with signal-based hard timeout)
    _TAP_HARD_TIMEOUT = 30  # seconds — interrupts even stuck sockets
    query = _build_query(ra, dec, radius_arcsec)
    log.info("Querying IRSA TAP for NEOWISE time-series at (%.4f, %.4f)", ra, dec)

    class _TapTimeoutError(Exception):
        pass

    def _tap_alarm_handler(signum, frame):
        raise _TapTimeoutError(
            f"IRSA TAP query exceeded {_TAP_HARD_TIMEOUT}s hard timeout"
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _tap_alarm_handler)
        signal.alarm(_TAP_HARD_TIMEOUT)
        from astroquery.utils.tap.core import TapPlus
        tap = TapPlus(url=IRSA_TAP_URL)
        job = tap.launch_job(query, verbose=False)
        table = job.get_results()
        signal.alarm(0)  # cancel alarm on success
    except _TapTimeoutError:
        log.warning("IRSA TAP query timed out after %ds for (%.4f, %.4f)",
                     _TAP_HARD_TIMEOUT, ra, dec)
        log.info("Falling back to simulated NEOWISE data")
        result = _simulate_neowise(ra, dec)
        return result
    except Exception as exc:
        signal.alarm(0)  # cancel alarm on other exceptions
        log.error("IRSA TAP query failed: %s", exc)
        log.info("Falling back to simulated NEOWISE data")
        result = _simulate_neowise(ra, dec)
        # Do NOT cache simulated data — allow retry when IRSA recovers
        return result
    finally:
        signal.signal(signal.SIGALRM, old_handler or signal.SIG_DFL)

    if table is None or len(table) == 0:
        log.info("No NEOWISE data found for (%.4f, %.4f)", ra, dec)
        return NEOWISETimeSeries(target_ra=ra, target_dec=dec, n_epochs=0)

    # Parse results — column names may vary between TAP versions.
    # IRSA TAP sometimes returns generic col_0..col_N names instead of
    # the requested column names.  We map them back using SELECT order:
    #   SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, ra, dec, qual_frame
    _EXPECTED_COLS = ["mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro",
                      "ra", "dec", "qual_frame"]

    epochs = []
    try:
        col_names = [c.lower() for c in table.colnames]
    except Exception:
        col_names = []

    # Build a column-name mapping so we can access rows by expected name
    col_map = {}
    if "mjd" in col_names:
        # Columns have proper names — use them directly
        for c in _EXPECTED_COLS:
            col_map[c] = c
    elif len(col_names) == len(_EXPECTED_COLS) and col_names[0].startswith("col_"):
        # Generic col_0..col_7 — map by SELECT position
        log.info("IRSA TAP returned generic column names; remapping by SELECT order")
        for i, expected in enumerate(_EXPECTED_COLS):
            col_map[expected] = col_names[i]
    else:
        log.warning("IRSA TAP returned unexpected columns: %s", col_names)
        log.info("Falling back to simulated NEOWISE data")
        result = _simulate_neowise(ra, dec)
        # Do NOT cache simulated data — allow retry when IRSA recovers
        return result

    for row in table:
        try:
            epoch = NEOWISEEpoch(
                mjd=float(row[col_map["mjd"]]),
                w1_mag=float(row[col_map["w1mpro"]]),
                w1_err=float(row[col_map["w1sigmpro"]]),
                w2_mag=float(row[col_map["w2mpro"]]) if row[col_map["w2mpro"]] is not None else np.nan,
                w2_err=float(row[col_map["w2sigmpro"]]) if row[col_map["w2sigmpro"]] is not None else np.nan,
                ra=float(row[col_map["ra"]]),
                dec=float(row[col_map["dec"]]),
                qual_frame=int(row[col_map["qual_frame"]]),
            )
            # Quality cuts: W1 error, W2 error, qual_frame
            w2_ok = np.isnan(epoch.w2_mag) or epoch.w2_err <= MAX_W2_ERR
            if (epoch.w1_err <= MAX_W1_ERR
                    and w2_ok
                    and epoch.qual_frame >= MIN_QUAL_FRAME):
                epochs.append(epoch)
        except (ValueError, TypeError, KeyError):
            continue

    result = _build_timeseries(ra, dec, epochs)
    result.data_source = "real"
    log.info(
        "NEOWISE time-series: %d epochs over %.1f years for (%.4f, %.4f)",
        result.n_epochs, result.time_baseline_years, ra, dec,
    )

    if use_cache:
        save_cache(cache_key, result.to_dict())

    return result


def get_neowise_lightcurve(
    ra: float,
    dec: float,
    sigma_clip: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cleaned, outlier-rejected W1/W2 light curves.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in degrees.
    sigma_clip : float
        Reject points deviating more than this many sigma from the median.

    Returns
    -------
    mjd, w1, w1_err, w2, w2_err : np.ndarray
        Cleaned time-series arrays.
    """
    ts = query_neowise_timeseries(ra, dec)

    if ts.n_epochs == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty

    mjd = ts.mjd.copy()
    w1 = ts.w1_mag.copy()
    w1_err = ts.w1_err.copy()
    w2 = ts.w2_mag.copy()
    w2_err = ts.w2_err.copy()

    # Sigma clip on W1
    med_w1 = np.nanmedian(w1)
    mad_w1 = np.nanmedian(np.abs(w1 - med_w1))
    std_w1 = 1.4826 * mad_w1 if mad_w1 > 0 else np.nanstd(w1)

    mask = np.abs(w1 - med_w1) < sigma_clip * std_w1
    mask &= np.isfinite(w1) & np.isfinite(mjd)

    return mjd[mask], w1[mask], w1_err[mask], w2[mask], w2_err[mask]


def batch_query(
    targets: List[Dict[str, float]],
    delay_sec: float = 0.5,
) -> Dict[str, NEOWISETimeSeries]:
    """Query NEOWISE time-series for multiple targets.

    Parameters
    ----------
    targets : list of dict
        Each dict must have 'ra', 'dec', and optionally 'source_id'.
    delay_sec : float
        Delay between queries to be polite to IRSA.

    Returns
    -------
    dict
        source_id -> NEOWISETimeSeries
    """
    results = {}
    for i, tgt in enumerate(targets):
        ra = tgt["ra"]
        dec = tgt["dec"]
        sid = tgt.get("source_id", f"target_{i:04d}")

        log.info("Batch query %d/%d: %s (%.4f, %.4f)", i + 1, len(targets), sid, ra, dec)
        ts = query_neowise_timeseries(ra, dec)
        results[sid] = ts

        if delay_sec > 0 and i < len(targets) - 1:
            _time.sleep(delay_sec)

    return results


# =====================================================================
#  Internal helpers
# =====================================================================

def _filter_bad_epochs(epochs: List[NEOWISEEpoch]) -> List[NEOWISEEpoch]:
    """Filter out NEOWISE epochs with known calibration issues.

    Removes:
    1. Epochs before NEOWISE reactivation stabilised (MJD < 56700)
       — ±1.6% flat-field residuals in early reactivation data
    2. W2 data in MJD 57000-57071 bad zero-point epoch
       — W2 magnitudes are nullified (set to NaN) for this range
       since the zero-point correction was incorrect

    Note: we keep the W1 data for MJD 57000-57071 since W1 was unaffected.
    Only W2 is corrupted in this range.
    """
    filtered = []
    for ep in epochs:
        # Skip pre-stabilisation epochs entirely
        if ep.mjd < NEOWISE_STABLE_START_MJD:
            continue

        # For the W2 bad epoch range: keep epoch but null out W2
        if W2_BAD_MJD_START <= ep.mjd <= W2_BAD_MJD_END:
            ep = NEOWISEEpoch(
                mjd=ep.mjd,
                w1_mag=ep.w1_mag,
                w1_err=ep.w1_err,
                w2_mag=np.nan,   # W2 bad zero-point — discard
                w2_err=np.nan,
                ra=ep.ra,
                dec=ep.dec,
                qual_frame=ep.qual_frame,
            )

        filtered.append(ep)

    n_removed = len(epochs) - len(filtered)
    n_w2_nulled = sum(
        1 for ep in epochs
        if W2_BAD_MJD_START <= ep.mjd <= W2_BAD_MJD_END
        and ep.mjd >= NEOWISE_STABLE_START_MJD
    )
    if n_removed > 0 or n_w2_nulled > 0:
        log.info(
            "NEOWISE calibration filter: removed %d early epochs, "
            "nulled W2 for %d bad-ZP epochs",
            n_removed, n_w2_nulled,
        )

    return filtered


def compute_epoch_averages(ts: NEOWISETimeSeries) -> Dict[str, Any]:
    """Compute epoch-averaged photometry grouped by NEOWISE sky pass.

    NEOWISE scans each sky position ~2 times per year in ~1-day windows,
    with ~12 individual exposures per sky pass. Averaging within each
    pass suppresses per-exposure scatter and provides ~10-20 high-S/N
    data points over the mission baseline — ideal for secular trend
    detection.

    Returns
    -------
    dict with keys:
        epoch_mjd : np.ndarray — mean MJD of each sky pass
        epoch_w1 : np.ndarray — mean W1 mag per pass
        epoch_w1_err : np.ndarray — error on the mean W1
        epoch_w2 : np.ndarray — mean W2 mag per pass (NaN where no valid W2)
        epoch_w2_err : np.ndarray — error on the mean W2
        n_per_epoch : np.ndarray — number of exposures per pass
        n_epochs : int — number of sky passes
    """
    if ts.n_epochs == 0:
        empty = np.array([])
        return {
            "epoch_mjd": empty, "epoch_w1": empty, "epoch_w1_err": empty,
            "epoch_w2": empty, "epoch_w2_err": empty,
            "n_per_epoch": empty, "n_epochs": 0,
        }

    mjd = ts.mjd
    w1 = ts.w1_mag
    w1_err = ts.w1_err
    w2 = ts.w2_mag
    w2_err = ts.w2_err

    # Group by sky pass: consecutive observations within 30 days
    # belong to the same sky pass
    PASS_GAP_DAYS = 30.0

    # Find pass boundaries
    pass_starts = [0]
    for i in range(1, len(mjd)):
        if mjd[i] - mjd[i - 1] > PASS_GAP_DAYS:
            pass_starts.append(i)
    pass_starts.append(len(mjd))  # sentinel

    epoch_mjd = []
    epoch_w1 = []
    epoch_w1_err = []
    epoch_w2 = []
    epoch_w2_err = []
    n_per_epoch = []

    for k in range(len(pass_starts) - 1):
        sl = slice(pass_starts[k], pass_starts[k + 1])
        n = pass_starts[k + 1] - pass_starts[k]
        if n < 2:
            continue  # skip passes with only 1 exposure

        # W1 — weighted mean
        w1_sl = w1[sl]
        w1e_sl = w1_err[sl]
        w1_valid = np.isfinite(w1_sl) & np.isfinite(w1e_sl) & (w1e_sl > 0)
        if np.sum(w1_valid) < 2:
            continue

        w1_w = 1.0 / w1e_sl[w1_valid] ** 2
        w1_mean = np.sum(w1_w * w1_sl[w1_valid]) / np.sum(w1_w)
        w1_mean_err = 1.0 / np.sqrt(np.sum(w1_w))

        # W2 — weighted mean (may be all NaN in bad epoch)
        w2_sl = w2[sl]
        w2e_sl = w2_err[sl]
        w2_valid = np.isfinite(w2_sl) & np.isfinite(w2e_sl) & (w2e_sl > 0)
        if np.sum(w2_valid) >= 2:
            w2_w = 1.0 / w2e_sl[w2_valid] ** 2
            w2_mean = np.sum(w2_w * w2_sl[w2_valid]) / np.sum(w2_w)
            w2_mean_err = 1.0 / np.sqrt(np.sum(w2_w))
        else:
            w2_mean = np.nan
            w2_mean_err = np.nan

        epoch_mjd.append(float(np.mean(mjd[sl])))
        epoch_w1.append(float(w1_mean))
        epoch_w1_err.append(float(w1_mean_err))
        epoch_w2.append(float(w2_mean))
        epoch_w2_err.append(float(w2_mean_err))
        n_per_epoch.append(int(n))

    return {
        "epoch_mjd": np.array(epoch_mjd),
        "epoch_w1": np.array(epoch_w1),
        "epoch_w1_err": np.array(epoch_w1_err),
        "epoch_w2": np.array(epoch_w2),
        "epoch_w2_err": np.array(epoch_w2_err),
        "n_per_epoch": np.array(n_per_epoch),
        "n_epochs": len(epoch_mjd),
    }


def _build_timeseries(ra: float, dec: float, epochs: List[NEOWISEEpoch]) -> NEOWISETimeSeries:
    """Build a NEOWISETimeSeries from a list of epochs."""
    if not epochs:
        return NEOWISETimeSeries(target_ra=ra, target_dec=dec, n_epochs=0)

    # Apply calibration filters (bad epochs, early data)
    epochs = _filter_bad_epochs(epochs)

    if not epochs:
        return NEOWISETimeSeries(target_ra=ra, target_dec=dec, n_epochs=0)

    # Sort by MJD
    epochs.sort(key=lambda e: e.mjd)

    mjd = np.array([e.mjd for e in epochs])
    w1 = np.array([e.w1_mag for e in epochs])
    w1_err = np.array([e.w1_err for e in epochs])
    w2 = np.array([e.w2_mag for e in epochs])
    w2_err = np.array([e.w2_err for e in epochs])

    baseline_days = mjd[-1] - mjd[0]
    baseline_years = baseline_days / 365.25

    return NEOWISETimeSeries(
        target_ra=ra,
        target_dec=dec,
        n_epochs=len(epochs),
        epochs=epochs,
        mjd=mjd,
        w1_mag=w1,
        w1_err=w1_err,
        w2_mag=w2,
        w2_err=w2_err,
        mean_w1=float(np.nanmean(w1)),
        mean_w2=float(np.nanmean(w2)),
        std_w1=float(np.nanstd(w1)),
        std_w2=float(np.nanstd(w2)),
        time_baseline_years=baseline_years,
    )


def _dict_to_timeseries(data: Dict[str, Any], ra: float, dec: float) -> NEOWISETimeSeries:
    """Reconstruct a NEOWISETimeSeries from a cached dict."""
    # Audit fix N1b: JSON null → Python None gives dtype=object.
    # Force float64 so np.isfinite()/np.nanstd() work everywhere downstream.
    mjd = np.asarray(data.get("mjd", []), dtype=np.float64)
    w1 = np.asarray(data.get("w1_mag", []), dtype=np.float64)
    w1_err = np.asarray(data.get("w1_err", []), dtype=np.float64)
    w2 = np.asarray(data.get("w2_mag", []), dtype=np.float64)
    w2_err = np.asarray(data.get("w2_err", []), dtype=np.float64)

    return NEOWISETimeSeries(
        target_ra=ra,
        target_dec=dec,
        n_epochs=len(mjd),
        mjd=mjd,
        w1_mag=w1,
        w1_err=w1_err,
        w2_mag=w2,
        w2_err=w2_err,
        mean_w1=float(data.get("mean_w1", 0)),
        mean_w2=float(data.get("mean_w2", 0)),
        std_w1=float(data.get("std_w1", 0)),
        std_w2=float(data.get("std_w2", 0)),
        time_baseline_years=float(data.get("time_baseline_years", 0)),
        data_source=str(data.get("data_source", "unknown")),
    )


def _simulate_neowise(ra: float, dec: float, n_epochs: int = 250) -> NEOWISETimeSeries:
    """Generate realistic simulated NEOWISE data for testing.

    Simulates ~10 years of NEOWISE observations with seasonal gaps,
    realistic photometric scatter, and the characteristic W1/W2
    observation cadence.
    """
    rng = np.random.default_rng(seed=int(abs(ra * 1000 + dec * 100)) % (2**31))

    # Baseline W1/W2 magnitudes (solar-type star at ~100pc)
    base_w1 = 8.0 + rng.uniform(-1.0, 1.0)
    base_w2 = base_w1 + 0.01 + rng.normal(0, 0.02)

    # Generate observation times: ~6 months visible per year, ~2 visits per week
    mjd_start = 56700.0  # ~ Jan 2014 (NEOWISE reactivation)
    mjd_end = 60700.0    # ~ early 2025

    epochs = []
    current_mjd = mjd_start

    while current_mjd < mjd_end:
        # Observing season: ~180 days
        season_start = current_mjd + rng.uniform(0, 30)
        season_end = season_start + 180 + rng.uniform(-20, 20)

        # ~12-15 epochs per season
        n_season = rng.integers(10, 18)
        season_mjds = np.sort(rng.uniform(season_start, season_end, n_season))

        for mjd in season_mjds:
            w1_noise = rng.normal(0, 0.03)
            w2_noise = rng.normal(0, 0.05)

            # Small slow trend (secular variability)
            years_elapsed = (mjd - mjd_start) / 365.25
            trend = 0.002 * years_elapsed * rng.choice([-1, 1])

            epoch = NEOWISEEpoch(
                mjd=float(mjd),
                w1_mag=float(base_w1 + w1_noise + trend),
                w1_err=float(0.02 + abs(rng.normal(0, 0.01))),
                w2_mag=float(base_w2 + w2_noise + trend),
                w2_err=float(0.04 + abs(rng.normal(0, 0.02))),
                ra=ra + rng.normal(0, 0.0003),
                dec=dec + rng.normal(0, 0.0003),
                qual_frame=rng.integers(1, 10),
            )
            epochs.append(epoch)

        current_mjd = season_end + rng.uniform(150, 200)  # gap between seasons

    result = _build_timeseries(ra, dec, epochs)
    result.data_source = "simulated"
    log.info(
        "Simulated NEOWISE: %d epochs over %.1f years for (%.4f, %.4f)",
        result.n_epochs, result.time_baseline_years, ra, dec,
    )
    return result


# =====================================================================
#  CLI demo
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- NEOWISE Time-Series Ingestion Demo")
    print("=" * 70)

    # Tabby's Star (KIC 8462852)
    ra_tabby = 301.5643
    dec_tabby = 44.4568

    print(f"\n  Querying NEOWISE for Tabby's Star (RA={ra_tabby}, Dec={dec_tabby})")
    ts = query_neowise_timeseries(ra_tabby, dec_tabby)
    print(f"  Epochs:    {ts.n_epochs}")
    print(f"  Baseline:  {ts.time_baseline_years:.1f} years")
    print(f"  Mean W1:   {ts.mean_w1:.3f} mag")
    print(f"  Mean W2:   {ts.mean_w2:.3f} mag")
    print(f"  Std W1:    {ts.std_w1:.4f} mag")
    print(f"  Std W2:    {ts.std_w2:.4f} mag")

    if ts.n_epochs > 0:
        print(f"  MJD range: {ts.mjd[0]:.1f} - {ts.mjd[-1]:.1f}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
