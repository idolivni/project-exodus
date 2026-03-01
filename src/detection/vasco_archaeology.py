"""
VASCO Temporal Archaeology — NEOWISE IR analysis at vanished-star positions.

This module implements a targeted search for infrared emission at positions
where optical sources have vanished (VASCO catalog: Villarroel et al. 2020,
AJ 159:8). The physics case for Dyson spheres:

  1. An advanced civilization encloses a star → optical luminosity drops
  2. Waste heat is radiated in the mid-infrared → WISE/NEOWISE detects IR
  3. Signature: optical source VANISHES but IR source PERSISTS or BRIGHTENS

This is the *exact* predicted Dyson sphere observational signature. VASCO's
127 candidates (28 highest-quality in Table 3) are positions where USNO B1.0
sources are missing in Pan-STARRS DR1.

What this module does
---------------------
For each VASCO vanishing position:
  1. Query NEOWISE multi-epoch photometry (W1/W2, 10+ years, 2013-2024)
  2. Check if ANY IR source is detected at the vanished position
  3. If detected: analyze flux trend (brightening = construction phase)
  4. Query radio continuum (FIRST/NVSS) for persistent emission
  5. Cross-check AllWISE (static epoch ~2010) for pre-vanishing IR state
  6. Compute archaeology_score: IR persistence + trend + radio + multi-band

Interpretation
--------------
  - IR detected + optical gone     → HIGH PRIORITY (Dyson sphere candidate)
  - IR brightening over 10yr       → VERY HIGH PRIORITY (active construction)
  - IR + radio detected            → Likely AGN/blazar (but still interesting)
  - No IR, no radio                → Plate artifact or genuine stellar death
  - IR fading                      → Could be dying star, less interesting for SETI

Public API
----------
    analyze_vanished_star(ra, dec, **kwargs) -> VanishedStarResult
    batch_analyze(targets) -> dict[str, VanishedStarResult]

Usage
-----
    python -m src.detection.vasco_archaeology \\
        --targets data/targets/vasco_vanishing_28.json \\
        --output data/reports/vasco_archaeology.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("detection.vasco_archaeology")


# ── Result dataclass ─────────────────────────────────────────────────

@dataclass
class VanishedStarResult:
    """Result of temporal archaeology for a single VASCO vanished position."""
    target_id: str = ""

    # NEOWISE detection
    neowise_detected: bool = False      # Any IR source within search radius?
    neowise_n_epochs: int = 0           # Number of good NEOWISE epochs
    neowise_baseline_yr: float = 0.0    # Time span of observations
    neowise_data_source: str = "none"   # "real", "none", "insufficient"

    # W1/W2 photometry (means over all epochs)
    w1_mean: float = 0.0               # Mean W1 magnitude
    w1_std: float = 0.0                # W1 scatter
    w2_mean: float = 0.0               # Mean W2 magnitude
    w2_std: float = 0.0                # W2 scatter

    # Secular trend (brightening = NEGATIVE slope in magnitudes)
    w1_trend_mag_per_yr: float = 0.0   # Linear trend in W1 mag/yr
    w1_trend_sigma: float = 0.0        # Significance of W1 trend
    w2_trend_mag_per_yr: float = 0.0
    w2_trend_sigma: float = 0.0
    is_brightening: bool = False       # Secular IR brightening detected?

    # Variability
    w1_excess_scatter: float = 0.0     # Scatter / median_error - 1
    is_variable: bool = False          # Excess scatter > 3x noise

    # AllWISE (static, ~2010 epoch)
    allwise_detected: bool = False
    allwise_w1: Optional[float] = None
    allwise_w2: Optional[float] = None
    allwise_w3: Optional[float] = None
    allwise_w4: Optional[float] = None

    # Radio continuum
    radio_detected: bool = False
    radio_flux_mJy: Optional[float] = None
    radio_survey: str = "none"         # "FIRST", "NVSS", "none"

    # Archaeology score (0-1, higher = more interesting)
    archaeology_score: float = 0.0

    # Classification
    classification: str = "UNKNOWN"
    # Possible: DYSON_CANDIDATE, BRIGHTENING_IR, PERSISTENT_IR,
    #           AGN_LIKELY, NO_DETECTION, PLATE_ARTIFACT, FADING_SOURCE
    priority: str = "LOW"
    # Possible: CRITICAL, HIGH, MODERATE, LOW
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "neowise_detected": self.neowise_detected,
            "neowise_n_epochs": self.neowise_n_epochs,
            "neowise_baseline_yr": round(self.neowise_baseline_yr, 2),
            "neowise_data_source": self.neowise_data_source,
            "w1_mean": round(self.w1_mean, 3) if self.w1_mean else 0.0,
            "w1_std": round(self.w1_std, 5) if self.w1_std else 0.0,
            "w2_mean": round(self.w2_mean, 3) if self.w2_mean else 0.0,
            "w2_std": round(self.w2_std, 5) if self.w2_std else 0.0,
            "w1_trend_mag_per_yr": round(self.w1_trend_mag_per_yr, 6),
            "w1_trend_sigma": round(self.w1_trend_sigma, 2),
            "w2_trend_mag_per_yr": round(self.w2_trend_mag_per_yr, 6),
            "w2_trend_sigma": round(self.w2_trend_sigma, 2),
            "is_brightening": self.is_brightening,
            "w1_excess_scatter": round(self.w1_excess_scatter, 3),
            "is_variable": self.is_variable,
            "allwise_detected": self.allwise_detected,
            "allwise_w1": self.allwise_w1,
            "allwise_w2": self.allwise_w2,
            "allwise_w3": self.allwise_w3,
            "allwise_w4": self.allwise_w4,
            "radio_detected": self.radio_detected,
            "radio_flux_mJy": self.radio_flux_mJy,
            "radio_survey": self.radio_survey,
            "archaeology_score": round(self.archaeology_score, 4),
            "classification": self.classification,
            "priority": self.priority,
            "note": self.note,
        }


# ── Thresholds ───────────────────────────────────────────────────────

NEOWISE_SEARCH_RADIUS_ARCSEC = 6.0   # WISE beam ~6" FWHM
MIN_NEOWISE_EPOCHS = 10               # Need at least 10 epochs
TREND_SIGMA_THRESH = 3.0              # Significant secular trend
EXCESS_SCATTER_THRESH = 3.0           # Variable at 3x expected noise
ALLWISE_SEARCH_RADIUS_ARCSEC = 6.0    # Match radius for AllWISE
RADIO_SEARCH_RADIUS_ARCSEC = 15.0     # Wider for radio (FIRST 5", NVSS 45")


# ── Main analysis ────────────────────────────────────────────────────

def analyze_vanished_star(
    ra: float,
    dec: float,
    target_id: str = "unknown",
    use_cache: bool = True,
    thresholds: Optional[Dict[str, Any]] = None,
) -> VanishedStarResult:
    """Analyze a single VASCO vanished-star position.

    Queries NEOWISE time-series, AllWISE static photometry, and radio
    continuum catalogs at the vanished position.

    Parameters
    ----------
    ra, dec : float
        Position in degrees (USNO B1.0 epoch).
    target_id : str
        Identifier for this target.
    use_cache : bool
        Whether to use cached data.

    Returns
    -------
    VanishedStarResult
    """
    result = VanishedStarResult(target_id=target_id)

    # Audit fix N6: allow blind protocol to inject locked thresholds.
    # If thresholds dict is provided, use those instead of module constants.
    _min_epochs = (thresholds or {}).get("min_neowise_epochs", MIN_NEOWISE_EPOCHS)
    _excess_thresh = (thresholds or {}).get("excess_scatter_thresh", EXCESS_SCATTER_THRESH)
    _trend_thresh = (thresholds or {}).get("trend_sigma_thresh", TREND_SIGMA_THRESH)

    # ── 1. NEOWISE time-series ───────────────────────────────────────
    neowise_data = _query_neowise(ra, dec, use_cache=use_cache)

    if neowise_data is not None and neowise_data.n_epochs >= _min_epochs:
        result.neowise_detected = True
        result.neowise_n_epochs = neowise_data.n_epochs
        result.neowise_baseline_yr = neowise_data.time_baseline_years
        result.neowise_data_source = getattr(neowise_data, "data_source", "real")

        # Photometry statistics
        w1 = neowise_data.w1_mag
        w1_err = neowise_data.w1_err
        w2 = neowise_data.w2_mag
        w2_err = neowise_data.w2_err

        valid_w1 = np.isfinite(w1) & np.isfinite(w1_err) & (w1_err > 0)
        if np.sum(valid_w1) >= _min_epochs:
            result.w1_mean = float(np.nanmean(w1[valid_w1]))
            result.w1_std = float(np.nanstd(w1[valid_w1]))

            # Excess scatter
            med_err = np.nanmedian(w1_err[valid_w1])
            result.w1_excess_scatter = float(
                (result.w1_std / max(med_err, 1e-6)) - 1.0
            )

            # Secular trend
            mjd = neowise_data.mjd
            t_years = (mjd[valid_w1] - mjd[valid_w1][0]) / 365.25
            result.w1_trend_mag_per_yr, result.w1_trend_sigma = _weighted_linear_trend(
                t_years, w1[valid_w1], w1_err[valid_w1]
            )

        valid_w2 = np.isfinite(w2) & np.isfinite(w2_err) & (w2_err > 0)
        if np.sum(valid_w2) >= _min_epochs:
            result.w2_mean = float(np.nanmean(w2[valid_w2]))
            result.w2_std = float(np.nanstd(w2[valid_w2]))

            mjd = neowise_data.mjd
            t_years = (mjd[valid_w2] - mjd[valid_w2][0]) / 365.25
            result.w2_trend_mag_per_yr, result.w2_trend_sigma = _weighted_linear_trend(
                t_years, w2[valid_w2], w2_err[valid_w2]
            )

        # Flags
        result.is_variable = result.w1_excess_scatter > _excess_thresh
        # Brightening = NEGATIVE magnitude trend (getting brighter)
        result.is_brightening = (
            result.w1_trend_sigma < -_trend_thresh or
            result.w2_trend_sigma < -_trend_thresh
        )

    elif neowise_data is not None:
        result.neowise_n_epochs = neowise_data.n_epochs
        result.neowise_baseline_yr = neowise_data.time_baseline_years
        result.neowise_data_source = "insufficient"
    else:
        result.neowise_data_source = "none"

    # ── 2. AllWISE static photometry ─────────────────────────────────
    allwise = _query_allwise(ra, dec)
    if allwise is not None:
        result.allwise_detected = True
        result.allwise_w1 = allwise.get("w1mpro")
        result.allwise_w2 = allwise.get("w2mpro")
        result.allwise_w3 = allwise.get("w3mpro")
        result.allwise_w4 = allwise.get("w4mpro")

    # ── 3. Radio continuum ───────────────────────────────────────────
    radio = _query_radio(ra, dec)
    if radio is not None:
        result.radio_detected = True
        result.radio_flux_mJy = radio.get("flux_mJy")
        result.radio_survey = radio.get("survey", "unknown")

    # ── 4. Compute archaeology score and classify ────────────────────
    _classify(result)

    return result


def batch_analyze(
    targets: List[Dict[str, Any]],
    use_cache: bool = True,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, VanishedStarResult]:
    """Analyze a batch of VASCO targets.

    Parameters
    ----------
    targets : list of dict
        Each must have 'ra', 'dec', 'target_id'.
    use_cache : bool

    Returns
    -------
    dict : target_id -> VanishedStarResult
    """
    results = {}
    n = len(targets)
    for i, tgt in enumerate(targets):
        ra = tgt["ra"]
        dec = tgt["dec"]
        tid = tgt.get("target_id", f"VASCO_{i:04d}")

        if (i + 1) % 10 == 0 or i == 0:
            log.info("VASCO archaeology: %d/%d targets", i + 1, n)

        results[tid] = analyze_vanished_star(
            ra, dec, target_id=tid, use_cache=use_cache,
            thresholds=thresholds,
        )

    # Summary
    n_detected = sum(1 for r in results.values() if r.neowise_detected)
    n_bright = sum(1 for r in results.values() if r.is_brightening)
    n_radio = sum(1 for r in results.values() if r.radio_detected)
    n_high = sum(1 for r in results.values() if r.priority in ("CRITICAL", "HIGH"))

    log.info(
        "VASCO archaeology complete: %d targets, %d IR detected, "
        "%d brightening, %d radio, %d HIGH+ priority",
        n, n_detected, n_bright, n_radio, n_high,
    )

    return results


# ── Classification logic ─────────────────────────────────────────────

def _classify(result: VanishedStarResult) -> None:
    """Classify and score a vanished star based on multi-band evidence."""
    score = 0.0
    notes = []

    # Component 1: IR persistence (0-0.3)
    # Optical vanished + IR still there = the core Dyson signature
    if result.neowise_detected:
        score += 0.3
        notes.append(f"IR persistent ({result.neowise_n_epochs} epochs, "
                      f"{result.neowise_baseline_yr:.1f} yr)")
    else:
        notes.append("No IR detection")

    # Component 2: Secular brightening (0-0.3)
    # IR getting brighter = active construction
    if result.is_brightening:
        # Weight by significance
        max_sig = max(abs(result.w1_trend_sigma), abs(result.w2_trend_sigma))
        bright_component = min(0.3, 0.1 * (max_sig / TREND_SIGMA_THRESH))
        score += bright_component
        notes.append(f"IR brightening ({max_sig:.1f}σ)")
    elif result.neowise_detected:
        # Steady IR = still interesting (persistent emission)
        if abs(result.w1_trend_sigma) < 1.0:
            score += 0.05
            notes.append("IR steady (no significant trend)")

    # Component 3: IR variability (0-0.1)
    if result.is_variable:
        score += 0.1
        notes.append(f"IR variable (excess scatter {result.w1_excess_scatter:.1f}x)")

    # Component 4: AllWISE confirmation (0-0.15)
    # If AllWISE (2010) ALSO detected = IR source was there BEFORE vanishing
    if result.allwise_detected:
        score += 0.1
        notes.append(f"AllWISE detection (W1={result.allwise_w1})")
        # Long-wavelength excess (W3/W4 bright relative to W1)
        if result.allwise_w3 is not None and result.allwise_w1 is not None:
            w1_w3 = result.allwise_w1 - result.allwise_w3
            if w1_w3 > 2.0:  # Significant W3 excess
                score += 0.05
                notes.append(f"W1-W3 excess = {w1_w3:.2f}")

    # Component 5: Radio detection (0-0.15)
    if result.radio_detected:
        score += 0.05
        notes.append(f"Radio: {result.radio_flux_mJy:.2f} mJy ({result.radio_survey})")
        # Radio + IR without optical = could be AGN, but also artificial
        if result.neowise_detected:
            score += 0.05
            notes.append("Multi-band non-optical emission")

    result.archaeology_score = min(1.0, score)

    # ── Classification hierarchy ─────────────────────────────────────
    # Use injected thresholds when available (blind protocol); fall back to module constants.
    _score_high = (thresholds or {}).get("archaeology_score_high_threshold", 0.4)
    _trend_dim_thresh = (thresholds or {}).get("trend_sigma_thresh", TREND_SIGMA_THRESH)

    if result.neowise_detected and result.is_brightening and not result.radio_detected:
        result.classification = "BRIGHTENING_IR"
        result.priority = "CRITICAL"
    elif result.neowise_detected and result.is_brightening and result.radio_detected:
        result.classification = "BRIGHTENING_IR"
        result.priority = "HIGH"  # Radio could be AGN, but brightening is rare
    elif result.neowise_detected and not result.radio_detected and score > _score_high:
        result.classification = "DYSON_CANDIDATE"
        result.priority = "HIGH"
    elif result.neowise_detected and result.radio_detected:
        result.classification = "AGN_LIKELY"
        result.priority = "MODERATE"
    elif result.neowise_detected:
        result.classification = "PERSISTENT_IR"
        result.priority = "MODERATE"
    elif result.neowise_data_source == "none":
        # No NEOWISE data at all — could be outside coverage or too faint
        if result.radio_detected:
            result.classification = "RADIO_ONLY"
            result.priority = "LOW"
        else:
            result.classification = "NO_DETECTION"
            result.priority = "LOW"
    else:
        # Insufficient epochs
        result.classification = "INSUFFICIENT_DATA"
        result.priority = "LOW"

    # Dimming IR source (opposite of construction)
    if result.neowise_detected:
        max_dim_sigma = max(result.w1_trend_sigma, result.w2_trend_sigma)
        if max_dim_sigma > _trend_dim_thresh:
            result.classification = "FADING_SOURCE"
            result.priority = "LOW"
            notes.append(f"IR fading ({max_dim_sigma:.1f}σ) — likely natural")

    result.note = "; ".join(notes)


# ── Data query helpers ───────────────────────────────────────────────

def _query_neowise(ra: float, dec: float, use_cache: bool = True):
    """Query NEOWISE time-series at a position.

    Returns NEOWISETimeSeries or None.
    """
    try:
        from src.ingestion.neowise_timeseries import query_neowise_timeseries
        ts = query_neowise_timeseries(ra, dec, use_cache=use_cache)
        # Reject simulated data — we need REAL detections
        if getattr(ts, "data_source", "unknown") == "simulated":
            log.debug("Rejecting simulated NEOWISE data at (%.4f, %.4f)", ra, dec)
            return None
        return ts
    except Exception as exc:
        log.warning("NEOWISE query failed for (%.4f, %.4f): %s", ra, dec, exc)
        return None


def _query_allwise(ra: float, dec: float) -> Optional[Dict]:
    """Query AllWISE for static photometry at a position.

    Returns dict with w1mpro..w4mpro or None.
    """
    try:
        from astroquery.ipac.irsa import Irsa
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        table = Irsa.query_region(
            coord,
            catalog="allwise_p3as_psd",
            spatial="Cone",
            radius=ALLWISE_SEARCH_RADIUS_ARCSEC * u.arcsec,
        )

        if table is None or len(table) == 0:
            return None

        # Take closest source
        row = table[0]
        result = {}
        for col in ("w1mpro", "w2mpro", "w3mpro", "w4mpro"):
            val = row[col] if col in table.colnames else None
            if val is not None and np.isfinite(float(val)):
                result[col] = round(float(val), 3)
            else:
                result[col] = None
        return result

    except Exception as exc:
        log.debug("AllWISE query failed for (%.4f, %.4f): %s", ra, dec, exc)
        return None


def _query_radio(ra: float, dec: float) -> Optional[Dict]:
    """Query FIRST + NVSS for radio continuum at a position.

    Returns dict with flux_mJy and survey name, or None.
    """
    try:
        from src.ingestion.vlass_catalog import query_radio_continuum
        result = query_radio_continuum(ra, dec)
        if result is not None and result.get("detected"):
            return {
                "flux_mJy": result.get("flux_mJy"),
                "survey": result.get("survey", "unknown"),
            }
        return None
    except Exception as exc:
        log.debug("Radio query failed for (%.4f, %.4f): %s", ra, dec, exc)
        return None


def _weighted_linear_trend(
    t: np.ndarray, mag: np.ndarray, err: np.ndarray
) -> tuple:
    """Weighted linear regression: mag = a + b * t.

    Returns (slope_mag_per_year, slope_significance_sigma).
    Negative slope = brightening in magnitudes.
    """
    valid = np.isfinite(t) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if np.sum(valid) < 5:
        return 0.0, 0.0

    t_v = t[valid]
    m_v = mag[valid]
    w = 1.0 / (err[valid] ** 2)

    S = np.sum(w)
    Sx = np.sum(w * t_v)
    Sy = np.sum(w * m_v)
    Sxx = np.sum(w * t_v ** 2)
    Sxy = np.sum(w * t_v * m_v)

    denom = S * Sxx - Sx ** 2
    if abs(denom) < 1e-30:
        return 0.0, 0.0

    slope = (S * Sxy - Sx * Sy) / denom
    slope_err = np.sqrt(S / denom)
    sigma = slope / max(slope_err, 1e-15)

    return float(slope), float(sigma)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VASCO Temporal Archaeology — IR analysis at vanished-star positions"
    )
    parser.add_argument("--targets", required=True,
                        help="Path to VASCO target JSON file")
    parser.add_argument("--output",
                        help="Output JSON path (default: stdout summary)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip cache for fresh queries")
    args = parser.parse_args()

    # Load targets
    with open(args.targets) as f:
        data = json.load(f)

    targets = data.get("targets", data) if isinstance(data, dict) else data
    print(f"{'=' * 70}")
    print(f"  VASCO Temporal Archaeology — {len(targets)} vanished-star positions")
    print(f"{'=' * 70}")

    results = batch_analyze(targets, use_cache=not args.no_cache)

    # Print results
    for tid, r in sorted(results.items(),
                          key=lambda x: -x[1].archaeology_score):
        icon = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MODERATE": "🟡",
            "LOW": "⚪",
        }.get(r.priority, "?")

        ir_str = (f"W1={r.w1_mean:.2f}±{r.w1_std:.3f}" if r.neowise_detected
                  else "no IR")
        trend_str = ""
        if r.is_brightening:
            trend_str = " ↑BRIGHTENING"
        elif r.neowise_detected and abs(r.w1_trend_sigma) > 2:
            trend_str = f" trend={r.w1_trend_sigma:+.1f}σ"

        radio_str = (f" radio={r.radio_flux_mJy:.2f}mJy" if r.radio_detected
                     else "")

        print(f"  {icon} [{r.priority:8s}] {tid}: "
              f"score={r.archaeology_score:.3f} | {ir_str}{trend_str}{radio_str} "
              f"| {r.classification}")
        if r.note:
            print(f"     {r.note}")

    # Summary
    print(f"\n{'=' * 70}")
    n_det = sum(1 for r in results.values() if r.neowise_detected)
    n_bright = sum(1 for r in results.values() if r.is_brightening)
    n_crit = sum(1 for r in results.values() if r.priority == "CRITICAL")
    n_high = sum(1 for r in results.values() if r.priority == "HIGH")
    n_radio = sum(1 for r in results.values() if r.radio_detected)
    print(f"  IR detected: {n_det}/{len(results)}")
    print(f"  Brightening: {n_bright}")
    print(f"  Radio:       {n_radio}")
    print(f"  Priority:    {n_crit} CRITICAL, {n_high} HIGH")
    print(f"{'=' * 70}")

    if args.output:
        out = {
            "campaign": "vasco_archaeology",
            "n_targets": len(results),
            "n_ir_detected": n_det,
            "n_brightening": n_bright,
            "n_critical": n_crit,
            "n_high": n_high,
            "results": {tid: r.to_dict() for tid, r in results.items()},
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n  Saved to: {args.output}")
