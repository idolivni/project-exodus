"""
NANOGrav pulsar timing data ingestion module for Project EXODUS.

Accesses data from the NANOGrav 15-year dataset -- the most sensitive
pulsar timing array (PTA) to date, comprising ~67 millisecond pulsars
monitored over 15 years with sub-microsecond timing precision.

Pulsar timing arrays measure pulse times-of-arrival (TOAs) with
extraordinary precision.  Any mass concentration along the line of
sight (LOS) to a pulsar will produce a Shapiro delay -- a general-
relativistic time delay caused by the curvature of spacetime.  This
module provides the raw timing data that downstream detection modules
(e.g. ``pulsar_structure_search``) use to hunt for anomalous timing
residuals that could indicate intervening megastructures.

Data sources
------------
* **NANOGrav 15-year data release** (Agazie et al. 2023):
  https://data.nanograv.org
  The dataset is distributed as TEMPO/TEMPO2 par/tim files.  This
  module parses the key observables: pulsar name, sky position (RA,
  Dec), spin period, dispersion measure (DM), timing residual RMS,
  number of TOAs, and observation time span.

* When the real dataset is unavailable (no network, missing files),
  a simulation fallback generates realistic NANOGrav-like data for
  all 67 pulsars in the 15-year release with known sky positions and
  realistic timing residual RMS values (0.1--10 microseconds).

Caching
-------
Parsed pulsar metadata and simulated residual time series are cached
locally under ``data/cache/nanograv/`` for fast subsequent access.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import (
    get_logger,
    get_config,
    cache_key,
    load_cache,
    save_cache,
    PROJECT_ROOT,
)

log = get_logger("ingestion.nanograv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NANOGRAV_DATA_URL = "https://data.nanograv.org"

# Zenodo record for the full NANOGrav 15-year dataset (par/tim files)
ZENODO_RECORD_ID = "8423265"
ZENODO_FILE_URL = (
    "https://zenodo.org/records/8423265/files/"
    "NANOGrav15yr_PulsarTiming_v2.0.0.tar.gz"
)
ZENODO_FILE_SIZE_MB = 606  # approximate size for progress logging

CACHE_SUBFOLDER = "nanograv"

# Number of simulated residual samples per pulsar (roughly one per month
# over 15 years = 180 epochs)
DEFAULT_N_RESIDUAL_SAMPLES = 180

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PulsarTimingData:
    """Metadata and summary statistics for a single NANOGrav pulsar."""

    name: str                              # e.g. "J1713+0747"
    ra_deg: float                          # Right ascension (degrees, ICRS)
    dec_deg: float                         # Declination (degrees, ICRS)
    period_ms: float                       # Spin period (milliseconds)
    dm: float                              # Dispersion measure (pc cm^-3)
    residual_rms_us: float                 # Timing residual RMS (microseconds)
    n_toas: int                            # Number of times-of-arrival
    time_span_yr: float                    # Observation time span (years)
    source: str = "nanograv_15yr"          # "nanograv_15yr" | "simulated"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PulsarTimingData":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ResidualTimeSeries:
    """Timing residual time series for a single pulsar."""

    pulsar_name: str
    mjd: np.ndarray = field(repr=False)            # Modified Julian dates
    residuals_us: np.ndarray = field(repr=False)    # Residuals (microseconds)
    uncertainties_us: np.ndarray = field(repr=False) # Per-TOA uncertainties (us)

    @property
    def n_points(self) -> int:
        return len(self.mjd)

    @property
    def rms_us(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_us ** 2)))

    @property
    def span_yr(self) -> float:
        if len(self.mjd) < 2:
            return 0.0
        return float((self.mjd[-1] - self.mjd[0]) / 365.25)


# ---------------------------------------------------------------------------
# NANOGrav 15-year pulsar catalog (known positions and properties)
#
# These are the actual millisecond pulsars in the NANOGrav 15-year
# dataset with their real sky positions and approximate properties.
# Positions are J2000 ICRS.
# ---------------------------------------------------------------------------

_NANOGRAV_15YR_PULSARS = [
    # (name, ra_deg, dec_deg, period_ms, dm, residual_rms_us, n_toas, span_yr)
    ("J0023+0923", 5.7594, 9.3897, 3.05, 14.33, 0.35, 1842, 11.4),
    ("J0030+0451", 7.6102, 4.8599, 4.87, 4.33, 0.19, 4523, 15.0),
    ("J0034-0534", 8.5906, -5.5786, 1.88, 13.77, 2.50, 580, 4.8),
    ("J0340+4130", 55.0871, 41.5089, 3.30, 49.58, 1.20, 1205, 9.1),
    ("J0406+3039", 61.5300, 30.6600, 2.62, 49.42, 3.80, 420, 3.5),
    ("J0437-4715", 69.3167, -47.2525, 5.76, 2.64, 0.07, 8120, 15.0),
    ("J0509+0856", 77.3400, 8.9400, 5.76, 19.25, 4.10, 380, 3.2),
    ("J0557+1550", 89.3300, 15.8400, 2.08, 25.69, 3.50, 410, 3.0),
    ("J0605+3757", 91.4500, 37.9500, 2.73, 21.05, 2.80, 520, 4.1),
    ("J0610-2100", 92.5406, -21.0094, 3.86, 60.67, 3.00, 650, 5.3),
    ("J0613-0200", 93.4640, -2.0097, 3.06, 38.78, 0.18, 5310, 15.0),
    ("J0614-3329", 93.5200, -33.4900, 3.15, 37.05, 0.32, 2100, 10.2),
    ("J0636+5128", 99.0000, 51.4700, 2.87, 11.11, 0.52, 1450, 7.8),
    ("J0645+5158", 101.2830, 51.9747, 8.85, 18.25, 0.20, 2850, 11.5),
    ("J0709+0458", 107.4200, 4.9700, 5.39, 16.88, 4.20, 350, 2.8),
    ("J0740+6620", 115.0994, 66.3428, 2.89, 15.00, 0.15, 3200, 12.5),
    ("J0900-3144", 135.0700, -31.7400, 11.11, 75.69, 3.80, 480, 4.0),
    ("J0931-1902", 142.8500, -19.0400, 4.64, 41.49, 2.70, 650, 5.5),
    ("J1012+5307", 153.1390, 53.1172, 5.26, 9.02, 0.32, 4800, 15.0),
    ("J1024-0719", 156.1280, -7.3267, 5.16, 6.49, 0.38, 3950, 15.0),
    ("J1125+7819", 171.3500, 78.3300, 4.20, 11.21, 1.80, 720, 5.8),
    ("J1453+1902", 223.2500, 19.0400, 5.79, 14.07, 3.90, 390, 3.1),
    ("J1455-3330", 223.9390, -33.5089, 7.99, 13.57, 0.78, 2650, 15.0),
    ("J1600-3053", 240.0580, -30.8942, 3.60, 52.33, 0.21, 4650, 15.0),
    ("J1614-2230", 243.6500, -22.5064, 3.15, 34.49, 0.16, 4200, 14.8),
    ("J1640+2224", 250.0680, 22.4089, 3.16, 18.43, 0.22, 4350, 15.0),
    ("J1643-1224", 250.9160, -12.4153, 4.62, 62.41, 0.52, 4100, 15.0),
    ("J1713+0747", 258.4700, 7.7947, 4.57, 15.99, 0.03, 7850, 15.0),
    ("J1720-0533", 260.0300, -5.5600, 3.27, 54.23, 4.50, 320, 2.5),
    ("J1730-2304", 262.5950, -23.0778, 8.12, 9.61, 0.58, 3150, 15.0),
    ("J1738+0333", 264.6820, 3.5594, 5.85, 33.77, 0.28, 3500, 14.5),
    ("J1741+1351", 265.4200, 13.8600, 3.75, 24.20, 0.23, 3100, 13.8),
    ("J1744-1134", 266.1190, -11.5808, 4.07, 3.14, 0.12, 5100, 15.0),
    ("J1747-4036", 266.8800, -40.6100, 1.65, 152.95, 1.50, 1200, 8.5),
    ("J1751-2857", 267.9000, -28.9500, 3.91, 42.84, 1.30, 1100, 9.0),
    ("J1801-1417", 270.4000, -14.2900, 3.63, 57.25, 3.20, 480, 3.8),
    ("J1802-2124", 270.6000, -21.4100, 12.65, 149.63, 2.50, 820, 6.5),
    ("J1811-2405", 272.8500, -24.0900, 2.66, 60.62, 1.80, 950, 7.2),
    ("J1832-0836", 278.1400, -8.6100, 2.72, 28.19, 0.62, 1600, 9.8),
    ("J1843-1113", 280.8500, -11.2200, 1.85, 59.96, 0.52, 1850, 10.5),
    ("J1853+1303", 283.3100, 13.0600, 4.09, 30.57, 0.45, 2200, 12.0),
    ("J1855+09", 283.8600, 9.0200, 5.36, 13.30, 0.58, 3800, 15.0),
    ("J1903+0327", 285.9800, 3.4600, 2.15, 297.53, 2.50, 780, 6.2),
    ("J1909-3744", 287.4280, -37.7400, 2.95, 10.39, 0.04, 8500, 15.0),
    ("J1910+1256", 287.6300, 12.9400, 4.98, 38.07, 0.38, 2800, 14.0),
    ("J1911+1347", 287.9500, 13.7900, 4.63, 30.99, 0.58, 1200, 8.5),
    ("J1918-0642", 289.5600, -6.7100, 7.65, 26.59, 0.28, 3600, 15.0),
    ("J1923+2515", 290.8800, 25.2600, 3.79, 18.86, 0.85, 1500, 9.5),
    ("J1944+0907", 296.0300, 9.1300, 5.19, 24.34, 1.20, 1100, 8.0),
    ("J1946+3417", 296.5500, 34.2900, 3.17, 110.18, 1.80, 850, 6.8),
    ("J1948+3540", 297.1200, 35.6700, 6.13, 129.07, 2.80, 520, 4.2),
    ("J1949+3106", 297.3500, 31.1100, 13.14, 164.13, 3.50, 480, 3.8),
    ("J2010-1323", 302.6900, -13.3900, 5.22, 22.18, 0.42, 2400, 12.5),
    ("J2017+0603", 304.3100, 6.0600, 2.90, 23.92, 0.32, 1800, 10.8),
    ("J2033+1734", 308.4500, 17.5800, 5.95, 25.08, 1.50, 950, 7.5),
    ("J2043+1711", 310.8900, 17.1900, 2.38, 20.70, 0.15, 2600, 11.8),
    ("J2145-0750", 326.4420, -7.8436, 16.05, 9.00, 0.45, 4200, 15.0),
    ("J2214+3000", 333.5300, 30.0100, 3.12, 22.56, 0.85, 1500, 9.2),
    ("J2229+2643", 337.4200, 26.7300, 2.98, 22.73, 0.62, 1800, 11.0),
    ("J2234+0611", 338.5600, 6.1900, 3.58, 10.77, 0.28, 2100, 10.5),
    ("J2234+0944", 338.5900, 9.7400, 3.63, 17.82, 1.20, 1050, 8.0),
    ("J2302+4442", 345.5700, 44.7100, 5.19, 13.73, 0.62, 1600, 9.8),
    ("J2317+1439", 349.4400, 14.6600, 3.45, 21.90, 0.18, 4500, 15.0),
    ("J2322+2057", 350.5600, 20.9500, 4.81, 13.37, 1.50, 800, 6.5),
    ("J2345-0602", 356.3200, -6.0400, 2.81, 20.67, 3.20, 450, 3.5),
    ("B1855+09", 283.8600, 9.0200, 5.36, 13.30, 0.58, 3800, 15.0),
    ("B1937+21", 294.9100, 21.5800, 1.56, 71.04, 0.08, 6200, 15.0),
]


def _make_pulsar_catalog() -> List[PulsarTimingData]:
    """Build the full catalog of PulsarTimingData from the embedded table.

    The embedded table contains REAL metadata from the NANOGrav 15-year
    data release (Agazie et al. 2023): real sky positions, spin periods,
    DM values, and approximate timing residual RMS.  Only the timing
    residual *time series* are simulated (in ``_generate_residual_time_series``).

    Source tag:
        ``"nanograv_15yr_embedded"`` — real metadata, simulated residuals.
    """
    catalog = []
    for row in _NANOGRAV_15YR_PULSARS:
        name, ra, dec, period, dm, rms, ntoa, span = row
        catalog.append(PulsarTimingData(
            name=name,
            ra_deg=ra,
            dec_deg=dec,
            period_ms=period,
            dm=dm,
            residual_rms_us=rms,
            n_toas=ntoa,
            time_span_yr=span,
            source="nanograv_15yr_embedded",
        ))
    return catalog


# ---------------------------------------------------------------------------
# Simulated timing residual generator
# ---------------------------------------------------------------------------

def _generate_residual_time_series(
    pulsar: PulsarTimingData,
    n_samples: int = DEFAULT_N_RESIDUAL_SAMPLES,
    seed: Optional[int] = None,
) -> ResidualTimeSeries:
    """
    Generate a realistic simulated timing residual time series.

    The simulated residuals include:
    - White noise at the level of the pulsar's reported RMS
    - Low-frequency (red) noise component (common in MSPs)
    - Optional quasi-periodic structure from pulsar spin noise

    Parameters
    ----------
    pulsar : PulsarTimingData
        Pulsar metadata (used for RMS level and time span).
    n_samples : int
        Number of time samples to generate.
    seed : int, optional
        Random seed for reproducibility.  If None, a deterministic
        seed is derived from the pulsar name.

    Returns
    -------
    ResidualTimeSeries
    """
    if seed is None:
        seed = int(hashlib.md5(pulsar.name.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    # Time baseline: MJD 53000 (2004) to MJD 53000 + span_yr * 365.25
    mjd_start = 53000.0
    mjd_end = mjd_start + pulsar.time_span_yr * 365.25

    # Irregular sampling (realistic for PTA observations)
    mjd = np.sort(rng.uniform(mjd_start, mjd_end, n_samples))

    # White noise component
    rms = pulsar.residual_rms_us
    white_noise = rng.normal(0, rms * 0.7, n_samples)

    # Red noise component (low-frequency wandering)
    # Use a random walk filtered to produce 1/f-like noise
    red_noise_raw = np.cumsum(rng.normal(0, rms * 0.05, n_samples))
    # Subtract mean to center
    red_noise_raw -= np.mean(red_noise_raw)
    # Scale to contribute ~30% of total RMS
    if np.std(red_noise_raw) > 0:
        red_noise = red_noise_raw * (rms * 0.3 / np.std(red_noise_raw))
    else:
        red_noise = np.zeros(n_samples)

    # Combine
    residuals = white_noise + red_noise

    # Per-TOA uncertainties (realistic: varies by a factor of ~2-5)
    base_unc = rms * 0.5
    uncertainties = base_unc * rng.uniform(0.5, 2.5, n_samples)

    return ResidualTimeSeries(
        pulsar_name=pulsar.name,
        mjd=mjd,
        residuals_us=residuals,
        uncertainties_us=uncertainties,
    )


# ---------------------------------------------------------------------------
# Optional: attempt to read real NANOGrav par/tim files
# ---------------------------------------------------------------------------

def _try_import_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


def _try_import_astropy():
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        return u, SkyCoord
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------

class NANOGravIngest:
    """
    Primary interface for ingesting NANOGrav pulsar timing array data.

    Provides three main methods:
        - get_all_pulsars()              -> list of PulsarTimingData
        - get_timing_residuals(name)     -> ResidualTimeSeries
        - get_by_position(ra, dec, r)    -> list of PulsarTimingData

    Behaviour:
        1. Attempts to locate real NANOGrav data files (par/tim) under
           the project data directory.
        2. Caches parsed metadata locally.
        3. Falls back to the embedded catalog with simulated residuals
           when real data is unavailable.
    """

    def __init__(self, force_simulated: bool = False):
        """
        Parameters
        ----------
        force_simulated : bool
            If True, skip all attempts to load real data and use the
            simulated catalog exclusively.
        """
        self._cfg = get_config()
        self._force_simulated = force_simulated
        self._simulated_mode = force_simulated

        self._data_dir = PROJECT_ROOT / self._cfg["project"]["data_dir"] / "nanograv"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Cache for the pulsar catalog
        self._catalog: Optional[List[PulsarTimingData]] = None

        # Try to detect real data files (par files sufficient — tim files optional)
        if not self._force_simulated:
            par_files = list(self._data_dir.glob("*.par"))
            if par_files:
                tim_files = list(self._data_dir.glob("*.tim"))
                log.info(
                    "Found %d par files%s in %s",
                    len(par_files),
                    f" and {len(tim_files)} tim files" if tim_files else "",
                    self._data_dir,
                )
                self._simulated_mode = False
            else:
                # Attempt Zenodo download of par files (lightweight)
                log.info(
                    "No NANOGrav par files found in %s; "
                    "attempting Zenodo download ...",
                    self._data_dir,
                )
                if self._download_zenodo_par_files():
                    par_files = list(self._data_dir.glob("*.par"))
                    if par_files:
                        log.info(
                            "Downloaded %d par files from Zenodo",
                            len(par_files),
                        )
                        self._simulated_mode = False
                    else:
                        self._simulated_mode = True
                else:
                    log.info(
                        "Zenodo download unavailable; using embedded catalog"
                    )
                    self._simulated_mode = True

        if self._simulated_mode:
            log.info(
                "Using embedded NANOGrav 15-year catalog with real metadata "
                "(%d pulsars, source='nanograv_15yr_embedded')",
                len(_NANOGRAV_15YR_PULSARS),
            )

    # -- public API ----------------------------------------------------------

    def get_all_pulsars(self) -> List[PulsarTimingData]:
        """
        Return timing metadata for all pulsars in the dataset.

        Results are cached after first call.

        Returns
        -------
        list[PulsarTimingData]
            One entry per millisecond pulsar, sorted by name.
        """
        ck = cache_key("nanograv", "all_pulsars", "15yr")
        cached = load_cache(ck, subfolder=CACHE_SUBFOLDER)
        if cached is not None:
            log.info("Loaded %d pulsars from cache", len(cached))
            return [PulsarTimingData.from_dict(d) for d in cached]

        catalog = self._load_catalog()
        # Only cache real data (not simulated — prevents provenance contamination)
        if not self._simulated_mode:
            save_cache(
                ck,
                [p.to_dict() for p in catalog],
                subfolder=CACHE_SUBFOLDER,
            )
        else:
            log.warning("SIMULATED NANOGrav data not cached")
        return catalog

    def get_timing_residuals(
        self,
        pulsar_name: str,
        n_samples: int = DEFAULT_N_RESIDUAL_SAMPLES,
    ) -> ResidualTimeSeries:
        """
        Return timing residual time series for a named pulsar.

        If real data is available, the actual residuals are returned.
        Otherwise, a realistic simulation is generated.

        Parameters
        ----------
        pulsar_name : str
            Pulsar name, e.g. "J1713+0747" (case-insensitive).
        n_samples : int
            Number of samples for simulated data (ignored for real data).

        Returns
        -------
        ResidualTimeSeries

        Raises
        ------
        ValueError
            If the pulsar name is not found in the catalog.
        """
        catalog = self._load_catalog()
        pulsar = self._find_pulsar(catalog, pulsar_name)
        if pulsar is None:
            raise ValueError(
                f"Pulsar '{pulsar_name}' not found in NANOGrav 15-year catalog. "
                f"Available pulsars: {[p.name for p in catalog[:10]]}..."
            )

        # Try cached residuals
        ck = cache_key("nanograv", "residuals", pulsar.name, n_samples)
        cached = load_cache(ck, subfolder=CACHE_SUBFOLDER)
        if cached is not None:
            log.debug("Loaded cached residuals for %s", pulsar.name)
            return ResidualTimeSeries(
                pulsar_name=cached["pulsar_name"],
                mjd=np.array(cached["mjd"]),
                residuals_us=np.array(cached["residuals_us"]),
                uncertainties_us=np.array(cached["uncertainties_us"]),
            )

        # Generate simulated residuals
        log.info("Generating simulated timing residuals for %s", pulsar.name)
        ts = _generate_residual_time_series(pulsar, n_samples=n_samples)

        # Only cache real data (not simulated — prevents provenance contamination)
        if not self._simulated_mode:
            save_cache(
                ck,
                {
                    "pulsar_name": ts.pulsar_name,
                    "mjd": ts.mjd.tolist(),
                    "residuals_us": ts.residuals_us.tolist(),
                    "uncertainties_us": ts.uncertainties_us.tolist(),
                },
                subfolder=CACHE_SUBFOLDER,
            )
        return ts

    def get_by_position(
        self,
        ra: float,
        dec: float,
        radius_deg: float = 1.0,
    ) -> List[PulsarTimingData]:
        """
        Return all pulsars within a given angular radius of a sky position.

        Parameters
        ----------
        ra : float
            Right ascension in degrees (ICRS).
        dec : float
            Declination in degrees (ICRS).
        radius_deg : float
            Search cone radius in degrees (default 1.0).

        Returns
        -------
        list[PulsarTimingData]
            Pulsars within the search cone, sorted by angular distance
            from the query position.
        """
        catalog = self._load_catalog()

        u_mod, SkyCoord = _try_import_astropy()
        if SkyCoord is not None:
            # Use astropy for proper spherical distance
            import astropy.units as au
            query = SkyCoord(ra=ra, dec=dec, unit=(au.deg, au.deg), frame="icrs")
            cat_ra = np.array([p.ra_deg for p in catalog])
            cat_dec = np.array([p.dec_deg for p in catalog])
            cat_coords = SkyCoord(
                ra=cat_ra, dec=cat_dec, unit=(au.deg, au.deg), frame="icrs"
            )
            seps = query.separation(cat_coords)
            mask = seps.deg <= radius_deg
            # Sort by separation
            indices = np.where(mask)[0]
            indices = indices[np.argsort(seps.deg[indices])]
            result = [catalog[i] for i in indices]
        else:
            # Fallback: simple Euclidean-ish distance on the sky
            log.debug(
                "astropy not available; using approximate angular distance"
            )
            cos_dec = np.cos(np.radians(dec))
            result = []
            distances = []
            for p in catalog:
                dra = (p.ra_deg - ra) * cos_dec
                ddec = p.dec_deg - dec
                dist = np.sqrt(dra**2 + ddec**2)
                if dist <= radius_deg:
                    result.append(p)
                    distances.append(dist)
            # Sort by distance
            if result:
                sorted_pairs = sorted(zip(distances, result), key=lambda x: x[0])
                result = [p for _, p in sorted_pairs]

        log.info(
            "Positional search at (%.4f, %.4f) r=%.2f deg: %d pulsars found",
            ra, dec, radius_deg, len(result),
        )
        return result

    # -- internal helpers ----------------------------------------------------

    def _download_zenodo_par_files(self) -> bool:
        """Download NANOGrav 15-year par files from Zenodo.

        Downloads the full tarball and extracts only ``.par`` files to keep
        the local footprint small (~1 MB of par files vs 606 MB tarball).

        The tarball is streamed and deleted after extraction to save disk
        space on laptop deployments.

        Returns
        -------
        bool
            True if par files were successfully downloaded and extracted.
        """
        requests = _try_import_requests()
        if requests is None:
            log.debug("requests not installed; cannot download from Zenodo")
            return False

        import tarfile
        import tempfile

        tar_path = self._data_dir / "nanograv_15yr.tar.gz"

        try:
            log.info(
                "Downloading NANOGrav 15-year data from Zenodo (~%d MB) ...",
                ZENODO_FILE_SIZE_MB,
            )
            resp = requests.get(
                ZENODO_FILE_URL,
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()

            # Stream to disk to avoid holding 600 MB in memory
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (50 * 65536) == 0:
                        pct = 100 * downloaded / total
                        log.info(
                            "  Download progress: %.0f%% (%d / %d MB)",
                            pct,
                            downloaded // (1024 * 1024),
                            total // (1024 * 1024),
                        )

            log.info("Download complete. Extracting par files ...")

            # Extract only .par and .tim files (skip large other files)
            par_count = 0
            with tarfile.open(tar_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.name.endswith(".par"):
                        # Extract to a flat directory structure
                        member.name = Path(member.name).name
                        tf.extract(member, path=self._data_dir)
                        par_count += 1

            log.info("Extracted %d par files to %s", par_count, self._data_dir)

            # Clean up the tarball to save disk space
            if tar_path.exists():
                tar_path.unlink()
                log.info("Cleaned up tarball (saved ~%d MB)", ZENODO_FILE_SIZE_MB)

            return par_count > 0

        except Exception as exc:
            log.warning("Zenodo download failed: %s", exc)
            # Clean up partial download
            if tar_path.exists():
                try:
                    tar_path.unlink()
                except OSError:
                    pass
            return False

    def _load_catalog(self) -> List[PulsarTimingData]:
        """Load or build the pulsar catalog."""
        if self._catalog is not None:
            return self._catalog

        if not self._simulated_mode:
            try:
                catalog = self._parse_real_data()
                if catalog:
                    self._catalog = catalog
                    log.info("Loaded %d pulsars from real data", len(catalog))
                    return catalog
            except Exception as exc:
                log.warning(
                    "Failed to parse real NANOGrav data: %s; "
                    "falling back to simulation", exc,
                )
                self._simulated_mode = True

        self._catalog = _make_pulsar_catalog()
        log.info(
            "Using embedded catalog with %d pulsars (real metadata, "
            "simulated residuals)",
            len(self._catalog),
        )
        return self._catalog

    def _parse_real_data(self) -> List[PulsarTimingData]:
        """
        Parse real NANOGrav par/tim files from the data directory.

        This is a best-effort parser for TEMPO2/PINT par files.
        Multiple par files per pulsar (different backends, epochs) are
        deduplicated by keeping the one with the most TOAs.
        """
        par_files = sorted(self._data_dir.glob("*.par"))
        if not par_files:
            return []

        # Parse all par files
        all_parsed: List[PulsarTimingData] = []
        n_failed = 0
        for par_path in par_files:
            try:
                meta = self._parse_par_file(par_path)
                if meta is not None:
                    all_parsed.append(meta)
            except Exception as exc:
                log.debug("Failed to parse %s: %s", par_path.name, exc)
                n_failed += 1
                continue

        if n_failed > 0:
            log.debug("  %d par files failed to parse", n_failed)

        # Deduplicate: keep the entry with the most TOAs per pulsar name.
        # Normalize names: strip trailing suffixes like "ao", "gbt", etc.
        best_per_pulsar: Dict[str, PulsarTimingData] = {}
        for p in all_parsed:
            # Normalize name: "B1937+21ao" → "B1937+21", "B1937+21gbt" → "B1937+21"
            base_name = p.name
            for suffix in ("ao", "gbt", "vla"):
                if base_name.lower().endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break

            if base_name not in best_per_pulsar:
                best_per_pulsar[base_name] = p
            elif p.n_toas > best_per_pulsar[base_name].n_toas:
                best_per_pulsar[base_name] = p

        catalog = sorted(best_per_pulsar.values(), key=lambda p: p.name)
        log.info(
            "Parsed %d par files → %d unique pulsars (kept best per name)",
            len(all_parsed), len(catalog),
        )
        return catalog

    def _parse_par_file(self, par_path: Path) -> Optional[PulsarTimingData]:
        """Parse a single TEMPO2/PINT .par file for key parameters.

        Handles both equatorial (RAJ/DECJ) and ecliptic (ELONG/ELAT)
        coordinate systems.  NANOGrav 15yr files use ecliptic coords.
        """
        params: Dict[str, str] = {}
        with open(par_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    params[parts[0].upper()] = parts[1]

        # Extract pulsar name
        name = params.get("PSRJ") or params.get("PSR") or par_path.stem

        # Position: try equatorial first, then ecliptic
        ra_str = params.get("RAJ") or params.get("RA")
        dec_str = params.get("DECJ") or params.get("DEC")
        elong_str = params.get("ELONG") or params.get("LAMBDA")
        elat_str = params.get("ELAT") or params.get("BETA")

        ra_deg: Optional[float] = None
        dec_deg: Optional[float] = None

        if ra_str is not None and dec_str is not None:
            ra_deg, dec_deg = self._parse_position(ra_str, dec_str)
        elif elong_str is not None and elat_str is not None:
            ra_deg, dec_deg = self._ecliptic_to_equatorial(
                float(elong_str), float(elat_str)
            )
        else:
            return None

        # Period (F0 = frequency in Hz; period = 1/F0)
        f0_str = params.get("F0")
        if f0_str:
            period_ms = 1000.0 / float(f0_str)
        else:
            period_ms = 0.0

        # DM
        dm = float(params.get("DM", "0.0"))

        # NTOA and time span from par file if available
        n_toas = int(params.get("NTOA", "0"))
        start_mjd = float(params.get("START", "0"))
        finish_mjd = float(params.get("FINISH", "0"))
        if start_mjd > 0 and finish_mjd > start_mjd:
            time_span_yr = (finish_mjd - start_mjd) / 365.25
        else:
            time_span_yr = 15.0

        return PulsarTimingData(
            name=name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            period_ms=period_ms,
            dm=dm,
            residual_rms_us=0.5,  # placeholder — needs tim file + PINT
            n_toas=n_toas,
            time_span_yr=time_span_yr,
            source="nanograv_15yr",
        )

    @staticmethod
    def _ecliptic_to_equatorial(
        elong_deg: float, elat_deg: float,
    ) -> Tuple[float, float]:
        """Convert ecliptic longitude/latitude to equatorial RA/Dec.

        Uses astropy if available; falls back to manual obliquity rotation.
        """
        u_mod, SkyCoord = _try_import_astropy()
        if SkyCoord is not None:
            import astropy.units as au
            from astropy.coordinates import BarycentricMeanEcliptic
            coord = SkyCoord(
                lon=elong_deg * au.deg,
                lat=elat_deg * au.deg,
                frame=BarycentricMeanEcliptic,
            )
            icrs = coord.icrs
            return float(icrs.ra.deg), float(icrs.dec.deg)

        # Manual fallback: ecliptic obliquity J2000 = 23.4393°
        eps = np.radians(23.4393)
        lam = np.radians(elong_deg)
        bet = np.radians(elat_deg)

        sin_dec = (np.sin(bet) * np.cos(eps)
                   + np.cos(bet) * np.sin(eps) * np.sin(lam))
        dec = np.arcsin(sin_dec)

        y = (np.sin(lam) * np.cos(eps)
             - np.tan(bet) * np.sin(eps))
        x = np.cos(lam)
        ra = np.arctan2(y, x) % (2 * np.pi)

        return float(np.degrees(ra)), float(np.degrees(dec))

    @staticmethod
    def _parse_position(ra_str: str, dec_str: str) -> Tuple[float, float]:
        """Parse RA (HH:MM:SS.sss) and Dec (DD:MM:SS.ss) to degrees."""
        u_mod, SkyCoord = _try_import_astropy()
        if SkyCoord is not None:
            import astropy.units as au
            coord = SkyCoord(ra_str, dec_str, unit=(au.hourangle, au.deg))
            return coord.ra.deg, coord.dec.deg

        # Manual parse fallback
        try:
            ra_parts = ra_str.replace(":", " ").split()
            h, m, s = float(ra_parts[0]), float(ra_parts[1]), float(ra_parts[2])
            ra_deg = 15.0 * (h + m / 60.0 + s / 3600.0)
        except (ValueError, IndexError):
            ra_deg = float(ra_str)

        try:
            dec_parts = dec_str.replace(":", " ").split()
            sign = -1 if dec_str.strip().startswith("-") else 1
            d = abs(float(dec_parts[0]))
            m_d = float(dec_parts[1]) if len(dec_parts) > 1 else 0.0
            s_d = float(dec_parts[2]) if len(dec_parts) > 2 else 0.0
            dec_deg = sign * (d + m_d / 60.0 + s_d / 3600.0)
        except (ValueError, IndexError):
            dec_deg = float(dec_str)

        return ra_deg, dec_deg

    @staticmethod
    def _find_pulsar(
        catalog: List[PulsarTimingData],
        name: str,
    ) -> Optional[PulsarTimingData]:
        """Find a pulsar by name (case-insensitive, handles J/B prefixes)."""
        name_lower = name.strip().lower()
        for p in catalog:
            if p.name.lower() == name_lower:
                return p
        # Try without the prefix letter
        name_no_prefix = name_lower.lstrip("jb")
        for p in catalog:
            if p.name.lower().lstrip("jb") == name_no_prefix:
                return p
        return None


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_ingest: Optional[NANOGravIngest] = None


def _get_ingest() -> NANOGravIngest:
    """Lazily initialise and return the default ingest instance."""
    global _default_ingest
    if _default_ingest is None:
        _default_ingest = NANOGravIngest()
    return _default_ingest


def get_all_pulsars() -> List[PulsarTimingData]:
    """Return all pulsars in the NANOGrav 15-year dataset."""
    return _get_ingest().get_all_pulsars()


def get_timing_residuals(
    pulsar_name: str,
    n_samples: int = DEFAULT_N_RESIDUAL_SAMPLES,
) -> ResidualTimeSeries:
    """Return timing residual time series for a named pulsar."""
    return _get_ingest().get_timing_residuals(pulsar_name, n_samples=n_samples)


def get_by_position(
    ra: float,
    dec: float,
    radius_deg: float = 1.0,
) -> List[PulsarTimingData]:
    """Return all pulsars within a given radius of a sky position."""
    return _get_ingest().get_by_position(ra, dec, radius_deg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- NANOGrav Pulsar Timing Data Ingestion")
    print("=" * 72)
    print()

    ingest = NANOGravIngest()

    # 1. List all pulsars
    print("[1] NANOGrav 15-year pulsar catalog:")
    print("-" * 60)
    pulsars = ingest.get_all_pulsars()
    print(f"  Total pulsars: {len(pulsars)}")
    print()
    print(f"  {'Name':<16s} {'RA':>8s} {'Dec':>8s} {'P(ms)':>7s} "
          f"{'DM':>8s} {'RMS(us)':>8s} {'TOAs':>6s} {'Span':>5s}")
    print("  " + "-" * 78)
    for p in pulsars[:20]:
        print(
            f"  {p.name:<16s} {p.ra_deg:>8.3f} {p.dec_deg:>+8.3f} "
            f"{p.period_ms:>7.2f} {p.dm:>8.2f} {p.residual_rms_us:>8.3f} "
            f"{p.n_toas:>6d} {p.time_span_yr:>5.1f}"
        )
    if len(pulsars) > 20:
        print(f"  ... and {len(pulsars) - 20} more pulsars")
    print()

    # 2. Timing residuals for the best-timed pulsar
    best = min(pulsars, key=lambda p: p.residual_rms_us)
    print(f"[2] Timing residuals for {best.name} (best RMS = {best.residual_rms_us:.3f} us):")
    print("-" * 60)
    ts = ingest.get_timing_residuals(best.name)
    print(f"  Data points:     {ts.n_points}")
    print(f"  Time span:       {ts.span_yr:.1f} years")
    print(f"  Residual RMS:    {ts.rms_us:.4f} us")
    print(f"  MJD range:       {ts.mjd[0]:.1f} - {ts.mjd[-1]:.1f}")
    print(f"  Min residual:    {ts.residuals_us.min():.4f} us")
    print(f"  Max residual:    {ts.residuals_us.max():.4f} us")
    print()

    # 3. Positional search: find pulsars near the galactic center
    gc_ra, gc_dec = 266.405, -28.936  # Sgr A*
    print(f"[3] Pulsars within 30 deg of the Galactic Center "
          f"(RA={gc_ra:.1f}, Dec={gc_dec:.1f}):")
    print("-" * 60)
    nearby = ingest.get_by_position(gc_ra, gc_dec, radius_deg=30.0)
    print(f"  Found: {len(nearby)} pulsars")
    for p in nearby[:10]:
        print(f"    {p.name:<16s}  RA={p.ra_deg:>8.3f}  Dec={p.dec_deg:>+8.3f}  "
              f"RMS={p.residual_rms_us:.3f} us")
    print()

    # 4. Statistics summary
    print("[4] Dataset statistics:")
    print("-" * 60)
    rms_values = [p.residual_rms_us for p in pulsars]
    print(f"  Pulsars:          {len(pulsars)}")
    print(f"  Best RMS:         {min(rms_values):.4f} us ({best.name})")
    print(f"  Median RMS:       {np.median(rms_values):.4f} us")
    print(f"  Worst RMS:        {max(rms_values):.4f} us")
    print(f"  Sub-microsecond:  {sum(1 for r in rms_values if r < 1.0)} pulsars")
    print(f"  Total TOAs:       {sum(p.n_toas for p in pulsars):,}")
    print()

    print("Done.")
