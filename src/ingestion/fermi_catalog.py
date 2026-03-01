"""
Fermi 4FGL-DR4 gamma-ray source catalog ingestion for Project EXODUS.

Downloads and parses the Fermi Large Area Telescope 4FGL-DR4 catalog
(14-year source catalog) of gamma-ray sources above 50 MeV.  The catalog
contains ~7000 sources, of which ~30% remain UNIDENTIFIED -- they have no
firm association with any known astrophysical object.

Unidentified gamma-ray sources are of extreme interest for SETI because a
technosignature producing gamma-ray emission (e.g. an energy-intensive
relativistic propulsion beam, or leakage from an antimatter-powered
civilization) would appear as an unidentified point source in Fermi data.
Any positional coincidence between an unidentified Fermi source and a
known exoplanet host star would warrant IMMEDIATE investigation.

Catalog access
--------------
The 4FGL-DR4 FITS file is downloaded from the Fermi Science Support Center:
    https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/

When the real catalog is unavailable (no network, missing astropy.io.fits,
etc.) the module falls back to a simulation mode that generates realistic
Fermi-like catalog data for development and testing.

Key columns extracted
---------------------
- Source_Name:          4FGL source designation
- RAJ2000 / DEJ2000:   J2000 sky position (degrees)
- Flux1000:            Energy flux above 1 GeV (ph/cm2/s)
- PL_Index:            Power-law photon spectral index
- CLASS1:              Source class (empty or "unk" = unidentified)
- Variability_Index:   Variability test statistic
- Conf_95_SemiMajor:   95% positional error ellipse semi-major axis (deg)
- Conf_95_SemiMinor:   95% positional error ellipse semi-minor axis (deg)
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

logger = get_logger("ingestion.fermi_catalog")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FERMI_4FGL_DR4_URL = (
    "https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/"
    "gll_psc_v35.fit"
)
CACHE_SUBFOLDER = "fermi_catalog"
CACHE_KEY_CATALOG = cache_key("fermi", "4fgl_dr4", "v35")
REQUEST_TIMEOUT = 120

# Column names in the 4FGL FITS binary table
_FITS_COLUMNS = [
    "Source_Name",
    "RAJ2000",
    "DEJ2000",
    "Flux1000",
    "PL_Index",
    "CLASS1",
    "Variability_Index",
    "Conf_95_SemiMajor",
    "Conf_95_SemiMinor",
]

# Internal rename map (FITS column -> our key)
_RENAME_MAP = {
    "Source_Name":        "source_name",
    "RAJ2000":            "ra",
    "DEJ2000":            "dec",
    "Flux1000":           "flux_1gev",
    "PL_Index":           "spectral_index",
    "CLASS1":             "source_class",
    "Variability_Index":  "variability_index",
    "Conf_95_SemiMajor":  "pos_err_semimajor_deg",
    "Conf_95_SemiMinor":  "pos_err_semiminor_deg",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FermiSource:
    """A single source from the Fermi 4FGL-DR4 catalog."""

    source_name: str
    ra: float                                 # Right ascension J2000 (deg)
    dec: float                                # Declination J2000 (deg)
    flux_1gev: Optional[float] = None         # Flux > 1 GeV (ph/cm2/s)
    spectral_index: Optional[float] = None    # Photon spectral index
    source_class: Optional[str] = None        # 4FGL source class
    variability_index: Optional[float] = None # Variability test statistic
    pos_err_semimajor_deg: Optional[float] = None  # 95% error semi-major (deg)
    pos_err_semiminor_deg: Optional[float] = None  # 95% error semi-minor (deg)
    is_unidentified: bool = False             # True if CLASS1 is empty/"unk"
    simulated: bool = False                   # True if from simulation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FermiSource":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def pos_err_arcmin(self) -> Optional[float]:
        """Return the average 95% positional uncertainty in arcminutes."""
        if self.pos_err_semimajor_deg is not None and self.pos_err_semiminor_deg is not None:
            return ((self.pos_err_semimajor_deg + self.pos_err_semiminor_deg) / 2.0) * 60.0
        return None


# ---------------------------------------------------------------------------
# Optional / lazy imports
# ---------------------------------------------------------------------------

def _try_import_astropy_fits():
    try:
        from astropy.io import fits
        return fits
    except ImportError:
        return None


def _try_import_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# FITS catalog download and parsing
# ---------------------------------------------------------------------------

def _download_fits(url: str, dest_path: Path) -> Path:
    """Download the 4FGL FITS file to *dest_path*."""
    requests = _try_import_requests()
    if requests is None:
        raise RuntimeError("requests library not installed; cannot download catalog")

    logger.info("Downloading Fermi 4FGL-DR4 catalog from %s", url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    logger.info("Download complete: %s (%d bytes)", dest_path, dest_path.stat().st_size)
    return dest_path


def _parse_fits(fits_path: Path) -> List[FermiSource]:
    """Parse the 4FGL FITS binary table and return a list of FermiSource."""
    fits = _try_import_astropy_fits()
    if fits is None:
        raise RuntimeError("astropy.io.fits not available; cannot parse FITS catalog")

    logger.info("Parsing FITS file: %s", fits_path)

    with fits.open(str(fits_path)) as hdul:
        # The source catalog is typically in the first binary table extension
        # (HDU index 1).  Try to locate it by name or index.
        table_hdu = None
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.columns is not None:
                col_names = [c.name for c in hdu.columns]
                if "Source_Name" in col_names or "RAJ2000" in col_names:
                    table_hdu = hdu
                    break

        if table_hdu is None:
            # Fallback: use the first binary table extension
            table_hdu = hdul[1]

        data = table_hdu.data
        n_rows = len(data)
        logger.info("FITS table has %d rows", n_rows)

        sources: List[FermiSource] = []
        for i in range(n_rows):
            row = data[i]

            # Extract source name (may be bytes in FITS)
            name_raw = row["Source_Name"]
            source_name = name_raw.strip() if isinstance(name_raw, str) else name_raw.decode("utf-8", errors="replace").strip()

            # Extract source class
            class_raw = row["CLASS1"]
            source_class = class_raw.strip() if isinstance(class_raw, str) else class_raw.decode("utf-8", errors="replace").strip()

            # Determine if unidentified
            is_unidentified = (
                source_class == ""
                or source_class.lower() == "unk"
                or source_class.lower() == "unknown"
            )

            # Safely extract numeric columns (handle masked/NaN)
            def _safe_float(val: Any) -> Optional[float]:
                try:
                    v = float(val)
                    return v if np.isfinite(v) else None
                except (ValueError, TypeError):
                    return None

            source = FermiSource(
                source_name=source_name,
                ra=float(row["RAJ2000"]),
                dec=float(row["DEJ2000"]),
                flux_1gev=_safe_float(row["Flux1000"]),
                spectral_index=_safe_float(row["PL_Index"]),
                source_class=source_class if source_class else None,
                variability_index=_safe_float(row["Variability_Index"]),
                pos_err_semimajor_deg=_safe_float(row["Conf_95_SemiMajor"]),
                pos_err_semiminor_deg=_safe_float(row["Conf_95_SemiMinor"]),
                is_unidentified=is_unidentified,
                simulated=False,
            )
            sources.append(source)

        logger.info("Parsed %d Fermi sources (%d unidentified)",
                     len(sources), sum(1 for s in sources if s.is_unidentified))
        return sources


# ---------------------------------------------------------------------------
# Simulated / fallback catalog generator
# ---------------------------------------------------------------------------

# Well-known Fermi source classes and their approximate fractions
_CLASS_DISTRIBUTION = [
    ("bll",  0.22),   # BL Lac type blazar
    ("fsrq", 0.12),   # Flat Spectrum Radio Quasar
    ("psr",  0.10),   # Pulsar
    ("spp",  0.04),   # Supernova remnant / pulsar wind nebula
    ("bcu",  0.08),   # Blazar Candidate of Uncertain type
    ("snr",  0.02),   # Supernova Remnant
    ("glc",  0.01),   # Globular Cluster
    ("agn",  0.03),   # Active Galactic Nucleus (other)
    ("",     0.30),   # UNIDENTIFIED
    ("unk",  0.08),   # Unknown
]

# Galactic plane has higher source density -- simulate that
_GAL_PLANE_DEC_RANGE = (-10, 10)  # crude galactic plane proxy in equatorial


def _generate_simulated_catalog(
    n_sources: int = 6659,
    seed: int = 4321,
) -> List[FermiSource]:
    """Generate a realistic simulated Fermi 4FGL-DR4 catalog.

    The simulated catalog mimics the statistical properties of the real
    4FGL-DR4: source density is higher along the Galactic plane, flux
    follows a power-law distribution, and class fractions approximate
    the real catalog.

    Parameters
    ----------
    n_sources : int
        Number of simulated sources (real 4FGL-DR4 has ~6659).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[FermiSource]
    """
    logger.info("Generating simulated Fermi 4FGL-DR4 catalog (%d sources)", n_sources)
    rng = np.random.RandomState(seed)

    # Build cumulative class distribution
    classes = [c for c, _ in _CLASS_DISTRIBUTION]
    weights = np.array([w for _, w in _CLASS_DISTRIBUTION])
    weights /= weights.sum()

    sources: List[FermiSource] = []
    for i in range(n_sources):
        # ---- Sky position ------------------------------------------------
        # ~60% of sources cluster near the Galactic plane
        if rng.random() < 0.6:
            ra = rng.uniform(0, 360)
            dec = rng.normal(0, 15)  # clustered near equator (proxy for Gal plane)
            dec = np.clip(dec, -90, 90)
        else:
            ra = rng.uniform(0, 360)
            dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))  # uniform on sphere

        # ---- Source class ------------------------------------------------
        source_class = rng.choice(classes, p=weights)
        is_unid = (source_class == "" or source_class.lower() == "unk")

        # ---- Flux (log-normal, typical range 1e-12 to 1e-8 ph/cm2/s) ----
        log_flux = rng.normal(-10.5, 1.0)
        flux_1gev = 10.0 ** log_flux

        # ---- Spectral index (pulsars softer ~2.0-2.5, blazars harder ~1.8-2.2)
        if source_class == "psr":
            spectral_index = rng.normal(2.3, 0.3)
        elif source_class in ("bll", "fsrq", "bcu", "agn"):
            spectral_index = rng.normal(2.1, 0.3)
        else:
            spectral_index = rng.normal(2.2, 0.4)
        spectral_index = max(1.0, min(4.0, spectral_index))

        # ---- Variability index -------------------------------------------
        # Blazars are highly variable; pulsars are steady
        if source_class in ("bll", "fsrq", "agn"):
            var_index = rng.exponential(50.0) + 20.0
        elif source_class == "psr":
            var_index = rng.exponential(5.0) + 5.0
        else:
            var_index = rng.exponential(15.0) + 10.0

        # ---- Positional uncertainty (95% error ellipse) ------------------
        # Bright sources have small errors (~0.5 arcmin), faint ones larger
        # 4FGL typical range: 1-10 arcmin for unidentified sources
        base_err_deg = 10.0 ** (rng.normal(-1.3, 0.4))  # ~0.02-0.2 deg
        if is_unid:
            base_err_deg *= rng.uniform(1.5, 3.0)  # unID sources have larger errors
        semi_major = min(0.5, max(0.005, base_err_deg))
        semi_minor = semi_major * rng.uniform(0.5, 1.0)  # slight ellipticity

        source_name = f"4FGL J{_ra_to_hms(ra)}{_dec_to_dms(dec)}s"

        source = FermiSource(
            source_name=source_name,
            ra=float(ra),
            dec=float(dec),
            flux_1gev=float(flux_1gev),
            spectral_index=float(spectral_index),
            source_class=source_class if source_class else None,
            variability_index=float(var_index),
            pos_err_semimajor_deg=float(semi_major),
            pos_err_semiminor_deg=float(semi_minor),
            is_unidentified=is_unid,
            simulated=True,
        )
        sources.append(source)

    n_unid = sum(1 for s in sources if s.is_unidentified)
    logger.info("Simulated catalog: %d sources, %d unidentified (%.1f%%)",
                len(sources), n_unid, 100.0 * n_unid / len(sources))
    return sources


def _ra_to_hms(ra_deg: float) -> str:
    """Convert RA in degrees to HHMMm string for 4FGL-style names."""
    h = ra_deg / 15.0
    hh = int(h)
    mm = int((h - hh) * 60)
    return f"{hh:02d}{mm:02d}"


def _dec_to_dms(dec_deg: float) -> str:
    """Convert Dec in degrees to +DDMM string for 4FGL-style names."""
    sign = "+" if dec_deg >= 0 else "-"
    ad = abs(dec_deg)
    dd = int(ad)
    mm = int((ad - dd) * 60)
    return f"{sign}{dd:02d}{mm:02d}"


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

class FermiCatalogIngest:
    """
    Primary interface for ingesting the Fermi 4FGL-DR4 gamma-ray catalog.

    Provides three main accessors:
        - get_all_sources()       -> list of all catalog sources
        - get_unidentified()      -> list of unidentified sources only
        - get_by_position(ra, dec, radius_deg) -> cone search

    Behaviour:
        1. Checks for a local cached copy of the parsed catalog.
        2. Attempts to download the FITS file from the Fermi SSC.
        3. Falls back to simulated data if download/parse fails.
    """

    def __init__(self, force_simulated: bool = False):
        """
        Parameters
        ----------
        force_simulated : bool
            If True, skip all network access and use simulated data.
        """
        self._cfg = get_config()
        self._force_simulated = force_simulated
        self._catalog: Optional[List[FermiSource]] = None

        self._data_dir = PROJECT_ROOT / self._cfg["project"]["data_dir"] / "fermi"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._fits_path = self._data_dir / "gll_psc_v35.fit"

    # -- Catalog loading ---------------------------------------------------

    def _load_catalog(self) -> List[FermiSource]:
        """Load the catalog, using cache -> download -> simulation fallback."""
        if self._catalog is not None:
            return self._catalog

        # 1. Try JSON cache
        cached = load_cache(CACHE_KEY_CATALOG, subfolder=CACHE_SUBFOLDER)
        if cached is not None:
            logger.info("Loading Fermi catalog from cache (%d sources)", len(cached))
            self._catalog = [FermiSource.from_dict(d) for d in cached]
            return self._catalog

        # 2. Try parsing existing FITS file
        if not self._force_simulated and self._fits_path.exists():
            try:
                self._catalog = _parse_fits(self._fits_path)
                self._save_to_cache(self._catalog)
                return self._catalog
            except Exception as exc:
                logger.warning("Failed to parse existing FITS file: %s", exc)

        # 3. Try downloading the FITS file
        if not self._force_simulated:
            try:
                _download_fits(FERMI_4FGL_DR4_URL, self._fits_path)
                self._catalog = _parse_fits(self._fits_path)
                self._save_to_cache(self._catalog)
                return self._catalog
            except Exception as exc:
                logger.warning(
                    "Failed to download/parse Fermi catalog: %s "
                    "-- falling back to simulation",
                    exc,
                )

        # 4. Simulation fallback (NOT cached — prevents provenance contamination)
        logger.warning("Using SIMULATED Fermi 4FGL-DR4 catalog (not cached)")
        self._catalog = _generate_simulated_catalog()
        return self._catalog

    def _save_to_cache(self, sources: List[FermiSource]) -> None:
        """Persist the parsed catalog to the JSON cache."""
        data = [s.to_dict() for s in sources]
        path = save_cache(CACHE_KEY_CATALOG, data, subfolder=CACHE_SUBFOLDER)
        logger.info("Cached %d Fermi sources to %s", len(sources), path)

    # -- Public API --------------------------------------------------------

    def get_all_sources(self) -> List[FermiSource]:
        """Return all sources in the Fermi 4FGL-DR4 catalog.

        Returns
        -------
        list[FermiSource]
            Complete catalog of gamma-ray sources.
        """
        return list(self._load_catalog())

    def get_unidentified(self) -> List[FermiSource]:
        """Return only unidentified gamma-ray sources.

        These are sources with CLASS1 empty or ``unk`` -- they have no
        firm association with any known astrophysical object and are
        therefore of highest interest for anomaly detection.

        Returns
        -------
        list[FermiSource]
            Unidentified gamma-ray sources.
        """
        catalog = self._load_catalog()
        unid = [s for s in catalog if s.is_unidentified]
        logger.info(
            "Unidentified Fermi sources: %d / %d (%.1f%%)",
            len(unid), len(catalog),
            100.0 * len(unid) / len(catalog) if catalog else 0,
        )
        return unid

    def get_by_position(
        self,
        ra: float,
        dec: float,
        radius_deg: float = 1.0,
    ) -> List[FermiSource]:
        """Return Fermi sources within a cone of given position.

        Uses astropy SkyCoord for proper spherical matching when available;
        falls back to a simple Cartesian approximation otherwise.

        Parameters
        ----------
        ra : float
            Right ascension in degrees (J2000).
        dec : float
            Declination in degrees (J2000).
        radius_deg : float
            Search cone radius in degrees.

        Returns
        -------
        list[FermiSource]
            Sources within the search cone, sorted by angular separation.
        """
        catalog = self._load_catalog()

        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u_astropy

            cat_ra = np.array([s.ra for s in catalog])
            cat_dec = np.array([s.dec for s in catalog])

            center = SkyCoord(ra=ra, dec=dec, unit=(u_astropy.deg, u_astropy.deg), frame="icrs")
            cat_coords = SkyCoord(ra=cat_ra, dec=cat_dec, unit=(u_astropy.deg, u_astropy.deg), frame="icrs")

            seps = center.separation(cat_coords)
            mask = seps.deg <= radius_deg

            indices = np.where(mask)[0]
            sep_values = seps.deg[mask]
            # Sort by separation
            order = np.argsort(sep_values)
            result = [catalog[indices[i]] for i in order]

        except ImportError:
            logger.debug("astropy not available; using approximate position matching")
            result = []
            cos_dec = np.cos(np.radians(dec))
            for s in catalog:
                dra = (s.ra - ra) * cos_dec
                ddec = s.dec - dec
                sep_deg = np.sqrt(dra**2 + ddec**2)
                if sep_deg <= radius_deg:
                    result.append(s)
            # Approximate sort by separation
            result.sort(
                key=lambda s: ((s.ra - ra) * cos_dec) ** 2 + (s.dec - dec) ** 2
            )

        logger.info(
            "Cone search at (%.4f, %.4f) r=%.2f deg: %d sources found",
            ra, dec, radius_deg, len(result),
        )
        return result


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_ingest: Optional[FermiCatalogIngest] = None


def _get_ingest() -> FermiCatalogIngest:
    """Lazily initialise and return the default ingest instance."""
    global _default_ingest
    if _default_ingest is None:
        _default_ingest = FermiCatalogIngest()
    return _default_ingest


def get_all_sources() -> List[FermiSource]:
    """Return all sources in the Fermi 4FGL-DR4 catalog."""
    return _get_ingest().get_all_sources()


def get_unidentified() -> List[FermiSource]:
    """Return only unidentified gamma-ray sources from the 4FGL-DR4."""
    return _get_ingest().get_unidentified()


def get_by_position(ra: float, dec: float, radius_deg: float = 1.0) -> List[FermiSource]:
    """Return Fermi sources within a cone around (RA, Dec)."""
    return _get_ingest().get_by_position(ra, dec, radius_deg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- Fermi 4FGL-DR4 Gamma-Ray Catalog Ingestion")
    print("=" * 72)
    print()

    ingest = FermiCatalogIngest()

    # 1. Load full catalog
    print("[1] Loading Fermi 4FGL-DR4 catalog ...")
    all_sources = ingest.get_all_sources()
    print(f"    Total sources: {len(all_sources)}")
    is_simulated = any(s.simulated for s in all_sources[:10])
    if is_simulated:
        print("    (using simulated catalog data)")
    print()

    # 2. Unidentified sources
    print("[2] Unidentified sources (no firm astrophysical association):")
    print("-" * 60)
    unid = ingest.get_unidentified()
    n_unid = len(unid)
    pct = 100.0 * n_unid / len(all_sources) if all_sources else 0
    print(f"    Unidentified: {n_unid} / {len(all_sources)} ({pct:.1f}%)")
    print()

    # 3. Source class breakdown
    print("[3] Source class breakdown:")
    print("-" * 60)
    class_counts: Dict[str, int] = {}
    for s in all_sources:
        cls = s.source_class if s.source_class else "(unidentified)"
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls in sorted(class_counts, key=class_counts.get, reverse=True):
        count = class_counts[cls]
        bar = "#" * max(1, int(40 * count / len(all_sources)))
        print(f"    {cls:<20s} {count:>5d}  {bar}")
    print()

    # 4. Flux statistics
    print("[4] Flux statistics (> 1 GeV):")
    print("-" * 60)
    fluxes = np.array([s.flux_1gev for s in all_sources if s.flux_1gev is not None])
    if len(fluxes) > 0:
        print(f"    Min flux:    {fluxes.min():.3e} ph/cm2/s")
        print(f"    Max flux:    {fluxes.max():.3e} ph/cm2/s")
        print(f"    Median flux: {np.median(fluxes):.3e} ph/cm2/s")
        print(f"    Mean flux:   {fluxes.mean():.3e} ph/cm2/s")
    print()

    # 5. Positional uncertainty statistics for unidentified sources
    print("[5] Positional uncertainty (95% error) for unidentified sources:")
    print("-" * 60)
    err_arcmin = np.array([
        s.pos_err_arcmin for s in unid if s.pos_err_arcmin is not None
    ])
    if len(err_arcmin) > 0:
        print(f"    Min:    {err_arcmin.min():.2f} arcmin")
        print(f"    Max:    {err_arcmin.max():.2f} arcmin")
        print(f"    Median: {np.median(err_arcmin):.2f} arcmin")
        print(f"    Mean:   {err_arcmin.mean():.2f} arcmin")
    print()

    # 6. Sample unidentified sources
    print("[6] Sample unidentified sources (first 15):")
    print("-" * 60)
    for s in unid[:15]:
        err_str = f"{s.pos_err_arcmin:.1f}'" if s.pos_err_arcmin else "N/A"
        flux_str = f"{s.flux_1gev:.2e}" if s.flux_1gev else "N/A"
        print(
            f"    {s.source_name:<25s}  "
            f"RA={s.ra:8.4f}  Dec={s.dec:+8.4f}  "
            f"F={flux_str}  err={err_str}"
        )
    print()

    # 7. Cone search demo: Galactic center region
    print("[7] Cone search demo: Galactic center region (RA=266.4, Dec=-29.0, r=5 deg):")
    print("-" * 60)
    gc_sources = ingest.get_by_position(266.4, -29.0, radius_deg=5.0)
    n_gc_unid = sum(1 for s in gc_sources if s.is_unidentified)
    print(f"    Sources in cone:     {len(gc_sources)}")
    print(f"    Unidentified in cone: {n_gc_unid}")
    print()

    print("=" * 72)
    print("  Fermi catalog ingestion complete.")
    print("=" * 72)
