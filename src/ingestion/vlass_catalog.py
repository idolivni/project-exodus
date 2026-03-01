"""
Radio continuum catalog access for Project EXODUS.

Queries VLA radio surveys via VizieR to check for radio continuum
emission from stellar targets:

1. FIRST (1.4 GHz, 0.94M sources, beam 5") — primary
2. NVSS (1.4 GHz, 2M sources, beam 45") — backup/complementary

Scientific rationale for EXODUS:
  - Most FGK dwarfs at < 50 pc should NOT be radio continuum sources.
  - Detection implies: stellar activity (flares), binary interaction,
    instrumental artifact, or genuine anomaly.
  - Radio continuum is INDEPENDENT of IR/optical/UV channels.
  - Unexpected radio emission near a star with IR excess would be
    a strong multi-channel convergence signal.

References
----------
- Becker et al. 1995 (FIRST survey)
- Condon et al. 1998 (NVSS)
- VizieR: VIII/92/first14 (FIRST), VIII/65/nvss (NVSS)

Public API
----------
query_radio_continuum(ra, dec, radius_arcsec=15)
    Returns closest radio continuum source from FIRST or NVSS.

is_radio_continuum_detected(match, min_flux_mjy=1.0)
    Check if a match represents a significant radio detection.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, Optional

log = logging.getLogger("exodus.ingestion.radio_continuum")

# VizieR catalog IDs
_FIRST_CATALOG = "VIII/92/first14"   # FIRST survey (1.4 GHz, 5" beam)
_NVSS_CATALOG = "VIII/65/nvss"       # NVSS (1.4 GHz, 45" beam)

# FIRST columns
_FIRST_COLUMNS = [
    "RAJ2000",     # RA (J2000), deg
    "DEJ2000",     # Dec (J2000), deg
    "Fpeak",       # Peak flux density (mJy/beam)
    "Fint",        # Integrated flux density (mJy)
    "Rms",         # Local noise RMS (mJy)
    "Maj",         # Fitted major axis (arcsec)
    "Min",         # Fitted minor axis (arcsec)
    "p(S)",        # Probability of spurious detection
]

# NVSS columns
_NVSS_COLUMNS = [
    "RAJ2000",
    "DEJ2000",
    "S1.4",        # 1.4 GHz flux density (mJy)
    "e_S1.4",      # Error in flux density
    "MajAx",       # Major axis (arcsec)
    "MinAx",       # Minor axis (arcsec)
    "l_S1.4",      # Limit flag
]


def query_radio_continuum(
    ra: float,
    dec: float,
    radius_arcsec: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Query FIRST and NVSS for radio continuum sources near a position.

    Tries FIRST first (higher resolution, 5" beam), falls back to NVSS
    (lower resolution, 45" beam, more complete).

    Parameters
    ----------
    ra, dec : float
        J2000 coordinates in degrees.
    radius_arcsec : float
        Search radius in arcseconds (default 15").
        FIRST beam is ~5", so 15" is 3x beam width.

    Returns
    -------
    dict or None
        Best radio continuum match with keys:
        ``ra``, ``dec``, ``sep_arcsec``,
        ``peak_flux_mjy``, ``integrated_flux_mjy``,
        ``snr``, ``survey``, ``data_source``.
        Returns None if no match found.
    """
    try:
        from astroquery.vizier import Vizier
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError:
        log.debug("astroquery not available for radio continuum query")
        return None

    coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

    # Try FIRST first (higher resolution, Dec > -10.5 deg)
    if dec > -10.5:
        match = _query_first(coord, radius_arcsec)
        if match is not None:
            return match

    # Fallback/complement: NVSS (Dec > -40 deg)
    if dec > -40.0:
        match = _query_nvss(coord, radius_arcsec)
        if match is not None:
            return match

    return None


def _query_first(
    coord: "SkyCoord",
    radius_arcsec: float,
) -> Optional[Dict[str, Any]]:
    """Query FIRST survey (1.4 GHz, 5" beam)."""
    from astroquery.vizier import Vizier
    import astropy.units as u

    try:
        v = Vizier(columns=_FIRST_COLUMNS, row_limit=5, timeout=30)
        result = v.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            catalog=_FIRST_CATALOG,
        )
    except Exception as exc:
        log.debug("FIRST VizieR query failed: %s", exc)
        return None

    if not result or len(result) == 0 or len(result[0]) == 0:
        return None

    table = result[0]
    from astropy.coordinates import SkyCoord as SC
    # FIRST returns sexagesimal coords — parse with SkyCoord
    source_coords = SC(
        ra=table["RAJ2000"], dec=table["DEJ2000"],
        unit=("hourangle", "deg"),
    )
    seps = coord.separation(source_coords).arcsec
    best_idx = int(seps.argmin())
    row = table[best_idx]

    peak_flux = _safe_float(row, "Fpeak")
    int_flux = _safe_float(row, "Fint")
    rms = _safe_float(row, "Rms")

    snr = 0.0
    if peak_flux is not None and rms is not None and rms > 0:
        snr = peak_flux / rms

    source_ra = float(source_coords[best_idx].ra.deg)
    source_dec = float(source_coords[best_idx].dec.deg)

    match = {
        "ra": source_ra,
        "dec": source_dec,
        "sep_arcsec": float(seps[best_idx]),
        "peak_flux_mjy": peak_flux,
        "integrated_flux_mjy": int_flux,
        "rms_mjy": rms,
        "snr": float(snr),
        "survey": "FIRST",
        "frequency_ghz": 1.4,
        "beam_arcsec": 5.0,
        "data_source": "first_vizier",
    }

    log.info(
        "FIRST match at %.1f\": peak=%.2f mJy (SNR=%.1f)",
        match["sep_arcsec"], peak_flux or 0, snr,
    )
    return match


def _query_nvss(
    coord: "SkyCoord",
    radius_arcsec: float,
) -> Optional[Dict[str, Any]]:
    """Query NVSS (1.4 GHz, 45" beam)."""
    from astroquery.vizier import Vizier
    import astropy.units as u

    # Use wider radius for NVSS due to lower resolution
    nvss_radius = max(radius_arcsec, 45.0)

    try:
        v = Vizier(columns=_NVSS_COLUMNS, row_limit=5, timeout=30)
        result = v.query_region(
            coord,
            radius=nvss_radius * u.arcsec,
            catalog=_NVSS_CATALOG,
        )
    except Exception as exc:
        log.debug("NVSS VizieR query failed: %s", exc)
        return None

    if not result or len(result) == 0 or len(result[0]) == 0:
        return None

    table = result[0]
    from astropy.coordinates import SkyCoord as SC
    # NVSS returns sexagesimal coords — parse with SkyCoord
    source_coords = SC(
        ra=table["RAJ2000"], dec=table["DEJ2000"],
        unit=("hourangle", "deg"),
    )
    seps = coord.separation(source_coords).arcsec
    best_idx = int(seps.argmin())
    row = table[best_idx]

    flux = _safe_float(row, "S1.4")
    flux_err = _safe_float(row, "e_S1.4")

    snr = 0.0
    if flux is not None and flux_err is not None and flux_err > 0:
        snr = flux / flux_err

    source_ra = float(source_coords[best_idx].ra.deg)
    source_dec = float(source_coords[best_idx].dec.deg)

    match = {
        "ra": source_ra,
        "dec": source_dec,
        "sep_arcsec": float(seps[best_idx]),
        "peak_flux_mjy": flux,  # NVSS reports total, not peak
        "integrated_flux_mjy": flux,
        "flux_err_mjy": flux_err,
        "snr": float(snr),
        "survey": "NVSS",
        "frequency_ghz": 1.4,
        "beam_arcsec": 45.0,
        "data_source": "nvss_vizier",
    }

    log.info(
        "NVSS match at %.1f\": S1.4=%.1f mJy (SNR=%.1f)",
        match["sep_arcsec"], flux or 0, snr,
    )
    return match


def is_radio_continuum_detected(
    match: Optional[Dict[str, Any]],
    min_flux_mjy: float = 1.0,
    max_sep_arcsec: float = 10.0,
) -> bool:
    """Check if a radio continuum match represents a significant detection.

    Parameters
    ----------
    match : dict or None
        Result from :func:`query_radio_continuum`.
    min_flux_mjy : float
        Minimum flux density for detection (default 1.0 mJy).
    max_sep_arcsec : float
        Maximum separation for confident association (default 10").

    Returns
    -------
    bool
        True if source is a confident radio continuum detection.
    """
    if match is None:
        return False

    flux = match.get("peak_flux_mjy") or match.get("integrated_flux_mjy", 0)
    sep = match.get("sep_arcsec", 99)

    return flux >= min_flux_mjy and sep <= max_sep_arcsec


def _safe_float(row: Any, col: str) -> Optional[float]:
    """Safely extract a float from an astropy table row."""
    try:
        val = row[col]
        if val is None:
            return None
        val = float(val)
        if np.isfinite(val):
            return val
        return None
    except (KeyError, TypeError, ValueError):
        return None


# ── CLI demo ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 3C 273 — bright quasar, should be detected
    print("Querying FIRST/NVSS for 3C 273 (RA=187.28, Dec=+2.05)...")
    match = query_radio_continuum(187.2779, 2.0524)
    if match:
        print(f"  {match['survey']} match at {match['sep_arcsec']:.1f}\"")
        print(f"  Flux: {match['peak_flux_mjy']:.1f} mJy ({match['frequency_ghz']} GHz)")
        print(f"  Detected: {is_radio_continuum_detected(match)}")
    else:
        print("  No radio continuum source found")

    # Proxima Cen — radio-quiet M dwarf (southern sky, no FIRST/NVSS)
    print("\nQuerying for Proxima Cen (Dec=-62.7, outside FIRST/NVSS)...")
    match2 = query_radio_continuum(217.4292, -62.6794)
    if match2:
        print(f"  Flux: {match2['peak_flux_mjy']:.2f} mJy")
    else:
        print("  No coverage (expected — Dec < -40)")

    # HD 209458 — quiet FGK star
    print("\nQuerying for HD 209458 (RA=330.8, Dec=+18.9)...")
    match3 = query_radio_continuum(330.795, 18.884)
    if match3:
        print(f"  {match3['survey']} at {match3['sep_arcsec']:.1f}\": {match3['peak_flux_mjy']:.2f} mJy")
        print(f"  Detected: {is_radio_continuum_detected(match3)}")
    else:
        print("  No radio continuum (expected — quiet star)")
