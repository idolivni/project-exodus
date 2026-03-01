"""
eROSITA eRASS1 X-ray catalog access.

Queries the eRASS1 main catalog (930,203 sources) via VizieR
to check if a target has an X-ray counterpart.  X-ray bright
sources near EXODUS targets strongly suggest stellar activity
(YSO, flare star, active binary) rather than technosignatures.

References
----------
- Merloni et al. 2024, A&A 682, A34 (eRASS1 main catalog)
- Freund et al. 2024, A&A 684, A121 (HamStar coronal IDs)
- VizieR catalog: J/A+A/682/A34
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger("exodus.ingestion.erosita")

# VizieR catalog identifier for eRASS1 main catalog
_VIZIER_CATALOG = "J/A+A/682/A34/erass1m"

# Columns we need from VizieR
_COLUMNS = [
    "RAJ2000",    # RA (corrected), deg
    "DEJ2000",    # Dec (corrected), deg
    "e_Pos",      # positional uncertainty, arcsec
    "DL.0",       # detection likelihood (0.2-2.3 keV)
    "Flux.1",     # 0.2-2.3 keV flux, erg/s/cm^2
    "e_Flux.1",   # flux uncertainty
    "CRate.0",    # count rate 0.2-5 keV
]


def query_erosita_cone(
    ra: float,
    dec: float,
    radius_arcsec: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Query eRASS1 for X-ray sources near the given position.

    Parameters
    ----------
    ra, dec : float
        J2000 coordinates in degrees.
    radius_arcsec : float
        Search radius in arcseconds (default 30").
        eROSITA PSF is ~26" HEW, so 30" is appropriate.

    Returns
    -------
    dict or None
        Best (closest) eRASS1 match with keys:
        ``ra``, ``dec``, ``sep_arcsec``, ``det_like``,
        ``flux_0p2_2p3`` (erg/s/cm²), ``count_rate``.
        Returns None if no match found.
    """
    try:
        from astroquery.vizier import Vizier
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError:
        log.debug("astroquery not available for eROSITA query")
        return None

    try:
        v = Vizier(columns=_COLUMNS, row_limit=5, timeout=30)
        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        result = v.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            catalog=_VIZIER_CATALOG,
        )
    except Exception as exc:
        log.debug("eROSITA VizieR query failed for (%.4f, %.4f): %s", ra, dec, exc)
        return None

    if not result or len(result) == 0 or len(result[0]) == 0:
        return None

    table = result[0]

    # Pick closest source
    source_coords = SkyCoord(
        ra=table["RAJ2000"], dec=table["DEJ2000"], unit="deg"
    )
    target_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    seps = target_coord.separation(source_coords).arcsec
    best_idx = seps.argmin()
    row = table[best_idx]

    match = {
        "ra": float(row["RAJ2000"]),
        "dec": float(row["DEJ2000"]),
        "sep_arcsec": float(seps[best_idx]),
        "det_like": float(row["DL.0"]) if row["DL.0"] else None,
        "flux_0p2_2p3": float(row["Flux.1"]) if row["Flux.1"] else None,
        "count_rate": float(row["CRate.0"]) if row["CRate.0"] else None,
    }

    log.info(
        "eRASS1 match at %.1f\": flux=%.2e erg/s/cm², DL=%.1f",
        match["sep_arcsec"],
        match["flux_0p2_2p3"] or 0,
        match["det_like"] or 0,
    )

    return match


def is_xray_active(
    erosita_match: Optional[Dict[str, Any]],
    flux_threshold: float = 1e-13,
) -> bool:
    """Check if an eROSITA match indicates stellar X-ray activity.

    Parameters
    ----------
    erosita_match : dict or None
        Result from :func:`query_erosita_cone`.
    flux_threshold : float
        Minimum 0.2-2.3 keV flux in erg/s/cm² to flag as active.
        1e-13 is conservative — picks up active stars, flare stars,
        YSOs, and RS CVn systems in the solar neighborhood.

    Returns
    -------
    bool
        True if the source is likely X-ray active (natural explanation).
    """
    if erosita_match is None:
        return False

    flux = erosita_match.get("flux_0p2_2p3")
    if flux is not None and flux > flux_threshold:
        return True

    return False
