"""
SB9 Spectroscopic Binary Catalog ingestion for Project EXODUS.

Queries the 9th Catalogue of Spectroscopic Binary Orbits (SB9)
via VizieR (B/sb9).  SB9 contains ~3,600 spectroscopic binary
systems with orbital elements.

A match in SB9 is strong evidence that a target is a binary —
RUWE anomalies, PM discrepancies, and photometric variability
are all expected and NOT technosignature indicators.

Public API
----------
query_sb9_cone(ra, dec, radius_arcsec=30.0)
    Return SB9 matches within a cone search.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("ingestion.sb9_catalog")

# VizieR catalog ID for SB9
_SB9_CATALOG = "B/sb9/main"


def query_sb9_cone(
    ra: float,
    dec: float,
    radius_arcsec: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Query SB9 for spectroscopic binaries near a position.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in degrees (J2000).
    radius_arcsec : float
        Search radius in arcseconds (default 30").

    Returns
    -------
    dict or None
        If match found: {"match": True, "name": str, "period_days": float,
        "sep_arcsec": float, "n_matches": int}.
        If no match: {"match": False}.
        If query fails: None.
    """
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        import numpy as np

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        v = Vizier(
            catalog=_SB9_CATALOG,
            columns=["*"],
            row_limit=5,
            timeout=15,
        )

        result = v.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
        )

        if result is None or len(result) == 0:
            return {"match": False}

        table = result[0]
        if len(table) == 0:
            return {"match": False}

        # Take closest match
        if "Seq" in table.colnames:
            row = table[0]
        else:
            row = table[0]

        # Extract period if available
        period = None
        for col in ("Per", "Period", "P"):
            if col in table.colnames:
                val = row[col]
                if val is not None and not np.ma.is_masked(val):
                    try:
                        period = float(val)
                    except (TypeError, ValueError):
                        pass
                break

        # Extract name
        name = ""
        for col in ("Name", "Comp", "HD"):
            if col in table.colnames:
                val = row[col]
                if val is not None and not np.ma.is_masked(val):
                    name = str(val).strip()
                    break

        # Compute separation
        row_ra = None
        row_dec = None
        for ra_col in ("_RAJ2000", "RAJ2000", "RA"):
            if ra_col in table.colnames:
                val = row[ra_col]
                if val is not None and not np.ma.is_masked(val):
                    row_ra = float(val)
                    break
        for dec_col in ("_DEJ2000", "DEJ2000", "DE"):
            if dec_col in table.colnames:
                val = row[dec_col]
                if val is not None and not np.ma.is_masked(val):
                    row_dec = float(val)
                    break

        sep = 0.0
        if row_ra is not None and row_dec is not None:
            match_coord = SkyCoord(
                ra=row_ra * u.deg, dec=row_dec * u.deg, frame="icrs"
            )
            sep = coord.separation(match_coord).arcsec

        log.info(
            "SB9 match at (%.4f, %.4f): %s, sep=%.1f\", P=%s d",
            ra, dec, name or "unnamed", sep,
            f"{period:.1f}" if period else "N/A",
        )

        return {
            "match": True,
            "name": name,
            "period_days": period,
            "sep_arcsec": round(sep, 2),
            "n_matches": len(table),
        }

    except Exception as exc:
        log.debug("SB9 query failed for (%.4f, %.4f): %s", ra, dec, exc)
        return None
