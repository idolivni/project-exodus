"""
Project EXODUS — Herschel Point Source Catalogue Cross-Match
============================================================

Cross-matches targets against the Herschel Point Source Catalogue (HPSC)
using VizieR.  Herschel/PACS (70, 100, 160 µm) and SPIRE (250, 350, 500 µm)
provide far-IR photometry that is critical for distinguishing:

1. **Dyson spheres** — should show thermal emission peaking at ~10-100 µm,
   continuing into Herschel bands for partial enclosures (T ~ 100-300 K).
2. **Circumstellar dust** — warm dust (debris disk) peaks in WISE bands
   and falls off by Herschel wavelengths.  Cold dust extends into SPIRE.
3. **Background galaxies** — have different far-IR SEDs from point-source
   circumstellar emission.

VizieR catalogues:
- VIII/106 — Herschel/PACS Point Source Catalogue (HPPSC)
- VIII/107 — Herschel/SPIRE Point Source Catalogue (HSPSC)

Usage
-----
    from src.ingestion.herschel_catalog import query_herschel

    result = query_herschel(ra=180.0, dec=30.0, radius_arcsec=15.0)
    if result["has_data"]:
        print(f"PACS 70µm: {result['pacs_70']} mJy")
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("ingestion.herschel_catalog")


# ── Herschel beam sizes (FWHM in arcsec) ────────────────────────────
PACS_70_BEAM = 5.6    # arcsec
PACS_100_BEAM = 6.8
PACS_160_BEAM = 11.3
SPIRE_250_BEAM = 18.2
SPIRE_350_BEAM = 24.9
SPIRE_500_BEAM = 36.3

# Default cross-match radius (arcsec)
DEFAULT_RADIUS = 15.0  # generous to account for PACS/SPIRE pointing


def _safe_float(val) -> Optional[float]:
    """Convert to float, returning None for masked/invalid values."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if not math.isfinite(f) else f
    except (ValueError, TypeError):
        return None


def query_herschel(
    ra: float,
    dec: float,
    radius_arcsec: float = DEFAULT_RADIUS,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Cross-match a position against Herschel PACS and SPIRE catalogues.

    Parameters
    ----------
    ra, dec : float
        Position in degrees (ICRS).
    radius_arcsec : float
        Search radius in arcseconds.
    timeout : float
        VizieR query timeout in seconds.

    Returns
    -------
    dict with keys:
        has_data : bool
        pacs_70, pacs_100, pacs_160 : float or None (mJy)
        spire_250, spire_350, spire_500 : float or None (mJy)
        separation_arcsec : float or None
        data_source : str
        n_pacs_matches, n_spire_matches : int
    """
    result = {
        "has_data": False,
        "data_source": "none",
        "pacs_70": None,
        "pacs_100": None,
        "pacs_160": None,
        "spire_250": None,
        "spire_350": None,
        "spire_500": None,
        "separation_arcsec": None,
        "n_pacs_matches": 0,
        "n_spire_matches": 0,
    }

    try:
        from astroquery.vizier import Vizier
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError:
        log.debug("astroquery not available — Herschel query skipped")
        return result

    coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    radius = radius_arcsec * u.arcsec

    # ── PACS query (VIII/106) ─────────────────────────────────────
    try:
        v_pacs = Vizier(
            columns=["RAJ2000", "DEJ2000", "F70", "F100", "F160",
                      "e_F70", "e_F100", "e_F160"],
            row_limit=5,
            timeout=timeout,
        )
        pacs_tables = v_pacs.query_region(
            coord, radius=radius, catalog="VIII/106"
        )
        if pacs_tables and len(pacs_tables) > 0:
            pacs = pacs_tables[0]
            result["n_pacs_matches"] = len(pacs)

            if len(pacs) > 0:
                row = pacs[0]  # closest match
                result["pacs_70"] = _safe_float(row.get("F70"))
                result["pacs_100"] = _safe_float(row.get("F100"))
                result["pacs_160"] = _safe_float(row.get("F160"))
                result["has_data"] = True

                # Separation
                src_ra = _safe_float(row.get("RAJ2000"))
                src_dec = _safe_float(row.get("DEJ2000"))
                if src_ra is not None and src_dec is not None:
                    src_coord = SkyCoord(ra=src_ra, dec=src_dec, unit="deg")
                    sep = coord.separation(src_coord).arcsec
                    result["separation_arcsec"] = float(sep)

                log.info(
                    "PACS match at (%.4f, %.4f): F70=%s F100=%s F160=%s (sep=%.1f\")",
                    ra, dec,
                    result["pacs_70"], result["pacs_100"], result["pacs_160"],
                    result.get("separation_arcsec", -1),
                )
    except Exception as exc:
        log.debug("PACS query failed for (%.4f, %.4f): %s", ra, dec, exc)

    # ── SPIRE query (VIII/107) ────────────────────────────────────
    try:
        v_spire = Vizier(
            columns=["RAJ2000", "DEJ2000", "F250", "F350", "F500",
                      "e_F250", "e_F350", "e_F500"],
            row_limit=5,
            timeout=timeout,
        )
        spire_tables = v_spire.query_region(
            coord, radius=radius, catalog="VIII/107"
        )
        if spire_tables and len(spire_tables) > 0:
            spire = spire_tables[0]
            result["n_spire_matches"] = len(spire)

            if len(spire) > 0:
                row = spire[0]
                result["spire_250"] = _safe_float(row.get("F250"))
                result["spire_350"] = _safe_float(row.get("F350"))
                result["spire_500"] = _safe_float(row.get("F500"))
                result["has_data"] = True

                # Update separation if SPIRE is closer
                src_ra = _safe_float(row.get("RAJ2000"))
                src_dec = _safe_float(row.get("DEJ2000"))
                if src_ra is not None and src_dec is not None:
                    src_coord = SkyCoord(ra=src_ra, dec=src_dec, unit="deg")
                    sep = coord.separation(src_coord).arcsec
                    if result["separation_arcsec"] is None or sep < result["separation_arcsec"]:
                        result["separation_arcsec"] = float(sep)

                log.info(
                    "SPIRE match at (%.4f, %.4f): F250=%s F350=%s F500=%s",
                    ra, dec,
                    result["spire_250"], result["spire_350"], result["spire_500"],
                )
    except Exception as exc:
        log.debug("SPIRE query failed for (%.4f, %.4f): %s", ra, dec, exc)

    if result["has_data"]:
        result["data_source"] = "herschel"

    return result


def interpret_herschel_sed(
    herschel: Dict[str, Any],
    wise_w3_flux_mjy: Optional[float] = None,
    wise_w4_flux_mjy: Optional[float] = None,
) -> Dict[str, Any]:
    """Interpret Herschel+WISE SED for technosignature diagnostics.

    A partial Dyson sphere at T ~ 200-300 K would show:
    - Strong emission at WISE W3/W4 (12/22 µm)
    - Declining but present emission at PACS 70/100 µm
    - Little/no emission at SPIRE 250+ µm

    A debris disk would show:
    - Moderate emission at WISE, often peaking at 70-100 µm
    - Declining at 160+ µm

    A background galaxy would show:
    - Relatively flat or rising SED from 70 to 250 µm

    Returns
    -------
    dict with:
        classification : str ("dyson_consistent", "dust_disk", "galaxy_like",
                              "cold_dust", "unclassified")
        confidence : float (0-1)
        explanation : str
    """
    result = {
        "classification": "unclassified",
        "confidence": 0.0,
        "explanation": "Insufficient Herschel data for classification",
    }

    f70 = herschel.get("pacs_70")
    f100 = herschel.get("pacs_100")
    f160 = herschel.get("pacs_160")
    f250 = herschel.get("spire_250")

    if f70 is None and f100 is None:
        return result

    # Check SED shape
    has_pacs = f70 is not None or f100 is not None
    has_spire = f250 is not None

    # Declining SED from mid-IR to far-IR → consistent with warm dust or Dyson
    if wise_w4_flux_mjy is not None and f70 is not None:
        ratio_w4_to_70 = wise_w4_flux_mjy / f70 if f70 > 0 else 0

        if ratio_w4_to_70 > 2.0:
            # Much brighter at 22 µm than 70 µm → warm source
            result["classification"] = "dyson_consistent"
            result["confidence"] = min(0.5, ratio_w4_to_70 / 10.0)
            result["explanation"] = (
                f"SED declines sharply from W4 to 70µm (ratio={ratio_w4_to_70:.1f}) — "
                f"consistent with T~200-300K thermal emission (warm dust or Dyson)"
            )
        elif 0.5 < ratio_w4_to_70 <= 2.0:
            result["classification"] = "dust_disk"
            result["confidence"] = 0.3
            result["explanation"] = (
                f"SED moderately declining W4→70µm (ratio={ratio_w4_to_70:.1f}) — "
                f"consistent with debris disk peaking near 70µm"
            )

    # Rising SED from 70 to 250 µm → galaxy or cold dust
    if f70 is not None and f250 is not None and f70 > 0:
        ratio_250_to_70 = f250 / f70

        if ratio_250_to_70 > 3.0:
            result["classification"] = "galaxy_like"
            result["confidence"] = 0.6
            result["explanation"] = (
                f"SED rises from 70µm to 250µm (ratio={ratio_250_to_70:.1f}) — "
                f"consistent with background galaxy dust emission"
            )
        elif ratio_250_to_70 > 1.0:
            result["classification"] = "cold_dust"
            result["confidence"] = 0.4
            result["explanation"] = (
                f"Significant cold dust component (250/70 ratio={ratio_250_to_70:.1f})"
            )

    return result
