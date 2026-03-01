"""
SIMBAD astronomical database access for EXODUS pipeline (audit fix D1).

Queries SIMBAD TAP to identify known object types for any target.
Stars classified as YSO, CV, QSO, AGN, Symbiotic, WR etc. are flagged
as high-risk astrophysical explanations in the red-team vetting.

References
----------
- SIMBAD: https://simbad.cds.unistra.fr/simbad/
- Wenger et al. 2000, A&AS 143, 9 (SIMBAD description)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger("exodus.ingestion.simbad")

# Object types that indicate the "anomaly" has a known astrophysical cause.
# Keyed by SIMBAD otype code, valued as (risk_level, human_readable).
HIGH_RISK_OTYPES = {
    "YSO":  (0.9, "Young Stellar Object — IR excess is protostellar"),
    "Y*O":  (0.9, "Young Stellar Object — IR excess is protostellar"),
    "TT*":  (0.8, "T Tauri star — IR excess from circumstellar disk"),
    "Or*":  (0.8, "Orion variable (PMS) — IR excess is protostellar"),
    "Ae*":  (0.7, "Herbig Ae/Be star — IR excess from disk"),
    "CV*":  (0.9, "Cataclysmic Variable — UV/optical anomaly expected"),
    "No*":  (0.8, "Classical Nova — UV/optical anomaly expected"),
    "SN*":  (0.8, "Supernova — transient, not technosignature"),
    "QSO":  (0.95, "Quasar — extragalactic, not a star at all"),
    "AGN":  (0.95, "Active Galactic Nucleus — extragalactic"),
    "Sy*":  (0.9, "Seyfert Galaxy — extragalactic"),
    "Bla":  (0.95, "Blazar — extragalactic jet"),
    "rG":   (0.9, "Radio Galaxy — extragalactic"),
    "IG":   (0.8, "Interacting Galaxy — extragalactic"),
    "GiC":  (0.7, "Galaxy in Cluster — extragalactic"),
    "BiC":  (0.7, "Brightest galaxy in Cluster — extragalactic"),
    "GiG":  (0.7, "Galaxy in Group — extragalactic"),
    "GiP":  (0.7, "Galaxy in Pair — extragalactic"),
    "G":    (0.7, "Galaxy — extragalactic"),
    "Em*":  (0.5, "Emission-line Star — possible activity indicator"),
    "Be*":  (0.5, "Be Star — decretion disk causes IR excess"),
    "WR*":  (0.7, "Wolf-Rayet star — strong winds cause IR excess"),
    "PN":   (0.8, "Planetary Nebula — extended IR emission"),
    "pA*":  (0.6, "Post-AGB star — circumstellar dust expected"),
    "AB*":  (0.6, "Asymptotic Giant Branch — dust shell IR excess"),
    "Mi*":  (0.5, "Mira variable — pulsation causes variability"),
    "LP*":  (0.4, "Long Period Variable — pulsation/dust"),
    "RS*":  (0.6, "RS CVn variable — active binary"),
    "BY*":  (0.5, "BY Dra variable — chromospheric activity"),
    "Fl*":  (0.5, "Flare star — magnetic activity"),
    "SB*":  (0.7, "Spectroscopic Binary — binarity confirmed"),
    "**":   (0.5, "Double/Multiple star — binary system"),
    "EB*":  (0.7, "Eclipsing Binary — binary system confirmed"),
    "Al*":  (0.6, "Algol-type binary — semi-detached binary"),
    "bL*":  (0.6, "Beta Lyrae binary — mass transfer binary"),
    "WU*":  (0.6, "W UMa binary — contact binary"),
    "El*":  (0.5, "Ellipsoidal variable — close binary"),
    "SB":   (0.7, "Spectroscopic binary (generic)"),
    "EB":   (0.7, "Eclipsing binary (generic)"),
}

# Moderate-risk: interesting but not disqualifying
MODERATE_RISK_OTYPES = {
    "PM*":  (0.1, "High Proper Motion star"),
    "HV*":  (0.1, "High Velocity star"),
    "*":    (0.0, "Known star (generic) — no additional risk"),
}


def query_simbad_cone(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Query SIMBAD for objects near the given position.

    Parameters
    ----------
    ra, dec : float
        J2000 coordinates in degrees.
    radius_arcsec : float
        Search radius in arcseconds (default 5").

    Returns
    -------
    dict or None
        Best (closest) match with keys:
        ``main_id``, ``otype``, ``sp_type``, ``rvz_radvel``,
        ``sep_arcsec``, ``risk_level``, ``risk_explanation``.
        Returns ``{"match": False}`` if no match found.
        Returns None if query fails.
    """
    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        log.debug("astroquery not available for SIMBAD query")
        return None

    try:
        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

        # Use SIMBAD TAP for efficient cone search
        adql = f"""
        SELECT TOP 5
            main_id, ra, dec, otype, sp_type, rvz_radvel, rvz_type
        FROM basic
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_arcsec / 3600.0:.8f})
        ) = 1
        ORDER BY DISTANCE(
            POINT('ICRS', ra, dec),
            POINT('ICRS', {ra:.6f}, {dec:.6f})
        ) ASC
        """
        result = Simbad.query_tap(adql, maxrec=5)

        if result is None or len(result) == 0:
            return {"match": False}

        # Take closest match
        row = result[0]
        main_id = str(row["main_id"]).strip() if row["main_id"] else "?"
        otype = str(row["otype"]).strip() if row["otype"] else "?"

        # Compute separation
        match_coord = SkyCoord(
            ra=float(row["ra"]) * u.deg,
            dec=float(row["dec"]) * u.deg,
            frame="icrs",
        )
        sep = coord.separation(match_coord).arcsec

        # Get spectral type and RV if available
        sp_type = str(row["sp_type"]).strip() if row["sp_type"] else None
        rv = None
        try:
            rv = float(row["rvz_radvel"])
        except (TypeError, ValueError):
            pass
        rv_type = str(row["rvz_type"]).strip() if row["rvz_type"] else None

        # Assess risk based on object type
        risk_level = 0.0
        risk_explanation = f"SIMBAD type '{otype}' ({main_id})"

        if otype in HIGH_RISK_OTYPES:
            risk_level, explanation = HIGH_RISK_OTYPES[otype]
            risk_explanation = f"{main_id}: {explanation}"
        elif otype in MODERATE_RISK_OTYPES:
            risk_level, explanation = MODERATE_RISK_OTYPES[otype]
            risk_explanation = f"{main_id}: {explanation}"

        return {
            "match": True,
            "main_id": main_id,
            "otype": otype,
            "sp_type": sp_type,
            "rvz_radvel": rv,
            "rvz_type": rv_type,
            "sep_arcsec": round(sep, 3),
            "risk_level": risk_level,
            "risk_explanation": risk_explanation,
        }

    except Exception as exc:
        log.debug("SIMBAD query failed for (%.4f, %.4f): %s", ra, dec, exc)
        return None
