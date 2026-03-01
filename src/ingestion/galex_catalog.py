"""
GALEX UV catalog access for Project EXODUS.

Queries the GALEX GR6+7 revised catalog (GUVcat_AIS, Bianchi+ 2017)
via VizieR to retrieve FUV and NUV photometry for targets.

UV photometry provides genuinely independent information:
  - UV deficit: unexplained UV absorption (megastructure occultation?)
  - UV excess: stellar activity indicator (chromospheric emission)
  - FUV-NUV color: constrains effective temperature independently of IR

This complements the IR-dominated detection channels by probing the
*opposite* end of the SED. A Dyson sphere would absorb UV light and
re-emit it in the IR, creating correlated UV deficit + IR excess.

References
----------
- Bianchi et al. 2017, ApJS 230, 24 (GUVcat_AIS GR6+7)
- VizieR catalog: II/335/galex_ais

Public API
----------
query_galex_cone(ra, dec, radius_arcsec=30)
    Returns closest GALEX match with FUV/NUV photometry.

compute_uv_metrics(galex_match, teff_est=None)
    Compute UV anomaly metrics from GALEX photometry.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, Optional

log = logging.getLogger("exodus.ingestion.galex")

# VizieR catalog identifier for GALEX GR6+7 AIS
_VIZIER_CATALOG = "II/335/galex_ais"

# Columns we need from VizieR
_COLUMNS = [
    "RAJ2000",      # RA (J2000), deg
    "DEJ2000",      # Dec (J2000), deg
    "FUVmag",       # FUV magnitude (AB system)
    "e_FUVmag",     # FUV magnitude error
    "NUVmag",       # NUV magnitude (AB system)
    "e_NUVmag",     # NUV magnitude error
    "Fafl",         # FUV artifact flag
    "Nafl",         # NUV artifact flag
    "Fexf",         # FUV extraction flag
    "Nexf",         # NUV extraction flag
]

# Artifact flag bits to reject (unreliable photometry)
# Bit 1 (2): window reflection, Bit 2 (4): dichroic reflection,
# Bit 5 (32): detector rim, Bit 9 (512): bright star ghost
_BAD_ARTIFACT_MASK = 2 | 4 | 32 | 512


def query_galex_cone(
    ra: float,
    dec: float,
    radius_arcsec: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Query GALEX GR6+7 for UV photometry near the given position.

    Parameters
    ----------
    ra, dec : float
        J2000 coordinates in degrees.
    radius_arcsec : float
        Search radius in arcseconds (default 30").
        GALEX PSF FWHM is ~4.2" (FUV) / ~5.3" (NUV), so 30" is
        generous enough to capture the target while minimizing
        confusion.

    Returns
    -------
    dict or None
        Best (closest) GALEX match with keys:
        ``ra``, ``dec``, ``sep_arcsec``,
        ``fuv_mag``, ``fuv_err``, ``nuv_mag``, ``nuv_err``,
        ``artifact_clean`` (bool), ``data_source``.
        Returns None if no match found.
    """
    try:
        from astroquery.vizier import Vizier
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError:
        log.debug("astroquery not available for GALEX query")
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
        log.debug("GALEX VizieR query failed for (%.4f, %.4f): %s", ra, dec, exc)
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
    best_idx = int(seps.argmin())
    row = table[best_idx]

    # Extract magnitudes (may be masked/NaN for non-detections)
    fuv_mag = _safe_float(row, "FUVmag")
    fuv_err = _safe_float(row, "e_FUVmag")
    nuv_mag = _safe_float(row, "NUVmag")
    nuv_err = _safe_float(row, "e_NUVmag")

    # Check artifact flags
    fuv_artifact = int(row["Fafl"]) if row["Fafl"] is not None else 0
    nuv_artifact = int(row["Nafl"]) if row["Nafl"] is not None else 0

    try:
        fuv_artifact = int(fuv_artifact)
    except (ValueError, TypeError):
        fuv_artifact = 0
    try:
        nuv_artifact = int(nuv_artifact)
    except (ValueError, TypeError):
        nuv_artifact = 0

    fuv_clean = (fuv_artifact & _BAD_ARTIFACT_MASK) == 0
    nuv_clean = (nuv_artifact & _BAD_ARTIFACT_MASK) == 0

    match = {
        "ra": float(row["RAJ2000"]),
        "dec": float(row["DEJ2000"]),
        "sep_arcsec": float(seps[best_idx]),
        "fuv_mag": fuv_mag,
        "fuv_err": fuv_err,
        "nuv_mag": nuv_mag,
        "nuv_err": nuv_err,
        "fuv_artifact_flag": fuv_artifact,
        "nuv_artifact_flag": nuv_artifact,
        "artifact_clean": fuv_clean and nuv_clean,
        "data_source": "galex_vizier",
    }

    log.info(
        "GALEX match at %.1f\": FUV=%.2f NUV=%.2f (clean=%s)",
        match["sep_arcsec"],
        fuv_mag if fuv_mag is not None else -99,
        nuv_mag if nuv_mag is not None else -99,
        match["artifact_clean"],
    )

    return match


def compute_uv_metrics(
    galex_match: Optional[Dict[str, Any]],
    gaia_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute UV anomaly metrics from GALEX photometry.

    Metrics computed:
    1. FUV-NUV color: bluer than expected → activity; redder → absorption
    2. UV deficit: NUV fainter than predicted from Teff → possible absorption
    3. UV excess: NUV brighter than predicted → chromospheric activity

    Parameters
    ----------
    galex_match : dict or None
        Result from :func:`query_galex_cone`.
    gaia_params : dict or None
        Gaia stellar parameters (bp_rp, phot_g_mean_mag for Teff proxy).

    Returns
    -------
    dict
        UV anomaly metrics including:
        ``fuv_nuv_color``, ``uv_anomaly_score`` (0-1),
        ``is_uv_active``, ``is_uv_deficit``.
    """
    result = {
        "has_data": False,
        "fuv_nuv_color": None,
        "uv_anomaly_score": 0.0,
        "is_uv_active": False,
        "is_uv_deficit": False,
        "data_source": "none",
    }

    if galex_match is None:
        return result

    nuv = galex_match.get("nuv_mag")
    fuv = galex_match.get("fuv_mag")
    artifact_clean = galex_match.get("artifact_clean", True)

    if nuv is None or not np.isfinite(nuv):
        return result

    result["has_data"] = True
    result["data_source"] = galex_match.get("data_source", "galex")

    # FUV-NUV color (if both bands available)
    if fuv is not None and np.isfinite(fuv):
        fuv_nuv = fuv - nuv
        result["fuv_nuv_color"] = float(fuv_nuv)

        # Expected FUV-NUV for main-sequence stars: ~1.5-3.0 for FGK stars
        # Chromospherically active stars: < 1.0
        # Heavy UV absorption: > 4.0
        if fuv_nuv < 1.0:
            result["is_uv_active"] = True

    # NUV anomaly relative to expected from optical color
    # For FGK stars (Bp-Rp 0.5-2.0), expected NUV ≈ G + 3.5 ± 1.5 (rough)
    bp_rp = None
    g_mag = None
    if gaia_params:
        bp_rp = gaia_params.get("bp_rp")
        g_mag = gaia_params.get("phot_g_mean_mag")

    if bp_rp is not None and g_mag is not None and np.isfinite(bp_rp) and np.isfinite(g_mag):
        # Empirical NUV-G relation for main-sequence stars (AB system):
        # NUV ≈ G + 1.5 + 2.0 * (BP-RP) for FGK dwarfs
        # This is approximate but sufficient for anomaly detection.
        expected_nuv = g_mag + 1.5 + 2.0 * bp_rp
        nuv_residual = nuv - expected_nuv  # positive = fainter than expected

        result["expected_nuv"] = float(expected_nuv)
        result["nuv_residual"] = float(nuv_residual)

        # UV deficit: NUV significantly fainter than expected (absorption)
        if nuv_residual > 2.0:
            result["is_uv_deficit"] = True

    # Anomaly flag check
    if not artifact_clean:
        result["artifact_warning"] = True
        # Reduce confidence in metrics
        result["uv_anomaly_score"] = 0.0
        return result

    # Compute UV anomaly score (0-1)
    scores = []

    # FUV-NUV color anomaly
    if result["fuv_nuv_color"] is not None:
        color = result["fuv_nuv_color"]
        # Expected for FGK: ~1.5-3.0; anomalous if < 0.5 or > 5.0
        if color < 0.5:
            # Extremely blue → strong activity
            scores.append(min(1.0, (0.5 - color) / 2.0))
        elif color > 4.0:
            # Very red → unusual absorption
            scores.append(min(1.0, (color - 4.0) / 3.0))

    # NUV residual anomaly
    nuv_resid = result.get("nuv_residual")
    if nuv_resid is not None:
        # UV deficit (positive residual > 2 mag)
        if nuv_resid > 2.0:
            scores.append(min(1.0, (nuv_resid - 2.0) / 3.0))
        # UV excess (negative residual < -2 mag)
        elif nuv_resid < -2.0:
            scores.append(min(1.0, (-nuv_resid - 2.0) / 3.0))

    if scores:
        result["uv_anomaly_score"] = float(max(scores))

    return result


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

    # Vega — bright A0V star, should have UV detection
    print("Querying GALEX for Vega (18h 36m, +38° 47')...")
    match = query_galex_cone(279.2347, 38.7837)
    if match:
        print(f"  Match at {match['sep_arcsec']:.1f}\"")
        print(f"  FUV = {match['fuv_mag']}, NUV = {match['nuv_mag']}")
        print(f"  Artifact clean: {match['artifact_clean']}")
        metrics = compute_uv_metrics(match)
        print(f"  UV anomaly score: {metrics['uv_anomaly_score']:.3f}")
    else:
        print("  No GALEX match found (Vega may be too bright / saturated)")

    # Proxima Centauri — M dwarf, UV active
    print("\nQuerying GALEX for Proxima Centauri...")
    match2 = query_galex_cone(217.4292, -62.6794)
    if match2:
        print(f"  Match at {match2['sep_arcsec']:.1f}\"")
        print(f"  FUV = {match2['fuv_mag']}, NUV = {match2['nuv_mag']}")
        metrics2 = compute_uv_metrics(match2)
        print(f"  UV active: {metrics2['is_uv_active']}")
    else:
        print("  No GALEX match (low declination / coverage)")
