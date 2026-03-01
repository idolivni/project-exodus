"""
Infrared survey ingestion for Project EXODUS.

Queries 2MASS (J, H, Ks near-IR) and WISE/NEOWISE (W1-W4 mid-IR) photometry
via IRSA.  Combined with Gaia optical photometry, these bands let us model
expected stellar spectral energy distributions and flag anomalous mid-infrared
excess -- the primary observable signature of a Dyson-sphere-class structure.

Catalogs queried
----------------
- 2MASS Point Source Catalog (``fp_psc``): J (1.25 um), H (1.65 um), Ks (2.17 um)
- AllWISE Source Catalog (``allwise_p3as_psc``): W1 (3.4 um), W2 (4.6 um),
  W3 (12 um), W4 (22 um)

Queries go through ``astroquery.irsa.Irsa`` with automatic fallback to
VizieR (CDS Strasbourg) when IRSA is unreachable.  Results are locally
cached to avoid redundant network calls.
"""

from __future__ import annotations

import signal
import sys
from typing import Any, Dict, List, Optional, Tuple

import astropy.units as u
from astropy.coordinates import SkyCoord

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))

from src.utils import (
    cache_key,
    get_config,
    get_logger,
    load_cache,
    save_cache,
)

log = get_logger("ingestion.ir_surveys")

# ── Catalog & column definitions ─────────────────────────────────────
TWOMASS_CATALOG = "fp_psc"
ALLWISE_CATALOG = "allwise_p3as_psd"

# Columns we actually need from each catalog (keeps downloads small).
TWOMASS_COLUMNS = [
    "ra", "dec", "designation",
    "j_m", "j_msigcom",   # J-band mag + combined uncertainty
    "h_m", "h_msigcom",   # H-band
    "k_m", "k_msigcom",   # Ks-band
    "ph_qual",             # photometric quality flag (AAA = best)
]

ALLWISE_COLUMNS = [
    "ra", "dec", "designation",
    "w1mpro", "w1sigmpro",  # W1 profile-fit mag + uncertainty
    "w2mpro", "w2sigmpro",  # W2
    "w3mpro", "w3sigmpro",  # W3
    "w4mpro", "w4sigmpro",  # W4
    "cc_flags",              # contamination/confusion flags
    "ext_flg",               # extended-source flag
    "ph_qual",               # photometric quality
    # Per-band centroid positions (may not exist in all AllWISE table
    # versions — the query gracefully drops missing columns).
    # Used by ir_excess contamination check to detect background AGN
    # via short-wave vs. long-wave centroid offset.
    "w1ra", "w1dec",         # W1 centroid position
    "w2ra", "w2dec",         # W2 centroid position
    "w3ra", "w3dec",         # W3 centroid position
    "w4ra", "w4dec",         # W4 centroid position
]

# Friendly names we expose in the returned dict.
TWOMASS_BAND_MAP = {
    "j_m":  "J",
    "h_m":  "H",
    "k_m":  "Ks",
}

TWOMASS_ERR_MAP = {
    "j_msigcom":  "J_err",
    "h_msigcom":  "H_err",
    "k_msigcom":  "Ks_err",
}

WISE_BAND_MAP = {
    "w1mpro": "W1",
    "w2mpro": "W2",
    "w3mpro": "W3",
    "w4mpro": "W4",
}

WISE_ERR_MAP = {
    "w1sigmpro": "W1_err",
    "w2sigmpro": "W2_err",
    "w3sigmpro": "W3_err",
    "w4sigmpro": "W4_err",
}

# CatWISE2020 (II/365): proper-motion-corrected W1/W2 photometry from
# combined WISE+NEOWISE (2010-2020).  Better astrometry and centroid
# positions than AllWISE, plus source proper motions.
CATWISE_CATALOG_VIZIER = "II/365/catwise"
CATWISE_COLUMNS_VIZIER = [
    "RA_ICRS", "DE_ICRS", "Name",
    "W1mproPM", "W2mproPM",     # W1/W2 PM-corrected profile-fit mag
    "pmRA", "e_pmRA",            # Proper motion RA (mas/yr)
    "pmDE", "e_pmDE",            # Proper motion Dec
    "abf",                       # artifact/blend flags
]

_VIZIER_CATWISE_COLS = {
    "RA_ICRS": "ra", "DE_ICRS": "dec", "Name": "designation",
    "W1mproPM": "w1mpro_pm", "W2mproPM": "w2mpro_pm",
    "pmRA": "pmra_wise", "e_pmRA": "e_pmra_wise",
    "pmDE": "pmdec_wise", "e_pmDE": "e_pmdec_wise",
    "abf": "ab_flags",
}

CATWISE_BAND_MAP = {
    "w1mpro_pm": "W1_catwise",
    "w2mpro_pm": "W2_catwise",
}

CATWISE_ERR_MAP = {
    # CatWISE VizieR table provides flux errors (FW1pm, e_FW1pm) but not
    # direct mag errors for PM-corrected photometry.  Errors are estimated
    # downstream from flux uncertainties if needed.
}

CACHE_SUBFOLDER = "ir_surveys"


# ── Internal helpers ─────────────────────────────────────────────────

def _default_radius() -> float:
    """Return the default cross-match radius (arcsec) from project config."""
    try:
        cfg = get_config()
        return float(cfg["search"]["crossmatch_radius_arcsec"])
    except Exception:
        return 3.0


class _IrsaQueryError(Exception):
    """Raised when an IRSA query fails due to network/service error."""


# IRSA query timeout (seconds).  IRSA TAP can hang indefinitely when their
# backend is overloaded — a finite timeout lets us fall back to VizieR.
_IRSA_TIMEOUT = 30
_IRSA_HARD_TIMEOUT = 60  # hard alarm-based deadline (catches stuck sockets)


class _HardTimeoutError(Exception):
    """Raised when a query exceeds the hard timeout (signal alarm)."""


def _hard_timeout_handler(signum, frame):
    raise _HardTimeoutError(f"IRSA query exceeded {_IRSA_HARD_TIMEOUT}s hard timeout")

# VizieR catalog IDs and column mappings for the IRSA fallback path.
_VIZIER_2MASS_CATALOG = "II/246/out"
_VIZIER_ALLWISE_CATALOG = "II/328/allwise"

_VIZIER_2MASS_COLS = {
    "RAJ2000": "ra", "DEJ2000": "dec", "_2MASS": "designation",
    "Jmag": "j_m", "e_Jmag": "j_msigcom",
    "Hmag": "h_m", "e_Hmag": "h_msigcom",
    "Kmag": "k_m", "e_Kmag": "k_msigcom",
    "Qflg": "ph_qual",
}

_VIZIER_ALLWISE_COLS = {
    "RAJ2000": "ra", "DEJ2000": "dec", "AllWISE": "designation",
    "W1mag": "w1mpro", "e_W1mag": "w1sigmpro",
    "W2mag": "w2mpro", "e_W2mag": "w2sigmpro",
    "W3mag": "w3mpro", "e_W3mag": "w3sigmpro",
    "W4mag": "w4mpro", "e_W4mag": "w4sigmpro",
    "ccf": "cc_flags", "ex": "ext_flg", "qph": "ph_qual",
}


def _vizier_fallback(
    vizier_catalog: str,
    col_map: Dict[str, str],
    ra: float,
    dec: float,
    radius_arcsec: float,
) -> Optional[Any]:
    """Query VizieR as fallback when IRSA is unreachable.

    Returns an astropy Table with columns renamed to match IRSA naming,
    or *None* if no sources found.  Raises on network failure.
    """
    from astroquery.vizier import Vizier

    viz = Vizier(columns=list(col_map.keys()), row_limit=20)
    coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    tables = viz.query_region(
        coord, radius=radius_arcsec * u.arcsec, catalog=vizier_catalog,
    )
    if not tables or len(tables) == 0:
        return None
    table = tables[0]
    if len(table) == 0:
        return None
    for viz_name, irsa_name in col_map.items():
        if viz_name in table.colnames:
            table.rename_column(viz_name, irsa_name)
    return table


def _irsa_query(
    catalog: str,
    ra: float,
    dec: float,
    radius_arcsec: float,
    columns: List[str],
) -> Optional[Any]:
    """
    Execute a cone search on an IRSA catalog.

    Returns an ``astropy.table.Table`` or *None* if no sources are found.
    Raises ``_IrsaQueryError`` on network/service failure so callers can
    distinguish "no match" from "query failed" and avoid caching errors.
    """
    try:
        from astroquery.ipac.irsa import Irsa
    except ImportError:
        from astroquery.irsa import Irsa

    Irsa.TIMEOUT = _IRSA_TIMEOUT

    coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    radius = radius_arcsec * u.arcsec

    # Set a hard alarm-based timeout to catch hung socket connections
    # that bypass the HTTP-level timeout.
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _hard_timeout_handler)
        signal.alarm(_IRSA_HARD_TIMEOUT)
        table = Irsa.query_region(
            coord,
            catalog=catalog,
            radius=radius,
        )
        signal.alarm(0)  # cancel alarm on success
    except _HardTimeoutError as exc:
        log.warning("IRSA hard timeout for catalog=%s ra=%.6f dec=%.6f",
                    catalog, ra, dec)
        raise _IrsaQueryError(str(exc)) from exc
    except Exception as exc:
        signal.alarm(0)  # cancel alarm on other exceptions
        log.warning("IRSA query failed for catalog=%s ra=%.6f dec=%.6f: %s",
                    catalog, ra, dec, exc)
        raise _IrsaQueryError(str(exc)) from exc
    finally:
        signal.signal(signal.SIGALRM, old_handler or signal.SIG_DFL)

    if table is None or len(table) == 0:
        return None

    # Keep only the columns we care about (if they exist in the result).
    existing = [c for c in columns if c in table.colnames]
    return table[existing]


def _pick_closest(table, ra: float, dec: float) -> Optional[Dict[str, Any]]:
    """
    Given an astropy Table of cone-search results, return the single row
    closest to (ra, dec) as a plain dict.
    """
    if table is None or len(table) == 0:
        return None

    target = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    catalog_coords = SkyCoord(
        ra=table["ra"], dec=table["dec"], unit="deg", frame="icrs"
    )
    sep = target.separation(catalog_coords)
    idx = sep.argmin()

    row = {}
    for col in table.colnames:
        val = table[col][idx]
        # Convert numpy/masked types to plain Python for JSON serialisation.
        try:
            if hasattr(val, "item"):
                val = val.item()
            if val is None or (hasattr(val, "__class__") and "masked" in type(val).__name__.lower()):
                val = None
        except Exception:
            val = None
        row[col] = val

    row["_sep_arcsec"] = float(sep[idx].arcsec)
    return row


def _row_to_bands(
    row: Optional[Dict[str, Any]],
    band_map: Dict[str, str],
    err_map: Dict[str, str],
) -> Dict[str, Any]:
    """Translate raw catalog column names to friendly band names."""
    result: Dict[str, Any] = {}
    if row is None:
        return result

    for col, band in band_map.items():
        val = row.get(col)
        result[band] = float(val) if val is not None else None

    for col, name in err_map.items():
        val = row.get(col)
        result[name] = float(val) if val is not None else None

    # Carry through useful metadata.
    if "designation" in row:
        result["designation"] = row["designation"]
    if "ph_qual" in row:
        result["ph_qual"] = row["ph_qual"]
    if "_sep_arcsec" in row:
        result["match_sep_arcsec"] = row["_sep_arcsec"]

    return result


# ── Public API ───────────────────────────────────────────────────────

def get_2mass(
    ra: float,
    dec: float,
    radius_arcsec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query 2MASS for the nearest point source within *radius_arcsec* of
    (ra, dec).

    Returns a dict with keys ``J``, ``H``, ``Ks`` (magnitudes), their
    ``*_err`` uncertainties, ``designation``, ``ph_qual``, and
    ``match_sep_arcsec``.  Returns an empty dict if nothing is found.
    """
    if radius_arcsec is None:
        radius_arcsec = _default_radius()

    key = cache_key("2mass", round(ra, 6), round(dec, 6), radius_arcsec)
    cached = load_cache(key, subfolder=CACHE_SUBFOLDER)
    if cached is not None:
        log.debug("2MASS cache hit for ra=%.6f dec=%.6f", ra, dec)
        return cached

    log.info("Querying 2MASS for ra=%.6f dec=%.6f radius=%.1f\"", ra, dec, radius_arcsec)
    try:
        table = _irsa_query(TWOMASS_CATALOG, ra, dec, radius_arcsec, TWOMASS_COLUMNS)
    except _IrsaQueryError:
        log.info("IRSA failed — trying VizieR fallback for 2MASS")
        try:
            table = _vizier_fallback(
                _VIZIER_2MASS_CATALOG, _VIZIER_2MASS_COLS,
                ra, dec, radius_arcsec,
            )
        except Exception as exc:
            log.warning("VizieR 2MASS fallback also failed: %s", exc)
            return {}  # Do NOT cache failed queries

    row = _pick_closest(table, ra, dec)
    result = _row_to_bands(row, TWOMASS_BAND_MAP, TWOMASS_ERR_MAP)

    save_cache(key, result, subfolder=CACHE_SUBFOLDER, fmt="json")
    return result


def get_wise(
    ra: float,
    dec: float,
    radius_arcsec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query AllWISE for the nearest source within *radius_arcsec* of (ra, dec).

    Returns a dict with keys ``W1`` .. ``W4`` (magnitudes), their ``*_err``
    uncertainties, ``designation``, ``ph_qual``, and ``match_sep_arcsec``.
    Returns an empty dict if nothing is found.
    """
    if radius_arcsec is None:
        radius_arcsec = _default_radius()

    key = cache_key("wise", round(ra, 6), round(dec, 6), radius_arcsec)
    cached = load_cache(key, subfolder=CACHE_SUBFOLDER)
    if cached is not None:
        log.debug("WISE cache hit for ra=%.6f dec=%.6f", ra, dec)
        return cached

    log.info("Querying AllWISE for ra=%.6f dec=%.6f radius=%.1f\"", ra, dec, radius_arcsec)
    try:
        table = _irsa_query(ALLWISE_CATALOG, ra, dec, radius_arcsec, ALLWISE_COLUMNS)
    except _IrsaQueryError:
        log.info("IRSA failed — trying VizieR fallback for AllWISE")
        try:
            table = _vizier_fallback(
                _VIZIER_ALLWISE_CATALOG, _VIZIER_ALLWISE_COLS,
                ra, dec, radius_arcsec,
            )
        except Exception as exc:
            log.warning("VizieR AllWISE fallback also failed: %s", exc)
            return {}  # Do NOT cache failed queries

    row = _pick_closest(table, ra, dec)
    result = _row_to_bands(row, WISE_BAND_MAP, WISE_ERR_MAP)

    # Stash a few WISE-specific quality flags that downstream filters use.
    if row is not None:
        result["cc_flags"] = row.get("cc_flags")
        result["ext_flg"] = row.get("ext_flg")
        # Per-band centroid positions for contamination check.
        # Map IRSA names (w1ra) → ir_excess expected names (w1_ra).
        for band_n in (1, 2, 3, 4):
            for coord in ("ra", "dec"):
                irsa_key = f"w{band_n}{coord}"    # e.g. "w1ra"
                out_key = f"w{band_n}_{coord}"     # e.g. "w1_ra"
                val = row.get(irsa_key)
                if val is not None:
                    try:
                        result[out_key] = float(val)
                    except (TypeError, ValueError):
                        pass

    save_cache(key, result, subfolder=CACHE_SUBFOLDER, fmt="json")
    return result


def get_catwise(
    ra: float,
    dec: float,
    radius_arcsec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query CatWISE2020 via VizieR for the nearest source within *radius_arcsec*.

    CatWISE2020 provides proper-motion-corrected W1/W2 photometry from
    10 years of combined WISE+NEOWISE data (2010-2020), plus source proper
    motions measured from the WISE astrometry alone.

    Returns a dict with keys ``W1_catwise``, ``W2_catwise`` (PM-corrected
    magnitudes), their ``*_err`` uncertainties, ``pmra_wise``,
    ``pmdec_wise`` (proper motion from WISE astrometry in mas/yr),
    ``designation``, and ``match_sep_arcsec``.

    Returns an empty dict if nothing is found.
    """
    if radius_arcsec is None:
        radius_arcsec = _default_radius()

    key = cache_key("catwise", round(ra, 6), round(dec, 6), radius_arcsec)
    cached = load_cache(key, subfolder=CACHE_SUBFOLDER)
    if cached is not None:
        log.debug("CatWISE cache hit for ra=%.6f dec=%.6f", ra, dec)
        return cached

    log.info("Querying CatWISE2020 for ra=%.6f dec=%.6f radius=%.1f\"",
             ra, dec, radius_arcsec)
    try:
        table = _vizier_fallback(
            CATWISE_CATALOG_VIZIER, _VIZIER_CATWISE_COLS,
            ra, dec, radius_arcsec,
        )
    except Exception as exc:
        log.warning("CatWISE2020 query failed: %s", exc)
        return {}

    row = _pick_closest(table, ra, dec)
    result = _row_to_bands(row, CATWISE_BAND_MAP, CATWISE_ERR_MAP)

    # Add proper motion from WISE astrometry
    if row is not None:
        for pm_key in ("pmra_wise", "e_pmra_wise", "pmdec_wise", "e_pmdec_wise"):
            val = row.get(pm_key)
            if val is not None:
                try:
                    result[pm_key] = float(val)
                except (TypeError, ValueError):
                    pass
        result["ab_flags"] = row.get("ab_flags")

    save_cache(key, result, subfolder=CACHE_SUBFOLDER, fmt="json")
    return result


def get_ir_photometry(
    ra: float,
    dec: float,
    radius_arcsec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return all available IR photometry (2MASS + AllWISE + CatWISE2020)
    for the position (ra, dec) as a single flat dict.

    Keys present depend on catalog matches:
      - ``J``, ``H``, ``Ks``       (2MASS, if matched)
      - ``W1``, ``W2``, ``W3``, ``W4``  (AllWISE, if matched)
      - ``W1_catwise``, ``W2_catwise``  (CatWISE2020 PM-corrected, if matched)
      - ``pmra_wise``, ``pmdec_wise``  (WISE-measured proper motion, mas/yr)
      - Corresponding ``*_err`` uncertainty keys
      - ``twomass_designation``, ``wise_designation``, ``catwise_designation``
      - ``twomass_sep_arcsec``, ``wise_sep_arcsec``, ``catwise_sep_arcsec``

    An empty dict means no IR counterpart was found in any catalog.
    """
    if radius_arcsec is None:
        radius_arcsec = _default_radius()

    merged: Dict[str, Any] = {"ra": ra, "dec": dec}

    # -- 2MASS --
    twomass = get_2mass(ra, dec, radius_arcsec=radius_arcsec)
    if twomass:
        for band in ("J", "H", "Ks", "J_err", "H_err", "Ks_err"):
            if band in twomass:
                merged[band] = twomass[band]
        merged["twomass_designation"] = twomass.get("designation")
        merged["twomass_ph_qual"] = twomass.get("ph_qual")
        merged["twomass_sep_arcsec"] = twomass.get("match_sep_arcsec")

    # -- WISE --
    wise = get_wise(ra, dec, radius_arcsec=radius_arcsec)
    if wise:
        for band in ("W1", "W2", "W3", "W4", "W1_err", "W2_err", "W3_err", "W4_err"):
            if band in wise:
                merged[band] = wise[band]
        merged["wise_designation"] = wise.get("designation")
        merged["wise_ph_qual"] = wise.get("ph_qual")
        merged["wise_sep_arcsec"] = wise.get("match_sep_arcsec")
        merged["wise_cc_flags"] = wise.get("cc_flags")
        merged["wise_ext_flg"] = wise.get("ext_flg")
        # Per-band centroid positions for contamination check.
        for band_n in (1, 2, 3, 4):
            for coord in ("ra", "dec"):
                key = f"w{band_n}_{coord}"  # e.g. "w1_ra"
                if key in wise:
                    merged[key] = wise[key]

    # -- CatWISE2020 (improved W1/W2 + proper motions from WISE) --
    catwise = get_catwise(ra, dec, radius_arcsec=radius_arcsec)
    if catwise:
        for band in ("W1_catwise", "W2_catwise"):
            if band in catwise:
                merged[band] = catwise[band]
        merged["catwise_designation"] = catwise.get("designation")
        merged["catwise_sep_arcsec"] = catwise.get("match_sep_arcsec")
        # WISE-measured proper motions (independent of Gaia)
        for pm_key in ("pmra_wise", "e_pmra_wise", "pmdec_wise", "e_pmdec_wise"):
            if pm_key in catwise:
                merged[pm_key] = catwise[pm_key]
        merged["catwise_ab_flags"] = catwise.get("ab_flags")

    return merged


def get_ir_photometry_batch(
    targets: List[Tuple[float, float]],
    radius_arcsec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Query IR photometry for a list of (ra, dec) targets.

    Returns a list of dicts (same format as :func:`get_ir_photometry`),
    one per input target.  Failed/missing entries are empty dicts.
    """
    if radius_arcsec is None:
        radius_arcsec = _default_radius()

    results = []
    total = len(targets)
    for idx, (ra, dec) in enumerate(targets, 1):
        log.info("IR photometry %d/%d: ra=%.6f dec=%.6f", idx, total, ra, dec)
        try:
            phot = get_ir_photometry(ra, dec, radius_arcsec=radius_arcsec)
        except Exception as exc:
            log.error("Failed IR photometry for ra=%.6f dec=%.6f: %s", ra, dec, exc)
            phot = {"ra": ra, "dec": dec}
        results.append(phot)

    return results


# ── CLI demo ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json

    # Tabby's Star (KIC 8462852) -- the canonical Dyson-sphere candidate.
    # Its anomalous dimming events make it a perfect smoke-test target.
    DEMO_RA = 301.5644    # degrees (J2000)
    DEMO_DEC = 44.4568    # degrees (J2000)
    DEMO_NAME = "KIC 8462852 (Tabby's Star)"

    print(f"\n{'=' * 60}")
    print(f"  Project EXODUS -- IR Survey Ingestion Demo")
    print(f"  Target: {DEMO_NAME}")
    print(f"  RA={DEMO_RA:.4f}  Dec={DEMO_DEC:.4f}")
    print(f"{'=' * 60}\n")

    print("--- 2MASS (J, H, Ks) ---")
    twomass_data = get_2mass(DEMO_RA, DEMO_DEC)
    if twomass_data:
        print(_json.dumps(twomass_data, indent=2))
    else:
        print("  No 2MASS match found.")

    print("\n--- AllWISE (W1, W2, W3, W4) ---")
    wise_data = get_wise(DEMO_RA, DEMO_DEC)
    if wise_data:
        print(_json.dumps(wise_data, indent=2))
    else:
        print("  No AllWISE match found.")

    print(f"\n--- Combined IR photometry ---")
    combined = get_ir_photometry(DEMO_RA, DEMO_DEC)
    print(_json.dumps(combined, indent=2))

    # Quick sanity check: flag if W3/W4 look anomalously bright relative
    # to Ks (a crude IR-excess indicator).
    ks = combined.get("Ks")
    w3 = combined.get("W3")
    w4 = combined.get("W4")
    if ks is not None and w3 is not None:
        excess_w3 = ks - w3
        print(f"\n  Ks - W3 = {excess_w3:+.3f} mag", end="")
        if excess_w3 > 0.5:
            print("  ** potential W3 excess **")
        else:
            print("  (normal)")
    if ks is not None and w4 is not None:
        excess_w4 = ks - w4
        print(f"  Ks - W4 = {excess_w4:+.3f} mag", end="")
        if excess_w4 > 1.0:
            print("  ** potential W4 excess **")
        else:
            print("  (normal)")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}\n")
