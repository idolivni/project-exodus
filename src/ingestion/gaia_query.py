"""
Gaia DR3 TAP query module for Project EXODUS.

Queries Gaia DR3 via the TAP service (using astroquery) for:
  - Photometry: G, BP, RP magnitudes for stellar characterization
  - Astrometry: parallax, proper motion for distance / motion anomalies
  - Epoch photometry: time-series brightness data for anomaly hunting

All results are cached locally to avoid redundant network calls.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _FutureTimeout
from typing import Dict, List, Optional, Tuple

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
# NOTE: `from astroquery.gaia import Gaia` is deliberately NOT imported at
# module level.  The import creates a GaiaClass() singleton that contacts
# the ESA TAP server — which can hang for 2+ minutes when ESA is down.
# Instead, we import it lazily only where needed (epoch photometry datalink).

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from src.utils import (
    cache_key,
    get_config,
    get_logger,
    load_cache,
    save_cache,
)

log = get_logger("ingestion.gaia")

# ---------------------------------------------------------------------------
# Gaia TAP mirror endpoints
# ---------------------------------------------------------------------------
# Primary (ESA) is tried first.  On timeout or transient failure, we fall
# back to ARI Heidelberg, then VizieR/CDS.  Table names differ per mirror.

_GAIA_MIRRORS: List[Dict[str, str]] = [
    {
        "name": "ESA",
        "url": "https://gea.esac.esa.int/tap-server/tap",
        "table": "gaiadr3.gaia_source",
    },
    {
        "name": "ARI Heidelberg",
        "url": "https://gaia.ari.uni-heidelberg.de/tap",
        "table": "gaiadr3.gaia_source",
    },
    {
        "name": "VizieR/CDS",
        "url": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
        "table": '"I/355/gaiadr3"',
    },
]

_DEFAULT_GAIA_TIMEOUT = 60  # seconds per query attempt

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

def _configure_gaia_tap() -> None:
    """Read the configured Gaia TAP URL and update the primary mirror.

    All queries now use ``TapPlus`` directly (not the global ``Gaia``
    class), so this only needs to update ``_GAIA_MIRRORS[0]``.
    """
    cfg = get_config()
    tap_url = cfg.get("catalogs", {}).get("gaia", {}).get(
        "url", "https://gea.esac.esa.int/tap-server/tap"
    )
    _GAIA_MIRRORS[0]["url"] = tap_url


_configure_gaia_tap()

_DEFAULT_RADIUS_ARCSEC: float = (
    get_config().get("search", {}).get("crossmatch_radius_arcsec", 3.0)
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _table_to_dataframe(tbl: Table) -> pd.DataFrame:
    """Convert an astropy Table to a pandas DataFrame, dropping mask cols."""
    df = tbl.to_pandas()
    return df


def _cone_adql(
    ra: float,
    dec: float,
    radius_arcsec: float,
    columns: str = "*",
    table: str = "gaiadr3.gaia_source",
    top_n: int = 10,
    order_by: str = "",
) -> str:
    """Build an ADQL cone-search query.

    Note: we avoid the DISTANCE() function and complex ORDER BY
    expressions because some TAP servers (including the Gaia Archive
    after infrastructure upgrades) have limited ADQL parser support.
    For our sub-arcminute search radii this doesn't matter -- we
    typically get 0-2 sources and pick the closest downstream.

    For wide searches (> 20"), top_n is automatically raised to ensure
    the actual target is included in the results, since high proper
    motion stars can move far from their J2000 catalogue positions.

    Parameters
    ----------
    order_by : str
        Optional ORDER BY clause (e.g. ``"parallax DESC"``).
        Used by the astrometry query to prioritize nearby stars.
    """
    # Scale top_n with search radius to avoid missing targets in wide cones
    if radius_arcsec > 20.0 and top_n < 200:
        top_n = 200

    radius_deg = radius_arcsec / 3600.0
    query = (
        f"SELECT TOP {top_n} {columns} "
        f"FROM {table} "
        f"WHERE CONTAINS("
        f"POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})"
        f") = 1"
    )
    if order_by:
        query += f" ORDER BY {order_by}"
    return query


def _execute_with_timeout(
    adql: str, tap_url: str, timeout_sec: int,
) -> pd.DataFrame:
    """Execute a Gaia TAP query in a worker thread with timeout.

    Uses ``TapPlus`` directly (not the global ``Gaia`` class) so each
    call is independent — safe for concurrent use from thread pools.
    Thread-based timeout (not ``signal.SIGALRM``) so it works inside
    ``ThreadPoolExecutor`` workers.

    Important: we use ``shutdown(wait=False)`` so the main thread isn't
    blocked by a hung worker when the timeout fires.  The orphaned daemon
    thread will eventually be cleaned up when the process exits.
    """
    from astroquery.utils.tap.core import TapPlus

    def _do_query() -> pd.DataFrame:
        tap = TapPlus(url=tap_url)
        job = tap.launch_job(adql)
        tbl = job.get_results()
        return _table_to_dataframe(tbl)

    pool = _TPE(max_workers=1)
    try:
        fut = pool.submit(_do_query)
        return fut.result(timeout=timeout_sec if timeout_sec > 0 else None)
    finally:
        # wait=False — don't block on a hung worker thread
        pool.shutdown(wait=False)


def _run_sync_query(
    adql: str,
    max_retries: int = 3,
    timeout_sec: int = 0,
    mirror_fallback: bool = True,
) -> pd.DataFrame:
    """Execute a synchronous TAP query with timeout and mirror fallback.

    Tries ESA first, then ARI Heidelberg, then VizieR/CDS.  Each mirror
    gets ``max_retries`` attempts with exponential backoff.  Timeout is
    enforced per attempt via a thread-based mechanism.

    Parameters
    ----------
    adql : str
        ADQL query string using the primary table name
        (``gaiadr3.gaia_source``).
    max_retries : int
        Retry attempts per mirror endpoint.
    timeout_sec : int
        Per-attempt timeout in seconds.  0 = use config default.
    mirror_fallback : bool
        If True, try alternative TAP mirrors on failure.
    """
    import time as _time

    cfg = get_config()
    perf = cfg.get("performance", {})
    if timeout_sec <= 0:
        timeout_sec = perf.get("gaia_timeout_sec", _DEFAULT_GAIA_TIMEOUT)
    if not perf.get("gaia_mirror_fallback", True):
        mirror_fallback = False

    mirrors = list(_GAIA_MIRRORS) if mirror_fallback else [_GAIA_MIRRORS[0]]
    primary_table = _GAIA_MIRRORS[0]["table"]

    log.debug("ADQL: %s", adql)

    last_exc: Optional[Exception] = None
    for mirror in mirrors:
        # Rewrite table name for this mirror
        mirror_adql = adql.replace(primary_table, mirror["table"])

        for attempt in range(max_retries):
            try:
                result = _execute_with_timeout(
                    mirror_adql, mirror["url"], timeout_sec,
                )
                if mirror["name"] != "ESA":
                    log.info("Gaia query succeeded via %s mirror", mirror["name"])
                return result

            except _FutureTimeout:
                log.warning(
                    "Gaia query timed out (%ds) on %s (attempt %d/%d)",
                    timeout_sec, mirror["name"], attempt + 1, max_retries,
                )
                last_exc = TimeoutError(
                    f"Gaia query timed out after {timeout_sec}s on {mirror['name']}"
                )
                if attempt < max_retries - 1:
                    _time.sleep(2 ** (attempt + 1))

            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                is_transient = any(tok in exc_str for tok in (
                    "408", "500", "timeout", "timed out", "connection",
                    "reset", "aborted",
                ))
                if is_transient and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    log.warning(
                        "Gaia query on %s attempt %d/%d failed: %s. "
                        "Retrying in %ds...",
                        mirror["name"], attempt + 1, max_retries, exc, wait,
                    )
                    _time.sleep(wait)
                elif is_transient:
                    log.warning(
                        "Gaia query exhausted retries on %s: %s",
                        mirror["name"], exc,
                    )
                    break  # Try next mirror
                else:
                    raise  # Non-transient error — propagate immediately

    raise RuntimeError(
        f"All Gaia TAP mirrors exhausted for query. Last error: {last_exc}"
    )

# ---------------------------------------------------------------------------
# Public API: stellar parameters (photometry)
# ---------------------------------------------------------------------------

def get_stellar_params(
    ra: float,
    dec: float,
    radius_arcsec: float = _DEFAULT_RADIUS_ARCSEC,
) -> Optional[pd.DataFrame]:
    """Return Gaia DR3 photometry for sources near (RA, Dec).

    Columns returned include source_id, ra, dec, phot_g_mean_mag,
    phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, teff_gspphot, logg_gspphot,
    and mh_gspphot (where available).

    Parameters
    ----------
    ra : float
        Right ascension in decimal degrees (ICRS).
    dec : float
        Declination in decimal degrees (ICRS).
    radius_arcsec : float
        Search cone radius in arcseconds (default from config).

    Returns
    -------
    pd.DataFrame or None
        Query results, or *None* if the query fails or returns no rows.
    """
    key = cache_key("gaia_stellar", ra, dec, radius_arcsec)
    cached = load_cache(key, subfolder="gaia")
    if cached is not None:
        log.info("Stellar params cache hit for (%.4f, %.4f)", ra, dec)
        if isinstance(cached, pd.DataFrame):
            return cached
        return pd.DataFrame(cached)

    columns = (
        "source_id, ra, dec, "
        "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, "
        "teff_gspphot, logg_gspphot, mh_gspphot"
    )
    adql = _cone_adql(ra, dec, radius_arcsec, columns=columns)

    log.info("Querying Gaia DR3 stellar params for (%.4f, %.4f) r=%.1f\"", ra, dec, radius_arcsec)
    try:
        df = _run_sync_query(adql)
    except Exception as exc:
        log.error("Gaia stellar-param query failed: %s", exc)
        return None

    if df.empty:
        log.warning("No Gaia sources within %.1f\" of (%.4f, %.4f)", radius_arcsec, ra, dec)
        return None

    save_cache(key, df, subfolder="gaia", fmt="csv")
    log.info("Cached %d stellar-param rows for (%.4f, %.4f)", len(df), ra, dec)
    return df

# ---------------------------------------------------------------------------
# Public API: astrometry
# ---------------------------------------------------------------------------

def get_astrometry(
    ra: float,
    dec: float,
    radius_arcsec: float = _DEFAULT_RADIUS_ARCSEC,
) -> Optional[pd.DataFrame]:
    """Return Gaia DR3 astrometry for sources near (RA, Dec).

    Columns include parallax, parallax_error, pmra, pmdec, and RUWE
    (re-normalised unit weight error) which flags problematic astrometric
    solutions and can hint at unresolved companions.

    Parameters
    ----------
    ra : float
        Right ascension in decimal degrees (ICRS).
    dec : float
        Declination in decimal degrees (ICRS).
    radius_arcsec : float
        Search cone radius in arcseconds.

    Returns
    -------
    pd.DataFrame or None
    """
    key = cache_key("gaia_astrom", ra, dec, radius_arcsec)
    cached = load_cache(key, subfolder="gaia")
    if cached is not None:
        log.info("Astrometry cache hit for (%.4f, %.4f)", ra, dec)
        if isinstance(cached, pd.DataFrame):
            return cached
        return pd.DataFrame(cached)

    columns = (
        "source_id, ra, dec, "
        "parallax, parallax_error, parallax_over_error, "
        "pmra, pmra_error, pmdec, pmdec_error, "
        "ruwe, astrometric_excess_noise, astrometric_excess_noise_sig, "
        "ipd_frac_multi_peak, non_single_star"
    )
    # ORDER BY parallax DESC ensures nearby/high-PM stars appear first in
    # the TOP N results, preventing them from being displaced by faint
    # background sources in wide cone searches.
    adql = _cone_adql(
        ra, dec, radius_arcsec, columns=columns,
        order_by="parallax DESC",
    )

    log.info("Querying Gaia DR3 astrometry for (%.4f, %.4f) r=%.1f\"", ra, dec, radius_arcsec)
    try:
        df = _run_sync_query(adql)
    except Exception as exc:
        log.error("Gaia astrometry query failed: %s", exc)
        return None

    if df.empty:
        log.warning("No astrometry within %.1f\" of (%.4f, %.4f)", radius_arcsec, ra, dec)
        return None

    save_cache(key, df, subfolder="gaia", fmt="csv")
    log.info("Cached %d astrometry rows for (%.4f, %.4f)", len(df), ra, dec)
    return df

# ---------------------------------------------------------------------------
# Public API: epoch photometry (time-series light curves)
# ---------------------------------------------------------------------------

def get_epoch_photometry(source_id: int) -> Optional[pd.DataFrame]:
    """Retrieve Gaia DR3 epoch photometry for a single source via DataLink.

    Epoch photometry provides individual-transit G, BP, and RP brightness
    measurements -- the raw light curve needed for anomaly detection.

    **Important:** The ``gaiadr3.epoch_photometry`` table is NOT accessible
    via ADQL TAP queries on the ESA Gaia Archive.  This function uses the
    DataLink protocol (``Gaia.load_data()``) to retrieve the data.

    The DataLink INDIVIDUAL format returns one row per transit per band
    (G/BP/RP) with columns: source_id, transit_id, band, time, flux,
    flux_error, mag.  We filter to G-band and rename columns for
    backward compatibility with all downstream consumers.

    Parameters
    ----------
    source_id : int
        Gaia DR3 source identifier.

    Returns
    -------
    pd.DataFrame or None
        G-band epoch photometry with columns: *source_id*, *g_transit_id*,
        *g_obs_time*, *g_transit_mag*, *g_transit_flux*,
        *g_transit_flux_error*.  Sorted by g_obs_time ascending.
        Returns None if no data is available or on error.
    """
    key = cache_key("gaia_epoch", source_id)
    cached = load_cache(key, subfolder="gaia")
    if cached is not None:
        log.info("Epoch photometry cache hit for source_id=%s", source_id)
        if isinstance(cached, pd.DataFrame):
            return cached
        return pd.DataFrame(cached)

    log.info("Retrieving Gaia DR3 epoch photometry via DataLink for source_id=%s", source_id)
    try:
        # Lazy import — creating the Gaia singleton contacts ESA and can hang
        from astroquery.gaia import Gaia as _Gaia
        datalink = _Gaia.load_data(
            ids=[source_id],
            data_release='Gaia DR3',
            retrieval_type='EPOCH_PHOTOMETRY',
            data_structure='INDIVIDUAL',
            valid_data=True,
            format='votable',
            verbose=False,
        )
    except Exception as exc:
        log.error("DataLink epoch photometry failed for source_id=%s: %s",
                  source_id, exc)
        return None

    # DataLink returns a dict: {filename: [Table, ...]}
    if not datalink:
        log.warning("No epoch photometry available for source_id=%s", source_id)
        return None

    dl_keys = list(datalink.keys())
    if not dl_keys:
        log.warning("DataLink returned empty dict for source_id=%s", source_id)
        return None

    # Extract the first table from the first key
    table_list = datalink[dl_keys[0]]
    if not table_list:
        log.warning("DataLink returned empty table list for source_id=%s", source_id)
        return None

    df_all = table_list[0].to_pandas()

    if df_all.empty:
        log.warning("No epoch photometry rows for source_id=%s", source_id)
        return None

    log.info("DataLink returned %d rows across bands for source_id=%s",
             len(df_all), source_id)

    # Filter to G-band only (downstream consumers expect G-band columns)
    if "band" in df_all.columns:
        df = df_all[df_all["band"] == "G"].copy()
    else:
        # Unexpected format -- use all rows
        df = df_all.copy()

    if df.empty:
        log.warning("No G-band epoch photometry for source_id=%s", source_id)
        return None

    # Rename DataLink INDIVIDUAL columns to backward-compatible names
    # DataLink: time, mag, flux, flux_error, transit_id
    # Expected: g_obs_time, g_transit_mag, g_transit_flux, g_transit_flux_error, g_transit_id
    rename_map = {
        "time": "g_obs_time",
        "mag": "g_transit_mag",
        "flux": "g_transit_flux",
        "flux_error": "g_transit_flux_error",
        "transit_id": "g_transit_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure source_id column exists
    if "source_id" not in df.columns:
        df["source_id"] = source_id

    # Sort by observation time
    if "g_obs_time" in df.columns:
        df = df.sort_values("g_obs_time").reset_index(drop=True)

    # ── Quality filtering ──────────────────────────────────────────
    n_before = len(df)

    if "g_transit_flux" in df.columns and "g_transit_flux_error" in df.columns:
        valid_flux = df["g_transit_flux"].abs() > 0
        frac_err = df["g_transit_flux_error"].abs() / df["g_transit_flux"].abs()
        good_mask = valid_flux & (frac_err < 0.2)
        df = df[good_mask].copy()

    n_after = len(df)
    if n_after < n_before:
        log.info(
            "Epoch photometry quality filter: %d -> %d rows (removed %d high-error)",
            n_before, n_after, n_before - n_after,
        )

    if df.empty:
        log.warning("All epoch photometry filtered out for source_id=%s", source_id)
        return None

    save_cache(key, df, subfolder="gaia", fmt="csv")
    log.info("Cached %d G-band epoch-photometry rows for source_id=%s",
             len(df), source_id)
    return df

# ---------------------------------------------------------------------------
# Batched cone searches (controls phase optimisation)
# ---------------------------------------------------------------------------

def batch_cone_search(
    positions: List[Tuple[float, float]],
    radius_arcsec: float = 600.0,
    top_n_per_position: int = 100,
    batch_size: int = 10,
) -> Dict[int, Optional[pd.DataFrame]]:
    """Execute batched cone searches for multiple sky positions.

    Groups positions into batches and runs each batch as a single ADQL
    ``UNION ALL`` query, dramatically reducing round-trips to the TAP
    server.  Individual results are cached using the same keys as
    :func:`cone_search` so existing cached data is reused.

    Parameters
    ----------
    positions : list of (ra, dec)
        Sky positions in decimal degrees (ICRS).
    radius_arcsec : float
        Cone radius for each position.
    top_n_per_position : int
        Maximum results per position.
    batch_size : int
        Number of positions per ADQL query.

    Returns
    -------
    dict
        Mapping of position index → DataFrame (or None).
    """
    results: Dict[int, Optional[pd.DataFrame]] = {}

    # Check cache first — skip positions with cached results
    uncached: List[Tuple[int, float, float]] = []
    for i, (ra, dec) in enumerate(positions):
        key = cache_key("gaia_cone", ra, dec, radius_arcsec, top_n_per_position)
        cached = load_cache(key, subfolder="gaia")
        if cached is not None:
            if isinstance(cached, pd.DataFrame):
                results[i] = cached if not cached.empty else None
            else:
                df = pd.DataFrame(cached)
                results[i] = df if not df.empty else None
        else:
            uncached.append((i, ra, dec))

    if not uncached:
        log.info("All %d cone searches served from cache", len(positions))
        return results

    log.info(
        "Batch cone search: %d positions (%d cached, %d to query, batch=%d)",
        len(positions),
        len(positions) - len(uncached),
        len(uncached),
        batch_size,
    )

    # Process uncached positions in batches
    for batch_start in range(0, len(uncached), batch_size):
        batch = uncached[batch_start : batch_start + batch_size]
        batch_results = _execute_batch_cone(
            batch, radius_arcsec, top_n_per_position,
        )
        for pos_idx, ra, dec in batch:
            df = batch_results.get(pos_idx)
            results[pos_idx] = df
            # Cache individual results (compatible with cone_search keys)
            key = cache_key("gaia_cone", ra, dec, radius_arcsec, top_n_per_position)
            save_cache(
                key,
                df if df is not None else pd.DataFrame(),
                subfolder="gaia",
                fmt="csv",
            )

    return results


def _execute_batch_cone(
    batch: List[Tuple[int, float, float]],
    radius_arcsec: float,
    top_n: int,
) -> Dict[int, Optional[pd.DataFrame]]:
    """Run a single UNION ALL query for a batch of cone searches.

    Each sub-query is tagged with ``_batch_idx`` so results can be split
    per-position afterward.
    """
    columns = (
        "source_id, ra, dec, "
        "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, "
        "parallax, parallax_error, pmra, pmdec, ruwe"
    )
    radius_deg = radius_arcsec / 3600.0

    sub_queries = []
    for pos_idx, ra, dec in batch:
        sq = (
            f"SELECT TOP {top_n} {columns}, {pos_idx} AS _batch_idx "
            f"FROM gaiadr3.gaia_source "
            f"WHERE CONTAINS("
            f"POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})"
            f") = 1"
        )
        sub_queries.append(sq)

    adql = " UNION ALL ".join(sub_queries)
    log.debug("Batch ADQL: %d positions, %d chars", len(batch), len(adql))

    try:
        df = _run_sync_query(adql)
    except Exception as exc:
        log.warning(
            "Batch cone search failed (%s). Falling back to individual queries.",
            exc,
        )
        return _fallback_individual_cone(batch, radius_arcsec, top_n)

    if df is None or df.empty:
        return {pos_idx: None for pos_idx, _, _ in batch}

    # Split results by _batch_idx
    results: Dict[int, Optional[pd.DataFrame]] = {}
    for pos_idx, _ra, _dec in batch:
        if "_batch_idx" in df.columns:
            subset = df[df["_batch_idx"] == pos_idx].drop(
                columns=["_batch_idx"],
            ).copy()
        else:
            subset = pd.DataFrame()
        results[pos_idx] = subset if not subset.empty else None

    return results


def _fallback_individual_cone(
    batch: List[Tuple[int, float, float]],
    radius_arcsec: float,
    top_n: int,
) -> Dict[int, Optional[pd.DataFrame]]:
    """Fall back to individual ``cone_search()`` calls if UNION ALL fails."""
    results: Dict[int, Optional[pd.DataFrame]] = {}
    for pos_idx, ra, dec in batch:
        results[pos_idx] = cone_search(
            ra, dec, radius_arcsec=radius_arcsec, top_n=top_n,
        )
    return results


# ---------------------------------------------------------------------------
# Batch queries for target lists
# ---------------------------------------------------------------------------

def query_target_list(
    targets: List[Tuple[float, float]],
    radius_arcsec: float = _DEFAULT_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Query Gaia DR3 photometry + astrometry for a list of (RA, Dec) targets.

    For each target the function fetches stellar parameters and astrometry,
    merges them on ``source_id``, and concatenates the results into a single
    DataFrame.  All individual queries are cached.

    Parameters
    ----------
    targets : list of (float, float)
        List of (RA, Dec) pairs in decimal degrees.
    radius_arcsec : float
        Cone radius in arcseconds for each target.

    Returns
    -------
    pd.DataFrame
        Combined results for all targets (may be empty).
    """
    frames: list[pd.DataFrame] = []
    for ra, dec in targets:
        stellar = get_stellar_params(ra, dec, radius_arcsec)
        astrom = get_astrometry(ra, dec, radius_arcsec)
        if stellar is not None and astrom is not None:
            merged = stellar.merge(astrom, on="source_id", suffixes=("", "_astrom"))
            frames.append(merged)
        elif stellar is not None:
            frames.append(stellar)
        elif astrom is not None:
            frames.append(astrom)

    if not frames:
        log.warning("No Gaia results for any of the %d targets", len(targets))
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined Gaia data: %d rows from %d targets", len(combined), len(targets))
    return combined


def cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 60.0,
    top_n: int = 500,
) -> Optional[pd.DataFrame]:
    """Broad cone search returning core Gaia columns for many sources.

    This is useful for field-level analysis rather than single-target
    cross-matching.

    Parameters
    ----------
    ra, dec : float
        Centre of the search cone (ICRS degrees).
    radius_arcsec : float
        Cone radius (default 60 arcsec = 1 arcmin).
    top_n : int
        Maximum number of results.

    Returns
    -------
    pd.DataFrame or None
    """
    key = cache_key("gaia_cone", ra, dec, radius_arcsec, top_n)
    cached = load_cache(key, subfolder="gaia")
    if cached is not None:
        log.info("Cone-search cache hit for (%.4f, %.4f) r=%.1f\"", ra, dec, radius_arcsec)
        if isinstance(cached, pd.DataFrame):
            return cached
        return pd.DataFrame(cached)

    columns = (
        "source_id, ra, dec, "
        "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, "
        "parallax, parallax_error, pmra, pmdec, ruwe"
    )
    adql = _cone_adql(ra, dec, radius_arcsec, columns=columns, top_n=top_n)

    log.info("Gaia cone search at (%.4f, %.4f) r=%.1f\" top=%d", ra, dec, radius_arcsec, top_n)
    try:
        df = _run_sync_query(adql)
    except Exception as exc:
        log.error("Gaia cone search failed: %s", exc)
        return None

    if df.empty:
        log.warning("Cone search returned no results")
        return None

    save_cache(key, df, subfolder="gaia", fmt="csv")
    log.info("Cached %d cone-search rows", len(df))
    return df

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Query Gaia DR3 for Proxima Centauri
    PROXIMA_RA = 217.4289
    PROXIMA_DEC = -62.6795

    log.info("=== Project EXODUS -- Gaia DR3 query for Proxima Centauri ===")

    # -- Stellar parameters (photometry) -----------------------------------
    stellar = get_stellar_params(PROXIMA_RA, PROXIMA_DEC)
    if stellar is not None and not stellar.empty:
        row = stellar.iloc[0]
        print("\n--- Stellar Parameters (photometry) ---")
        print(f"  Source ID : {int(row['source_id'])}")
        print(f"  G mag     : {row.get('phot_g_mean_mag', 'N/A'):.4f}")
        print(f"  BP mag    : {row.get('phot_bp_mean_mag', 'N/A'):.4f}")
        print(f"  RP mag    : {row.get('phot_rp_mean_mag', 'N/A'):.4f}")
        print(f"  BP-RP     : {row.get('bp_rp', 'N/A'):.4f}")
        print(f"  Teff      : {row.get('teff_gspphot', 'N/A')}")
        source_id = int(row["source_id"])
    else:
        print("No stellar parameters returned for Proxima Centauri.")
        source_id = None

    # -- Astrometry --------------------------------------------------------
    astrom = get_astrometry(PROXIMA_RA, PROXIMA_DEC)
    if astrom is not None and not astrom.empty:
        row = astrom.iloc[0]
        print("\n--- Astrometry ---")
        print(f"  Parallax       : {row.get('parallax', 'N/A'):.4f} mas")
        print(f"  Parallax error : {row.get('parallax_error', 'N/A'):.4f} mas")
        if row.get("parallax") and row["parallax"] > 0:
            distance_pc = 1000.0 / row["parallax"]
            print(f"  Distance       : {distance_pc:.4f} pc")
        print(f"  PM RA          : {row.get('pmra', 'N/A'):.4f} mas/yr")
        print(f"  PM Dec         : {row.get('pmdec', 'N/A'):.4f} mas/yr")
        print(f"  RUWE           : {row.get('ruwe', 'N/A')}")
    else:
        print("No astrometry returned for Proxima Centauri.")

    # -- Epoch photometry --------------------------------------------------
    if source_id is not None:
        epoch = get_epoch_photometry(source_id)
        if epoch is not None and not epoch.empty:
            print(f"\n--- Epoch Photometry ({len(epoch)} transits) ---")
            print(epoch[["g_obs_time", "g_transit_mag"]].head(10).to_string(index=False))
        else:
            print("\nNo epoch photometry available (may not be published for this source).")
    else:
        print("\nSkipping epoch photometry (no source_id).")

    print("\nDone.")
