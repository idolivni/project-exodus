"""
NASA Exoplanet Archive ingestion module for Project EXODUS.

Queries the NASA Exoplanet Archive TAP service (pscomppars table) for
confirmed exoplanets, caches results locally as CSV, and provides
convenience accessors for habitable-zone filtering and host-star
cross-referencing.

Habitable zone proxy
--------------------
A planet is flagged as a habitable-zone candidate when **either**:
  * its insolation flux (``pl_insol``) falls within 0.25--1.75 Earth
    insolation, **or**
  * its equilibrium temperature (``pl_eqt``) falls within 180--310 K.

If both columns are present the insolation criterion takes precedence
because it is a more physically meaningful indicator.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import (
    cache_key,
    get_config,
    get_logger,
    load_cache,
    save_cache,
)

log = get_logger("ingestion.exoplanet_archive")

# ── Column mapping ───────────────────────────────────────────────────
# Columns we request from the pscomppars table and how we rename them
# for internal use.
_ARCHIVE_COLUMNS = [
    "pl_name",       # planet name
    "hostname",      # host star name
    "ra",            # right ascension  [deg]
    "dec",           # declination      [deg]
    "pl_orbper",     # orbital period   [days]
    "pl_rade",       # planet radius    [Earth radii]
    "pl_eqt",        # equilibrium temp [K]
    "pl_insol",      # insolation flux  [Earth flux]
    "sy_dist",       # distance         [pc]
]

_RENAME_MAP = {
    "pl_name":   "planet_name",
    "hostname":  "host_star",
    "ra":        "ra_deg",
    "dec":       "dec_deg",
    "pl_orbper": "orbital_period_days",
    "pl_rade":   "radius_earth",
    "pl_eqt":    "eq_temp_k",
    "pl_insol":  "insol_flux_earth",
    "sy_dist":   "distance_pc",
}

# ── Habitable-zone thresholds ────────────────────────────────────────
HZ_INSOL_MIN = 0.25   # Earth insolation
HZ_INSOL_MAX = 1.75
HZ_TEMP_MIN  = 180.0  # Kelvin
HZ_TEMP_MAX  = 310.0

# ── Cache settings ───────────────────────────────────────────────────
_CACHE_SUBFOLDER = "exoplanet_archive"
_CACHE_KEY       = cache_key("pscomppars", "confirmed", "v2")


# =====================================================================
# Internal helpers
# =====================================================================

def _compute_hz_flag(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series indicating habitable-zone candidacy.

    Insolation-based criterion takes precedence when data is available;
    falls back to equilibrium temperature.
    """
    has_insol = "insol_flux_earth" in df.columns
    has_temp  = "eq_temp_k" in df.columns

    # Start with all False
    flag = pd.Series(False, index=df.index)

    if has_insol:
        insol_ok = df["insol_flux_earth"].between(HZ_INSOL_MIN, HZ_INSOL_MAX)
        flag = flag | insol_ok

    if has_temp:
        temp_ok = df["eq_temp_k"].between(HZ_TEMP_MIN, HZ_TEMP_MAX)
        # Only use temperature where insolation is missing
        if has_insol:
            insol_missing = df["insol_flux_earth"].isna()
            flag = flag | (temp_ok & insol_missing)
        else:
            flag = flag | temp_ok

    return flag


def _build_adql_query() -> str:
    """Build the ADQL SELECT for the pscomppars table."""
    cols = ", ".join(_ARCHIVE_COLUMNS)
    return f"SELECT {cols} FROM pscomppars"


# =====================================================================
# Core query function
# =====================================================================

def query_exoplanet_archive(*, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch confirmed exoplanets from the NASA Exoplanet Archive.

    Results are cached as a CSV under ``data/cache/exoplanet_archive/``.
    Subsequent calls return the cached copy unless *force_refresh* is
    ``True``.

    Returns
    -------
    pandas.DataFrame
        One row per confirmed planet with columns defined by
        ``_RENAME_MAP.values()`` plus ``hz_flag``.
    """
    # ---- 1. Try cache first -------------------------------------------
    if not force_refresh:
        cached = load_cache(_CACHE_KEY, subfolder=_CACHE_SUBFOLDER)
        if cached is not None:
            log.info(
                "Loaded %d exoplanets from cache.", len(cached)
            )
            # Ensure boolean dtype after CSV round-trip
            if "hz_flag" in cached.columns:
                cached["hz_flag"] = cached["hz_flag"].astype(bool)
            return cached

    # ---- 2. Query the archive ----------------------------------------
    log.info("Querying NASA Exoplanet Archive (pscomppars) ...")
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import (
            NasaExoplanetArchive,
        )

        result_table = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select=", ".join(_ARCHIVE_COLUMNS),
        )
        df = result_table.to_pandas()
        log.info(
            "Received %d rows from Exoplanet Archive.", len(df)
        )

    except ImportError:
        log.warning(
            "astroquery is not installed -- falling back to direct TAP query."
        )
        df = _fallback_tap_query()

    except Exception as exc:
        log.error("astroquery query failed: %s", exc)
        log.info("Attempting direct TAP fallback ...")
        try:
            df = _fallback_tap_query()
        except Exception as tap_exc:
            log.error("TAP fallback also failed: %s", tap_exc)
            raise RuntimeError(
                "Unable to retrieve data from NASA Exoplanet Archive."
            ) from tap_exc

    # ---- 3. Normalise & enrich ----------------------------------------
    df = df.rename(columns=_RENAME_MAP)
    df["hz_flag"] = _compute_hz_flag(df)

    # ---- 4. Persist to cache ------------------------------------------
    path = save_cache(
        _CACHE_KEY, df, subfolder=_CACHE_SUBFOLDER, fmt="csv"
    )
    log.info("Cached exoplanet data to %s", path)

    return df


def _fallback_tap_query() -> pd.DataFrame:
    """Query the archive via a raw HTTP TAP request (no astroquery)."""
    import requests

    cfg = get_config()
    tap_url = cfg["catalogs"]["exoplanet_archive"]["url"]
    query = _build_adql_query()

    log.info("TAP endpoint: %s", tap_url)
    log.debug("ADQL: %s", query)

    response = requests.get(
        f"{tap_url}/sync",
        params={
            "query": query,
            "format": "csv",
        },
        timeout=120,
    )
    response.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(response.text))
    log.info("TAP fallback returned %d rows.", len(df))
    return df


# =====================================================================
# Public convenience accessors
# =====================================================================

def get_hz_planets(
    max_distance_pc: float = 100,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return habitable-zone candidates within *max_distance_pc* parsecs.

    Parameters
    ----------
    max_distance_pc : float, default 100
        Maximum stellar distance in parsecs.
    force_refresh : bool, default False
        If ``True``, bypass the local cache and re-query the archive.

    Returns
    -------
    pandas.DataFrame
        Subset of the full catalogue where ``hz_flag`` is ``True`` and
        ``distance_pc <= max_distance_pc``, sorted by distance.
    """
    df = query_exoplanet_archive(force_refresh=force_refresh)

    mask = df["hz_flag"]
    if "distance_pc" in df.columns:
        mask = mask & (df["distance_pc"].notna())
        mask = mask & (df["distance_pc"] <= max_distance_pc)

    hz = df.loc[mask].copy()
    hz.sort_values("distance_pc", inplace=True)
    hz.reset_index(drop=True, inplace=True)

    log.info(
        "Found %d habitable-zone candidates within %s pc.",
        len(hz),
        max_distance_pc,
    )
    return hz


def get_all_hosts(
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a deduplicated table of host stars.

    Useful for cross-referencing with other catalogues (Gaia, SIMBAD,
    etc.).  Each host appears once; columns include ``host_star``,
    ``ra_deg``, ``dec_deg``, and ``distance_pc``.

    Returns
    -------
    pandas.DataFrame
        Unique host stars with positional/distance information.
    """
    df = query_exoplanet_archive(force_refresh=force_refresh)

    host_cols = ["host_star", "ra_deg", "dec_deg", "distance_pc"]
    available = [c for c in host_cols if c in df.columns]

    hosts = (
        df[available]
        .drop_duplicates(subset="host_star")
        .sort_values("host_star")
        .reset_index(drop=True)
    )
    log.info("Unique host stars: %d", len(hosts))
    return hosts


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("  Project EXODUS -- NASA Exoplanet Archive Ingest")
    print("=" * 64)

    try:
        all_planets = query_exoplanet_archive()
    except RuntimeError as err:
        print(f"\nFATAL: {err}")
        sys.exit(1)

    print(f"\nConfirmed exoplanets retrieved: {len(all_planets)}")

    hz = get_hz_planets(max_distance_pc=100)
    print(f"Habitable-zone candidates within 100 pc: {len(hz)}")

    if len(hz) > 0:
        top = hz.head(10)
        print("\n  10 nearest habitable-zone candidates:")
        print("  " + "-" * 60)
        for _, row in top.iterrows():
            name = row.get("planet_name", "N/A")
            dist = row.get("distance_pc", np.nan)
            temp = row.get("eq_temp_k", np.nan)
            insol = row.get("insol_flux_earth", np.nan)
            rad = row.get("radius_earth", np.nan)

            dist_str  = f"{dist:8.2f} pc"  if pd.notna(dist)  else "     N/A"
            temp_str  = f"{temp:6.0f} K"   if pd.notna(temp)  else "   N/A"
            insol_str = f"{insol:5.2f} Se" if pd.notna(insol) else "  N/A"
            rad_str   = f"{rad:5.2f} Re"   if pd.notna(rad)   else "  N/A"

            print(
                f"    {name:<30s}  d={dist_str}  "
                f"Teq={temp_str}  S={insol_str}  R={rad_str}"
            )
    else:
        print("\n  (no habitable-zone candidates found within 100 pc)")

    print()
