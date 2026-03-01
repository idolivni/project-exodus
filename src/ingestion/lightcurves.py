"""
Light-curve ingestion for Project EXODUS.

Downloads Kepler, K2, and TESS light curves via the ``lightkurve`` package,
cleans / normalizes them, and caches the results locally so repeated queries
hit the network only once.

Public API
----------
get_lightcurve(target, mission="any")
    Download (or load from cache) a single cleaned, normalized light curve.

get_all_sectors(target)
    Return a list of available TESS sectors for a target.

stitch_lightcurves(target, mission="TESS")
    Download every available quarter / sector for *target* and return a
    single stitched light curve.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Union

import numpy as np
import pandas as pd

try:
    import lightkurve as lk
except ImportError:
    raise ImportError(
        "lightkurve is required for light-curve ingestion.  "
        "Install it with:  pip install lightkurve"
    )

# Project utilities --------------------------------------------------------
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from utils import get_logger, get_config, cache_key, load_cache, save_cache  # noqa: E402

logger = get_logger("ingestion.lightcurves")

# Supported missions -------------------------------------------------------
SUPPORTED_MISSIONS = ("Kepler", "K2", "TESS")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_target(target: Union[str, tuple]) -> str:
    """Return a canonical string representation suitable for cache keys."""
    if isinstance(target, (list, tuple)):
        return f"ra{float(target[0]):.6f}_dec{float(target[1]):.6f}"
    return str(target).strip().replace(" ", "_")


def _search(
    target: Union[str, tuple],
    mission: Optional[str] = None,
    author: Optional[str] = None,
) -> "lk.SearchResult":
    """
    Thin wrapper around ``lightkurve.search_lightcurve`` with logging.

    Parameters
    ----------
    target : str or (RA, Dec) tuple
        Target identifier or sky coordinates.
    mission : str or None
        One of 'Kepler', 'K2', 'TESS', or None (all missions).
    author : str or None
        Pipeline author filter (e.g. 'SPOC', 'Kepler', 'K2').

    Returns
    -------
    lk.SearchResult
    """
    kwargs: dict = {}
    if mission and mission.lower() != "any":
        kwargs["mission"] = mission
    if author:
        kwargs["author"] = author

    logger.info(
        "Searching for light curves: target=%s  mission=%s  author=%s",
        target, mission or "any", author or "any",
    )
    result = lk.search_lightcurve(target, **kwargs)
    logger.info("  -> found %d result(s)", len(result))
    return result


def _clean(lc: "lk.LightCurve") -> "lk.LightCurve":
    """Remove NaNs, clip outliers, and normalize a light curve."""
    return lc.remove_nans().remove_outliers(sigma=5.0).normalize()


def _lc_to_dataframe(lc: "lk.LightCurve") -> pd.DataFrame:
    """Convert a lightkurve LightCurve to a lean DataFrame for caching."""
    mission = getattr(lc, "mission", None)
    if mission is None and hasattr(lc, "meta"):
        mission = lc.meta.get("MISSION", "unknown")
    df = pd.DataFrame({
        "time": np.asarray(lc.time.value, dtype=np.float64),
        "flux": np.asarray(lc.flux.value, dtype=np.float64),
        "flux_err": np.asarray(lc.flux_err.value, dtype=np.float64),
        "mission": str(mission or "unknown"),  # persisted through CSV
    })
    return df


def _dataframe_to_lc(df: pd.DataFrame) -> "lk.LightCurve":
    """Reconstruct a lightkurve LightCurve from a cached DataFrame."""
    mission = "unknown"
    if "mission" in df.columns:
        mission = str(df["mission"].iloc[0])
    result = lk.LightCurve(
        time=df["time"].values,
        flux=df["flux"].values,
        flux_err=df["flux_err"].values,
    )
    result.meta["MISSION"] = mission
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_lightcurve(
    target: Union[str, tuple],
    mission: str = "any",
    author: Optional[str] = None,
    use_cache: bool = True,
) -> Optional["lk.LightCurve"]:
    """
    Download and return a single cleaned, normalized light curve.

    The function picks the *first* search result that matches the requested
    mission.  For multi-sector / multi-quarter retrieval use
    :func:`stitch_lightcurves` instead.

    Parameters
    ----------
    target : str or (RA, Dec)
        Target identifier (e.g. ``"TOI-700"``, ``"KIC 8462852"``) or a
        tuple of ``(ra_deg, dec_deg)``.
    mission : str
        ``'Kepler'``, ``'K2'``, ``'TESS'``, or ``'any'`` (default).
    author : str or None
        Pipeline author filter forwarded to ``lightkurve.search_lightcurve``.
    use_cache : bool
        If *True* (default), look in the local cache before hitting MAST.

    Returns
    -------
    lk.LightCurve or None
        Cleaned, normalized light curve, or *None* if nothing was found.
    """
    norm_target = _normalize_target(target)
    ckey = cache_key("lc", norm_target, mission)

    # --- cache check -------------------------------------------------------
    if use_cache:
        cached = load_cache(ckey, subfolder="lightcurves")
        if cached is not None:
            logger.info("Cache hit for %s (mission=%s)", target, mission)
            if isinstance(cached, pd.DataFrame):
                return _dataframe_to_lc(cached)
            # JSON metadata-only record (shouldn't normally happen)
            logger.warning("Unexpected cache format; re-downloading.")

    # --- download ----------------------------------------------------------
    try:
        search_result = _search(target, mission=mission, author=author)
        if len(search_result) == 0:
            logger.warning("No light curves found for %s (mission=%s)", target, mission)
            return None

        first = search_result[0]
        # Extract mission from search result before download
        _mission = getattr(first, "mission", None)
        if _mission is None:
            try:
                _mission = str(first.table["mission"][0]) if "mission" in first.table.colnames else None
            except Exception:
                _mission = None
        if _mission is not None:
            if hasattr(_mission, '__len__') and not isinstance(_mission, str):
                _mission = str(_mission[0])  # numpy array → string
            _mission = str(_mission).split()[0]  # "TESS Sector 51" -> "TESS"

        logger.info("Downloading first result: %s (mission=%s)", first, _mission or "unknown")
        lc_raw = first.download()
        if lc_raw is None:
            logger.error("Download returned None for %s", target)
            return None

        lc = _clean(lc_raw)
        # Propagate mission into lightcurve metadata
        if _mission and hasattr(lc, "meta"):
            lc.meta["MISSION"] = _mission

    except Exception as exc:
        import traceback as _tb
        logger.error("Failed to retrieve light curve for %s: %s\n%s",
                     target, exc, _tb.format_exc())
        return None

    # --- cache result ------------------------------------------------------
    try:
        df = _lc_to_dataframe(lc)
        save_cache(ckey, df, subfolder="lightcurves", fmt="csv")
        logger.info("Cached light curve for %s -> key %s", target, ckey)
    except Exception as exc:
        logger.warning("Could not cache light curve: %s", exc)

    return lc


def get_all_sectors(target: Union[str, tuple]) -> List[int]:
    """
    Return a sorted list of TESS sector numbers available for *target*.

    Parameters
    ----------
    target : str or (RA, Dec)
        Target identifier or sky coordinates.

    Returns
    -------
    list[int]
        Sorted TESS sector numbers, or an empty list if none found.
    """
    norm_target = _normalize_target(target)
    ckey = cache_key("tess_sectors", norm_target)

    cached = load_cache(ckey, subfolder="lightcurves")
    if cached is not None and isinstance(cached, (list, dict)):
        sectors = cached if isinstance(cached, list) else cached.get("sectors", [])
        if sectors:
            logger.info("Cache hit: %d TESS sectors for %s", len(sectors), target)
            return sorted(sectors)

    try:
        search_result = _search(target, mission="TESS")
        if len(search_result) == 0:
            logger.warning("No TESS data found for %s", target)
            return []

        # Extract sector numbers from the search-result table
        table = search_result.table
        if "sequence_number" in table.colnames:
            sectors = sorted(set(int(s) for s in table["sequence_number"] if s is not None))
        else:
            # Fallback: count distinct rows as "sectors"
            sectors = list(range(1, len(search_result) + 1))
            logger.warning(
                "Could not determine sector numbers; returning index-based list"
            )

    except Exception as exc:
        logger.error("Failed to query TESS sectors for %s: %s", target, exc)
        return []

    # Cache the sector list
    try:
        save_cache(ckey, {"sectors": sectors, "target": str(target)},
                   subfolder="lightcurves", fmt="json")
    except Exception as exc:
        logger.warning("Could not cache sector list: %s", exc)

    return sectors


def stitch_lightcurves(
    target: Union[str, tuple],
    mission: str = "TESS",
    author: Optional[str] = None,
    use_cache: bool = True,
) -> Optional["lk.LightCurve"]:
    """
    Download every available quarter / sector for *target* and stitch
    them into a single continuous light curve.

    Parameters
    ----------
    target : str or (RA, Dec)
        Target identifier or sky coordinates.
    mission : str
        ``'Kepler'``, ``'K2'``, or ``'TESS'`` (default).
    author : str or None
        Pipeline author filter.
    use_cache : bool
        Look in the local cache first.

    Returns
    -------
    lk.LightCurve or None
        Stitched, cleaned, normalized light curve, or *None* if nothing
        was found.
    """
    norm_target = _normalize_target(target)
    ckey = cache_key("stitched", norm_target, mission)

    # --- cache check -------------------------------------------------------
    if use_cache:
        cached = load_cache(ckey, subfolder="lightcurves")
        if cached is not None and isinstance(cached, pd.DataFrame):
            logger.info("Cache hit for stitched %s (mission=%s)", target, mission)
            return _dataframe_to_lc(cached)

    # --- download all and stitch -------------------------------------------
    try:
        search_result = _search(target, mission=mission, author=author)
        if len(search_result) == 0:
            logger.warning(
                "No light curves to stitch for %s (mission=%s)", target, mission
            )
            return None

        logger.info(
            "Downloading %d light curve(s) for stitching (%s, %s)",
            len(search_result), target, mission,
        )
        lc_collection = search_result.download_all()

        if lc_collection is None or len(lc_collection) == 0:
            logger.error("download_all returned nothing for %s", target)
            return None

        # Clean each individual light curve before stitching
        cleaned = lk.LightCurveCollection(
            [_clean(lc) for lc in lc_collection]
        )

        stitched = cleaned.stitch()
        logger.info(
            "Stitched %d segments -> %d data points for %s",
            len(cleaned), len(stitched.flux), target,
        )

    except Exception as exc:
        logger.error(
            "Failed to stitch light curves for %s (mission=%s): %s",
            target, mission, exc,
        )
        return None

    # --- cache result ------------------------------------------------------
    try:
        df = _lc_to_dataframe(stitched)
        save_cache(ckey, df, subfolder="lightcurves", fmt="csv")
        logger.info("Cached stitched light curve -> key %s", ckey)
    except Exception as exc:
        logger.warning("Could not cache stitched light curve: %s", exc)

    return stitched


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS  --  Light-Curve Ingestion Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. TESS example: TOI-700 (known exoplanet host in TESS data)
    # ------------------------------------------------------------------
    tess_target = "TOI-700"
    print(f"\n[1] Searching TESS light curves for {tess_target} ...")

    sectors = get_all_sectors(tess_target)
    print(f"    Available TESS sectors: {sectors}")

    lc_tess = get_lightcurve(tess_target, mission="TESS")
    if lc_tess is not None:
        duration_days = float(lc_tess.time[-1].value - lc_tess.time[0].value)
        print(f"    Single-sector light curve:")
        print(f"      Data points : {len(lc_tess.flux)}")
        print(f"      Duration    : {duration_days:.1f} days")
        print(f"      Flux range  : {float(lc_tess.flux.min()):.6f} "
              f"- {float(lc_tess.flux.max()):.6f} (normalized)")
    else:
        print("    Could not retrieve a single-sector light curve.")

    # Try stitching all sectors
    print(f"\n    Stitching all available sectors for {tess_target} ...")
    lc_stitched = stitch_lightcurves(tess_target, mission="TESS")
    if lc_stitched is not None:
        duration_days = float(lc_stitched.time[-1].value - lc_stitched.time[0].value)
        print(f"    Stitched light curve:")
        print(f"      Data points : {len(lc_stitched.flux)}")
        print(f"      Duration    : {duration_days:.1f} days")
    else:
        print("    Could not stitch TESS sectors.")

    # ------------------------------------------------------------------
    # 2. Kepler example: Tabby's Star (KIC 8462852)
    # ------------------------------------------------------------------
    kepler_target = "KIC 8462852"
    print(f"\n[2] Tabby's Star ({kepler_target}) -- Kepler mission")
    print("    This star is famous for its irregular, deep dimming events")
    print("    that remain one of the most intriguing anomalies found by Kepler.")

    lc_kepler = get_lightcurve(kepler_target, mission="Kepler")
    if lc_kepler is not None:
        duration_days = float(lc_kepler.time[-1].value - lc_kepler.time[0].value)
        print(f"    Single-quarter light curve:")
        print(f"      Data points : {len(lc_kepler.flux)}")
        print(f"      Duration    : {duration_days:.1f} days")
    else:
        print("    Could not retrieve Kepler light curve (network or MAST issue).")

    # ------------------------------------------------------------------
    # 3. Summary of missions available for each target
    # ------------------------------------------------------------------
    print(f"\n[3] Missions available per target:")
    for name in [tess_target, kepler_target]:
        missions_found = []
        for m in SUPPORTED_MISSIONS:
            try:
                sr = lk.search_lightcurve(name, mission=m)
                if len(sr) > 0:
                    missions_found.append(f"{m} ({len(sr)})")
            except Exception:
                pass
        print(f"    {name:20s} : {', '.join(missions_found) if missions_found else 'none'}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)
