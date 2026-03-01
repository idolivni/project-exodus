#!/usr/bin/env python3
"""
Pre-cache NEOWISE time-series for a target list.

Queries IRSA TAP for all targets upfront, saving results to the NEOWISE
cache directory. When the pipeline runs later, it hits cache instead of
making live 60-120s TAP queries per target.

Usage
-----
    # Pre-cache for the 500-target blitz
    ./venv/bin/python scripts/precache_neowise.py \
        --targets data/targets/smart_targets.json \
        --workers 3

    # Pre-cache for calibration targets
    ./venv/bin/python scripts/precache_neowise.py \
        --targets data/targets/calibration_binary.json \
        --workers 2

Performance
-----------
With --workers 3, three IRSA TAP queries run concurrently. Each takes
60-120s, so 500 targets complete in ~3-4 hours instead of ~12 hours.
With cached data, the pipeline's _gather_data step is ~5x faster.

Notes
-----
- Respects existing cache (skips already-cached targets)
- Uses the corrected NEOWISE TAP column mapping
- Rate-limited to avoid IRSA throttling (max 4 workers recommended)
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, load_cache

log = get_logger("precache_neowise")


def precache_targets(
    targets: list,
    max_workers: int = 3,
    force_refresh: bool = False,
) -> dict:
    """Pre-cache NEOWISE time-series for a list of targets.

    Parameters
    ----------
    targets : list of dict
        Each must have 'ra', 'dec', 'target_id'.
    max_workers : int
        Number of concurrent IRSA TAP queries.
    force_refresh : bool
        If True, re-query even if cached.

    Returns
    -------
    dict with summary statistics
    """
    from src.ingestion.neowise_timeseries import query_neowise_timeseries

    n_total = len(targets)
    n_cached = 0
    n_queried = 0
    n_failed = 0
    n_real = 0
    n_sim = 0
    n_nodata = 0

    # Check which targets already have cache
    to_query = []
    for t in targets:
        ra, dec = t["ra"], t["dec"]
        tid = t.get("target_id", f"({ra:.4f},{dec:.4f})")
        cache_key = f"neowise_ts_{ra:.6f}_{dec:.6f}_3.6"

        if not force_refresh:
            cached = load_cache(cache_key)
            if cached is not None:
                n_cached += 1
                ds = cached.get("data_source", "unknown")
                if ds == "real":
                    n_real += 1
                elif ds == "simulated":
                    n_sim += 1
                continue

        to_query.append(t)

    log.info(
        "Pre-cache: %d targets total, %d already cached (%d real, %d simulated), "
        "%d to query",
        n_total, n_cached, n_real, n_sim, len(to_query),
    )

    if not to_query:
        log.info("All targets already cached. Nothing to do.")
        return {
            "total": n_total,
            "cached": n_cached,
            "queried": 0,
            "failed": 0,
            "real": n_real,
            "simulated": n_sim,
            "no_data": n_nodata,
        }

    def _query_one(target):
        """Query NEOWISE for a single target. Thread-safe."""
        ra = target["ra"]
        dec = target["dec"]
        tid = target.get("target_id", f"({ra:.4f},{dec:.4f})")
        try:
            ts = query_neowise_timeseries(ra, dec, use_cache=True)
            ds = getattr(ts, "data_source", "unknown")
            return {
                "target_id": tid,
                "status": "ok",
                "n_epochs": ts.n_epochs,
                "data_source": ds,
                "baseline_yr": ts.time_baseline_years,
            }
        except Exception as exc:
            return {
                "target_id": tid,
                "status": "error",
                "error": str(exc),
            }

    # Run concurrent queries
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_query_one, t): t.get("target_id", "?")
            for t in to_query
        }

        for i, future in enumerate(as_completed(future_map)):
            tid = future_map[future]
            try:
                result = future.result(timeout=300)
                results.append(result)

                if result["status"] == "ok":
                    n_queried += 1
                    ds = result["data_source"]
                    if ds == "real":
                        n_real += 1
                    elif ds == "simulated":
                        n_sim += 1
                    if result["n_epochs"] == 0:
                        n_nodata += 1

                    if (i + 1) % 10 == 0 or (i + 1) == len(to_query):
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed * 3600  # per hour
                        eta_hr = (len(to_query) - i - 1) / max(rate, 1)
                        log.info(
                            "  [%d/%d] %s: %d epochs (%s) — %.0f/hr, ETA %.1fh",
                            i + 1, len(to_query), tid,
                            result["n_epochs"], ds, rate, eta_hr,
                        )
                else:
                    n_failed += 1
                    log.warning("  [%d/%d] %s: FAILED — %s",
                                i + 1, len(to_query), tid, result.get("error"))

            except Exception as exc:
                n_failed += 1
                log.error("  [%d/%d] %s: EXCEPTION — %s",
                          i + 1, len(to_query), tid, exc)

    elapsed = time.time() - start_time
    log.info(
        "Pre-cache complete: %d queried, %d failed in %.1f min",
        n_queried, n_failed, elapsed / 60,
    )

    return {
        "total": n_total,
        "cached": n_cached,
        "queried": n_queried,
        "failed": n_failed,
        "real": n_real + n_cached,
        "simulated": n_sim,
        "no_data": n_nodata,
        "elapsed_sec": round(elapsed, 1),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-cache NEOWISE time-series for target list"
    )
    parser.add_argument("--targets", required=True,
                        help="Path to target JSON file")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of concurrent TAP queries (default: 3)")
    parser.add_argument("--force", action="store_true",
                        help="Re-query even if already cached")
    args = parser.parse_args()

    # Load targets
    with open(args.targets) as f:
        data = json.load(f)

    targets = data.get("targets", data) if isinstance(data, dict) else data
    print(f"{'=' * 70}")
    print(f"  NEOWISE Pre-Cache — {len(targets)} targets, {args.workers} workers")
    print(f"{'=' * 70}")

    stats = precache_targets(targets, max_workers=args.workers,
                              force_refresh=args.force)

    print(f"\n  Summary:")
    print(f"    Total targets:  {stats['total']}")
    print(f"    Already cached: {stats['cached']}")
    print(f"    Newly queried:  {stats['queried']}")
    print(f"    Failed:         {stats['failed']}")
    print(f"    Real data:      {stats['real']}")
    print(f"    Simulated:      {stats['simulated']}")
    print(f"    No data:        {stats['no_data']}")
    if stats.get("elapsed_sec"):
        print(f"    Elapsed:        {stats['elapsed_sec']:.1f}s")
    print(f"{'=' * 70}")
