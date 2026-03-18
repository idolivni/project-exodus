#!/usr/bin/env python3
"""
Deterministic throughput benchmark for EXODUS ingestion paths.

Purpose
-------
Provide a stable, network-free metric for autonomous performance tuning.
The benchmark intentionally targets two current bottlenecks:

1. `src.ingestion.gaia_query.query_target_list`
2. `src.ingestion.ir_surveys.get_ir_photometry_batch`

The synthetic backend injects fixed per-call latency so improvements come from
fewer round-trips, better batching, or better orchestration instead of archive
availability luck.
"""

from __future__ import annotations

import re
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion import gaia_query, ir_surveys  # noqa: E402


N_GAIA_TARGETS = 48
N_IR_TARGETS = 48
GAIA_CALL_LATENCY_SEC = 0.008
IR_CALL_LATENCY_SEC = 0.006


logging.getLogger("exodus.ingestion.gaia").setLevel(logging.CRITICAL)
logging.getLogger("exodus.ingestion.ir_surveys").setLevel(logging.CRITICAL)


def _synthetic_targets(n: int) -> List[Tuple[float, float]]:
    base_ra = 10.0
    base_dec = 40.0
    return [
        (base_ra + 0.01 * i, base_dec + 0.005 * (i % 7))
        for i in range(n)
    ]


def _mock_gaia_run_sync_query(adql: str) -> pd.DataFrame:
    time.sleep(GAIA_CALL_LATENCY_SEC)

    batch_matches = re.findall(r"(\d+)\s+AS\s+_batch_idx", adql)
    if batch_matches:
        rows: List[Dict[str, float]] = []
        for idx_str in batch_matches:
            idx = int(idx_str)
            rows.append(
                {
                    "source_id": 1_000_000 + idx,
                    "ra": 10.0 + 0.01 * idx,
                    "dec": 40.0 + 0.01 * idx,
                    "phot_g_mean_mag": 14.0,
                    "phot_bp_mean_mag": 14.6,
                    "phot_rp_mean_mag": 13.4,
                    "bp_rp": 1.2,
                    "parallax": 1.0,
                    "parallax_error": 0.05,
                    "pmra": 3.0,
                    "pmdec": -1.0,
                    "ruwe": 1.05,
                    "teff_gspphot": 6100.0,
                    "logg_gspphot": 4.2,
                    "mh_gspphot": -0.1,
                    "_batch_idx": idx,
                }
            )
        return pd.DataFrame(rows)

    return pd.DataFrame(
        [
            {
                "source_id": 42,
                "ra": 10.0,
                "dec": 40.0,
                "phot_g_mean_mag": 14.0,
                "phot_bp_mean_mag": 14.6,
                "phot_rp_mean_mag": 13.4,
                "bp_rp": 1.2,
                "parallax": 1.0,
                "parallax_error": 0.05,
                "pmra": 3.0,
                "pmdec": -1.0,
                "ruwe": 1.05,
                "teff_gspphot": 6100.0,
                "logg_gspphot": 4.2,
                "mh_gspphot": -0.1,
            }
        ]
    )


def _mock_2mass(ra: float, dec: float, radius_arcsec: float = 5.0):
    time.sleep(IR_CALL_LATENCY_SEC)
    return {
        "J": 13.8,
        "H": 13.5,
        "Ks": 13.4,
        "J_err": 0.03,
        "H_err": 0.03,
        "Ks_err": 0.03,
        "designation": f"2MASS_{ra:.4f}_{dec:.4f}",
        "match_sep_arcsec": 0.2,
    }


def _mock_wise(ra: float, dec: float, radius_arcsec: float = 5.0):
    time.sleep(IR_CALL_LATENCY_SEC)
    return {
        "W1": 13.2,
        "W2": 13.1,
        "W3": 12.7,
        "W4": 11.8,
        "W1_err": 0.03,
        "W2_err": 0.03,
        "W3_err": 0.08,
        "W4_err": 0.15,
        "designation": f"WISE_{ra:.4f}_{dec:.4f}",
        "match_sep_arcsec": 0.3,
    }


def _mock_catwise(ra: float, dec: float, radius_arcsec: float = 5.0):
    return {
        "W1_catwise": 13.2,
        "W2_catwise": 13.1,
        "designation": f"CATWISE_{ra:.4f}_{dec:.4f}",
        "match_sep_arcsec": 0.3,
        "pmra_wise": 2.8,
        "e_pmra_wise": 0.4,
        "pmdec_wise": -1.2,
        "e_pmdec_wise": 0.4,
        "ab_flags": "00",
    }


def benchmark_gaia_query_target_list() -> Tuple[float, int]:
    targets = _synthetic_targets(N_GAIA_TARGETS)
    call_count = {"n": 0}

    def _wrapped(adql: str):
        call_count["n"] += 1
        return _mock_gaia_run_sync_query(adql)

    with patch("src.ingestion.gaia_query.load_cache", return_value=None):
        with patch("src.ingestion.gaia_query.save_cache"):
            with patch("src.ingestion.gaia_query._run_sync_query", side_effect=_wrapped):
                t0 = time.perf_counter()
                df = gaia_query.query_target_list(targets, radius_arcsec=5.0)
                dt = time.perf_counter() - t0

    assert len(df) >= N_GAIA_TARGETS, "Gaia benchmark returned too few rows"
    return dt, call_count["n"]


def benchmark_ir_batch() -> Tuple[float, int]:
    targets = _synthetic_targets(N_IR_TARGETS)
    call_count = {"n": 0}

    def _counted_2mass(*args, **kwargs):
        call_count["n"] += 1
        return _mock_2mass(*args, **kwargs)

    def _counted_wise(*args, **kwargs):
        call_count["n"] += 1
        return _mock_wise(*args, **kwargs)

    def _counted_catwise(*args, **kwargs):
        call_count["n"] += 1
        return _mock_catwise(*args, **kwargs)

    with patch("src.ingestion.ir_surveys.get_2mass", side_effect=_counted_2mass):
        with patch("src.ingestion.ir_surveys.get_wise", side_effect=_counted_wise):
            with patch("src.ingestion.ir_surveys.get_catwise", side_effect=_counted_catwise):
                t0 = time.perf_counter()
                rows = ir_surveys.get_ir_photometry_batch(targets, radius_arcsec=5.0)
                dt = time.perf_counter() - t0

    assert len(rows) == N_IR_TARGETS, "IR benchmark returned wrong target count"
    assert all("ra" in row and "dec" in row for row in rows), "IR schema regression"
    return dt, call_count["n"]


def main() -> None:
    gaia_seconds, gaia_calls = benchmark_gaia_query_target_list()
    ir_seconds, ir_calls = benchmark_ir_batch()
    total_seconds = gaia_seconds + ir_seconds

    print("---")
    print(f"benchmark_total_seconds: {total_seconds:.6f}")
    print(f"gaia_query_target_list_seconds: {gaia_seconds:.6f}")
    print(f"gaia_backend_calls: {gaia_calls}")
    print(f"ir_batch_seconds: {ir_seconds:.6f}")
    print(f"ir_backend_calls: {ir_calls}")
    print(f"gaia_targets: {N_GAIA_TARGETS}")
    print(f"ir_targets: {N_IR_TARGETS}")


if __name__ == "__main__":
    main()
