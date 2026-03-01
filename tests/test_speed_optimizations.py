"""
Tests for pipeline speed optimizations.

Covers:
  - Gaia TAP timeout enforcement (_execute_with_timeout)
  - Mirror fallback when ESA fails (_run_sync_query)
  - ADQL table-name rewriting for VizieR
  - Batch cone search result splitting (_execute_batch_cone)
  - Batch fallback to individual cone_search
  - Batch reuses existing cone_search cache keys
  - Config disabling all optimizations = identical to old behavior

All tests are self-contained (no network calls) and run fast (<5s total).
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pre-import the module so that patching TapPlus doesn't break
# the initial module-level `from astroquery.gaia import Gaia`.
import src.ingestion.gaia_query as _gaia_mod  # noqa: E402


# =====================================================================
# Test 1: _execute_with_timeout fires on slow queries
# =====================================================================

def test_execute_with_timeout_fires():
    """A query that exceeds the timeout should raise TimeoutError-like."""
    from concurrent.futures import TimeoutError as _FutureTimeout

    # Patch TapPlus at the location where it's imported from
    with patch("astroquery.utils.tap.core.TapPlus") as mock_cls:
        # Make launch_job sleep longer than the timeout
        def slow_query(*a, **kw):
            time.sleep(10)
            return MagicMock()

        mock_instance = MagicMock()
        mock_instance.launch_job.side_effect = slow_query
        mock_cls.return_value = mock_instance

        with pytest.raises(_FutureTimeout):
            _gaia_mod._execute_with_timeout("SELECT 1", "http://example.com/tap", timeout_sec=1)


# =====================================================================
# Test 2: _execute_with_timeout returns result on fast queries
# =====================================================================

def test_execute_with_timeout_returns_on_success():
    """A fast query should return the DataFrame result normally."""
    from astropy.table import Table

    with patch("astroquery.utils.tap.core.TapPlus") as mock_cls:
        # Return a quick result
        result_table = Table({"source_id": [123], "ra": [10.0]})
        mock_job = MagicMock()
        mock_job.get_results.return_value = result_table
        mock_instance = MagicMock()
        mock_instance.launch_job.return_value = mock_job
        mock_cls.return_value = mock_instance

        df = _gaia_mod._execute_with_timeout("SELECT 1", "http://example.com/tap", timeout_sec=30)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["source_id"].iloc[0] == 123


# =====================================================================
# Test 3: Mirror fallback triggers when ESA raises
# =====================================================================

def test_mirror_fallback_on_esa_failure():
    """When ESA times out / errors, _run_sync_query should try next mirror."""
    from src.ingestion.gaia_query import _GAIA_MIRRORS

    # We need at least 2 mirrors configured
    assert len(_GAIA_MIRRORS) >= 2, "Need multiple mirrors for fallback test"

    call_log: List[str] = []

    def mock_execute_timeout(adql, tap_url, timeout_sec):
        call_log.append(tap_url)
        if "esac.esa" in tap_url:
            raise ConnectionError("connection refused by ESA")
        # ARI Heidelberg or VizieR succeeds
        return pd.DataFrame({"source_id": [999], "ra": [42.0]})

    with patch("src.ingestion.gaia_query._execute_with_timeout", side_effect=mock_execute_timeout):
        with patch("src.ingestion.gaia_query.get_config", return_value={
            "performance": {
                "gaia_timeout_sec": 10,
                "gaia_mirror_fallback": True,
            }
        }):
            from src.ingestion.gaia_query import _run_sync_query

            df = _run_sync_query("SELECT TOP 1 * FROM gaiadr3.gaia_source", max_retries=1)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    # ESA was tried first (3 attempts with default max_retries=1), then a fallback mirror
    assert any("esac.esa" in url for url in call_log)
    assert any("esac.esa" not in url for url in call_log)


# =====================================================================
# Test 4: ADQL table name rewriting for VizieR
# =====================================================================

def test_adql_table_rewriting_for_vizier():
    """When fallback reaches VizieR, gaiadr3.gaia_source → '"I/355/gaiadr3"'."""
    from src.ingestion.gaia_query import _GAIA_MIRRORS

    # Find the VizieR mirror
    vizier = [m for m in _GAIA_MIRRORS if "vizier" in m["name"].lower() or "vizier" in m["url"].lower()]
    assert vizier, "VizieR mirror not configured"

    received_adql: List[str] = []

    def mock_execute_timeout(adql, tap_url, timeout_sec):
        received_adql.append(adql)
        if "vizier" not in tap_url.lower():
            raise ConnectionError("connection reset by peer")
        return pd.DataFrame({"source_id": [1]})

    with patch("src.ingestion.gaia_query._execute_with_timeout", side_effect=mock_execute_timeout):
        with patch("src.ingestion.gaia_query.get_config", return_value={
            "performance": {
                "gaia_timeout_sec": 10,
                "gaia_mirror_fallback": True,
            }
        }):
            from src.ingestion.gaia_query import _run_sync_query

            _run_sync_query(
                "SELECT TOP 1 * FROM gaiadr3.gaia_source WHERE ra > 10",
                max_retries=1,
            )

    # The last ADQL (the one that succeeded on VizieR) should have rewritten table name
    vizier_adql = [q for q in received_adql if "I/355" in q]
    assert vizier_adql, f"Expected VizieR table name in ADQL, got: {received_adql}"
    assert "gaiadr3.gaia_source" not in vizier_adql[-1]


# =====================================================================
# Test 5: _execute_batch_cone splits results by _batch_idx
# =====================================================================

def test_batch_cone_splits_by_batch_idx():
    """_execute_batch_cone should split UNION ALL results by _batch_idx."""
    # Build a mock result as if the TAP server returned 3 rows for 2 positions
    mock_df = pd.DataFrame({
        "source_id": [100, 101, 200],
        "ra": [10.0, 10.01, 20.0],
        "dec": [43.0, 43.01, 55.0],
        "phot_g_mean_mag": [12.0, 12.5, 14.0],
        "phot_bp_mean_mag": [12.5, 13.0, 14.5],
        "phot_rp_mean_mag": [11.5, 12.0, 13.5],
        "bp_rp": [1.0, 1.0, 1.0],
        "parallax": [0.5, 0.5, 0.3],
        "parallax_error": [0.1, 0.1, 0.1],
        "pmra": [1.0, 1.0, 2.0],
        "pmdec": [0.5, 0.5, 1.0],
        "ruwe": [1.1, 1.2, 1.0],
        "_batch_idx": [0, 0, 1],  # 2 rows for position 0, 1 for position 1
    })

    with patch("src.ingestion.gaia_query._run_sync_query", return_value=mock_df):
        from src.ingestion.gaia_query import _execute_batch_cone

        batch = [(0, 10.0, 43.0), (1, 20.0, 55.0)]
        results = _execute_batch_cone(batch, radius_arcsec=600.0, top_n=100)

    assert 0 in results and 1 in results
    assert len(results[0]) == 2  # 2 rows for position 0
    assert len(results[1]) == 1  # 1 row for position 1
    assert "_batch_idx" not in results[0].columns  # dropped after split
    assert results[0]["source_id"].tolist() == [100, 101]
    assert results[1]["source_id"].tolist() == [200]


# =====================================================================
# Test 6: Batch fallback to individual cone_search
# =====================================================================

def test_batch_fallback_to_individual():
    """When UNION ALL fails, _execute_batch_cone falls back to individual queries."""
    individual_calls: List[Tuple[float, float]] = []

    def mock_cone_search(ra, dec, radius_arcsec=600, top_n=100):
        individual_calls.append((ra, dec))
        return pd.DataFrame({"source_id": [int(ra * 10)], "ra": [ra]})

    with patch("src.ingestion.gaia_query._run_sync_query", side_effect=RuntimeError("UNION ALL unsupported")):
        with patch("src.ingestion.gaia_query.cone_search", side_effect=mock_cone_search):
            from src.ingestion.gaia_query import _execute_batch_cone

            batch = [(0, 10.0, 43.0), (1, 20.0, 55.0)]
            results = _execute_batch_cone(batch, radius_arcsec=600.0, top_n=100)

    # Should have fallen back to 2 individual queries
    assert len(individual_calls) == 2
    assert (10.0, 43.0) in individual_calls
    assert (20.0, 55.0) in individual_calls
    assert 0 in results and 1 in results


# =====================================================================
# Test 7: batch_cone_search reuses existing cache keys
# =====================================================================

def test_batch_reuses_existing_cache_keys():
    """Cache keys for batch_cone_search must be identical to cone_search keys.

    This ensures cached data from previous sequential runs is reused, and
    newly-cached data from batch runs is usable by cone_search.
    """
    from src.utils import cache_key

    ra, dec, radius, top_n = 83.633, 22.014, 600.0, 100  # Crab Nebula region

    # Simulate the cache key that cone_search would generate
    expected_key = cache_key("gaia_cone", ra, dec, radius, top_n)

    # Now check batch_cone_search uses the same key pattern
    # by reading the source: it calls cache_key("gaia_cone", ra, dec, radius, top_n)
    # We verify by mocking load_cache and checking the key passed
    seen_keys: List[str] = []

    def track_load_cache(key, subfolder=None):
        seen_keys.append(key)
        # Return a cached result so no network calls needed
        return pd.DataFrame({"source_id": [123], "ra": [ra]})

    with patch("src.ingestion.gaia_query.load_cache", side_effect=track_load_cache):
        with patch("src.ingestion.gaia_query.save_cache"):  # don't actually write
            from src.ingestion.gaia_query import batch_cone_search

            results = batch_cone_search(
                [(ra, dec)],
                radius_arcsec=radius,
                top_n_per_position=top_n,
            )

    assert expected_key in seen_keys, (
        f"Expected cache key {expected_key!r} not found in {seen_keys}"
    )
    assert 0 in results


# =====================================================================
# Test 8: Config set to 1/false = identical to old sequential behavior
# =====================================================================

def test_config_disables_all_optimizations():
    """With parallel_targets=1 and mirror_fallback=false, code path is sequential."""
    from src.ingestion.gaia_query import _GAIA_MIRRORS

    call_log: List[str] = []

    def mock_execute_timeout(adql, tap_url, timeout_sec):
        call_log.append(tap_url)
        return pd.DataFrame({"source_id": [1]})

    config_sequential = {
        "performance": {
            "parallel_targets": 1,
            "gaia_timeout_sec": 60,
            "gaia_mirror_fallback": False,
            "controls_cone_batch_size": 10,
        }
    }

    with patch("src.ingestion.gaia_query._execute_with_timeout", side_effect=mock_execute_timeout):
        with patch("src.ingestion.gaia_query.get_config", return_value=config_sequential):
            from src.ingestion.gaia_query import _run_sync_query

            df = _run_sync_query("SELECT TOP 1 * FROM gaiadr3.gaia_source")

    # Only the primary (ESA) mirror should have been called
    assert all("esac.esa" in url or _GAIA_MIRRORS[0]["url"] in url for url in call_log)
    assert isinstance(df, pd.DataFrame)


# =====================================================================
# Test 9: Empty batch returns empty dict
# =====================================================================

def test_batch_cone_empty_positions():
    """batch_cone_search with empty positions list returns empty dict."""
    from src.ingestion.gaia_query import batch_cone_search

    with patch("src.ingestion.gaia_query.load_cache", return_value=None):
        with patch("src.ingestion.gaia_query._run_sync_query", return_value=pd.DataFrame()):
            results = batch_cone_search([], radius_arcsec=600.0)

    assert results == {}
