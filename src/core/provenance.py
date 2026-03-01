"""
Provenance and reproducibility infrastructure for Project EXODUS.

Every data query is logged with its parameters, results, and timing.
Run manifests capture the full state of a research iteration for
exact reproducibility.

Public API
----------
QueryLogEntry (dataclass)
    endpoint, query_string, params, timestamp, n_rows_returned,
    cache_hit, duration_ms

RunManifest (dataclass)
    run_id, timestamp, config_hash, random_seed, dataset_versions,
    n_targets, thresholds, notes

ProvenanceLogger
    log_query(endpoint, query_string, params, n_rows, cache_hit, duration_ms)
        Append a query record to the session log.

    get_queries() -> List[QueryLogEntry]
        Return all queries logged so far.

    create_manifest(config, seed, n_targets, thresholds) -> RunManifest
        Construct a manifest capturing the current run state.

    save_manifest(manifest)
        Persist to ``data/runs/<run_id>/manifest.json``.

    save_query_log(run_id)
        Persist all logged queries to ``data/runs/<run_id>/queries.json``.

    clear()
        Reset the logger for the next run.

Module-level singleton
    ``provenance_logger`` -- ready-to-use instance of ProvenanceLogger.
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import get_logger, PROJECT_ROOT

log = get_logger("core.provenance")


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class QueryLogEntry:
    """A single logged data query."""

    endpoint: str
    query_string: str
    params: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    n_rows_returned: int = 0
    cache_hit: bool = False
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryLogEntry:
        """Reconstruct from a dictionary."""
        return cls(**data)


@dataclass
class RunManifest:
    """Captures the full state of a single research run for reproducibility."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str = ""
    random_seed: int = 0
    dataset_versions: Dict[str, str] = field(default_factory=dict)
    n_targets: int = 0
    thresholds: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunManifest:
        """Reconstruct from a dictionary."""
        return cls(**data)


# =====================================================================
#  ProvenanceLogger
# =====================================================================

class ProvenanceLogger:
    """Session-scoped logger for data queries and run manifests.

    Use the module-level ``provenance_logger`` instance for a singleton-
    like experience.  Multiple instances are possible for testing.
    """

    def __init__(self) -> None:
        self._queries: List[QueryLogEntry] = []

    # ── Query logging ─────────────────────────────────────────────

    def log_query(
        self,
        endpoint: str,
        query_string: str,
        params: Dict[str, Any],
        n_rows: int = 0,
        cache_hit: bool = False,
        duration_ms: float = 0.0,
    ) -> QueryLogEntry:
        """Record a data query.

        Parameters
        ----------
        endpoint : str
            The service or API endpoint queried (e.g. "Gaia TAP").
        query_string : str
            The query text (ADQL, SQL, URL, etc.).
        params : dict
            Query parameters / filters applied.
        n_rows : int
            Number of rows returned by the query.
        cache_hit : bool
            True if the result came from local cache.
        duration_ms : float
            Wall-clock time for the query in milliseconds.

        Returns
        -------
        QueryLogEntry
            The newly created log entry.
        """
        entry = QueryLogEntry(
            endpoint=endpoint,
            query_string=query_string,
            params=params,
            n_rows_returned=n_rows,
            cache_hit=cache_hit,
            duration_ms=duration_ms,
        )
        self._queries.append(entry)
        log.debug(
            "Logged query: %s (%d rows, %.1f ms, cache=%s)",
            endpoint, n_rows, duration_ms, cache_hit,
        )
        return entry

    def get_queries(self) -> List[QueryLogEntry]:
        """Return all queries logged in this session."""
        return list(self._queries)

    # ── Manifest creation ─────────────────────────────────────────

    def create_manifest(
        self,
        config: Dict[str, Any],
        seed: int,
        n_targets: int,
        thresholds: Dict[str, float],
        notes: str = "",
    ) -> RunManifest:
        """Construct a run manifest capturing reproducibility state.

        Parameters
        ----------
        config : dict
            The full configuration dict (will be hashed).
        seed : int
            The random seed used for this run.
        n_targets : int
            Number of targets processed.
        thresholds : dict
            Detection thresholds keyed by channel name.
        notes : str, optional
            Free-text notes about this run.

        Returns
        -------
        RunManifest
        """
        config_json = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

        manifest = RunManifest(
            config_hash=config_hash,
            random_seed=seed,
            n_targets=n_targets,
            thresholds=thresholds,
            notes=notes,
        )
        log.info(
            "Created manifest run_id=%s  config_hash=%s  targets=%d",
            manifest.run_id, config_hash, n_targets,
        )
        return manifest

    # ── Persistence ───────────────────────────────────────────────

    def save_manifest(self, manifest: RunManifest) -> Path:
        """Save a run manifest to ``data/runs/<run_id>/manifest.json``.

        Parameters
        ----------
        manifest : RunManifest
            The manifest to persist.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        run_dir = PROJECT_ROOT / "data" / "runs" / manifest.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        with open(path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2, default=str)
        log.info("Saved manifest to %s", path)
        return path

    def save_query_log(self, run_id: str) -> Path:
        """Save all logged queries to ``data/runs/<run_id>/queries.json``.

        Parameters
        ----------
        run_id : str
            The run ID whose directory will hold the log.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        run_dir = PROJECT_ROOT / "data" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "queries.json"
        entries = [e.to_dict() for e in self._queries]
        with open(path, "w") as f:
            json.dump(entries, f, indent=2, default=str)
        log.info("Saved %d query log entries to %s", len(entries), path)
        return path

    # ── Reset ─────────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset the query log for the next run."""
        n = len(self._queries)
        self._queries.clear()
        log.debug("Cleared %d query log entries", n)

    # ── Repr ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ProvenanceLogger(queries={len(self._queries)})"


# ── Module-level singleton ────────────────────────────────────────────
provenance_logger = ProvenanceLogger()


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Provenance & Reproducibility Module Demo")
    print("=" * 70)

    pl = ProvenanceLogger()

    # ------------------------------------------------------------------
    # 1. Log some queries
    # ------------------------------------------------------------------
    print("\n[1] Logging data queries")
    print("-" * 50)

    pl.log_query(
        endpoint="Gaia TAP",
        query_string="SELECT * FROM gaiadr3.gaia_source WHERE source_id IN (...)",
        params={"source_ids": ["12345", "67890"], "columns": "ra,dec,phot_g_mean_mag"},
        n_rows=2,
        cache_hit=False,
        duration_ms=1250.3,
    )
    pl.log_query(
        endpoint="WISE AllSky",
        query_string="SELECT * FROM allwise_p3as_psd WHERE ...",
        params={"ra": 297.695, "dec": 46.952, "radius_deg": 0.001},
        n_rows=1,
        cache_hit=True,
        duration_ms=45.7,
    )
    pl.log_query(
        endpoint="Kepler/TESS MAST",
        query_string="astroquery.mast.Observations.query_criteria(...)",
        params={"target_name": "KIC 8462852", "obs_collection": "Kepler"},
        n_rows=18,
        cache_hit=False,
        duration_ms=3200.0,
    )
    pl.log_query(
        endpoint="Breakthrough Listen",
        query_string="SELECT * FROM bl_observations WHERE ...",
        params={"target": "KIC 8462852"},
        n_rows=0,
        cache_hit=False,
        duration_ms=890.1,
    )
    pl.log_query(
        endpoint="Fermi 4FGL",
        query_string="cone search around target coordinates",
        params={"ra": 297.695, "dec": 46.952, "radius_deg": 0.5},
        n_rows=3,
        cache_hit=True,
        duration_ms=120.0,
    )

    queries = pl.get_queries()
    print(f"  Logged {len(queries)} queries")
    for q in queries:
        print(f"    {q.endpoint:<25s}  rows={q.n_rows_returned:3d}  "
              f"cache={str(q.cache_hit):<5s}  {q.duration_ms:.0f} ms")

    # ------------------------------------------------------------------
    # 2. Create a run manifest
    # ------------------------------------------------------------------
    print("\n[2] Creating run manifest")
    print("-" * 50)

    demo_config = {
        "channels": ["ir_excess", "transit_anomaly", "radio", "gaia_photo"],
        "thresholds": {"ir_excess": 0.3, "transit": 0.25},
        "max_targets": 1000,
    }

    manifest = pl.create_manifest(
        config=demo_config,
        seed=42,
        n_targets=847,
        thresholds={"ir_excess": 0.3, "transit_anomaly": 0.25, "radio": 0.4},
        notes="Demo run for provenance module testing",
    )

    print(f"  run_id:      {manifest.run_id}")
    print(f"  timestamp:   {manifest.timestamp}")
    print(f"  config_hash: {manifest.config_hash}")
    print(f"  seed:        {manifest.random_seed}")
    print(f"  n_targets:   {manifest.n_targets}")
    print(f"  thresholds:  {manifest.thresholds}")
    print(f"  notes:       {manifest.notes}")

    # ------------------------------------------------------------------
    # 3. Save manifest and query log
    # ------------------------------------------------------------------
    print("\n[3] Saving manifest and query log")
    print("-" * 50)

    manifest_path = pl.save_manifest(manifest)
    queries_path = pl.save_query_log(manifest.run_id)
    print(f"  Manifest saved to: {manifest_path}")
    print(f"  Query log saved to: {queries_path}")

    # Verify files exist and are valid JSON
    with open(manifest_path) as f:
        loaded_manifest = json.load(f)
    assert loaded_manifest["run_id"] == manifest.run_id
    print("  Manifest JSON valid: PASS")

    with open(queries_path) as f:
        loaded_queries = json.load(f)
    assert len(loaded_queries) == 5
    print(f"  Query log JSON valid: PASS ({len(loaded_queries)} entries)")

    # ------------------------------------------------------------------
    # 4. Round-trip dataclass serialisation
    # ------------------------------------------------------------------
    print("\n[4] Dataclass round-trip serialisation")
    print("-" * 50)

    manifest_dict = manifest.to_dict()
    restored_manifest = RunManifest.from_dict(manifest_dict)
    assert restored_manifest.run_id == manifest.run_id
    assert restored_manifest.config_hash == manifest.config_hash
    assert restored_manifest.random_seed == manifest.random_seed
    print("  RunManifest to_dict/from_dict: PASS")

    entry = queries[0]
    entry_dict = entry.to_dict()
    restored_entry = QueryLogEntry.from_dict(entry_dict)
    assert restored_entry.endpoint == entry.endpoint
    assert restored_entry.n_rows_returned == entry.n_rows_returned
    print("  QueryLogEntry to_dict/from_dict: PASS")

    # ------------------------------------------------------------------
    # 5. Clear and verify
    # ------------------------------------------------------------------
    print("\n[5] Clear and verify")
    print("-" * 50)

    pl.clear()
    assert len(pl.get_queries()) == 0
    print("  After clear(): 0 queries (PASS)")

    print("\n" + "=" * 70)
    print("  All provenance module tests passed.")
    print("=" * 70)
