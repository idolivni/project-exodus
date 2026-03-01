"""
Evidence bundle system for Project EXODUS.

Every breakthrough candidate (Level 3+) gets a formal evidence bundle
that packages all relevant data, analysis results, and provenance
for independent verification.

Public API
----------
EvidenceBundle (dataclass)
    target_id, run_id, timestamp, channel_results, multi_messenger_results,
    raw_data_refs, query_provenance, plots, detector_versions,
    breakthrough_level, analyst_result, notes

    to_dict() / from_dict(data)
        Round-trip serialisation.

save_evidence_bundle(bundle)
    Persist to ``data/evidence/<target_id>/bundle_<run_id>.json``.

load_evidence_bundle(target_id, run_id)
    Load a previously saved bundle.

list_evidence_bundles(target_id)
    List all bundle run IDs for a given target.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import get_logger, PROJECT_ROOT

log = get_logger("core.evidence")


# =====================================================================
#  Evidence bundle data class
# =====================================================================

@dataclass
class EvidenceBundle:
    """Formal evidence package for a breakthrough candidate.

    Every target that reaches escalation Level 3 or above receives an
    evidence bundle that captures the full chain of data, analysis,
    and provenance so an independent researcher can reproduce and
    verify the finding.

    Attributes
    ----------
    target_id : str
        Unique identifier for the astronomical target.
    run_id : str
        The EXODUS run that produced this bundle.
    timestamp : str
        ISO-8601 creation timestamp (auto-generated).
    channel_results : dict
        Per-channel detector outputs keyed by channel name.
    multi_messenger_results : dict
        Cross-messenger analysis results (gamma, neutrino, GW, etc.).
    raw_data_refs : list of str
        Paths or URIs to raw data files used in the analysis.
    query_provenance : list of dict
        Serialised QueryLogEntry records that produced this data.
    plots : list of str
        Paths to generated diagnostic / summary plots.
    detector_versions : dict
        Version strings keyed by detector module name.
    breakthrough_level : int
        The escalation level reached (1-6).
    analyst_result : dict or None
        Output from the Analyst engine, if available.
    notes : str
        Free-text notes attached by the pipeline or human reviewer.
    """

    target_id: str = ""
    run_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    channel_results: Dict[str, Any] = field(default_factory=dict)
    multi_messenger_results: Dict[str, Any] = field(default_factory=dict)
    raw_data_refs: List[str] = field(default_factory=list)
    query_provenance: List[Dict[str, Any]] = field(default_factory=list)
    plots: List[str] = field(default_factory=list)
    detector_versions: Dict[str, str] = field(default_factory=dict)
    breakthrough_level: int = 0
    analyst_result: Optional[Dict[str, Any]] = None
    notes: str = ""

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceBundle:
        """Reconstruct an EvidenceBundle from a dictionary.

        Parameters
        ----------
        data : dict
            Output of ``to_dict()``.

        Returns
        -------
        EvidenceBundle
        """
        return cls(**data)

    # ── Repr ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EvidenceBundle(target={self.target_id!r}, "
            f"run={self.run_id!r}, level={self.breakthrough_level})"
        )


# =====================================================================
#  Persistence helpers
# =====================================================================

def _evidence_dir(target_id: str) -> Path:
    """Return (and create) the evidence directory for a target."""
    d = PROJECT_ROOT / "data" / "evidence" / target_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_evidence_bundle(bundle: EvidenceBundle) -> Path:
    """Persist an evidence bundle to disk.

    Saves to ``data/evidence/<target_id>/bundle_<run_id>.json``.

    Parameters
    ----------
    bundle : EvidenceBundle
        The bundle to save.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    d = _evidence_dir(bundle.target_id)
    path = d / f"bundle_{bundle.run_id}.json"
    with open(path, "w") as f:
        json.dump(bundle.to_dict(), f, indent=2, default=str)
    log.info(
        "Saved evidence bundle for %s (run %s, level %d) to %s",
        bundle.target_id, bundle.run_id, bundle.breakthrough_level, path,
    )
    return path


def load_evidence_bundle(target_id: str, run_id: str) -> EvidenceBundle:
    """Load a previously saved evidence bundle.

    Parameters
    ----------
    target_id : str
        Target identifier.
    run_id : str
        Run identifier.

    Returns
    -------
    EvidenceBundle

    Raises
    ------
    FileNotFoundError
        If the bundle JSON file does not exist.
    """
    d = PROJECT_ROOT / "data" / "evidence" / target_id
    path = d / f"bundle_{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No evidence bundle at {path}")
    with open(path) as f:
        data = json.load(f)
    log.info("Loaded evidence bundle from %s", path)
    return EvidenceBundle.from_dict(data)


def list_evidence_bundles(target_id: str) -> List[str]:
    """List all evidence bundle run IDs for a target.

    Parameters
    ----------
    target_id : str
        Target identifier.

    Returns
    -------
    list of str
        Run IDs that have saved bundles, sorted alphabetically.
    """
    d = PROJECT_ROOT / "data" / "evidence" / target_id
    if not d.exists():
        return []
    run_ids = []
    for p in sorted(d.glob("bundle_*.json")):
        # Extract run_id from "bundle_<run_id>.json"
        stem = p.stem  # "bundle_<run_id>"
        run_id = stem[len("bundle_"):]
        run_ids.append(run_id)
    return run_ids


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    import uuid

    print("=" * 70)
    print("  Project EXODUS -- Evidence Bundle Module Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Create a demo evidence bundle
    # ------------------------------------------------------------------
    print("\n[1] Creating evidence bundle")
    print("-" * 50)

    demo_run_id = str(uuid.uuid4())
    demo_target = "KIC_8462852"

    bundle = EvidenceBundle(
        target_id=demo_target,
        run_id=demo_run_id,
        channel_results={
            "ir_excess": {"score": 0.72, "w1_w2": 0.35, "p_value": 0.003},
            "transit_anomaly": {"score": 0.88, "depth_var": 0.15, "p_value": 0.0001},
            "radio_anomaly": {"score": 0.12, "p_value": 0.45},
            "gaia_photometric": {"score": 0.65, "ruwe": 1.8, "p_value": 0.02},
        },
        multi_messenger_results={
            "gamma_crossmatch": {"n_sources": 1, "angular_sep_deg": 0.3, "significance": 2.1},
            "neutrino_crossmatch": {"n_events": 0, "upper_limit": True},
            "gw_crossmatch": {"n_events": 0},
        },
        raw_data_refs=[
            "data/cache/gaia/kic8462852_dr3.csv",
            "data/cache/wise/kic8462852_allwise.csv",
            "data/cache/kepler/kic8462852_lc.fits",
        ],
        query_provenance=[
            {"endpoint": "Gaia TAP", "query_string": "SELECT ...", "n_rows_returned": 1},
            {"endpoint": "WISE AllSky", "query_string": "SELECT ...", "n_rows_returned": 1},
            {"endpoint": "Kepler MAST", "query_string": "query_criteria(...)", "n_rows_returned": 18},
        ],
        plots=[
            "data/results/plots/kic8462852_ir_sed.png",
            "data/results/plots/kic8462852_lightcurve.png",
            "data/results/plots/kic8462852_radio_waterfall.png",
        ],
        detector_versions={
            "ir_excess": "1.2.0",
            "transit_anomaly": "2.0.1",
            "radio_anomaly": "0.9.3",
            "gaia_photometric": "1.1.0",
        },
        breakthrough_level=4,
        analyst_result={
            "verdict": "UNEXPLAINED",
            "confidence": 0.92,
            "natural_explanations_tested": [
                "variable_star", "eclipsing_binary", "dust_cloud",
                "instrumental_artifact",
            ],
            "all_rejected": True,
        },
        notes="Tabby's Star demo bundle -- strongest multi-channel candidate",
    )

    print(f"  Target:      {bundle.target_id}")
    print(f"  Run ID:      {bundle.run_id}")
    print(f"  Timestamp:   {bundle.timestamp}")
    print(f"  Level:       {bundle.breakthrough_level}")
    print(f"  Channels:    {len(bundle.channel_results)}")
    print(f"  Multi-msg:   {len(bundle.multi_messenger_results)}")
    print(f"  Raw refs:    {len(bundle.raw_data_refs)}")
    print(f"  Provenance:  {len(bundle.query_provenance)} queries")
    print(f"  Plots:       {len(bundle.plots)}")
    print(f"  Verdict:     {bundle.analyst_result['verdict']}")

    # ------------------------------------------------------------------
    # 2. Save the bundle
    # ------------------------------------------------------------------
    print("\n[2] Saving evidence bundle")
    print("-" * 50)

    path = save_evidence_bundle(bundle)
    print(f"  Saved to: {path}")
    assert path.exists(), "Bundle file should exist"
    print("  File exists: PASS")

    # ------------------------------------------------------------------
    # 3. Load it back
    # ------------------------------------------------------------------
    print("\n[3] Loading evidence bundle")
    print("-" * 50)

    loaded = load_evidence_bundle(demo_target, demo_run_id)
    assert loaded.target_id == bundle.target_id
    assert loaded.run_id == bundle.run_id
    assert loaded.breakthrough_level == bundle.breakthrough_level
    assert loaded.channel_results == bundle.channel_results
    assert loaded.analyst_result == bundle.analyst_result
    print(f"  Loaded: {loaded}")
    print("  All fields match: PASS")

    # ------------------------------------------------------------------
    # 4. List bundles for target
    # ------------------------------------------------------------------
    print("\n[4] Listing evidence bundles")
    print("-" * 50)

    # Save a second bundle for the same target
    second_run_id = str(uuid.uuid4())
    second_bundle = EvidenceBundle(
        target_id=demo_target,
        run_id=second_run_id,
        channel_results={"ir_excess": {"score": 0.68}},
        breakthrough_level=3,
        notes="Second demo bundle",
    )
    save_evidence_bundle(second_bundle)

    bundles = list_evidence_bundles(demo_target)
    print(f"  Found {len(bundles)} bundles for {demo_target}:")
    for rid in bundles:
        print(f"    - {rid}")
    assert len(bundles) >= 2, "Should have at least 2 bundles"
    assert demo_run_id in bundles
    assert second_run_id in bundles
    print("  Bundle listing: PASS")

    # ------------------------------------------------------------------
    # 5. Round-trip serialisation
    # ------------------------------------------------------------------
    print("\n[5] Dataclass round-trip serialisation")
    print("-" * 50)

    bundle_dict = bundle.to_dict()
    restored = EvidenceBundle.from_dict(bundle_dict)
    assert restored.to_dict() == bundle_dict
    print("  to_dict/from_dict round-trip: PASS")
    print(f"  JSON size: {len(json.dumps(bundle_dict, indent=2))} bytes")

    # ------------------------------------------------------------------
    # 6. Missing bundle error handling
    # ------------------------------------------------------------------
    print("\n[6] Error handling for missing bundles")
    print("-" * 50)

    try:
        load_evidence_bundle("NONEXISTENT_TARGET", "fake_run_id")
        print("  ERROR: Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  FileNotFoundError raised: PASS")
        print(f"    {e}")

    missing = list_evidence_bundles("NONEXISTENT_TARGET")
    assert missing == [], "Listing non-existent target should return empty list"
    print("  Empty listing for missing target: PASS")

    print("\n" + "=" * 70)
    print("  All evidence bundle tests passed.")
    print("=" * 70)
