"""
Candidate retention policy for Project EXODUS.

Borderline candidates that fall just below significance thresholds are
retained with documented reasons rather than silently discarded. This
prevents the system from throwing away potentially real signals that
could be validated with additional data.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, PROJECT_ROOT

log = get_logger("core.retention")

# Default persistence path
_DEFAULT_RETENTION_DIR = PROJECT_ROOT / "data" / "retention"
_DEFAULT_RETENTION_FILE = _DEFAULT_RETENTION_DIR / "boundary_candidates.json"


# =====================================================================
#  RetentionEntry — one borderline candidate record
# =====================================================================

@dataclass
class RetentionEntry:
    """A single borderline candidate that was retained rather than discarded.

    Attributes
    ----------
    target_id : str
        Unique identifier for the target (e.g. Gaia DR3 source ID, TIC, KIC).
    exodus_score : float
        The composite EXODUS anomaly score at the time of evaluation.
    combined_p : float
        Calibrated combined p-value (Fisher or Stouffer) across active channels.
    why_rejected : str
        The specific threshold or criterion that was *not* met. For example:
        ``"FDR-corrected p = 0.08 > 0.05 cutoff"`` or
        ``"only 1/3 active channels (minimum 2 required)"``.
    why_retained : str
        Human-readable justification for keeping the candidate. For example:
        ``"2/3 active channels but score below threshold"`` or
        ``"strong IR excess + marginal radio; pending deeper observation"``.
    evidence_summary : Dict[str, Any]
        Free-form dictionary summarising the evidence that motivated retention.
        Typical keys: ``active_channels``, ``channel_scores``, ``notes``.
    timestamp : str
        ISO-8601 UTC timestamp; auto-populated at creation time.
    """

    target_id: str
    exodus_score: float
    combined_p: float
    why_rejected: str
    why_retained: str
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default="")

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# =====================================================================
#  RetentionLog — collection of retained borderline candidates
# =====================================================================

class RetentionLog:
    """Manages a log of borderline candidates retained for future review.

    The log enforces the EXODUS directive: *never silently discard borderline
    candidates*.  Every near-miss is recorded with machine- and human-readable
    justification so that it can be revisited when new data or improved
    thresholds become available.
    """

    def __init__(self) -> None:
        self._entries: List[RetentionEntry] = []

    # ---- mutators ---------------------------------------------------

    def add(self, entry: RetentionEntry) -> None:
        """Append a retention entry to the log.

        Parameters
        ----------
        entry : RetentionEntry
            The borderline candidate record to retain.
        """
        self._entries.append(entry)
        log.info(
            "Retained borderline candidate %s  (score=%.4f, p=%.4e) — %s",
            entry.target_id,
            entry.exodus_score,
            entry.combined_p,
            entry.why_retained,
        )

    # ---- queries ----------------------------------------------------

    def get_all(self) -> List[RetentionEntry]:
        """Return every retained entry (chronological order)."""
        return list(self._entries)

    def get_for_target(self, target_id: str) -> List[RetentionEntry]:
        """Return all retained entries for a specific target.

        Parameters
        ----------
        target_id : str
            The target identifier to filter on.

        Returns
        -------
        List[RetentionEntry]
            Matching entries (may be empty).
        """
        return [e for e in self._entries if e.target_id == target_id]

    # ---- persistence ------------------------------------------------

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Serialise the retention log to JSON.

        Parameters
        ----------
        path : str or Path, optional
            Destination file.  Defaults to
            ``data/retention/boundary_candidates.json`` under the project root.

        Returns
        -------
        Path
            The path that was written.
        """
        out = Path(path) if path else _DEFAULT_RETENTION_FILE
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = [asdict(e) for e in self._entries]
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

        log.info("Saved %d retention entries to %s", len(self._entries), out)
        return out

    def load(self, path: Optional[str | Path] = None) -> None:
        """Load retention entries from a JSON file (replaces current state).

        Parameters
        ----------
        path : str or Path, optional
            Source file.  Defaults to
            ``data/retention/boundary_candidates.json`` under the project root.
        """
        src = Path(path) if path else _DEFAULT_RETENTION_FILE

        if not src.exists():
            log.warning("Retention file not found at %s — starting empty", src)
            self._entries = []
            return

        with open(src, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        self._entries = [RetentionEntry(**record) for record in raw]
        log.info("Loaded %d retention entries from %s", len(self._entries), src)

    # ---- reporting --------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the retention log.

        Returns
        -------
        str
            Multi-line summary suitable for console output or reports.
        """
        n = len(self._entries)
        if n == 0:
            return "Retention log is empty — no borderline candidates recorded."

        unique_targets = sorted({e.target_id for e in self._entries})
        scores = [e.exodus_score for e in self._entries]
        p_values = [e.combined_p for e in self._entries]

        lines = [
            "=" * 65,
            "  EXODUS Candidate Retention Summary",
            "=" * 65,
            f"  Total retained entries : {n}",
            f"  Unique targets         : {len(unique_targets)}",
            f"  Score range            : {min(scores):.4f} – {max(scores):.4f}",
            f"  Combined-p range       : {min(p_values):.2e} – {max(p_values):.2e}",
            "-" * 65,
        ]

        for tid in unique_targets:
            t_entries = self.get_for_target(tid)
            latest = t_entries[-1]
            lines.append(
                f"  {tid:30s}  score={latest.exodus_score:.4f}  "
                f"p={latest.combined_p:.2e}  ({len(t_entries)} record(s))"
            )
            lines.append(f"    rejected : {latest.why_rejected}")
            lines.append(f"    retained : {latest.why_retained}")

        lines.append("=" * 65)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"RetentionLog(entries={len(self._entries)})"


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("  Project EXODUS — Candidate Retention Module Demo")
    print("=" * 70)

    rlog = RetentionLog()

    # ------------------------------------------------------------------
    # Demo 1: Add borderline candidates
    # ------------------------------------------------------------------
    print("\n[1] Adding borderline candidates to retention log")
    print("-" * 50)

    rlog.add(RetentionEntry(
        target_id="TIC-470710327",
        exodus_score=0.42,
        combined_p=0.08,
        why_rejected="FDR-corrected p = 0.08 > 0.05 cutoff",
        why_retained="2/3 active channels (IR excess + marginal radio); "
                     "score just below threshold, worth revisiting with deeper obs",
        evidence_summary={
            "active_channels": ["neowise_ir", "radio_seti"],
            "channel_scores": {"neowise_ir": 0.65, "radio_seti": 0.31, "gaia_dimming": 0.05},
            "notes": "Strong W1-W2 color excess; radio hit at 1420 MHz marginal SNR",
        },
    ))

    rlog.add(RetentionEntry(
        target_id="Gaia-DR3-4056795432158",
        exodus_score=0.38,
        combined_p=0.12,
        why_rejected="Only 1/3 active channels (minimum 2 required for candidacy)",
        why_retained="Extreme Gaia dimming event (>3 sigma) but no corroborating channel; "
                     "pending NEOWISE reprocessing may reveal IR counterpart",
        evidence_summary={
            "active_channels": ["gaia_dimming"],
            "channel_scores": {"gaia_dimming": 0.82, "neowise_ir": 0.02, "radio_seti": 0.00},
            "notes": "Single deep dip in Gaia epoch photometry; reminiscent of Boyajian's Star",
        },
    ))

    rlog.add(RetentionEntry(
        target_id="KIC-8462852",
        exodus_score=0.55,
        combined_p=0.06,
        why_rejected="FDR-corrected p = 0.06 > 0.05 cutoff (barely missed)",
        why_retained="Benchmark Boyajian's Star — 3/3 active channels, known anomalous "
                     "target used for pipeline validation; borderline after FDR is expected",
        evidence_summary={
            "active_channels": ["neowise_ir", "gaia_dimming", "radio_seti"],
            "channel_scores": {"neowise_ir": 0.48, "gaia_dimming": 0.71, "radio_seti": 0.22},
            "notes": "Benchmark target; score should increase with real data calibration",
        },
    ))

    rlog.add(RetentionEntry(
        target_id="TIC-470710327",
        exodus_score=0.45,
        combined_p=0.07,
        why_rejected="FDR-corrected p = 0.07 > 0.05 cutoff",
        why_retained="Score improved after recalibration; still borderline but trending "
                     "toward significance with additional radio epochs",
        evidence_summary={
            "active_channels": ["neowise_ir", "radio_seti"],
            "channel_scores": {"neowise_ir": 0.67, "radio_seti": 0.35, "gaia_dimming": 0.05},
            "notes": "Re-evaluation after radio epoch 2; SNR improved marginally",
        },
    ))

    print(f"  Entries in log: {len(rlog)}")
    assert len(rlog) == 4

    # ------------------------------------------------------------------
    # Demo 2: Query by target
    # ------------------------------------------------------------------
    print("\n[2] Querying retention log by target")
    print("-" * 50)

    tic_entries = rlog.get_for_target("TIC-470710327")
    print(f"  TIC-470710327 has {len(tic_entries)} retention record(s)")
    assert len(tic_entries) == 2

    kic_entries = rlog.get_for_target("KIC-8462852")
    print(f"  KIC-8462852   has {len(kic_entries)} retention record(s)")
    assert len(kic_entries) == 1

    missing = rlog.get_for_target("NONEXISTENT-TARGET")
    print(f"  Nonexistent   has {len(missing)} retention record(s)")
    assert len(missing) == 0
    print("  PASS")

    # ------------------------------------------------------------------
    # Demo 3: Summary output
    # ------------------------------------------------------------------
    print("\n[3] Human-readable summary")
    print("-" * 50)
    print(rlog.summary())

    # ------------------------------------------------------------------
    # Demo 4: Save and reload
    # ------------------------------------------------------------------
    print("\n[4] Save / load round-trip")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_retention.json"
        saved_path = rlog.save(tmp_path)
        print(f"  Saved to: {saved_path}")

        rlog2 = RetentionLog()
        rlog2.load(tmp_path)
        print(f"  Loaded {len(rlog2)} entries from file")
        assert len(rlog2) == len(rlog), "Round-trip entry count mismatch"

        # Verify field integrity
        orig = rlog.get_all()
        loaded = rlog2.get_all()
        for o, l in zip(orig, loaded):
            assert o.target_id == l.target_id, f"target_id mismatch: {o.target_id} vs {l.target_id}"
            assert abs(o.exodus_score - l.exodus_score) < 1e-9, "exodus_score mismatch"
            assert abs(o.combined_p - l.combined_p) < 1e-9, "combined_p mismatch"
            assert o.why_rejected == l.why_rejected, "why_rejected mismatch"
            assert o.why_retained == l.why_retained, "why_retained mismatch"
            assert o.evidence_summary == l.evidence_summary, "evidence_summary mismatch"
            assert o.timestamp == l.timestamp, "timestamp mismatch"
        print("  PASS: All fields match after round-trip")

    # ------------------------------------------------------------------
    # Demo 5: Load from nonexistent file (graceful)
    # ------------------------------------------------------------------
    print("\n[5] Graceful load from nonexistent file")
    print("-" * 50)

    rlog3 = RetentionLog()
    rlog3.load("/tmp/nonexistent_retention_file.json")
    assert len(rlog3) == 0
    print("  PASS: Empty log returned for missing file")

    # ------------------------------------------------------------------
    # Demo 6: Empty log summary
    # ------------------------------------------------------------------
    print("\n[6] Empty log summary")
    print("-" * 50)

    empty_log = RetentionLog()
    empty_summary = empty_log.summary()
    print(f"  {empty_summary}")
    assert "empty" in empty_summary.lower()
    print("  PASS")

    print("\n" + "=" * 70)
    print("  All retention module tests passed.")
    print("=" * 70)
