"""
Hypothesis Generator engine for Project EXODUS.

Creates, tracks, and evolves testable hypotheses about where and how to find
technosignatures.  Each hypothesis specifies a claim, the dataset(s) to test it
against, the analysis method, and a Kardashev-level tag indicating the
civilisation type that would produce the expected signal.

Hypotheses live in a SQLite database (``data/hypotheses/hypotheses.db``) managed
via SQLAlchemy.  The generator seeds the database with an initial bank of
high-priority hypotheses and can autonomously create follow-up hypotheses when
earlier results surface interesting targets.

Three tiers of hypothesis generation
-------------------------------------
1. **Pattern-based** -- "Target X shows anomaly A; does it also show anomaly B?"
2. **Systematic**    -- "Scan all targets with property A for property B."
3. **Creative**      -- Novel search strategies drawn from a curated idea bank,
   plus the ability to inject new strategies at runtime.

Key references
--------------
- Wright et al. 2014, ApJ 792, 26  (Glimpsing Heat from Alien Technologies)
- Sheikh et al. 2020, PASP 132, 044502  (Breakthrough Listen)
- Suazo et al. 2024, MNRAS 527, 1  (Project Hephaistos)
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import PROJECT_ROOT, get_logger

log = get_logger("engines.generator")

# =====================================================================
#  SQLAlchemy model
# =====================================================================

Base = declarative_base()

VALID_STATUSES = ("pending", "tested", "confirmed", "rejected")


class Hypothesis(Base):
    """ORM model for a single testable hypothesis."""

    __tablename__ = "hypotheses"

    hypothesis_id = Column(String(64), primary_key=True)
    claim         = Column(Text, nullable=False)
    dataset       = Column(String(256), nullable=False)
    method        = Column(String(128), nullable=False)
    status        = Column(
        Enum(*VALID_STATUSES, name="hypothesis_status"),
        nullable=False,
        default="pending",
    )
    results       = Column(Text, nullable=True)          # JSON blob
    timestamp     = Column(DateTime, nullable=False)
    kardashev     = Column(String(32), nullable=True)
    parent_id     = Column(String(64), nullable=True)    # FK to generating hypothesis

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim":         self.claim,
            "dataset":       self.dataset,
            "method":        self.method,
            "status":        self.status,
            "results":       json.loads(self.results) if self.results else None,
            "timestamp":     self.timestamp.isoformat() if self.timestamp else None,
            "kardashev":     self.kardashev,
            "parent_id":     self.parent_id,
        }

    def __repr__(self) -> str:
        return (
            f"<Hypothesis {self.hypothesis_id} "
            f"[{self.status}] {self.claim[:60]}...>"
        )


# =====================================================================
#  Initial hypothesis bank
# =====================================================================

INITIAL_HYPOTHESES: List[Dict[str, str]] = [
    {
        "id":       "H001",
        "claim":    "Stars with confirmed HZ exoplanets within 100pc have higher "
                    "IR excess than control sample",
        "dataset":  "gaia+wise+exoplanet_archive",
        "method":   "ir_excess_comparison",
        "kardashev": "Type II",
    },
    {
        "id":       "H002",
        "claim":    "Gaia epoch photometry of HZ exoplanet hosts contains "
                    "non-periodic asymmetric dimming events",
        "dataset":  "gaia_epoch_photometry",
        "method":   "anomaly_detection_lightcurve",
        "kardashev": "Type II",
    },
    {
        "id":       "H003",
        "claim":    "BL radio observations of stars with IR excess contain more "
                    "narrowband signal candidates than control stars",
        "dataset":  "breakthrough_listen+wise",
        "method":   "radio_search_targeted",
        "kardashev": "Type I",
    },
    {
        "id":       "H004",
        "claim":    "Kepler/TESS single-transit events preferentially occur "
                    "around stars with IR excess",
        "dataset":  "kepler+tess+wise",
        "method":   "single_transit_ir_correlation",
        "kardashev": "Type II",
    },
    {
        "id":       "H005",
        "claim":    "Cross-matching radio survey catalogs across decades reveals "
                    "sources that appeared near exoplanet hosts",
        "dataset":  "nvss+vlass+exoplanet_archive",
        "method":   "temporal_differencing",
        "kardashev": "Type I",
    },
]


# =====================================================================
#  Creative search strategies (Tier 3)
# =====================================================================

CREATIVE_STRATEGIES: List[Dict[str, str]] = [
    {
        "claim":    "Gaia astrometric excess noise correlates with IR excess in "
                    "HZ exoplanet hosts, possibly indicating unresolved megastructure",
        "dataset":  "gaia+wise+exoplanet_archive",
        "method":   "astrometric_noise_ir_correlation",
        "kardashev": "Type II",
    },
    {
        "claim":    "Stellar metallicity outliers among HZ exoplanet hosts show "
                    "preferential clustering in galactic position, suggesting a "
                    "common origin or expansion front",
        "dataset":  "gaia+galah+exoplanet_archive",
        "method":   "metallicity_spatial_clustering",
        "kardashev": "Type III",
    },
    {
        "claim":    "TESS full-frame images contain co-moving faint transients "
                    "along the ecliptic that are absent from known minor-body "
                    "catalogs",
        "dataset":  "tess_ffi+mpc",
        "method":   "ecliptic_transient_search",
        "kardashev": "Type I",
    },
    {
        "claim":    "Pairs of nearby stars with correlated unusual variability "
                    "suggest coordinated or shared artificial activity",
        "dataset":  "gaia_epoch_photometry+tess",
        "method":   "correlated_variability_pairs",
        "kardashev": "Type II",
    },
    {
        "claim":    "The void region between spiral arms contains anomalous "
                    "point-source mid-IR emitters inconsistent with known "
                    "astrophysical populations",
        "dataset":  "wise+gaia",
        "method":   "interarm_ir_anomaly_search",
        "kardashev": "Type III",
    },
]


# =====================================================================
#  Follow-up hypothesis templates
# =====================================================================
#
# When a parent hypothesis finds interesting targets, these templates are
# instantiated with target-specific details.  Each template is a callable
# that receives the parent hypothesis dict and a summary of results and
# returns a list of (claim, dataset, method, kardashev) tuples.

def _followup_radio_search(parent: Dict, results: Dict) -> List[tuple]:
    """Check if interesting targets have Breakthrough Listen radio data."""
    target_names = results.get("interesting_targets", [])
    if not target_names:
        return []
    targets_str = ", ".join(str(t) for t in target_names[:10])
    return [(
        f"Targets from {parent['hypothesis_id']} ({targets_str}) have "
        f"Breakthrough Listen radio observations containing narrowband "
        f"signal candidates",
        "breakthrough_listen",
        "radio_search_targeted",
        "Type I",
    )]


def _followup_spatial_clustering(parent: Dict, results: Dict) -> List[tuple]:
    """Check whether interesting targets cluster in galactic coordinates."""
    n_targets = results.get("n_interesting", 0)
    if n_targets < 3:
        return []
    return [(
        f"The {n_targets} interesting targets from {parent['hypothesis_id']} "
        f"cluster spatially in galactic coordinates beyond random expectation",
        "gaia",
        "spatial_clustering_galactic",
        parent.get("kardashev", "Type II"),
    )]


def _followup_proper_motion(parent: Dict, results: Dict) -> List[tuple]:
    """Check for unusual Gaia proper motions."""
    target_names = results.get("interesting_targets", [])
    if not target_names:
        return []
    targets_str = ", ".join(str(t) for t in target_names[:10])
    return [(
        f"Targets from {parent['hypothesis_id']} ({targets_str}) exhibit "
        f"unusual Gaia proper motions inconsistent with their spectral type "
        f"and distance",
        "gaia",
        "proper_motion_anomaly",
        parent.get("kardashev", "Type II"),
    )]


def _followup_lightcurve_check(parent: Dict, results: Dict) -> List[tuple]:
    """Look for anomalous dimming in lightcurves of interesting targets."""
    target_names = results.get("interesting_targets", [])
    if not target_names:
        return []
    targets_str = ", ".join(str(t) for t in target_names[:10])
    return [(
        f"Targets from {parent['hypothesis_id']} ({targets_str}) display "
        f"anomalous non-periodic dimming events in Kepler/TESS/Gaia "
        f"lightcurves",
        "kepler+tess+gaia_epoch_photometry",
        "anomaly_detection_lightcurve",
        "Type II",
    )]


def _followup_multiwavelength_convergence(parent: Dict, results: Dict) -> List[tuple]:
    """Cross-check interesting targets across all available wavelengths."""
    n_targets = results.get("n_interesting", 0)
    if n_targets < 2:
        return []
    return [(
        f"Interesting targets from {parent['hypothesis_id']} show anomalies "
        f"in at least two independent wavelength domains (IR + radio, or "
        f"IR + optical variability)",
        "gaia+wise+breakthrough_listen+kepler+tess",
        "multiwavelength_convergence",
        parent.get("kardashev", "Type II"),
    )]


FOLLOWUP_GENERATORS = [
    _followup_radio_search,
    _followup_spatial_clustering,
    _followup_proper_motion,
    _followup_lightcurve_check,
    _followup_multiwavelength_convergence,
]


# =====================================================================
#  HypothesisGenerator
# =====================================================================

class HypothesisGenerator:
    """Creates, tracks, and evolves technosignature search hypotheses.

    Parameters
    ----------
    db_path : str or Path, optional
        Path to the SQLite database file.  Defaults to
        ``<PROJECT_ROOT>/data/hypotheses/hypotheses.db``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = PROJECT_ROOT / "data" / "hypotheses" / "hypotheses.db"
        else:
            db_path = Path(db_path)

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            future=True,
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

        log.info("Hypothesis database initialised at %s", db_path)

        # Seed initial hypotheses (idempotent)
        self._seed_initial_hypotheses()

    # ── Seeding ──────────────────────────────────────────────────────

    def _seed_initial_hypotheses(self) -> None:
        """Insert the initial hypothesis bank if they do not already exist."""
        with self._Session() as session:
            existing_ids = {
                row[0]
                for row in session.query(Hypothesis.hypothesis_id).all()
            }

            added = 0
            for h in INITIAL_HYPOTHESES:
                hid = h["id"]
                if hid in existing_ids:
                    continue
                hypothesis = Hypothesis(
                    hypothesis_id=hid,
                    claim=h["claim"],
                    dataset=h["dataset"],
                    method=h["method"],
                    status="pending",
                    results=None,
                    timestamp=datetime.now(timezone.utc),
                    kardashev=h.get("kardashev"),
                    parent_id=None,
                )
                session.add(hypothesis)
                added += 1

            session.commit()

        if added:
            log.info("Seeded %d initial hypotheses into the database", added)
        else:
            log.debug("Initial hypotheses already present; nothing to seed")

    # ── Queries ──────────────────────────────────────────────────────

    def get_pending(self) -> List[Dict[str, Any]]:
        """Return all hypotheses with status ``'pending'``."""
        with self._Session() as session:
            rows = (
                session.query(Hypothesis)
                .filter(Hypothesis.status == "pending")
                .order_by(Hypothesis.timestamp)
                .all()
            )
            return [r.to_dict() for r in rows]

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """Return a single hypothesis by ID, or ``None`` if not found."""
        with self._Session() as session:
            row = session.get(Hypothesis, hypothesis_id)
            if row is None:
                log.warning("Hypothesis %s not found", hypothesis_id)
                return None
            return row.to_dict()

    def get_all(self) -> List[Dict[str, Any]]:
        """Return every hypothesis in the database."""
        with self._Session() as session:
            rows = (
                session.query(Hypothesis)
                .order_by(Hypothesis.timestamp)
                .all()
            )
            return [r.to_dict() for r in rows]

    # ── Mutations ────────────────────────────────────────────────────

    def update_status(
        self,
        hypothesis_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update the status (and optionally the results) of a hypothesis.

        Parameters
        ----------
        hypothesis_id : str
            The ID of the hypothesis to update.
        status : str
            New status.  Must be one of ``'pending'``, ``'tested'``,
            ``'confirmed'``, ``'rejected'``.
        results : dict, optional
            JSON-serialisable results payload to store alongside the status.

        Returns
        -------
        bool
            ``True`` if the hypothesis was found and updated.
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'; must be one of {VALID_STATUSES}"
            )

        with self._Session() as session:
            row = session.get(Hypothesis, hypothesis_id)
            if row is None:
                log.warning(
                    "Cannot update status: hypothesis %s not found",
                    hypothesis_id,
                )
                return False

            old_status = row.status
            row.status = status
            if results is not None:
                row.results = json.dumps(results, default=str)

            session.commit()

        log.info(
            "Hypothesis %s: %s -> %s%s",
            hypothesis_id,
            old_status,
            status,
            f" (results stored)" if results else "",
        )
        return True

    def add_hypothesis(
        self,
        claim: str,
        dataset: str,
        method: str,
        kardashev: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Add a new hypothesis to the database.

        Parameters
        ----------
        claim : str
            The testable claim.
        dataset : str
            Dataset(s) required (e.g. ``"gaia+wise"``).
        method : str
            Analysis method name.
        kardashev : str, optional
            Kardashev level tag (e.g. ``"Type I"``, ``"Type II"``).
        parent_id : str, optional
            The ID of the hypothesis that spawned this one.

        Returns
        -------
        str
            The newly generated hypothesis ID.
        """
        hid = f"H{uuid.uuid4().hex[:8].upper()}"

        with self._Session() as session:
            hypothesis = Hypothesis(
                hypothesis_id=hid,
                claim=claim,
                dataset=dataset,
                method=method,
                status="pending",
                results=None,
                timestamp=datetime.now(timezone.utc),
                kardashev=kardashev,
                parent_id=parent_id,
            )
            session.add(hypothesis)
            session.commit()

        log.info(
            "Added hypothesis %s (parent=%s): %s",
            hid,
            parent_id or "none",
            claim[:80],
        )
        return hid

    def generate_followups(
        self,
        hypothesis_id: str,
        results: Dict[str, Any],
    ) -> List[str]:
        """Generate follow-up hypotheses based on the results of a tested hypothesis.

        Applies every registered follow-up template to the parent hypothesis
        and its results.  Each template can produce zero or more new hypotheses.

        Parameters
        ----------
        hypothesis_id : str
            The parent hypothesis that was tested.
        results : dict
            The results from testing the parent.  Expected keys:

            - ``"interesting_targets"`` -- list of target names / IDs
            - ``"n_interesting"``       -- count of interesting targets
              (auto-computed from the list if absent)

        Returns
        -------
        list of str
            The IDs of all newly created follow-up hypotheses.
        """
        parent = self.get_hypothesis(hypothesis_id)
        if parent is None:
            log.error(
                "Cannot generate follow-ups: hypothesis %s not found",
                hypothesis_id,
            )
            return []

        # Ensure n_interesting is populated
        if "n_interesting" not in results:
            targets = results.get("interesting_targets", [])
            results["n_interesting"] = len(targets)

        new_ids: List[str] = []

        for gen_fn in FOLLOWUP_GENERATORS:
            try:
                proposals = gen_fn(parent, results)
            except Exception as exc:
                log.error(
                    "Follow-up generator %s raised %s: %s",
                    gen_fn.__name__,
                    type(exc).__name__,
                    exc,
                )
                continue

            for claim, dataset, method, kardashev in proposals:
                hid = self.add_hypothesis(
                    claim=claim,
                    dataset=dataset,
                    method=method,
                    kardashev=kardashev,
                    parent_id=hypothesis_id,
                )
                new_ids.append(hid)

        log.info(
            "Generated %d follow-up hypotheses from %s",
            len(new_ids),
            hypothesis_id,
        )
        return new_ids

    # ── Statistics ───────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the hypothesis database.

        Returns
        -------
        dict
            Keys: ``total``, ``by_status`` (dict of status -> count),
            ``by_kardashev`` (dict of level -> count), ``with_parent``
            (number of follow-up hypotheses), ``oldest``, ``newest``.
        """
        with self._Session() as session:
            total = session.query(func.count(Hypothesis.hypothesis_id)).scalar()

            status_counts = dict(
                session.query(Hypothesis.status, func.count(Hypothesis.hypothesis_id))
                .group_by(Hypothesis.status)
                .all()
            )

            kardashev_counts = dict(
                session.query(
                    Hypothesis.kardashev, func.count(Hypothesis.hypothesis_id)
                )
                .group_by(Hypothesis.kardashev)
                .all()
            )

            with_parent = (
                session.query(func.count(Hypothesis.hypothesis_id))
                .filter(Hypothesis.parent_id.isnot(None))
                .scalar()
            )

            oldest = session.query(func.min(Hypothesis.timestamp)).scalar()
            newest = session.query(func.max(Hypothesis.timestamp)).scalar()

        return {
            "total":        total or 0,
            "by_status":    status_counts,
            "by_kardashev": kardashev_counts,
            "with_parent":  with_parent or 0,
            "oldest":       oldest.isoformat() if oldest else None,
            "newest":       newest.isoformat() if newest else None,
        }

    # ── Creative hypothesis injection ────────────────────────────────

    def inject_creative_strategies(self) -> List[str]:
        """Seed the creative / novel hypothesis strategies into the database.

        Only strategies not already present (matched by claim text) are added.

        Returns
        -------
        list of str
            IDs of newly created hypotheses.
        """
        with self._Session() as session:
            existing_claims = {
                row[0] for row in session.query(Hypothesis.claim).all()
            }

        new_ids: List[str] = []
        for strat in CREATIVE_STRATEGIES:
            if strat["claim"] in existing_claims:
                continue
            hid = self.add_hypothesis(
                claim=strat["claim"],
                dataset=strat["dataset"],
                method=strat["method"],
                kardashev=strat.get("kardashev"),
            )
            new_ids.append(hid)

        if new_ids:
            log.info("Injected %d creative strategies", len(new_ids))
        else:
            log.debug("All creative strategies already present")

        return new_ids


# =====================================================================
#  CLI demo
# =====================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  Project EXODUS -- Hypothesis Generator Engine Demo")
    print("=" * 70)

    # Use a temporary in-memory-style DB for demo (but still on disk so we
    # can inspect it).  A separate path avoids clobbering real data.
    demo_db = PROJECT_ROOT / "data" / "hypotheses" / "demo_hypotheses.db"
    # Clean slate for the demo
    if demo_db.exists():
        demo_db.unlink()

    gen = HypothesisGenerator(db_path=str(demo_db))

    # ── Print initial hypotheses ─────────────────────────────────────
    print("\n--- Initial hypotheses (seeded on first run) ---")
    for h in gen.get_all():
        print(
            f"  [{h['hypothesis_id']}] [{h['status']:>9s}] "
            f"[{h['kardashev'] or '?':>8s}]  {h['claim'][:72]}..."
        )

    # ── Inject creative strategies ───────────────────────────────────
    print("\n--- Injecting creative search strategies ---")
    creative_ids = gen.inject_creative_strategies()
    print(f"  Added {len(creative_ids)} creative hypotheses: {creative_ids}")

    # ── Stats after seeding ──────────────────────────────────────────
    print("\n--- Database statistics ---")
    stats = gen.get_stats()
    print(f"  Total hypotheses : {stats['total']}")
    print(f"  By status        : {stats['by_status']}")
    print(f"  By Kardashev     : {stats['by_kardashev']}")
    print(f"  Follow-ups       : {stats['with_parent']}")

    # ── Simulate testing H001 ────────────────────────────────────────
    print("\n--- Simulating test of H001 ---")
    h001 = gen.get_hypothesis("H001")
    print(f"  Claim: {h001['claim']}")
    print(f"  Status before: {h001['status']}")

    # Suppose H001 found 5 interesting stars
    simulated_results = {
        "interesting_targets": [
            "Gaia DR3 4567890123456",
            "Gaia DR3 1234567890123",
            "Gaia DR3 9876543210987",
            "Gaia DR3 5555555555555",
            "Gaia DR3 7777777777777",
        ],
        "n_interesting": 5,
        "mean_ir_excess_sigma": 4.2,
        "control_sample_size": 500,
        "p_value": 0.0012,
        "summary": "5 stars with HZ exoplanets within 100pc show IR excess "
                   "at >3 sigma; statistically significant compared to "
                   "matched control sample (p=0.0012).",
    }

    gen.update_status("H001", "tested", results=simulated_results)
    h001_updated = gen.get_hypothesis("H001")
    print(f"  Status after : {h001_updated['status']}")
    print(f"  Results      : {json.dumps(h001_updated['results'], indent=4)}")

    # ── Generate follow-up hypotheses ────────────────────────────────
    print("\n--- Generating follow-up hypotheses from H001 results ---")
    followup_ids = gen.generate_followups("H001", simulated_results)
    print(f"  Generated {len(followup_ids)} follow-up hypotheses:")
    for fid in followup_ids:
        fh = gen.get_hypothesis(fid)
        print(
            f"    [{fid}] (parent={fh['parent_id']}) "
            f"[{fh['kardashev'] or '?':>8s}]  {fh['claim'][:65]}..."
        )

    # ── Final stats ──────────────────────────────────────────────────
    print("\n--- Final database statistics ---")
    stats = gen.get_stats()
    print(f"  Total hypotheses : {stats['total']}")
    print(f"  By status        : {stats['by_status']}")
    print(f"  By Kardashev     : {stats['by_kardashev']}")
    print(f"  Follow-ups       : {stats['with_parent']}")

    # ── List all pending ─────────────────────────────────────────────
    print("\n--- All pending hypotheses (ready for testing) ---")
    pending = gen.get_pending()
    for h in pending:
        parent_str = f"  (from {h['parent_id']})" if h["parent_id"] else ""
        print(
            f"  [{h['hypothesis_id']}] {h['claim'][:65]}...{parent_str}"
        )
    print(f"\n  {len(pending)} hypotheses ready for testing")

    # ── Cleanup demo DB ──────────────────────────────────────────────
    if demo_db.exists():
        demo_db.unlink()
        log.info("Removed demo database %s", demo_db)

    print()
    print("=" * 70)
    print("  Demo complete.")
    print("=" * 70)
    print()
