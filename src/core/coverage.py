"""
Coverage matrix tracking for Project EXODUS.

Tracks which astronomical datasets have been successfully queried for
each target. Used to apply a missingness penalty during scoring so
targets cannot rank highly solely due to having more data available.

Public API
----------
CoverageMatrix
    register(target_id, dataset_name)
        Mark a dataset as available for a target.

    has(target_id, dataset_name) -> bool
        Check whether a dataset is registered for a target.

    completeness_fraction(target_id) -> float
        Ratio of available datasets to total known datasets.

    missingness_penalty(target_id) -> float
        Returns completeness_fraction ** 0.5 (sqrt softens the penalty).

    summary() -> dict
        Coverage statistics across all targets.

    to_dict() / from_dict(data)
        Round-trip serialisation.

Constants
---------
KNOWN_DATASETS : list[str]
    The canonical list of dataset names tracked by the coverage matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import get_logger, PROJECT_ROOT

log = get_logger("core.coverage")


# =====================================================================
#  Known datasets
# =====================================================================

KNOWN_DATASETS: List[str] = [
    "Gaia",
    "WISE",
    "NEOWISE",
    "Kepler_TESS",
    "BL_radio",
    "Fermi",
    "IceCube",
    "GWTC",
    "NANOGrav",
    "FRB",
    "ExoplanetArchive",
]


# =====================================================================
#  CoverageMatrix
# =====================================================================

class CoverageMatrix:
    """Tracks dataset availability per target.

    Internally stores a dict mapping target_id -> set of dataset names.
    Only datasets listed in ``KNOWN_DATASETS`` are counted toward
    completeness; unknown dataset names are still stored but ignored
    when computing fractions and penalties.
    """

    def __init__(self) -> None:
        self._coverage: Dict[str, Set[str]] = {}

    # ── Mutation ──────────────────────────────────────────────────

    def register(self, target_id: str, dataset_name: str) -> None:
        """Mark *dataset_name* as available for *target_id*.

        Parameters
        ----------
        target_id : str
            Unique identifier for the astronomical target.
        dataset_name : str
            Name of the dataset (should be one of ``KNOWN_DATASETS``).
        """
        if dataset_name not in KNOWN_DATASETS:
            log.warning(
                "Dataset '%s' is not in KNOWN_DATASETS; registering anyway",
                dataset_name,
            )
        self._coverage.setdefault(target_id, set()).add(dataset_name)
        log.debug("Registered %s -> %s", target_id, dataset_name)

    # ── Queries ───────────────────────────────────────────────────

    def has(self, target_id: str, dataset_name: str) -> bool:
        """Return True if *dataset_name* is registered for *target_id*."""
        return dataset_name in self._coverage.get(target_id, set())

    def completeness_fraction(self, target_id: str) -> float:
        """Fraction of known datasets available for *target_id*.

        Returns
        -------
        float
            Value in [0, 1].  1.0 means every dataset in
            ``KNOWN_DATASETS`` has been registered for this target.
        """
        available = self._coverage.get(target_id, set())
        n_known = sum(1 for d in available if d in KNOWN_DATASETS)
        return n_known / len(KNOWN_DATASETS) if KNOWN_DATASETS else 0.0

    def missingness_penalty(self, target_id: str) -> float:
        """Penalty multiplier for incomplete data coverage.

        Uses the square root of the completeness fraction so that
        missing a few datasets does not obliterate the score, but
        having very sparse coverage still incurs a meaningful penalty.

        Returns
        -------
        float
            Value in [0, 1].  Multiply the EXODUS score by this value.
        """
        return self.completeness_fraction(target_id) ** 0.5

    # ── Aggregate statistics ──────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Coverage statistics across all registered targets.

        Returns
        -------
        dict
            Keys: n_targets, n_known_datasets, mean_completeness,
            min_completeness, max_completeness, fully_covered_targets,
            per_dataset_availability.
        """
        if not self._coverage:
            return {
                "n_targets": 0,
                "n_known_datasets": len(KNOWN_DATASETS),
                "mean_completeness": 0.0,
                "min_completeness": 0.0,
                "max_completeness": 0.0,
                "fully_covered_targets": 0,
                "per_dataset_availability": {},
            }

        fractions = [
            self.completeness_fraction(tid) for tid in self._coverage
        ]
        n_full = sum(1 for f in fractions if f >= 1.0)

        # Per-dataset availability count
        per_ds: Dict[str, int] = {}
        for ds in KNOWN_DATASETS:
            per_ds[ds] = sum(
                1 for tid in self._coverage if ds in self._coverage[tid]
            )

        return {
            "n_targets": len(self._coverage),
            "n_known_datasets": len(KNOWN_DATASETS),
            "mean_completeness": sum(fractions) / len(fractions),
            "min_completeness": min(fractions),
            "max_completeness": max(fractions),
            "fully_covered_targets": n_full,
            "per_dataset_availability": per_ds,
        }

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the coverage matrix to a plain dict.

        Returns
        -------
        dict
            JSON-safe representation with sets converted to sorted lists.
        """
        return {
            "known_datasets": KNOWN_DATASETS,
            "coverage": {
                tid: sorted(ds_set) for tid, ds_set in self._coverage.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoverageMatrix:
        """Reconstruct a CoverageMatrix from a serialised dict.

        Parameters
        ----------
        data : dict
            Output of ``to_dict()``.

        Returns
        -------
        CoverageMatrix
        """
        matrix = cls()
        for tid, ds_list in data.get("coverage", {}).items():
            matrix._coverage[tid] = set(ds_list)
        return matrix

    # ── Repr ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CoverageMatrix(targets={len(self._coverage)}, "
            f"known_datasets={len(KNOWN_DATASETS)})"
        )


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  Project EXODUS -- Coverage Matrix Module Demo")
    print("=" * 70)

    matrix = CoverageMatrix()

    # ------------------------------------------------------------------
    # Register datasets for 5 demo targets
    # ------------------------------------------------------------------
    demo_targets = {
        "KIC_8462852": [
            "Gaia", "WISE", "NEOWISE", "Kepler_TESS", "BL_radio",
            "Fermi", "ExoplanetArchive",
        ],
        "HD_164922": [
            "Gaia", "WISE", "Kepler_TESS", "ExoplanetArchive",
        ],
        "PSR_B1919+21": [
            "Gaia", "BL_radio", "Fermi", "NANOGrav",
        ],
        "TYC_1234": [
            "Gaia", "WISE",
        ],
        "TRAPPIST-1": list(KNOWN_DATASETS),  # full coverage
    }

    print("\n[1] Registering datasets for 5 targets")
    print("-" * 50)
    for tid, datasets in demo_targets.items():
        for ds in datasets:
            matrix.register(tid, ds)
        frac = matrix.completeness_fraction(tid)
        penalty = matrix.missingness_penalty(tid)
        print(f"  {tid:<20s}  datasets={len(datasets):2d}/{len(KNOWN_DATASETS)}  "
              f"completeness={frac:.3f}  penalty={penalty:.3f}")

    # ------------------------------------------------------------------
    # Query checks
    # ------------------------------------------------------------------
    print("\n[2] Has-dataset checks")
    print("-" * 50)
    assert matrix.has("KIC_8462852", "Gaia") is True
    assert matrix.has("KIC_8462852", "IceCube") is False
    assert matrix.has("TRAPPIST-1", "FRB") is True
    assert matrix.has("TYC_1234", "Fermi") is False
    print("  All has() checks passed")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n[3] Coverage summary")
    print("-" * 50)
    summary = matrix.summary()
    for k, v in summary.items():
        if k != "per_dataset_availability":
            print(f"  {k}: {v}")
    print("  Per-dataset availability:")
    for ds, count in summary["per_dataset_availability"].items():
        print(f"    {ds:<20s}: {count}/{summary['n_targets']} targets")

    # ------------------------------------------------------------------
    # Round-trip serialisation
    # ------------------------------------------------------------------
    print("\n[4] Serialisation round-trip")
    print("-" * 50)
    serialised = matrix.to_dict()
    restored = CoverageMatrix.from_dict(serialised)
    assert restored.to_dict() == serialised
    print("  to_dict/from_dict round-trip: PASS")
    print(f"  JSON size: {len(json.dumps(serialised, indent=2))} bytes")

    # ------------------------------------------------------------------
    # Penalty behaviour
    # ------------------------------------------------------------------
    print("\n[5] Missingness penalty behaviour")
    print("-" * 50)
    for tid in demo_targets:
        frac = matrix.completeness_fraction(tid)
        penalty = matrix.missingness_penalty(tid)
        print(f"  {tid:<20s}  completeness={frac:.3f}  "
              f"penalty={penalty:.3f}  (sqrt softening)")
    assert matrix.missingness_penalty("TRAPPIST-1") == 1.0
    assert matrix.missingness_penalty("TYC_1234") < matrix.missingness_penalty("KIC_8462852")
    print("  Penalty ordering: PASS")

    print("\n" + "=" * 70)
    print("  All coverage matrix tests passed.")
    print("=" * 70)
