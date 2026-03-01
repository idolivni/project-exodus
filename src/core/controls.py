"""
Matched control sample selection for Project EXODUS.

Selects control stars that are statistically similar to the target sample
on confounding variables (magnitude, colour, distance, galactic latitude)
but are NOT in the target list.  The control distribution is then used to
calibrate detector scores into p-values, ensuring that a "high score"
genuinely means the target is unusual rather than just being a certain
stellar type.

Public API
----------
select_matched_controls(targets, catalog, n_per_target=10, match_on=...)
    Return a control cohort matched to the target distribution.

validate_matching(targets, controls, match_on)
    KS-test validation that the matching succeeded.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("core.controls")

# Canonical matching features (keys expected in target/catalog dicts)
DEFAULT_MATCH_FEATURES = ["phot_g_mean_mag", "bp_rp", "distance_pc", "b_gal"]


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class ControlCohort:
    """Result of matched control selection.

    Attributes
    ----------
    controls : list of dict
        Selected control sources with all original catalog fields.
    n_per_target : int
        Number of controls requested per target.
    match_features : list of str
        Features used for matching.
    target_ids : set of str
        Target IDs that were excluded from the control pool.
    ks_results : dict
        KS test results per feature: {feature: (statistic, p_value)}.
    matching_caveats : list of str
        Any caveats about the matching quality (e.g., distance-agnostic
        matching for very nearby targets).
    """
    controls: List[Dict[str, Any]] = field(default_factory=list)
    n_per_target: int = 10
    match_features: List[str] = field(default_factory=list)
    target_ids: Set[str] = field(default_factory=set)
    ks_results: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    matching_caveats: List[str] = field(default_factory=list)

    @property
    def n_controls(self) -> int:
        return len(self.controls)

    def is_well_matched(self, alpha: float = 0.05) -> bool:
        """Check if all matching features pass KS test at level alpha.

        A passing KS test (p > alpha) means we CANNOT reject the null
        hypothesis that control and target distributions are the same —
        which is what we want.
        """
        if not self.ks_results:
            return False
        return all(p > alpha for _, p in self.ks_results.values())

    def summary(self) -> str:
        lines = [
            f"ControlCohort: {self.n_controls} controls for "
            f"{len(self.target_ids)} targets",
            f"  Match features: {self.match_features}",
            f"  KS test results:",
        ]
        for feat, (stat, pval) in self.ks_results.items():
            status = "PASS" if pval > 0.05 else "FAIL"
            lines.append(f"    {feat:<24s}  D={stat:.4f}  p={pval:.4f}  [{status}]")
        lines.append(
            f"  Overall: {'WELL-MATCHED' if self.is_well_matched() else 'POORLY MATCHED'}"
        )
        if self.matching_caveats:
            lines.append("  Caveats:")
            for c in self.matching_caveats:
                lines.append(f"    - {c}")
        return "\n".join(lines)


# =====================================================================
#  Core matching logic
# =====================================================================

def _extract_features(
    sources: List[Dict[str, Any]],
    features: List[str],
) -> Tuple[np.ndarray, List[int]]:
    """Extract feature matrix from sources, returning valid indices.

    Returns
    -------
    feature_matrix : np.ndarray of shape (n_valid, n_features)
    valid_indices : list of int
        Indices into the original list for sources with all features present.
    """
    valid_idx = []
    rows = []

    for i, src in enumerate(sources):
        vals = []
        ok = True
        for feat in features:
            v = src.get(feat)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                ok = False
                break
            vals.append(float(v))
        if ok:
            valid_idx.append(i)
            rows.append(vals)

    if not rows:
        return np.empty((0, len(features))), []

    return np.array(rows, dtype=np.float64), valid_idx


def select_matched_controls(
    targets: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
    n_per_target: int = 10,
    match_on: Optional[List[str]] = None,
    target_id_key: str = "target_id",
) -> ControlCohort:
    """Select matched control sources from a catalog.

    For each target, finds the ``n_per_target`` nearest neighbours in
    the normalised feature space defined by ``match_on``, excluding
    any source that is itself a target.

    Parameters
    ----------
    targets : list of dict
        Target sources.  Each must have the ``target_id_key`` and all
        features in ``match_on``.
    catalog : list of dict
        Full catalog to draw controls from.  Must be a superset of the
        targets (or a separate field catalog).
    n_per_target : int
        Number of controls per target.  Default 10.
    match_on : list of str, optional
        Feature names to match on.  Default: magnitude, colour, distance,
        galactic latitude.
    target_id_key : str
        Key used to identify targets.  Default ``"target_id"``.

    Returns
    -------
    ControlCohort
        The selected control sample with KS validation.
    """
    if match_on is None:
        match_on = DEFAULT_MATCH_FEATURES.copy()

    # Adaptive matching: for very nearby targets (< 10 pc), distance matching
    # is unreliable because field stars at similar distances are too sparse.
    # Drop distance_pc from matching features and add a caveat.
    caveats: List[str] = []
    if "distance_pc" in match_on:
        distances = [t.get("distance_pc") for t in targets
                     if t.get("distance_pc") is not None]
        if distances:
            median_dist = float(np.median(distances))
            min_dist = float(np.min(distances))
            if median_dist < 10.0:
                match_on = [f for f in match_on if f != "distance_pc"]
                caveats.append(
                    f"Distance-agnostic matching: median target distance "
                    f"{median_dist:.1f} pc (min {min_dist:.1f} pc) is too "
                    f"nearby for reliable distance matching. Controls matched "
                    f"on photometry + galactic latitude only. Calibrated "
                    f"p-values should be interpreted with caution."
                )
                log.warning(
                    "Nearby targets (median %.1f pc): dropping distance_pc "
                    "from matching features", median_dist,
                )

    log.info(
        "Selecting matched controls: %d targets, catalog size %d, "
        "%d controls/target, features=%s",
        len(targets), len(catalog), n_per_target, match_on,
    )

    # Identify target IDs to exclude from control pool.
    # Filter out None/empty IDs — those will use coordinate fallback.
    target_ids = set()
    target_coords = []  # (ra, dec) for coordinate-based fallback
    for i, t in enumerate(targets):
        tid = t.get(target_id_key)
        if tid is not None and str(tid).strip():
            target_ids.add(str(tid))
        # Always record coordinates for fallback exclusion
        _ra = t.get("ra")
        _dec = t.get("dec")
        if _ra is not None and _dec is not None:
            target_coords.append((_ra, _dec))

    if not target_ids and not target_coords:
        log.warning("No target identifiers or coordinates for exclusion")

    log.info(
        "  Exclusion: %d targets with %s key, %d with coord fallback",
        len(target_ids), target_id_key, len(target_coords),
    )

    # Extract features from targets
    target_matrix, target_valid = _extract_features(targets, match_on)

    if len(target_matrix) == 0:
        log.warning("No targets with complete matching features")
        return ControlCohort(
            match_features=match_on,
            n_per_target=n_per_target,
            target_ids=target_ids,
        )

    # Build control pool (catalog minus targets).
    # Primary: exclude by target_id_key match.
    # Fallback: exclude sources within 5 arcsec of any target position,
    # which catches cases where the target's key is missing.
    _EXCL_RADIUS_DEG = 5.0 / 3600.0  # 5 arcsec in degrees

    def _is_target(src: Dict[str, Any]) -> bool:
        """Return True if src should be excluded (is a target)."""
        sid = src.get(target_id_key)
        if sid is not None and str(sid) in target_ids:
            return True
        # Coordinate fallback: exclude sources very close to any target
        src_ra = src.get("ra")
        src_dec = src.get("dec")
        if src_ra is not None and src_dec is not None:
            for tra, tdec in target_coords:
                cos_dec = np.cos(np.radians(0.5 * (src_dec + tdec)))
                dra = abs(src_ra - tra) * max(cos_dec, 0.1)
                ddec = abs(src_dec - tdec)
                if (dra < _EXCL_RADIUS_DEG
                        and ddec < _EXCL_RADIUS_DEG):
                    return True
        return False

    pool = [src for src in catalog if not _is_target(src)]

    pool_matrix, pool_valid = _extract_features(pool, match_on)

    if len(pool_matrix) == 0:
        log.warning("No catalog sources with complete matching features")
        return ControlCohort(
            match_features=match_on,
            n_per_target=n_per_target,
            target_ids=target_ids,
        )

    # Normalise features (z-score) for distance computation
    combined = np.vstack([target_matrix, pool_matrix])
    means = combined.mean(axis=0)
    stds = combined.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero

    target_norm = (target_matrix - means) / stds
    pool_norm = (pool_matrix - means) / stds

    # For each target, find k nearest neighbours in pool
    # Using brute-force Euclidean distance (fine for < 100k sources)
    selected_pool_indices: Set[int] = set()

    for t_idx in range(len(target_norm)):
        t_vec = target_norm[t_idx]
        dists = np.sqrt(np.sum((pool_norm - t_vec) ** 2, axis=1))
        nearest = np.argsort(dists)[:n_per_target]
        for n_idx in nearest:
            selected_pool_indices.add(pool_valid[n_idx])

    # Gather selected controls
    controls = [pool[i] for i in sorted(selected_pool_indices)]

    log.info("Selected %d unique controls", len(controls))

    # Validate matching with KS tests
    control_matrix, _ = _extract_features(controls, match_on)
    ks_results = {}

    if len(control_matrix) > 0 and len(target_matrix) > 0:
        for j, feat in enumerate(match_on):
            stat, pval = sp_stats.ks_2samp(
                target_matrix[:, j], control_matrix[:, j]
            )
            ks_results[feat] = (float(stat), float(pval))
            log.debug(
                "  KS test for '%s': D=%.4f, p=%.4f",
                feat, stat, pval,
            )

    cohort = ControlCohort(
        controls=controls,
        n_per_target=n_per_target,
        match_features=match_on,
        target_ids=target_ids,
        ks_results=ks_results,
        matching_caveats=caveats,
    )

    log.info("Control selection complete:\n%s", cohort.summary())
    return cohort


def validate_matching(
    targets: List[Dict[str, Any]],
    controls: List[Dict[str, Any]],
    match_on: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float, bool]]:
    """Independently validate that controls match targets on all features.

    Parameters
    ----------
    targets, controls : list of dict
        Source dictionaries with feature values.
    match_on : list of str, optional
        Features to check.
    alpha : float
        Significance level for KS test.

    Returns
    -------
    dict
        {feature_name: (ks_statistic, p_value, passes)} where ``passes``
        is True if the distributions are statistically indistinguishable.
    """
    if match_on is None:
        match_on = DEFAULT_MATCH_FEATURES.copy()

    target_matrix, _ = _extract_features(targets, match_on)
    control_matrix, _ = _extract_features(controls, match_on)

    results = {}
    for j, feat in enumerate(match_on):
        if len(target_matrix) > 0 and len(control_matrix) > 0:
            stat, pval = sp_stats.ks_2samp(
                target_matrix[:, j], control_matrix[:, j]
            )
            results[feat] = (float(stat), float(pval), pval > alpha)
        else:
            results[feat] = (1.0, 0.0, False)

    return results


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Matched Control Selection Demo")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # Simulate a field catalog of 2000 stars
    n_catalog = 2000
    catalog = []
    for i in range(n_catalog):
        mag = rng.normal(12.0, 3.0)
        bp_rp = rng.normal(1.0, 0.5)
        dist = rng.lognormal(5.5, 0.8)  # ~250 pc median
        b_gal = rng.uniform(-90, 90)
        catalog.append({
            "target_id": f"FIELD_{i:05d}",
            "phot_g_mean_mag": float(mag),
            "bp_rp": float(bp_rp),
            "distance_pc": float(dist),
            "b_gal": float(b_gal),
        })

    # Select 30 "targets" — brighter than average, nearby
    targets = []
    for i in range(30):
        mag = rng.normal(10.0, 1.5)  # brighter
        bp_rp = rng.normal(1.2, 0.4)
        dist = rng.lognormal(4.5, 0.5)  # closer
        b_gal = rng.uniform(-60, 60)
        targets.append({
            "target_id": f"TARGET_{i:03d}",
            "phot_g_mean_mag": float(mag),
            "bp_rp": float(bp_rp),
            "distance_pc": float(dist),
            "b_gal": float(b_gal),
        })

    # ------------------------------------------------------------------
    # Select matched controls
    # ------------------------------------------------------------------
    print("\n[1] Selecting matched controls")
    print("-" * 50)

    cohort = select_matched_controls(
        targets, catalog, n_per_target=10,
        match_on=["phot_g_mean_mag", "bp_rp", "distance_pc", "b_gal"],
    )

    print(cohort.summary())

    # ------------------------------------------------------------------
    # Validate matching
    # ------------------------------------------------------------------
    print("\n[2] Independent validation")
    print("-" * 50)

    validation = validate_matching(
        targets, cohort.controls,
        match_on=["phot_g_mean_mag", "bp_rp", "distance_pc", "b_gal"],
    )

    all_pass = True
    for feat, (stat, pval, passes) in validation.items():
        status = "PASS" if passes else "FAIL"
        if not passes:
            all_pass = False
        print(f"  {feat:<24s}  D={stat:.4f}  p={pval:.4f}  [{status}]")

    print(f"\n  Overall: {'PASS' if all_pass else 'NEEDS LARGER CATALOG'}")

    # ------------------------------------------------------------------
    # Verify controls exclude targets
    # ------------------------------------------------------------------
    print("\n[3] Verifying no target contamination in controls")
    print("-" * 50)
    control_ids = {c["target_id"] for c in cohort.controls}
    target_id_set = {t["target_id"] for t in targets}
    overlap = control_ids & target_id_set
    if overlap:
        print(f"  FAIL: {len(overlap)} targets found in controls!")
    else:
        print(f"  PASS: 0 targets in {len(cohort.controls)} controls")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)
