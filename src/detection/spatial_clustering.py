"""
Project EXODUS — Spatial Clustering Analysis
=============================================

Post-scoring analysis that searches for spatial clusters of anomalous
targets.  The SETI motivation is simple: an expanding technological
civilisation would produce *clusters* of anomalous stars, not random
scatter across the sky.

Two statistical methods are implemented:

1. **Kulldorff scan statistic** — slides a variable-radius window across
   the target set and computes a likelihood ratio for each window, testing
   whether the mean anomaly score inside is significantly higher than
   outside.  Monte Carlo permutation gives significance.

2. **Ripley's K function** — measures the pair-correlation of anomalous
   targets as a function of separation.  If high-score targets cluster
   beyond what spatial randomness predicts, the K function will show excess.

Neither method is a "channel" in the scorer — they operate on *already
scored* target sets and produce supplementary reports.

Usage
-----
    from src.detection.spatial_clustering import (
        load_scored_targets,
        kulldorff_scan,
        ripleys_k,
        SpatialClusterResult,
    )

    targets = load_scored_targets("data/reports/quick_run_*.json")
    clusters = kulldorff_scan(targets, n_permutations=999)
    k_result = ripleys_k(targets, score_threshold=0.3)

CLI
---
    python -m src.detection.spatial_clustering --reports data/reports/*.json
"""

from __future__ import annotations

import json
import glob as _glob
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import get_logger

log = get_logger("detection.spatial_clustering")


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class ScoredTarget:
    """Minimal representation of a scored target for spatial analysis."""

    target_id: str
    ra: float        # degrees, ICRS
    dec: float       # degrees, ICRS
    distance_pc: float  # parsecs (NaN if unknown)
    total_score: float
    n_active_channels: int
    stouffer_p: Optional[float] = None

    # 3D Cartesian position (heliocentric, parsec)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def compute_cartesian(self) -> None:
        """Convert (RA, Dec, distance) to heliocentric Cartesian."""
        if not np.isfinite(self.distance_pc) or self.distance_pc <= 0:
            self.x = self.y = self.z = np.nan
            return
        ra_rad = math.radians(self.ra)
        dec_rad = math.radians(self.dec)
        cos_dec = math.cos(dec_rad)
        self.x = self.distance_pc * cos_dec * math.cos(ra_rad)
        self.y = self.distance_pc * cos_dec * math.sin(ra_rad)
        self.z = self.distance_pc * math.sin(dec_rad)


@dataclass
class ClusterCandidate:
    """A spatial cluster found by the Kulldorff scan."""

    center_target_id: str
    center_ra: float
    center_dec: float
    center_distance_pc: float
    radius_pc: float
    n_members: int
    member_ids: List[str]
    mean_score_inside: float
    mean_score_outside: float
    likelihood_ratio: float
    p_value: float  # Monte Carlo p-value
    is_significant: bool  # p < 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RipleysKResult:
    """Result of Ripley's K analysis."""

    distances_pc: List[float]      # evaluation distances
    k_observed: List[float]        # K(r) for anomalous targets
    k_expected_csr: List[float]    # K(r) under complete spatial randomness
    l_function: List[float]        # L(r) - r  (excess clustering)
    max_l_excess: float            # max(L(r) - r), peak clustering excess
    max_l_distance_pc: float       # distance at max excess
    n_anomalous: int               # number of anomalous targets used
    score_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialClusterResult:
    """Complete spatial clustering analysis result."""

    n_targets: int
    n_with_3d_coords: int
    n_anomalous: int  # targets with n_active >= 2
    clusters: List[ClusterCandidate]
    ripleys_k: Optional[RipleysKResult] = None
    flagged_targets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_targets": self.n_targets,
            "n_with_3d_coords": self.n_with_3d_coords,
            "n_anomalous": self.n_anomalous,
            "n_clusters_found": len(self.clusters),
            "n_significant_clusters": sum(1 for c in self.clusters if c.is_significant),
            "clusters": [c.to_dict() for c in self.clusters],
            "flagged_targets": self.flagged_targets,
        }
        if self.ripleys_k is not None:
            d["ripleys_k"] = self.ripleys_k.to_dict()
        return d


# ── Target loading ───────────────────────────────────────────────────


def load_scored_targets(report_paths: List[str]) -> List[ScoredTarget]:
    """Load scored targets from one or more EXODUS report JSON files.

    Deduplicates by target_id, keeping the highest-scoring entry.

    Parameters
    ----------
    report_paths : list of str
        Paths to EXODUS quick_run_*.json reports (glob patterns accepted).

    Returns
    -------
    list of ScoredTarget
    """
    # Expand glob patterns
    expanded: List[str] = []
    for p in report_paths:
        matches = _glob.glob(p)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(p)

    best: Dict[str, ScoredTarget] = {}

    for path in expanded:
        try:
            with open(path) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Cannot read report %s: %s", path, exc)
            continue

        targets = report.get("all_scored", [])
        if not targets:
            targets = report.get("top_targets", [])

        for t in targets:
            tid = t.get("target_id", "")
            ra = t.get("ra")
            dec = t.get("dec")
            if ra is None or dec is None:
                continue

            score = t.get("total_score", 0.0)
            n_active = t.get("n_active_channels", 0)
            dist = t.get("distance_pc")
            stouffer_p = t.get("stouffer_p")

            if dist is None or not np.isfinite(dist) or dist <= 0:
                dist = float("nan")

            st = ScoredTarget(
                target_id=tid,
                ra=float(ra),
                dec=float(dec),
                distance_pc=float(dist),
                total_score=float(score),
                n_active_channels=int(n_active),
                stouffer_p=stouffer_p,
            )
            st.compute_cartesian()

            # Keep highest score per target
            if tid not in best or st.total_score > best[tid].total_score:
                best[tid] = st

    result = list(best.values())
    log.info("Loaded %d unique scored targets from %d reports", len(result), len(expanded))
    return result


# ── Kulldorff scan statistic ─────────────────────────────────────────


def _pairwise_distances_3d(targets: List[ScoredTarget]) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix in 3D (parsecs)."""
    n = len(targets)
    coords = np.array([[t.x, t.y, t.z] for t in targets])
    # Using broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    sq = np.sum(coords ** 2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2 * coords @ coords.T
    # Clamp to avoid negative from floating point
    dist_sq = np.clip(dist_sq, 0, None)
    return np.sqrt(dist_sq)


def _scan_log_likelihood_ratio(
    scores: np.ndarray,
    inside_mask: np.ndarray,
) -> float:
    """Compute scan statistic for continuous scores.

    Tests whether the mean score inside a window is significantly higher
    than outside.  Uses the normal-model scan statistic (Kulldorff 1997
    adapted for continuous data):

        LR = n_in * (mean_in - mean_all)^2 / (2 * var_all)

    only when mean_in > mean_all (one-sided: looking for hot spots).

    Parameters
    ----------
    scores : 1D array of target scores
    inside_mask : boolean mask for targets inside the window

    Returns
    -------
    float : scan statistic (higher = stronger cluster signal)
    """
    n_in = int(inside_mask.sum())
    n_out = len(scores) - n_in
    if n_in < 2 or n_out < 2:
        return 0.0

    mean_in = scores[inside_mask].mean()
    mean_out = scores[~inside_mask].mean()
    mean_all = scores.mean()
    var_all = scores.var()

    if mean_in <= mean_all or var_all <= 0:
        return 0.0

    # Normal-model scan statistic: excess-mean weighted by count
    lr = n_in * (mean_in - mean_all) ** 2 / (2.0 * var_all)
    return float(lr)


def kulldorff_scan(
    targets: List[ScoredTarget],
    n_permutations: int = 999,
    max_cluster_fraction: float = 0.5,
    min_cluster_size: int = 3,
    significance_level: float = 0.05,
) -> List[ClusterCandidate]:
    """Run Kulldorff spatial scan statistic on scored targets.

    For each target as a potential cluster center, scans expanding radii
    and computes the likelihood ratio.  Monte Carlo permutation of scores
    gives p-values.

    Parameters
    ----------
    targets : list of ScoredTarget
        Must have finite 3D coordinates for inclusion.
    n_permutations : int
        Number of Monte Carlo permutations for significance.
    max_cluster_fraction : float
        Maximum fraction of targets allowed in a cluster (0.5 = 50%).
    min_cluster_size : int
        Minimum number of targets to form a cluster.
    significance_level : float
        P-value threshold for significance.

    Returns
    -------
    list of ClusterCandidate
        All clusters found, sorted by likelihood ratio (descending).
        Only the top non-overlapping clusters are returned.
    """
    # Filter to targets with valid 3D positions
    valid = [t for t in targets if np.isfinite(t.x)]
    if len(valid) < 10:
        log.warning("Only %d targets with 3D coordinates — need ≥10 for scan", len(valid))
        return []

    n = len(valid)
    max_in_cluster = int(n * max_cluster_fraction)
    scores = np.array([t.total_score for t in valid])

    log.info("Running Kulldorff scan on %d targets (%.0f permutations)", n, n_permutations)

    # Precompute distance matrix
    dist_matrix = _pairwise_distances_3d(valid)

    # ── Observed scan ────────────────────────────────────────────
    best_lr = 0.0
    best_center_idx = 0
    best_radius = 0.0
    best_inside_mask = np.zeros(n, dtype=bool)

    # For each potential center
    for i in range(n):
        # Sort other targets by distance from center i
        dists_from_i = dist_matrix[i]
        sorted_idx = np.argsort(dists_from_i)

        # Expand radius: include nearest, nearest 2, ..., up to max
        inside = np.zeros(n, dtype=bool)
        for k_idx in range(min(max_in_cluster, n)):
            j = sorted_idx[k_idx]
            inside[j] = True

            if inside.sum() < min_cluster_size:
                continue

            lr = _scan_log_likelihood_ratio(scores, inside)
            if lr > best_lr:
                best_lr = lr
                best_center_idx = i
                best_radius = dists_from_i[j]
                best_inside_mask = inside.copy()

    if best_lr == 0.0:
        log.info("No cluster signal found (all LR=0)")
        return []

    # ── Monte Carlo permutation significance ─────────────────────
    n_as_extreme = 0
    rng = np.random.default_rng(42)

    for perm in range(n_permutations):
        perm_scores = rng.permutation(scores)

        perm_best_lr = 0.0
        for i in range(n):
            sorted_idx = np.argsort(dist_matrix[i])
            inside = np.zeros(n, dtype=bool)
            for k_idx in range(min(max_in_cluster, n)):
                j = sorted_idx[k_idx]
                inside[j] = True
                if inside.sum() < min_cluster_size:
                    continue
                lr = _scan_log_likelihood_ratio(perm_scores, inside)
                if lr > perm_best_lr:
                    perm_best_lr = lr

        if perm_best_lr >= best_lr:
            n_as_extreme += 1

        # Progress reporting
        if (perm + 1) % 100 == 0:
            log.debug("  Permutation %d/%d", perm + 1, n_permutations)

    p_value = (n_as_extreme + 1) / (n_permutations + 1)

    # Build the cluster candidate
    center = valid[best_center_idx]
    members = [valid[j].target_id for j in range(n) if best_inside_mask[j]]
    mean_in = scores[best_inside_mask].mean()
    mean_out = scores[~best_inside_mask].mean() if (~best_inside_mask).sum() > 0 else 0.0

    cluster = ClusterCandidate(
        center_target_id=center.target_id,
        center_ra=center.ra,
        center_dec=center.dec,
        center_distance_pc=center.distance_pc,
        radius_pc=float(best_radius),
        n_members=len(members),
        member_ids=members,
        mean_score_inside=float(mean_in),
        mean_score_outside=float(mean_out),
        likelihood_ratio=float(best_lr),
        p_value=float(p_value),
        is_significant=p_value < significance_level,
    )

    log.info(
        "Best cluster: center=%s  radius=%.1f pc  n=%d  "
        "mean_in=%.3f  mean_out=%.3f  LR=%.3f  p=%.4f  %s",
        cluster.center_target_id,
        cluster.radius_pc,
        cluster.n_members,
        cluster.mean_score_inside,
        cluster.mean_score_outside,
        cluster.likelihood_ratio,
        cluster.p_value,
        "SIGNIFICANT" if cluster.is_significant else "not significant",
    )

    return [cluster]


# ── Ripley's K function ──────────────────────────────────────────────


def ripleys_k(
    targets: List[ScoredTarget],
    score_threshold: float = 0.3,
    n_distance_bins: int = 20,
    max_distance_pc: Optional[float] = None,
) -> Optional[RipleysKResult]:
    """Compute Ripley's K function for anomalous targets.

    Compares pair-correlation of high-score targets against CSR
    (complete spatial randomness).

    Parameters
    ----------
    targets : list of ScoredTarget
    score_threshold : float
        Minimum total_score to count as "anomalous".
    n_distance_bins : int
        Number of distance bins.
    max_distance_pc : float or None
        Maximum evaluation distance. Defaults to half the sample extent.

    Returns
    -------
    RipleysKResult or None
        None if insufficient anomalous targets.
    """
    # Get all targets with 3D coords
    valid = [t for t in targets if np.isfinite(t.x)]
    if len(valid) < 10:
        log.warning("Ripley's K: insufficient targets with 3D coords (%d)", len(valid))
        return None

    # Select anomalous subset
    anomalous = [t for t in valid if t.total_score >= score_threshold]
    n_anom = len(anomalous)

    if n_anom < 5:
        log.warning("Ripley's K: only %d anomalous targets — need ≥5", n_anom)
        return None

    # Compute extent of the sample (for CSR normalisation)
    coords_all = np.array([[t.x, t.y, t.z] for t in valid])
    extent = coords_all.max(axis=0) - coords_all.min(axis=0)
    volume = float(np.prod(extent))
    if volume <= 0:
        log.warning("Ripley's K: zero-volume sample — all targets co-located?")
        return None

    # Anomalous target coordinates
    coords_anom = np.array([[t.x, t.y, t.z] for t in anomalous])

    # Max distance
    if max_distance_pc is None:
        max_distance_pc = float(min(extent) / 2.0)
    if max_distance_pc <= 0:
        max_distance_pc = 50.0

    distances = np.linspace(0, max_distance_pc, n_distance_bins + 1)[1:]

    # Pairwise distances between anomalous targets
    if n_anom > 1:
        diff = coords_anom[:, None, :] - coords_anom[None, :, :]
        pair_dists = np.sqrt((diff ** 2).sum(axis=2))
    else:
        pair_dists = np.array([[0.0]])

    # Intensity (density of anomalous targets)
    intensity = n_anom / volume

    # Compute K(r) for each distance bin
    k_observed = []
    k_csr = []
    l_function = []

    for r in distances:
        # Count pairs within distance r (excluding self-pairs)
        n_pairs = ((pair_dists > 0) & (pair_dists <= r)).sum()
        # K(r) = (V / n^2) * sum of pairs within r
        k_r = (volume / (n_anom ** 2)) * n_pairs
        k_observed.append(float(k_r))

        # CSR expectation: K(r) = (4/3) * pi * r^3
        k_csr_r = (4.0 / 3.0) * math.pi * r ** 3
        k_csr.append(float(k_csr_r))

        # L function: L(r) = (K(r) * 3 / (4*pi))^(1/3)
        # L(r) - r measures excess clustering
        if k_r > 0:
            l_r = (k_r * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        else:
            l_r = 0.0
        l_function.append(float(l_r - r))

    # Find peak excess
    l_arr = np.array(l_function)
    max_idx = np.argmax(l_arr)
    max_l_excess = float(l_arr[max_idx])
    max_l_distance = float(distances[max_idx])

    result = RipleysKResult(
        distances_pc=[float(d) for d in distances],
        k_observed=k_observed,
        k_expected_csr=k_csr,
        l_function=l_function,
        max_l_excess=max_l_excess,
        max_l_distance_pc=max_l_distance,
        n_anomalous=n_anom,
        score_threshold=score_threshold,
    )

    log.info(
        "Ripley's K: %d anomalous targets in volume=%.0f pc³  "
        "max L-excess=%.2f at r=%.1f pc",
        n_anom, volume, max_l_excess, max_l_distance,
    )

    return result


# ── Full analysis ────────────────────────────────────────────────────


def run_spatial_analysis(
    targets: List[ScoredTarget],
    n_permutations: int = 999,
    score_threshold: float = 0.3,
    min_active_channels: int = 2,
) -> SpatialClusterResult:
    """Run complete spatial clustering analysis on scored targets.

    Parameters
    ----------
    targets : list of ScoredTarget
    n_permutations : int
        Monte Carlo permutations for Kulldorff scan.
    score_threshold : float
        Threshold for Ripley's K "anomalous" subset.
    min_active_channels : int
        Minimum active channels for a target to count as anomalous.

    Returns
    -------
    SpatialClusterResult
    """
    n_total = len(targets)
    n_3d = sum(1 for t in targets if np.isfinite(t.x))
    n_anomalous = sum(1 for t in targets if t.n_active_channels >= min_active_channels)

    log.info(
        "Spatial analysis: %d targets total, %d with 3D coords, %d anomalous (≥%d ch)",
        n_total, n_3d, n_anomalous, min_active_channels,
    )

    # Kulldorff scan
    clusters = kulldorff_scan(
        targets,
        n_permutations=n_permutations,
        min_cluster_size=3,
    )

    # Ripley's K
    k_result = ripleys_k(targets, score_threshold=score_threshold)

    # Flag targets: those inside significant clusters or with multi-channel convergence
    flagged: List[str] = []
    for c in clusters:
        if c.is_significant:
            flagged.extend(c.member_ids)
    flagged = sorted(set(flagged))

    return SpatialClusterResult(
        n_targets=n_total,
        n_with_3d_coords=n_3d,
        n_anomalous=n_anomalous,
        clusters=clusters,
        ripleys_k=k_result,
        flagged_targets=flagged,
    )


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    """Run spatial clustering analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EXODUS spatial clustering analysis"
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Report JSON files (glob patterns accepted)",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Monte Carlo permutations for Kulldorff scan (default: 999)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold for anomalous classification (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/reports/spatial_clustering.json)",
    )
    args = parser.parse_args()

    targets = load_scored_targets(args.reports)
    if not targets:
        log.error("No targets loaded — check report paths")
        return

    result = run_spatial_analysis(
        targets,
        n_permutations=args.permutations,
        score_threshold=args.score_threshold,
    )

    out_path = args.output or str(
        Path(__file__).resolve().parent.parent.parent
        / "data" / "reports" / "spatial_clustering.json"
    )
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    log.info("Results written to %s", out_path)

    # Summary
    print(f"\n=== EXODUS Spatial Clustering Analysis ===")
    print(f"Targets: {result.n_targets} total, {result.n_with_3d_coords} with 3D coords")
    print(f"Anomalous: {result.n_anomalous} (≥2 active channels)")
    print(f"Clusters found: {len(result.clusters)}")
    sig = sum(1 for c in result.clusters if c.is_significant)
    print(f"Significant clusters (p<0.05): {sig}")

    if result.ripleys_k:
        rk = result.ripleys_k
        print(f"\nRipley's K: {rk.n_anomalous} anomalous targets (score≥{rk.score_threshold})")
        print(f"  Max L-excess: {rk.max_l_excess:.3f} at r={rk.max_l_distance_pc:.1f} pc")
        if rk.max_l_excess > 0:
            print("  → Anomalous targets cluster MORE than random")
        else:
            print("  → No excess clustering detected")

    for i, c in enumerate(result.clusters):
        print(f"\nCluster {i+1}:")
        print(f"  Center: {c.center_target_id} (RA={c.center_ra:.4f}, Dec={c.center_dec:.4f})")
        print(f"  Radius: {c.radius_pc:.1f} pc")
        print(f"  Members: {c.n_members}")
        print(f"  Mean score inside: {c.mean_score_inside:.3f}")
        print(f"  Mean score outside: {c.mean_score_outside:.3f}")
        print(f"  Likelihood ratio: {c.likelihood_ratio:.3f}")
        print(f"  p-value: {c.p_value:.4f} {'✓ SIGNIFICANT' if c.is_significant else '✗'}")

    if result.flagged_targets:
        print(f"\nFlagged targets (in significant clusters): {result.flagged_targets}")


if __name__ == "__main__":
    main()
