#!/usr/bin/env python3
"""EBM permutation test: validate p=0.00055 non-parametrically via 10K reshuffles."""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.statistics import ebm_combine, compute_ebm_covariance

# Load the 355x6 control p-value matrix
with open(os.path.join(os.path.dirname(__file__), "..", "data", "reports", "ebm_covariance.json")) as f:
    data = json.load(f)

all_p = np.array(data["control_p_matrix"])  # (355, 6)
channel_names = data["channel_names"]
# Channels: ir_excess(0), proper_motion_anomaly(1), hr_anomaly(2), uv_anomaly(3), radio_emission(4), ir_variability(5)

# Candidate 1 uses 5 channels (no UV — no GALEX coverage): indices 0,1,2,4,5
use_idx = [0, 1, 2, 4, 5]  # ir_excess, pm, hr, radio, irvar
use_names = [channel_names[i] for i in use_idx]
print(f"Channels: {use_names}")

# Extract sub-matrix
ctrl_p = all_p[:, use_idx]  # (355, 5)
n_ctrl, n_ch = ctrl_p.shape
print(f"Control matrix: {n_ctrl} x {n_ch}")

# Candidate 1 p-values (same order)
cand_p = np.array([0.028, 0.059, 0.997, 0.003, 1.0])
print(f"Candidate p-values: {dict(zip(use_names, cand_p))}")

# Compute covariance from controls
ctrl_stats = -2 * np.log(np.clip(ctrl_p, 1e-15, 1.0))
cov = np.cov(ctrl_stats, rowvar=False)  # (5, 5)
print(f"\nCovariance matrix diagonal: {np.diag(cov)}")
print(f"Max |correlation|: {np.max(np.abs(np.corrcoef(ctrl_stats, rowvar=False) - np.eye(n_ch))):.4f}")

# Observed EBM p-value
observed_p = ebm_combine(cand_p, covariance_matrix=cov)
print(f"\nObserved EBM p = {observed_p:.6f}")

# Permutation test: 10K reshuffles
np.random.seed(42)
N_PERM = 10_000
n_extreme = 0

print(f"\nRunning {N_PERM} permutations...")
for i in range(N_PERM):
    # Independently shuffle each column (breaks inter-channel correlations
    # while preserving marginal distributions)
    shuffled = ctrl_p.copy()
    for col in range(n_ch):
        np.random.shuffle(shuffled[:, col])

    # Recompute covariance from shuffled data
    shuf_stats = -2 * np.log(np.clip(shuffled, 1e-15, 1.0))
    shuf_cov = np.cov(shuf_stats, rowvar=False)

    # Pick a random pseudo-candidate from shuffled rows
    idx = np.random.randint(n_ctrl)
    pseudo_p = shuffled[idx]

    # Compute EBM p for this pseudo-candidate
    perm_p = ebm_combine(pseudo_p, covariance_matrix=shuf_cov)

    if perm_p <= observed_p:
        n_extreme += 1

    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{N_PERM} done, {n_extreme} extreme so far")

empirical_p = n_extreme / N_PERM

# Wilson score 95% CI
from math import sqrt
z = 1.96
n = N_PERM
p_hat = empirical_p
denom = 1 + z**2 / n
center = (p_hat + z**2 / (2*n)) / denom
margin = z * sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denom
ci_low = max(0, center - margin)
ci_high = min(1, center + margin)

print(f"\n{'='*60}")
print(f"RESULT: {n_extreme}/{N_PERM} permutations had p <= {observed_p:.6f}")
print(f"Empirical p-value = {empirical_p:.4f}")
print(f"95% CI (Wilson): [{ci_low:.4f}, {ci_high:.4f}]")
print(f"{'='*60}")

# Save results
result = {
    "test": "EBM permutation test",
    "n_permutations": N_PERM,
    "observed_ebm_p": observed_p,
    "n_extreme": n_extreme,
    "empirical_p": empirical_p,
    "wilson_95ci": [round(ci_low, 6), round(ci_high, 6)],
    "channels_used": use_names,
    "candidate_p_values": dict(zip(use_names, cand_p.tolist())),
    "seed": 42,
    "method": "Independent column shuffle, recompute covariance each permutation, random pseudo-candidate row"
}

outpath = os.path.join(os.path.dirname(__file__), "..", "data", "reports", "ebm_permutation_test.json")
with open(outpath, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {outpath}")
