#!/usr/bin/env python3
"""
Project EXODUS — Build Chemically Anomalous Star Targets
==========================================================

Targets with genuinely unexplained stellar abundance patterns,
compiled from Perplexity research brief Prompt 7.

These stars have chemical signatures that resist ALL standard
nucleosynthetic explanations. If they ALSO show multi-channel
anomalies (IR excess, PM anomaly, UV deficit, radio), the
combination is extraordinarily difficult to explain naturally.

Usage
-----
    python scripts/build_chemical_anomaly_targets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "data" / "targets" / "chemical_anomaly.json"


TARGETS = [
    # --- Tier 1: Most anomalous, genuinely unexplained ---
    {
        "target_id": "Przybylskis_Star",
        "host_star": "HD_101065",
        "ra": 174.405,
        "dec": -46.709,
        "distance_pc": 109.0,
        "gaia_source_id": 5256513877287839744,
        "hz_flag": False,
        "spectral_type": "F3Ho",
        "notes": (
            "Most anomalous stellar spectrum known. Lanthanides 1000-10000x solar. "
            "Short-lived actinides (Tc, Pm?) claimed. Fe depleted 10x. Rotation "
            "period ~188 years. [Fe/H]=-2.40 (iron only). roAp pulsator."
        ),
        "source_paper": "Cowley et al. 2000 MNRAS 317:299",
        "discovery_reason": "extreme_abundance_anomaly",
        "anomaly_type": "actinide_overabundance",
    },
    {
        "target_id": "LAMOST_J0206_4941",
        "host_star": "LAMOST_J020623.21+494127.9",
        "ra": 31.597,
        "dec": 49.691,
        "distance_pc": None,  # Not published in source
        "hz_flag": False,
        "notes": (
            "r-II star in thin disk. [Eu/Fe]=+1.32, [Eu/H]=+0.78 (highest ever). "
            "[Fe/H]=-0.54. No binary evidence. No known stellar stream. "
            "r-process pattern doesn't match solar template — Sr, Y, Ce, Pr, Nd underabundant."
        ),
        "source_paper": "arXiv:2505.06494 (Xie et al. 2024/2025)",
        "discovery_reason": "extreme_r_process_anomaly",
        "anomaly_type": "r_process_thin_disk",
    },
    {
        "target_id": "2M1353_P_rich",
        "host_star": "2M13535604+4437076",
        "ra": 208.484,
        "dec": 44.619,
        "hz_flag": False,
        "notes": (
            "Brightest phosphorus-rich star. [P/Fe]=+2.2. Enhanced O, Mg, Si, Al, Ce. "
            "Anomalous [Ba/La]=+0.7 (predicted by neither s-process nor r-process). "
            "No known nucleosynthesis model explains this combination."
        ),
        "source_paper": "Masseron et al. 2020 Nature Comms",
        "discovery_reason": "unexplained_phosphorus_enrichment",
        "anomaly_type": "phosphorus_rich",
    },
    {
        "target_id": "HD_25354",
        "host_star": "HD_25354",
        "ra": 60.280,
        "dec": 26.380,
        "hz_flag": False,
        "notes": (
            "Possible second actinide-rich star. Hints of Am (Z=95) and Cm (Z=96). "
            "Not independently confirmed. On 'shakier footing' than Przybylski's."
        ),
        "source_paper": "Dzuba et al. 2017",
        "discovery_reason": "possible_actinide_anomaly",
        "anomaly_type": "possible_actinides",
    },
    # --- Tier 2: Known r-II calibrators with unexplained features ---
    {
        "target_id": "CS_22892_052",
        "host_star": "CS_22892-052",
        "ra": 333.267,
        "dec": -16.730,
        "gaia_source_id": 3646375476891952512,
        "hz_flag": False,
        "notes": (
            "Bright r-II star. [Fe/H]=-3.1, [Eu/Fe]=+1.6. Third-peak r-process "
            "(Os, Ir, Pt) deviates 0.5 dex from scaled solar r-process."
        ),
        "source_paper": "Sneden et al. 1996",
        "discovery_reason": "r_process_calibrator",
        "anomaly_type": "r_process_anomaly",
    },
    # --- Tier 3: Population-level chemical anomaly targets ---
    {
        "target_id": "ASASSN_21qj_chem",
        "host_star": "2MASS_J08152329-3859234",
        "ra": 123.847,
        "dec": -38.990,
        "distance_pc": 555.0,
        "gaia_source_id": 5539970601632026752,
        "hz_flag": False,
        "notes": (
            "ASASSN-21qj — extreme IR + optical anomaly. G2V host. "
            "Spectroscopic abundance analysis would test for accretion signatures."
        ),
        "source_paper": "Marshall et al. 2023 (arXiv:2309.16969)",
        "discovery_reason": "ir_optical_anomaly_spectroscopic_followup",
        "anomaly_type": "multi_channel_anomaly",
    },
]


def main():
    campaign = {
        "campaign": "chemical_anomaly",
        "description": (
            "Stars with genuinely unexplained stellar abundance patterns. Includes "
            "Przybylski's Star (extreme lanthanide/actinide overabundances), the highest "
            "[Eu/H] thin-disk star ever found, phosphorus-rich stars with no nucleosynthetic "
            "explanation, and r-II calibrators with anomalous third-peak elements. "
            "Running through EXODUS 10-channel pipeline tests for multi-channel convergence."
        ),
        "targets": TARGETS,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(campaign, f, indent=2)

    with_gaia = sum(1 for t in TARGETS if t.get("gaia_source_id"))
    with_dist = [t for t in TARGETS if t.get("distance_pc")]

    print(f"\n{'='*60}")
    print(f"  EXODUS — Chemically Anomalous Star Targets")
    print(f"{'='*60}")
    print(f"  Total targets: {len(TARGETS)}")
    print(f"  With Gaia DR3 source ID: {with_gaia}")
    if with_dist:
        dists = [t["distance_pc"] for t in with_dist]
        print(f"  Distance range: {min(dists):.1f} — {max(dists):.1f} pc")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
