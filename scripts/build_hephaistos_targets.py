#!/usr/bin/env python3
"""
Project EXODUS — Build Hephaistos Surviving DS Candidates + Benchmarks
=======================================================================

Creates a target file for the 4 surviving Hephaistos Dyson sphere
candidates (C, D, E, F from Suazo et al. 2024) plus 2 benchmark
anomalous stars (Tabby's Star, ASASSN-21qj).

Candidates A, B are contaminated by background radio sources.
Candidate G is CONFIRMED background AGN (Ren et al. 2025 VLBI).
Candidates C, D, E, F have NO confirmed contamination as of 2025.

Running these through the 10-channel EXODUS pipeline adds UV, radio,
HR anomaly, and PM channels that were NOT evaluated in Hephaistos.

Usage
-----
    python scripts/build_hephaistos_targets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "data" / "targets" / "hephaistos_survivors.json"


# All data from Suazo et al. 2024 Table 5 + Ren et al. 2024/2025
TARGETS = [
    # --- 4 surviving Hephaistos candidates ---
    {
        "target_id": "HEPH_C",
        "host_star": "Gaia_4649396037451459712",
        "ra": 74.0127,
        "dec": -74.1706,
        "distance_pc": 219.4,
        "gaia_source_id": 4649396037451459712,
        "hz_flag": False,
        "notes": "Hephaistos Cand C — M-dwarf, T_DS=187K, gamma=0.14. NO confirmed contamination.",
        "source_paper": "arXiv:2405.02927",
        "discovery_reason": "unexplained_wise_w3w4_excess",
        "hephaistos_tds_k": 187,
        "hephaistos_gamma": 0.14,
    },
    {
        "target_id": "HEPH_D",
        "host_star": "Gaia_2660349163149053824",
        "ra": 351.9637,
        "dec": 5.1073,
        "distance_pc": 211.5,
        "gaia_source_id": 2660349163149053824,
        "hz_flag": False,
        "notes": "Hephaistos Cand D — M-dwarf, T_DS=178K, gamma=0.16. NO confirmed contamination.",
        "source_paper": "arXiv:2405.02927",
        "discovery_reason": "unexplained_wise_w3w4_excess",
        "hephaistos_tds_k": 178,
        "hephaistos_gamma": 0.16,
    },
    {
        "target_id": "HEPH_E",
        "host_star": "Gaia_3190232820489766656",
        "ra": 60.5325,
        "dec": -10.9113,
        "distance_pc": 274.7,
        "gaia_source_id": 3190232820489766656,
        "hz_flag": False,
        "notes": "Hephaistos Cand E — M-dwarf, T_DS=180K, gamma=0.08. NO confirmed contamination.",
        "source_paper": "arXiv:2405.02927",
        "discovery_reason": "unexplained_wise_w3w4_excess",
        "hephaistos_tds_k": 180,
        "hephaistos_gamma": 0.08,
    },
    {
        "target_id": "HEPH_F",
        "host_star": "Gaia_2956570141274256512",
        "ra": 78.4438,
        "dec": -25.1865,
        "distance_pc": 265.0,
        "gaia_source_id": 2956570141274256512,
        "hz_flag": False,
        "notes": "Hephaistos Cand F — M-dwarf, T_DS=137K, gamma=0.03. NO confirmed contamination.",
        "source_paper": "arXiv:2405.02927",
        "discovery_reason": "unexplained_wise_w3w4_excess",
        "hephaistos_tds_k": 137,
        "hephaistos_gamma": 0.03,
    },
    # --- Benchmark anomalous stars ---
    {
        "target_id": "Tabbys_Star",
        "host_star": "KIC_8462852",
        "ra": 301.5644,
        "dec": 44.4569,
        "distance_pc": 454.0,
        "gaia_source_id": 2081900940499099136,
        "hz_flag": False,
        "notes": "Boyajian's Star — 3-channel anomaly (optical+UV+transit), no IR excess. Benchmark.",
        "source_paper": "arXiv:1509.03622",
        "discovery_reason": "multi_channel_anomaly",
    },
    {
        "target_id": "ASASSN_21qj",
        "host_star": "2MASS_J08152329-3859234",
        "ra": 123.847,
        "dec": -38.990,
        "distance_pc": 555.0,
        "gaia_source_id": 5539970601632026752,
        "hz_flag": False,
        "notes": "ASASSN-21qj — IR+Optical 2-channel anomaly. 6.5 mag dips + pre-dimming IR brightening.",
        "source_paper": "arXiv:2309.16969",
        "discovery_reason": "ir_optical_anomaly",
    },
]


def main():
    campaign = {
        "campaign": "hephaistos_survivors",
        "description": (
            "4 surviving Hephaistos Dyson sphere candidates (C, D, E, F) from "
            "Suazo et al. 2024 (arXiv:2405.02927), plus Tabby's Star and ASASSN-21qj "
            "as multi-channel benchmarks. Candidates A, B have background radio sources; "
            "G confirmed AGN (Ren et al. 2025). C-F have NO confirmed contamination. "
            "EXODUS 10-channel pipeline adds UV, radio, HR, PM analysis not in Hephaistos."
        ),
        "targets": TARGETS,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(campaign, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  EXODUS — Hephaistos Survivors + Benchmarks")
    print(f"{'='*60}")
    print(f"  Hephaistos survivors (C,D,E,F): 4")
    print(f"  Benchmarks (Tabby's Star, ASASSN-21qj): 2")
    print(f"  Total targets: {len(TARGETS)}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
