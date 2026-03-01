#!/usr/bin/env python3
"""
Project EXODUS — Build M-Dwarf HZ Planet Target File
======================================================

23 M-dwarf systems with confirmed habitable-zone planets, compiled
from Perplexity research brief (Prompt 12, Table E.1).

These are the nearest M dwarfs with confirmed HZ planets — optimal
for partial Dyson sphere detection because M-dwarf photospheres emit
negligibly at W3/W4, making even 1% covering fraction detectable.

Priority ranked by: proximity x age x number of HZ planets x quietness.

Usage
-----
    python scripts/build_mdwarf_hz_targets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "data" / "targets" / "mdwarf_hz_23.json"


# All data from research brief Table E.1
# Gaia DR3 source IDs verified from research
TARGETS = [
    # --- Priority 1: Nearest + oldest + quietest ---
    {
        "target_id": "Proxima_Centauri",
        "host_star": "Proxima_Centauri",
        "ra": 217.429,
        "dec": -62.680,
        "distance_pc": 1.30,
        "gaia_source_id": 5853498713190525696,
        "hz_flag": True,
        "spectral_type": "M5.5V",
        "notes": "b (HZ). Nearest star with HZ planet. Active flarer.",
    },
    {
        "target_id": "Barnards_Star",
        "host_star": "GJ_699",
        "ra": 269.452,
        "dec": 4.693,
        "distance_pc": 1.83,
        "gaia_source_id": 4472832130942575872,
        "hz_flag": False,
        "spectral_type": "M4V",
        "notes": "b,c,d confirmed 2024. None confirmed HZ but high interest due to proximity.",
    },
    {
        "target_id": "Ross_128",
        "host_star": "GJ_447",
        "ra": 176.936,
        "dec": 0.804,
        "distance_pc": 3.37,
        "gaia_source_id": 3795565035484557696,
        "hz_flag": True,
        "spectral_type": "M4V",
        "notes": "b (HZ). ~9.5 Gyr, very quiet. Priority 1 for DS search.",
    },
    {
        "target_id": "GJ_1061",
        "host_star": "GJ_1061",
        "ra": 53.230,
        "dec": -44.640,
        "distance_pc": 3.67,
        "gaia_source_id": 5136546404348129408,
        "hz_flag": True,
        "spectral_type": "M5.5V",
        "notes": "c, d both in HZ. Very quiet.",
    },
    {
        "target_id": "Teegarden",
        "host_star": "Teegarden's_Star",
        "ra": 43.294,
        "dec": 16.883,
        "distance_pc": 3.83,
        "gaia_source_id": 2879095865321671040,
        "hz_flag": True,
        "spectral_type": "M7V",
        "notes": "b, c HZ. d confirmed 2024. ~8 Gyr. High PM — care with cross-matching.",
    },
    {
        "target_id": "Luytens_Star",
        "host_star": "GJ_273",
        "ra": 109.862,
        "dec": 5.225,
        "distance_pc": 3.79,
        "gaia_source_id": 3095217440326309504,
        "hz_flag": True,
        "spectral_type": "M3.5V",
        "notes": "b (HZ). Moderately quiet.",
    },
    {
        "target_id": "Wolf_1061",
        "host_star": "GJ_628",
        "ra": 247.585,
        "dec": -12.660,
        "distance_pc": 4.29,
        "gaia_source_id": 4096047466801670272,
        "hz_flag": True,
        "spectral_type": "M3V",
        "notes": "c (HZ). 3-planet system.",
    },
    {
        "target_id": "GJ_1002",
        "host_star": "GJ_1002",
        "ra": 0.585,
        "dec": -7.541,
        "distance_pc": 4.84,
        "gaia_source_id": 2393777236402831872,
        "hz_flag": True,
        "spectral_type": "M5.5V",
        "notes": "b, c both in HZ. Very quiet. Priority 2 for DS search.",
    },
    {
        "target_id": "Gliese_251",
        "host_star": "GJ_251",
        "ra": 101.180,
        "dec": 33.260,
        "distance_pc": 5.58,
        "gaia_source_id": 3428018978965826048,
        "hz_flag": True,
        "spectral_type": "M3V",
        "notes": "c (HZ confirmed 2025).",
    },
    {
        "target_id": "Gliese_625",
        "host_star": "GJ_625",
        "ra": 244.600,
        "dec": 54.300,
        "distance_pc": 6.50,
        "gaia_source_id": 1432497219543050624,
        "hz_flag": True,
        "spectral_type": "M2V",
        "notes": "b (near HZ).",
    },
    # --- Priority 2: Slightly farther, still excellent ---
    {
        "target_id": "GJ_667C",
        "host_star": "GJ_667C",
        "ra": 259.730,
        "dec": -34.990,
        "distance_pc": 7.24,
        "hz_flag": True,
        "spectral_type": "M1.5V",
        "notes": "Multiple HZ planet candidates (c, e, f). Triple system.",
    },
    {
        "target_id": "Wolf_1069",
        "host_star": "Wolf_1069",
        "ra": 265.800,
        "dec": 11.060,
        "distance_pc": 9.56,
        "hz_flag": True,
        "spectral_type": "M5V",
        "notes": "b in HZ. Earth-mass.",
    },
    {
        "target_id": "Gliese_357",
        "host_star": "GJ_357",
        "ra": 140.830,
        "dec": -21.260,
        "distance_pc": 9.44,
        "hz_flag": True,
        "spectral_type": "M2.5V",
        "notes": "d (HZ candidate). Transiting system.",
    },
    {
        "target_id": "L_98_59",
        "host_star": "L_98-59",
        "ra": 124.530,
        "dec": -68.310,
        "distance_pc": 10.62,
        "hz_flag": True,
        "spectral_type": "M3V",
        "notes": "Multiple transiting planets. d near HZ.",
    },
    {
        "target_id": "Ross_508",
        "host_star": "Ross_508",
        "ra": 202.330,
        "dec": 26.990,
        "distance_pc": 11.25,
        "hz_flag": True,
        "spectral_type": "M4.5V",
        "notes": "b (inner edge of HZ). IRD/Subaru discovery.",
    },
    {
        "target_id": "Gliese_180",
        "host_star": "GJ_180",
        "ra": 72.480,
        "dec": -23.660,
        "distance_pc": 12.0,
        "hz_flag": True,
        "spectral_type": "M2V",
        "notes": "c (HZ). Quiet star.",
    },
    {
        "target_id": "TRAPPIST_1",
        "host_star": "TRAPPIST-1",
        "ra": 346.622,
        "dec": -5.041,
        "distance_pc": 12.43,
        "gaia_source_id": 2635476908753563008,
        "hz_flag": True,
        "spectral_type": "M8V",
        "notes": "e, f, g in HZ. 3 rocky HZ planets. ~7.6 Gyr. JWST priority.",
    },
    {
        "target_id": "LHS_1140",
        "host_star": "LHS_1140",
        "ra": 10.260,
        "dec": -15.267,
        "distance_pc": 14.99,
        "gaia_source_id": 2337599798969098752,
        "hz_flag": True,
        "spectral_type": "M4.5V",
        "notes": "b (HZ, transiting, ocean world candidate). ~5-8 Gyr.",
    },
    {
        "target_id": "GJ_725B",
        "host_star": "Struve_2398_B",
        "ra": 280.7,
        "dec": 59.63,
        "distance_pc": 3.52,
        "hz_flag": False,
        "spectral_type": "M3.5V",
        "notes": "Nearby M dwarf. Candidate planets under investigation.",
    },
    {
        "target_id": "Gliese_3293",
        "host_star": "GJ_3293",
        "ra": 65.560,
        "dec": -25.470,
        "distance_pc": 17.9,
        "hz_flag": True,
        "spectral_type": "M2.5V",
        "notes": "d (HZ candidate).",
    },
    {
        "target_id": "TOI_700",
        "host_star": "TOI-700",
        "ra": 97.170,
        "dec": -65.580,
        "distance_pc": 31.1,
        "hz_flag": True,
        "spectral_type": "M2V",
        "notes": "d and e (HZ). TESS discovery. d = first Earth-size HZ TESS planet.",
    },
    {
        "target_id": "LP_890_9",
        "host_star": "SPECULOOS-2",
        "ra": 109.890,
        "dec": -17.340,
        "distance_pc": 32.0,
        "hz_flag": True,
        "spectral_type": "M6V",
        "notes": "c in HZ. One of nearest HZ transiting planets around ultra-cool dwarf.",
    },
    {
        "target_id": "GJ_3470",
        "host_star": "GJ_3470",
        "ra": 119.480,
        "dec": 15.390,
        "distance_pc": 29.3,
        "hz_flag": False,
        "spectral_type": "M1.5V",
        "notes": "b (hot Neptune, not HZ but well-characterized). JWST atmospheric data.",
    },
]


def main():
    campaign = {
        "campaign": "mdwarf_hz_23",
        "description": (
            "23 M-dwarf systems with confirmed or candidate habitable-zone planets, "
            "prioritized by proximity, age, number of HZ planets, and stellar activity. "
            "M-dwarf photospheres are negligible at W3/W4, making even 1% Dyson sphere "
            "coverage fraction detectable. From Perplexity research Prompt 12."
        ),
        "targets": TARGETS,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(campaign, f, indent=2)

    hz_count = sum(1 for t in TARGETS if t.get("hz_flag"))
    with_gaia = sum(1 for t in TARGETS if t.get("gaia_source_id"))

    print(f"\n{'='*60}")
    print(f"  EXODUS — M-Dwarf HZ Planet Targets")
    print(f"{'='*60}")
    print(f"  Total targets: {len(TARGETS)}")
    print(f"  With HZ planets: {hz_count}")
    print(f"  With Gaia DR3 source ID: {with_gaia}")
    dists = [t["distance_pc"] for t in TARGETS]
    print(f"  Distance range: {min(dists):.1f} — {max(dists):.1f} pc")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
