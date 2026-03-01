#!/usr/bin/env python3
"""
Project EXODUS — Build Lower-RUWE SmartTargets
================================================

The original SmartTargets selected RUWE > 2.0 → all extreme binaries.
This script selects the "sweet spot" RUWE 1.4-3.0 where stars are
mildly anomalous but NOT screaming binaries.

A partial Dyson swarm doesn't produce RUWE=40 — it produces RUWE=1.5-2.0.
These moderately-weird stars are MORE interesting than extremely-weird ones.

Usage
-----
    python scripts/build_lower_ruwe_targets.py
    python scripts/build_lower_ruwe_targets.py --min-ruwe 1.4 --max-ruwe 3.0
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger

log = get_logger("build_lower_ruwe")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EXODUS Lower-RUWE SmartTargets Builder")
    parser.add_argument("--min-ruwe", type=float, default=1.4)
    parser.add_argument("--max-ruwe", type=float, default=3.0)
    parser.add_argument("--max-targets", type=int, default=500)
    parser.add_argument("--max-distance", type=float, default=50.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from src.targeting.smart_targeter import SmartTargeter

    output = args.output or str(
        PROJECT_ROOT / "data" / "targets"
        / f"smart_targets_ruwe_{args.min_ruwe}_{args.max_ruwe}.json"
    )

    st = SmartTargeter(
        max_targets=args.max_targets,
        max_distance_pc=args.max_distance,
        min_ruwe=args.min_ruwe,
        max_ruwe=args.max_ruwe,
    )

    targets = st.generate()
    st.save(output)

    print(f"\n{'='*60}")
    print(f"  EXODUS — Lower-RUWE SmartTargets")
    print(f"{'='*60}")
    print(f"  RUWE range: {args.min_ruwe} — {args.max_ruwe}")
    print(f"  Targets generated: {len(targets)}")
    if targets:
        ruwe_vals = [t.get("ruwe", 0) for t in targets if t.get("ruwe")]
        if ruwe_vals:
            print(f"  RUWE range in output: {min(ruwe_vals):.2f} — {max(ruwe_vals):.2f}")
    print(f"  Output: {output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
