#!/usr/bin/env python3
"""
Project EXODUS — Build 20-pc Volume-Limited Target List
========================================================

Query Gaia DR3 for ALL stars within 20 parsecs (parallax > 50 mas).
At 20 pc, a 1% partial Dyson swarm produces ~2σ IR excess — invisible
at 50 pc but detectable here. These are the best-characterised stars
in the sky.

Usage
-----
    python scripts/build_20pc_targets.py
    python scripts/build_20pc_targets.py --max-distance 15 --output data/targets/volume_15pc.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger

log = get_logger("build_20pc")


# Declination strips to avoid Gaia TAP 408 timeout on large queries
DEC_STRIPS = [
    (-90, -60), (-60, -30), (-30, -10), (-10, 10),
    (10, 30), (30, 60), (60, 90),
]


def query_gaia_volume(max_distance_pc: float = 20.0, max_gmag: float = 20.0) -> List[Dict[str, Any]]:
    """Query Gaia DR3 for all stars within max_distance_pc.

    Uses chunked declination strips to avoid TAP timeout.
    """
    from astroquery.gaia import Gaia

    min_parallax = 1000.0 / max_distance_pc  # mas

    all_rows = []

    for dec_lo, dec_hi in DEC_STRIPS:
        adql = f"""
        SELECT source_id, ra, dec, parallax, parallax_error,
               pmra, pmdec, phot_g_mean_mag, bp_rp,
               ruwe, astrometric_excess_noise_sig,
               teff_gspphot, logg_gspphot,
               non_single_star
        FROM gaiadr3.gaia_source
        WHERE parallax > {min_parallax}
          AND parallax_over_error > 10
          AND phot_g_mean_mag < {max_gmag}
          AND dec >= {dec_lo} AND dec < {dec_hi}
        ORDER BY parallax DESC
        """

        log.info("Querying Gaia: dec [%d, %d), parallax > %.1f mas ...",
                 dec_lo, dec_hi, min_parallax)

        for attempt in range(3):
            try:
                job = Gaia.launch_job(adql)
                table = job.get_results()
                df = table.to_pandas()
                log.info("  → %d stars in dec strip [%d, %d)", len(df), dec_lo, dec_hi)
                all_rows.append(df)
                break
            except Exception as e:
                log.warning("  Attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    log.error("  Failed all 3 attempts for dec strip [%d, %d)", dec_lo, dec_hi)

        time.sleep(1)  # Rate limiting

    if not all_rows:
        log.error("No data returned from Gaia.")
        return []

    import pandas as pd
    combined = pd.concat(all_rows, ignore_index=True)

    # Deduplicate by source_id
    combined = combined.drop_duplicates(subset=["source_id"])
    log.info("Total unique stars within %.0f pc: %d", max_distance_pc, len(combined))

    # Convert to target dicts
    targets = []
    for _, row in combined.iterrows():
        sid = str(int(row["source_id"]))
        plx = float(row["parallax"])
        dist_pc = 1000.0 / plx if plx > 0 else 999.0

        targets.append({
            "target_id": f"GAIA_{sid}",
            "host_star": f"Gaia DR3 {sid}",
            "ra": round(float(row["ra"]), 6),
            "dec": round(float(row["dec"]), 6),
            "distance_pc": round(dist_pc, 2),
            "phot_g_mean_mag": round(float(row["phot_g_mean_mag"]), 3) if not _isnan(row["phot_g_mean_mag"]) else None,
            "bp_rp": round(float(row["bp_rp"]), 3) if not _isnan(row["bp_rp"]) else None,
            "ruwe": round(float(row["ruwe"]), 3) if not _isnan(row["ruwe"]) else None,
            "teff": round(float(row["teff_gspphot"]), 0) if not _isnan(row["teff_gspphot"]) else None,
            "source_tier": "volume_limited",
            "discovery_reason": f"Within {max_distance_pc} pc volume",
        })

    # Sort by distance
    targets.sort(key=lambda t: t.get("distance_pc", 999))

    return targets


def _isnan(val) -> bool:
    """Check if value is NaN or None."""
    if val is None:
        return True
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return True


def build_target_file(
    targets: List[Dict[str, Any]],
    max_distance_pc: float,
    output_path: Path,
) -> None:
    """Write target JSON file."""
    campaign = {
        "campaign": f"volume_{int(max_distance_pc)}pc",
        "description": (
            f"Volume-limited survey: all Gaia DR3 stars within {max_distance_pc} pc "
            f"(parallax > {1000/max_distance_pc:.0f} mas, parallax_over_error > 10). "
            f"High sensitivity to partial Dyson swarms and subtle multi-channel anomalies."
        ),
        "phase": "volume_survey",
        "n_targets": len(targets),
        "targets": targets,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(campaign, f, indent=2, default=str)

    log.info("Wrote %d targets to %s", len(targets), output_path)


def main():
    parser = argparse.ArgumentParser(description="EXODUS 20-pc Volume Target Builder")
    parser.add_argument("--max-distance", type=float, default=20.0,
                        help="Maximum distance in parsecs (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "targets" / f"volume_{int(args.max_distance)}pc.json"
    )

    targets = query_gaia_volume(max_distance_pc=args.max_distance)

    if targets:
        build_target_file(targets, args.max_distance, output)
        print(f"\n{'='*60}")
        print(f"  EXODUS — {int(args.max_distance)}-pc Volume Target List")
        print(f"{'='*60}")
        print(f"  Stars found: {len(targets)}")
        print(f"  Distance range: {targets[0]['distance_pc']:.1f} — {targets[-1]['distance_pc']:.1f} pc")

        # Stats
        with_ruwe = [t for t in targets if t.get("ruwe") is not None]
        high_ruwe = [t for t in with_ruwe if t["ruwe"] > 1.4]
        print(f"  With RUWE data: {len(with_ruwe)}")
        print(f"  RUWE > 1.4 (potential binaries): {len(high_ruwe)}")
        print(f"  Output: {output}")
        print(f"{'='*60}")
    else:
        log.error("No targets found.")


if __name__ == "__main__":
    main()
