#!/usr/bin/env python3
"""
Project EXODUS — Build White Dwarf IR Survey Target List
=========================================================

Query Gaia DR3 for white dwarfs within 50 pc using the HR diagram
locus method (absolute G mag > 10, bp_rp < 1.5).

Why WDs are critical for SETI: the binary_system template that explains
EVERY current EXODUS detection CANNOT apply to white dwarfs. A WD with
mid-IR excess can't be "unresolved companion shifting photocentre" because
the WD IS the endpoint. WD + unexplained W3 excess = either disrupted
planetary debris (publishable astrophysics) or something weirder.

Any WD hit survives the binary template by construction.

Usage
-----
    python scripts/build_wd_targets.py
    python scripts/build_wd_targets.py --max-distance 100 --output data/targets/wd_100pc.json
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

log = get_logger("build_wd")


DEC_STRIPS = [
    (-90, -45), (-45, -15), (-15, 15), (15, 45), (45, 90),
]


def query_gaia_white_dwarfs(max_distance_pc: float = 50.0) -> List[Dict[str, Any]]:
    """Query Gaia DR3 for white dwarfs using HR diagram locus.

    WD selection criteria:
    - parallax > 1000/max_distance_pc (within distance limit)
    - parallax_over_error > 10 (reliable distance)
    - Absolute G magnitude > 10 (faint for their temperature)
    - bp_rp < 1.5 (blue/moderate color, excludes cool MS dwarfs)

    Absolute G = phot_g_mean_mag + 5*log10(parallax/1000) + 5
              = phot_g_mean_mag + 5*log10(parallax) - 10
    For M_G > 10: phot_g_mean_mag > 10 - 5*log10(parallax) + 10
    """
    from astroquery.gaia import Gaia

    min_parallax = 1000.0 / max_distance_pc

    all_rows = []

    for dec_lo, dec_hi in DEC_STRIPS:
        # Use Gaia's absolute magnitude calculation inline
        # M_G = phot_g_mean_mag + 5*log10(parallax/1000) + 5
        #     = phot_g_mean_mag + 5*log10(parallax) - 10
        # Want M_G > 10 → phot_g_mean_mag + 5*log10(parallax) - 10 > 10
        # → phot_g_mean_mag > 20 - 5*log10(parallax)
        adql = f"""
        SELECT source_id, ra, dec, parallax, parallax_error,
               pmra, pmdec, phot_g_mean_mag, bp_rp,
               ruwe, astrometric_excess_noise_sig,
               teff_gspphot, logg_gspphot,
               non_single_star
        FROM gaiadr3.gaia_source
        WHERE parallax > {min_parallax}
          AND parallax_over_error > 10
          AND phot_g_mean_mag + 5 * LOG10(parallax) - 10 > 10
          AND bp_rp < 1.5
          AND bp_rp IS NOT NULL
          AND dec >= {dec_lo} AND dec < {dec_hi}
        ORDER BY parallax DESC
        """

        log.info("Querying Gaia WDs: dec [%d, %d), parallax > %.1f mas ...",
                 dec_lo, dec_hi, min_parallax)

        for attempt in range(3):
            try:
                job = Gaia.launch_job(adql)
                table = job.get_results()
                df = table.to_pandas()
                log.info("  → %d WD candidates in dec strip [%d, %d)", len(df), dec_lo, dec_hi)
                all_rows.append(df)
                break
            except Exception as e:
                log.warning("  Attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    log.error("  Failed all 3 attempts for dec strip [%d, %d)", dec_lo, dec_hi)

        time.sleep(1)

    if not all_rows:
        log.error("No WD data returned from Gaia.")
        return []

    import pandas as pd
    import numpy as np
    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["source_id"])

    # Compute absolute magnitude and further filter
    combined["abs_g"] = combined["phot_g_mean_mag"] + 5 * np.log10(combined["parallax"]) - 10
    log.info("Total WD candidates within %.0f pc (M_G > 10, bp_rp < 1.5): %d",
             max_distance_pc, len(combined))

    # Further cleanup: remove obvious subdwarfs/MS stars that leaked through
    # True WDs typically have M_G > 11 and logg > 7 (if available)
    # But we keep the generous cut and let EXODUS scoring handle it

    targets = []
    for _, row in combined.iterrows():
        sid = str(int(row["source_id"]))
        plx = float(row["parallax"])
        dist_pc = 1000.0 / plx if plx > 0 else 999.0

        targets.append({
            "target_id": f"WD_GAIA_{sid}",
            "host_star": f"Gaia DR3 {sid}",
            "ra": round(float(row["ra"]), 6),
            "dec": round(float(row["dec"]), 6),
            "distance_pc": round(dist_pc, 2),
            "phot_g_mean_mag": round(float(row["phot_g_mean_mag"]), 3) if not _isnan(row["phot_g_mean_mag"]) else None,
            "bp_rp": round(float(row["bp_rp"]), 3) if not _isnan(row["bp_rp"]) else None,
            "abs_g": round(float(row["abs_g"]), 3),
            "ruwe": round(float(row["ruwe"]), 3) if not _isnan(row["ruwe"]) else None,
            "teff": round(float(row["teff_gspphot"]), 0) if not _isnan(row["teff_gspphot"]) else None,
            "logg": round(float(row["logg_gspphot"]), 2) if not _isnan(row["logg_gspphot"]) else None,
            "source_tier": "white_dwarf_survey",
            "discovery_reason": f"White dwarf candidate within {max_distance_pc} pc (M_G={float(row['abs_g']):.1f})",
        })

    targets.sort(key=lambda t: t.get("distance_pc", 999))
    return targets


def _isnan(val) -> bool:
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
    campaign = {
        "campaign": f"white_dwarf_{int(max_distance_pc)}pc",
        "description": (
            f"White dwarf IR survey: Gaia DR3 WDs within {max_distance_pc} pc "
            f"(M_G > 10, bp_rp < 1.5). Binary template cannot explain WD IR excess — "
            f"any hit survives by construction. Key population for SETI detection."
        ),
        "phase": "wd_survey",
        "n_targets": len(targets),
        "targets": targets,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(campaign, f, indent=2, default=str)

    log.info("Wrote %d WD targets to %s", len(targets), output_path)


def main():
    parser = argparse.ArgumentParser(description="EXODUS White Dwarf Target Builder")
    parser.add_argument("--max-distance", type=float, default=50.0,
                        help="Maximum distance in parsecs (default: 50)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "targets" / f"white_dwarfs_{int(args.max_distance)}pc.json"
    )

    targets = query_gaia_white_dwarfs(max_distance_pc=args.max_distance)

    if targets:
        build_target_file(targets, args.max_distance, output)
        print(f"\n{'='*60}")
        print(f"  EXODUS — White Dwarf Target List ({int(args.max_distance)} pc)")
        print(f"{'='*60}")
        print(f"  WDs found: {len(targets)}")
        print(f"  Distance range: {targets[0]['distance_pc']:.1f} — {targets[-1]['distance_pc']:.1f} pc")

        # Stats
        abs_g_vals = [t["abs_g"] for t in targets if t.get("abs_g") is not None]
        if abs_g_vals:
            print(f"  Abs G range: {min(abs_g_vals):.1f} — {max(abs_g_vals):.1f}")
        with_logg = [t for t in targets if t.get("logg") is not None and t["logg"] > 7.0]
        print(f"  Confirmed high-logg (>7): {len(with_logg)}")
        print(f"  Output: {output}")
        print(f"{'='*60}")
    else:
        log.error("No WD targets found.")


if __name__ == "__main__":
    main()
