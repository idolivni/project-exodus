#!/usr/bin/env python
"""
Build calibration target lists of known stellar populations.

Queries SIMBAD, VizieR, and Gaia DR3 to construct four calibration sets:
  1. Known visual/spectroscopic binaries (astrometric anomaly expected)
  2. Known circumstellar disk hosts (IR excess expected)
  3. Known YSOs / T Tauri stars (IR + variability expected)
  4. Known giants, luminosity class III (negative controls — no anomalies)

Usage
-----
  python scripts/build_calibration.py                # Build all 4 populations
  python scripts/build_calibration.py --population binaries  # Just one
  python scripts/build_calibration.py --max-per-pop 100      # Smaller test set
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

log = logging.getLogger("exodus.calibration")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Population-specific expected channel behavior
# ---------------------------------------------------------------------------

POPULATION_EXPECTATIONS = {
    "binary": {
        "expected_channels": {
            "proper_motion_anomaly": "positive",
        },
        "is_positive_control": True,
        "is_negative_control": False,
        "expected_behavior": "Astrometric anomaly from unresolved/resolved companion",
    },
    "disk": {
        "expected_channels": {
            "ir_excess": "positive",
        },
        "is_positive_control": True,
        "is_negative_control": False,
        "expected_behavior": "IR excess from circumstellar dust disk",
    },
    "yso": {
        "expected_channels": {
            "ir_excess": "positive",
        },
        "is_positive_control": True,
        "is_negative_control": False,
        "expected_behavior": "IR excess + photometric variability from protoplanetary disk and accretion",
    },
    "giant": {
        "expected_channels": {},
        "is_positive_control": False,
        "is_negative_control": True,
        "expected_behavior": "Negative control — evolved giant, no technosignature-mimicking anomalies expected",
    },
}


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def _query_simbad_by_otype(otype: str, max_results: int, max_distance_pc: float = 100.0) -> List[Dict[str, Any]]:
    """Query SIMBAD for objects of a given type within max_distance_pc."""
    from astroquery.simbad import Simbad

    s = Simbad()
    s.add_votable_fields("otype", "plx", "plx_error", "sp", "flux(V)")
    s.ROW_LIMIT = max_results * 3  # over-query to allow filtering

    # SIMBAD ADQL query for objects by type with parallax constraint
    min_plx = 1000.0 / max_distance_pc  # parallax in mas for distance limit

    query = f"""
    SELECT TOP {max_results * 3}
        main_id, ra, dec, plx_value, plx_error, sp_type, otype_txt
    FROM basic
    WHERE otype = '{otype}'
      AND plx_value > {min_plx}
      AND plx_value IS NOT NULL
      AND ra IS NOT NULL
      AND dec IS NOT NULL
    ORDER BY plx_value DESC
    """

    log.info("SIMBAD ADQL query for otype='%s', min_plx=%.1f mas ...", otype, min_plx)
    try:
        result = Simbad.query_tap(query)
        if result is None or len(result) == 0:
            log.warning("SIMBAD returned no results for otype='%s'", otype)
            return []

        targets = []
        seen_ids = set()
        for row in result:
            main_id = str(row["main_id"]).strip()
            ra = float(row["ra"])
            dec = float(row["dec"])
            plx = float(row["plx_value"])

            if plx <= 0:
                continue
            dist_pc = 1000.0 / plx
            if dist_pc > max_distance_pc:
                continue

            # Clean target_id
            target_id = main_id.replace(" ", "_").replace("*", "").replace("+", "p").strip("_")
            if target_id in seen_ids:
                continue
            seen_ids.add(target_id)

            sp_type = str(row["sp_type"]) if row["sp_type"] else ""

            targets.append({
                "target_id": target_id,
                "host_star": main_id,
                "ra": round(ra, 6),
                "dec": round(dec, 6),
                "distance_pc": round(dist_pc, 1),
                "spectral_type": sp_type,
            })

            if len(targets) >= max_results:
                break

        log.info("SIMBAD returned %d targets for otype='%s'", len(targets), otype)
        return targets

    except Exception as e:
        log.error("SIMBAD query failed for otype='%s': %s", otype, e)
        return []


def query_binaries(max_results: int = 500, max_distance_pc: float = 100.0) -> List[Dict[str, Any]]:
    """Query SIMBAD for known visual/spectroscopic binary stars."""
    from astroquery.simbad import Simbad

    # SB* = spectroscopic binary, ** = visual binary
    # We query both and merge
    min_plx = 1000.0 / max_distance_pc

    queries = [
        ("SB*", "Spectroscopic binary"),
        ("**", "Visual binary"),
    ]

    all_targets = []
    seen_ids = set()

    for otype, label in queries:
        query = f"""
        SELECT TOP {max_results * 2}
            main_id, ra, dec, plx_value, sp_type
        FROM basic
        WHERE otype = '{otype}'
          AND plx_value > {min_plx}
          AND plx_value IS NOT NULL
          AND ra IS NOT NULL
          AND dec IS NOT NULL
        ORDER BY plx_value DESC
        """

        log.info("Querying SIMBAD for %s (otype='%s') ...", label, otype)
        try:
            result = Simbad.query_tap(query)
            if result is None or len(result) == 0:
                log.warning("No results for %s", label)
                continue

            for row in result:
                main_id = str(row["main_id"]).strip()
                ra = float(row["ra"])
                dec = float(row["dec"])
                plx = float(row["plx_value"])

                if plx <= 0:
                    continue
                dist_pc = 1000.0 / plx
                if dist_pc > max_distance_pc:
                    continue

                target_id = main_id.replace(" ", "_").replace("*", "").replace("+", "p").strip("_")
                if target_id in seen_ids:
                    continue
                seen_ids.add(target_id)

                sp_type = str(row["sp_type"]) if row["sp_type"] else ""

                all_targets.append({
                    "target_id": target_id,
                    "host_star": main_id,
                    "ra": round(ra, 6),
                    "dec": round(dec, 6),
                    "distance_pc": round(dist_pc, 1),
                    "spectral_type": sp_type,
                    "binary_type": label,
                })

        except Exception as e:
            log.error("SIMBAD query failed for %s: %s", label, e)

    # Sort by distance (nearest first)
    all_targets.sort(key=lambda x: x["distance_pc"])
    log.info("Total binary candidates: %d (returning top %d)", len(all_targets), max_results)
    return all_targets[:max_results]


def query_disks(max_results: int = 500, max_distance_pc: float = 150.0) -> List[Dict[str, Any]]:
    """Query SIMBAD for known circumstellar disk hosts."""
    from astroquery.simbad import Simbad

    min_plx = 1000.0 / max_distance_pc

    # Object types: di* = circumstellar disk, PD* = post-AGB disk
    # Also try "PM*" (proper motion star with disk) variants
    # Main target: stars flagged with circumstellar matter
    query = f"""
    SELECT TOP {max_results * 3}
        main_id, ra, dec, plx_value, sp_type
    FROM basic
    WHERE (otype = 'di*' OR otype = 'PD*')
      AND plx_value > {min_plx}
      AND plx_value IS NOT NULL
      AND ra IS NOT NULL
      AND dec IS NOT NULL
    ORDER BY plx_value DESC
    """

    log.info("Querying SIMBAD for disk hosts (di*, PD*), max_dist=%.0f pc ...", max_distance_pc)
    targets = []
    seen_ids = set()

    try:
        result = Simbad.query_tap(query)
        if result is not None:
            for row in result:
                main_id = str(row["main_id"]).strip()
                ra = float(row["ra"])
                dec = float(row["dec"])
                plx = float(row["plx_value"])
                if plx <= 0:
                    continue
                dist_pc = 1000.0 / plx

                target_id = main_id.replace(" ", "_").replace("*", "").replace("+", "p").strip("_")
                if target_id in seen_ids:
                    continue
                seen_ids.add(target_id)

                sp_type = str(row["sp_type"]) if row["sp_type"] else ""
                targets.append({
                    "target_id": target_id,
                    "host_star": main_id,
                    "ra": round(ra, 6),
                    "dec": round(dec, 6),
                    "distance_pc": round(dist_pc, 1),
                    "spectral_type": sp_type,
                })
    except Exception as e:
        log.error("SIMBAD disk query failed: %s", e)

    # If we didn't get enough, try broader "young star with disk" types
    if len(targets) < max_results:
        log.info("Got %d disk hosts, trying TT* (T Tauri with disk) ...", len(targets))
        query2 = f"""
        SELECT TOP {(max_results - len(targets)) * 2}
            main_id, ra, dec, plx_value, sp_type
        FROM basic
        WHERE otype = 'TT*'
          AND plx_value > {min_plx}
          AND plx_value IS NOT NULL
          AND ra IS NOT NULL
          AND dec IS NOT NULL
        ORDER BY plx_value DESC
        """
        try:
            result2 = Simbad.query_tap(query2)
            if result2 is not None:
                for row in result2:
                    main_id = str(row["main_id"]).strip()
                    target_id = main_id.replace(" ", "_").replace("*", "").replace("+", "p").strip("_")
                    if target_id in seen_ids:
                        continue
                    seen_ids.add(target_id)
                    plx = float(row["plx_value"])
                    if plx <= 0:
                        continue
                    targets.append({
                        "target_id": target_id,
                        "host_star": main_id,
                        "ra": round(float(row["ra"]), 6),
                        "dec": round(float(row["dec"]), 6),
                        "distance_pc": round(1000.0 / plx, 1),
                        "spectral_type": str(row["sp_type"]) if row["sp_type"] else "",
                    })
        except Exception as e:
            log.error("SIMBAD TT* query failed: %s", e)

    targets.sort(key=lambda x: x["distance_pc"])
    log.info("Total disk host candidates: %d (returning top %d)", len(targets), max_results)
    return targets[:max_results]


def query_ysos(max_results: int = 500, max_distance_pc: float = 500.0) -> List[Dict[str, Any]]:
    """Query SIMBAD for known Young Stellar Objects."""
    from astroquery.simbad import Simbad

    min_plx = 1000.0 / max_distance_pc

    # YSO types: Y*O = YSO, TT* = T Tauri, Ae* = Herbig Ae/Be
    # YSOs are often at larger distances (star-forming regions at 100-500 pc)
    query = f"""
    SELECT TOP {max_results * 3}
        main_id, ra, dec, plx_value, sp_type, otype_txt
    FROM basic
    WHERE (otype = 'Y*O' OR otype = 'TT*' OR otype = 'Ae*' OR otype = 'Or*')
      AND plx_value > {min_plx}
      AND plx_value IS NOT NULL
      AND ra IS NOT NULL
      AND dec IS NOT NULL
    ORDER BY plx_value DESC
    """

    log.info("Querying SIMBAD for YSOs (Y*O, TT*, Ae*, Or*), max_dist=%.0f pc ...", max_distance_pc)
    targets = []
    seen_ids = set()

    try:
        result = Simbad.query_tap(query)
        if result is not None:
            for row in result:
                main_id = str(row["main_id"]).strip()
                ra = float(row["ra"])
                dec = float(row["dec"])
                plx = float(row["plx_value"])
                if plx <= 0:
                    continue
                dist_pc = 1000.0 / plx

                target_id = main_id.replace(" ", "_").replace("*", "").replace("+", "p").strip("_")
                if target_id in seen_ids:
                    continue
                seen_ids.add(target_id)

                sp_type = str(row["sp_type"]) if row["sp_type"] else ""
                otype = str(row["otype_txt"]) if row["otype_txt"] else ""

                targets.append({
                    "target_id": target_id,
                    "host_star": main_id,
                    "ra": round(ra, 6),
                    "dec": round(dec, 6),
                    "distance_pc": round(dist_pc, 1),
                    "spectral_type": sp_type,
                    "yso_subtype": otype,
                })
    except Exception as e:
        log.error("SIMBAD YSO query failed: %s", e)

    targets.sort(key=lambda x: x["distance_pc"])
    log.info("Total YSO candidates: %d (returning top %d)", len(targets), max_results)
    return targets[:max_results]


def query_giants(max_results: int = 500, max_distance_pc: float = 100.0) -> List[Dict[str, Any]]:
    """Query SIMBAD for giant stars (luminosity class III/IV) within max_distance_pc.

    Falls back to SIMBAD because Gaia archive frequently times out on
    logg_gspphot queries.
    """
    from astroquery.simbad import Simbad

    min_plx = 1000.0 / max_distance_pc

    # SIMBAD otypes: RG* = Red Giant, AB* = AGB, HB* = Horizontal Branch,
    # LP* = Long Period Variable (mostly giants), bC* = Carbon star
    queries = [
        ("RG*", "Red Giant", max_results * 2),
        ("AB*", "AGB star", max_results),
        ("HB*", "Horizontal Branch", max_results),
        ("LP*", "Long Period Variable", max_results),
        ("bC*", "Carbon star", max_results // 2),
    ]

    targets = []
    seen_ids = set()

    for otype, label, limit in queries:
        query = f"""
        SELECT TOP {limit}
            main_id, ra, dec, plx_value, sp_type
        FROM basic
        WHERE otype = '{otype}'
          AND plx_value > {min_plx}
          AND plx_value IS NOT NULL
          AND ra IS NOT NULL
          AND dec IS NOT NULL
        ORDER BY plx_value DESC
        """

        log.info("Querying SIMBAD for %s (otype='%s') ...", label, otype)
        try:
            result = Simbad.query_tap(query)
            if result is None or len(result) == 0:
                log.warning("No results for %s", label)
                continue

            for row in result:
                main_id = str(row["main_id"]).strip()
                plx = float(row["plx_value"])
                if plx <= 0:
                    continue
                dist_pc = 1000.0 / plx
                if dist_pc > max_distance_pc:
                    continue

                target_id = main_id.replace(" ", "_").replace("*", "").replace("+", "p").strip("_")
                if target_id in seen_ids:
                    continue
                seen_ids.add(target_id)

                sp_type = str(row["sp_type"]) if row["sp_type"] else ""
                targets.append({
                    "target_id": target_id,
                    "host_star": main_id,
                    "ra": round(float(row["ra"]), 6),
                    "dec": round(float(row["dec"]), 6),
                    "distance_pc": round(dist_pc, 1),
                    "spectral_type": sp_type,
                    "giant_type": label,
                })

                if len(targets) >= max_results:
                    break

        except Exception as e:
            log.error("SIMBAD query failed for %s: %s", label, e)

        if len(targets) >= max_results:
            break

    targets.sort(key=lambda x: x["distance_pc"])
    log.info("Total giant candidates: %d (returning top %d)", len(targets), max_results)
    return targets[:max_results]


# ---------------------------------------------------------------------------
# Target file writer
# ---------------------------------------------------------------------------

def write_calibration_file(
    population: str,
    targets: List[Dict[str, Any]],
    output_dir: str = "data/targets",
) -> str:
    """Write a calibration target file in campaign JSON format."""
    expectations = POPULATION_EXPECTATIONS[population]

    campaign_targets = []
    for t in targets:
        entry = {
            "target_id": t["target_id"],
            "host_star": t["host_star"],
            "ra": t["ra"],
            "dec": t["dec"],
            "distance_pc": t.get("distance_pc"),
            "hz_flag": False,
            "is_positive_control": expectations["is_positive_control"],
            "is_negative_control": expectations["is_negative_control"],
            "expected_behavior": expectations["expected_behavior"],
            "expected_channels": expectations["expected_channels"],
        }
        # Add population-specific metadata
        if "spectral_type" in t and t["spectral_type"]:
            entry["spectral_type"] = t["spectral_type"]
        if "phot_g_mean_mag" in t and t["phot_g_mean_mag"] is not None:
            entry["phot_g_mean_mag"] = t["phot_g_mean_mag"]
        if "bp_rp" in t and t["bp_rp"] is not None:
            entry["bp_rp"] = t["bp_rp"]
        if "binary_type" in t:
            entry["notes"] = f"Binary type: {t['binary_type']}"
        if "yso_subtype" in t:
            entry["notes"] = f"YSO subtype: {t['yso_subtype']}"
        if "logg" in t and t["logg"] is not None:
            entry["notes"] = f"logg={t['logg']}, Teff={t.get('teff', '?')} K"
        if "giant_type" in t:
            entry["notes"] = f"Giant type: {t['giant_type']}"

        campaign_targets.append(entry)

    campaign = {
        "campaign": f"calibration_{population}",
        "description": (
            f"Known population calibration: {len(campaign_targets)} {population} targets. "
            f"{expectations['expected_behavior']}."
        ),
        "phase": "calibration",
        "targets": campaign_targets,
    }

    output_path = Path(output_dir) / f"calibration_{population}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(campaign, f, indent=2)

    log.info("Wrote %d targets to %s", len(campaign_targets), output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

POPULATION_BUILDERS = {
    "binary": query_binaries,
    "disk": query_disks,
    "yso": query_ysos,
    "giant": query_giants,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build calibration target lists of known stellar populations."
    )
    parser.add_argument(
        "--population", choices=list(POPULATION_BUILDERS.keys()) + ["all"],
        default="all",
        help="Which population to build (default: all)",
    )
    parser.add_argument(
        "--max-per-pop", type=int, default=500,
        help="Maximum targets per population (default: 500)",
    )
    parser.add_argument(
        "--output-dir", default="data/targets",
        help="Output directory for target files",
    )
    args = parser.parse_args()

    populations = list(POPULATION_BUILDERS.keys()) if args.population == "all" else [args.population]

    print(f"\n{'='*60}")
    print(f"EXODUS Known Population Calibration Builder")
    print(f"{'='*60}")
    print(f"Populations: {', '.join(populations)}")
    print(f"Max targets per population: {args.max_per_pop}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    results = {}
    for pop in populations:
        print(f"\n--- {pop.upper()} ---")
        t0 = time.time()
        builder = POPULATION_BUILDERS[pop]

        # Adjust distance limits per population
        if pop == "yso":
            targets = builder(max_results=args.max_per_pop, max_distance_pc=500.0)
        elif pop == "disk":
            targets = builder(max_results=args.max_per_pop, max_distance_pc=200.0)
        elif pop == "giant":
            targets = builder(max_results=args.max_per_pop, max_distance_pc=200.0)
        else:
            targets = builder(max_results=args.max_per_pop)

        if targets:
            output = write_calibration_file(pop, targets, args.output_dir)
            results[pop] = {"count": len(targets), "file": output}
        else:
            results[pop] = {"count": 0, "file": None}
            log.warning("No targets found for population '%s'", pop)

        elapsed = time.time() - t0
        print(f"  {pop}: {results[pop]['count']} targets in {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = 0
    for pop, info in results.items():
        status = f"{info['count']} targets -> {info['file']}" if info['file'] else "FAILED"
        print(f"  {pop:<10} {status}")
        total += info["count"]
    print(f"\n  TOTAL: {total} calibration targets")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
