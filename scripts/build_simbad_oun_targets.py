#!/usr/bin/env python3
"""
Build SIMBAD 'Objects of Unknown Nature' target list for EXODUS.

Strategy:
  SIMBAD classifies objects it cannot identify as "Unknown" (otype='?').
  These are objects that professional astronomers have looked at, recorded,
  and published saying "we don't know what this is."

  We query for:
  1. Objects of Unknown Nature within 100 pc (via Gaia cross-match)
  2. Candidate YSOs that might be misclassified (otype='Y*?')
  3. Unclassified variables (otype='V*?')

  Each is then cross-matched with Gaia DR3 for distances.

Usage:
  ./venv/bin/python scripts/build_simbad_oun_targets.py [--max-distance 100] [--max-targets 500]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_FILE = "data/targets/simbad_unknown_objects.json"


def query_simbad_unknown_nearby(max_distance_pc: float = 100.0, max_targets: int = 500):
    """
    Query SIMBAD TAP for objects of unknown nature with Gaia distances.

    Uses SIMBAD's TAP service to find objects classified as 'Unknown'
    that have Gaia parallaxes indicating they're within max_distance_pc.
    """
    from astroquery.simbad import Simbad
    import warnings
    warnings.filterwarnings('ignore')

    min_parallax = 1000.0 / max_distance_pc  # mas

    print(f"  Querying SIMBAD TAP for unknown objects within {max_distance_pc} pc ...")
    print(f"  (parallax > {min_parallax:.1f} mas)")

    # SIMBAD TAP query: objects with unknown otype + Gaia parallax
    # SIMBAD otypes: '?' = unknown, 'Y*?' = YSO candidate, 'V*?' = variable candidate
    # We join with basic + allfluxes + mesDistance

    try:
        from astroquery.simbad import Simbad as SimbadQ

        # Use TAP for more flexible queries
        from pyvo import dal
        tap_url = "https://simbad.cds.unistra.fr/simbad/sim-tap"
        service = dal.TAPService(tap_url)

        # Query 1: Unknown objects with Gaia DR3 parallax
        # Note: SIMBAD ADQL does not support COALESCE — use simple WHERE instead
        adql = f"""
        SELECT TOP {max_targets}
            b.main_id, b.ra, b.dec, b.otype,
            b.pmra, b.pmdec, b.sp_type,
            m.plx_value AS parallax,
            m.plx_err AS parallax_err
        FROM basic AS b
        JOIN mesPlx AS m ON b.oid = m.oidref
        WHERE b.otype = '?'
          AND m.plx_value > {min_parallax:.2f}
          AND m.plx_err > 0
          AND m.plx_value / m.plx_err > 3
        ORDER BY m.plx_value DESC
        """

        print("  Running SIMBAD TAP query (unknown objects) ...")
        result = service.run_async(adql, timeout=120)
        table = result.to_table()
        print(f"  Found {len(table)} unknown objects within {max_distance_pc} pc")

        targets = []
        for row in table:
            main_id = str(row['main_id']).strip()
            ra = float(row['ra'])
            dec = float(row['dec'])
            plx = float(row['parallax'])
            plx_err = float(row['parallax_err']) if row['parallax_err'] else None
            dist = 1000.0 / plx if plx > 0 else None
            otype = str(row['otype']).strip()
            sp_type = str(row['sp_type']).strip() if row['sp_type'] and str(row['sp_type']).strip() != '' else None

            safe_id = main_id.replace(' ', '_').replace('+', 'p').replace('-', 'm').replace('*', 'x')
            target = {
                "target_id": f"SIMBAD_{safe_id}",
                "host_star": main_id,
                "ra": ra,
                "dec": dec,
                "distance_pc": dist,
                "parallax": plx,
                "parallax_err": plx_err,
                "hz_flag": False,
                "otype": otype,
                "sp_type": sp_type,
                "source_catalog": "SIMBAD",
                "discovery_channel": "unclassified_object",
                "discovery_reason": f"SIMBAD object of unknown nature (otype='{otype}'), d={dist:.1f} pc" if dist else f"SIMBAD unknown (otype='{otype}')",
            }
            targets.append(target)

        # Query 2: YSO candidates (might be misclassified)
        adql2 = f"""
        SELECT TOP {max(100, max_targets // 5)}
            b.main_id, b.ra, b.dec, b.otype,
            b.pmra, b.pmdec, b.sp_type,
            m.plx_value AS parallax,
            m.plx_err AS parallax_err
        FROM basic AS b
        JOIN mesPlx AS m ON b.oid = m.oidref
        WHERE b.otype = 'Y*?'
          AND m.plx_value > {min_parallax:.2f}
          AND m.plx_err > 0
          AND m.plx_value / m.plx_err > 3
        ORDER BY m.plx_value DESC
        """

        print("  Running SIMBAD TAP query (YSO candidates) ...")
        result2 = service.run_async(adql2, timeout=120)
        table2 = result2.to_table()
        print(f"  Found {len(table2)} YSO candidates within {max_distance_pc} pc")

        for row in table2:
            main_id = str(row['main_id']).strip()
            ra = float(row['ra'])
            dec = float(row['dec'])
            plx = float(row['parallax'])
            plx_err = float(row['parallax_err']) if row['parallax_err'] else None
            dist = 1000.0 / plx if plx > 0 else None
            otype = str(row['otype']).strip()

            safe_id = main_id.replace(' ', '_').replace('+', 'p').replace('-', 'm').replace('*', 'x')
            target = {
                "target_id": f"SIMBAD_{safe_id}",
                "host_star": main_id,
                "ra": ra,
                "dec": dec,
                "distance_pc": dist,
                "parallax": plx,
                "parallax_err": plx_err,
                "hz_flag": False,
                "otype": otype,
                "source_catalog": "SIMBAD",
                "discovery_channel": "yso_candidate",
                "discovery_reason": f"SIMBAD YSO candidate (otype='{otype}'), d={dist:.1f} pc" if dist else f"SIMBAD YSO candidate",
            }
            targets.append(target)

        return targets

    except Exception as e:
        print(f"  TAP query failed: {e}")
        print("  Falling back to basic SIMBAD query ...")
        return []


def build_targets(max_distance: float, max_targets: int):
    """Build the full target list."""
    print("=" * 60)
    print("  SIMBAD Unknown Objects → EXODUS Target Builder")
    print(f"  Max distance: {max_distance} pc")
    print(f"  Max targets: {max_targets}")
    print("=" * 60)

    targets = query_simbad_unknown_nearby(max_distance, max_targets)

    if not targets:
        print("\n  No targets found! Check SIMBAD connectivity.")
        return

    # Deduplicate by target_id
    seen = set()
    unique = []
    for t in targets:
        if t["target_id"] not in seen:
            seen.add(t["target_id"])
            unique.append(t)
    targets = unique

    # Sort by distance
    targets.sort(key=lambda t: t.get("distance_pc") or 9999)

    output = {
        "campaign": "simbad_unknown_objects",
        "phase": "novel_targets",
        "description": (
            f"SIMBAD objects of unknown nature within {max_distance} pc. "
            "These are astronomical objects that professional astronomers have "
            "observed but could not classify. EXODUS checks whether any show "
            "multi-channel anomalies consistent with technosignature hypotheses."
        ),
        "source": "SIMBAD astronomical database (CDS Strasbourg)",
        "targets": targets,
    }

    output_dir = Path("data/targets")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "simbad_unknown_objects.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Stats
    n_unknown = sum(1 for t in targets if t.get("otype") == "?")
    n_yso = sum(1 for t in targets if "Y*" in str(t.get("otype", "")))
    nearby = [t for t in targets if t.get("distance_pc") and t["distance_pc"] < 50]
    with_gaia = [t for t in targets if t.get("gaia_dr3_source_id")]

    print("\n" + "=" * 60)
    print("  SIMBAD TARGET BUILD COMPLETE")
    print("=" * 60)
    print(f"\n  Total targets: {len(targets)}")
    print(f"  Unknown nature: {n_unknown}")
    print(f"  YSO candidates: {n_yso}")
    print(f"  With Gaia ID: {len(with_gaia)}")
    print(f"  Within 50 pc: {len(nearby)}")
    if targets:
        print(f"  Nearest: {targets[0]['host_star']} at {targets[0].get('distance_pc', '?'):.1f} pc")

    print(f"\n  Output: data/targets/simbad_unknown_objects.json")
    print(f"\n  Next steps:")
    print(f"  1. Run: ./venv/bin/python scripts/run_quick.py --target-file data/targets/simbad_unknown_objects.json --tier 0")
    print(f"  2. Any multi-channel detection = IMMEDIATE ESCALATION")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SIMBAD unknown objects target list")
    parser.add_argument("--max-distance", type=float, default=100.0,
                        help="Maximum distance in parsecs (default: 100)")
    parser.add_argument("--max-targets", type=int, default=500,
                        help="Maximum number of targets (default: 500)")
    args = parser.parse_args()
    build_targets(args.max_distance, args.max_targets)
