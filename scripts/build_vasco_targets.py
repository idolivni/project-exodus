#!/usr/bin/env python3
"""
Build VASCO (Vanishing & Appearing Sources) target list for EXODUS.

Sources:
  - Villarroel et al. (2020), AJ 159:8 — 99 vanishing star candidates (Table 2)
  - Plus 28 "most interesting" subset (Table 3)
  - VizieR catalog: J/AJ/159/8

Strategy:
  These are USNO B1.0 objects that have no Pan-STARRS DR1 counterpart within 30".
  They may be genuine vanishing stars, extreme flare events, plate artifacts, or
  (the SETI angle) objects that have been enclosed by a Dyson-like structure.

  EXODUS will cross-check each candidate against ALL 6 detection channels:
  - IR excess (AllWISE): Is there residual mid-IR emission?
  - IR variability (NEOWISE): Has IR flux changed over 10 years?
  - PM anomaly (CatWISE vs Gaia): Unusual astrometric signature?
  - Transit anomaly: Any optical variability in TESS/ZTF?
  - Radio continuum: 1.4 GHz emission?
  - Multi-messenger: Near any Fermi/IceCube/FRB sources?

  A star that vanished in optical but shows up in IR or radio is the exact
  physical signature of an object being enclosed by a megastructure.

Usage:
  ./venv/bin/python scripts/build_vasco_targets.py
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

OUTPUT_FILE = "data/targets/vasco_vanishing_99.json"
OUTPUT_FILE_28 = "data/targets/vasco_vanishing_28.json"


def sexagesimal_to_degrees(ra_str: str, dec_str: str) -> tuple:
    """Convert sexagesimal RA (HH MM SS.ss) and Dec (±DD MM SS.ss) to degrees."""
    # RA
    parts = ra_str.strip().split()
    ra_h, ra_m, ra_s = float(parts[0]), float(parts[1]), float(parts[2])
    ra_deg = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0

    # Dec
    dec_str = dec_str.strip()
    sign = -1 if dec_str.startswith('-') else 1
    parts = dec_str.lstrip('+-').split()
    dec_d, dec_m, dec_s = float(parts[0]), float(parts[1]), float(parts[2])
    dec_deg = sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)

    return ra_deg, dec_deg


def query_gaia_for_position(ra_deg: float, dec_deg: float, radius_arcsec: float = 10.0):
    """Query Gaia DR3 for the nearest source at a position. Returns dict or None."""
    try:
        from astroquery.gaia import Gaia
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

        adql = f"""
        SELECT TOP 1 source_id, ra, dec, parallax, parallax_over_error,
               phot_g_mean_mag, bp_rp, ruwe, pmra, pmdec,
               astrometric_excess_noise_sig
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_deg:.6f}, {dec_deg:.6f}, {radius_arcsec / 3600.0:.6f})
        )
        AND parallax_over_error > 3
        ORDER BY phot_g_mean_mag ASC
        """
        job = Gaia.launch_job(adql)
        result = job.get_results()
        if len(result) == 0:
            return None
        row = result[0]
        plx = float(row['parallax'])
        dist_pc = 1000.0 / plx if plx > 0 else None
        return {
            "gaia_source_id": str(row['source_id']),
            "ra_gaia": float(row['ra']),
            "dec_gaia": float(row['dec']),
            "distance_pc": dist_pc,
            "parallax": plx,
            "phot_g_mean_mag": float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] else None,
            "bp_rp": float(row['bp_rp']) if row['bp_rp'] else None,
            "ruwe": float(row['ruwe']) if row['ruwe'] else None,
            "pmra": float(row['pmra']) if row['pmra'] else None,
            "pmdec": float(row['pmdec']) if row['pmdec'] else None,
        }
    except Exception as e:
        print(f"  Gaia query failed: {e}")
        return None


def build_vasco_targets():
    """Fetch VASCO candidates from VizieR and build target files."""
    from astroquery.vizier import Vizier

    print("=" * 60)
    print("  VASCO Vanishing Stars → EXODUS Target Builder")
    print("=" * 60)

    # ── Fetch from VizieR ──
    print("\n[1/4] Querying VizieR J/AJ/159/8 ...")
    v = Vizier(catalog='J/AJ/159/8', row_limit=-1)
    tables = v.get_catalogs('J/AJ/159/8')

    table2 = tables[0]  # 99 candidates
    table3 = tables[1]  # 28 most interesting

    print(f"  Table 2: {len(table2)} vanishing star candidates")
    print(f"  Table 3: {len(table3)} 'most interesting' subset")

    # ── Convert coordinates ──
    print("\n[2/4] Converting coordinates ...")
    candidates_99 = []
    for i, row in enumerate(table2):
        ra_deg, dec_deg = sexagesimal_to_degrees(str(row['RAJ2000']), str(row['DEJ2000']))
        rmag = float(row['rmag']) if row['rmag'] else None
        candidates_99.append({
            "index": i,
            "ra_usno": ra_deg,
            "dec_usno": dec_deg,
            "rmag_usno": rmag,
            "ra_str": str(row['RAJ2000']),
            "dec_str": str(row['DEJ2000']),
        })

    # Table 3 is a SEPARATE set of 28 highest-quality candidates (not a subset of Table 2)
    candidates_28_extra = []
    for i, row in enumerate(table3):
        ra_deg, dec_deg = sexagesimal_to_degrees(str(row['RAJ2000']), str(row['DEJ2000']))
        candidates_28_extra.append({
            "index": 99 + i,  # Continue numbering after Table 2
            "ra_usno": ra_deg,
            "dec_usno": dec_deg,
            "rmag_usno": None,  # Table 3 doesn't have rmag
            "ra_str": str(row['RAJ2000']),
            "dec_str": str(row['DEJ2000']),
            "table3": True,
        })

    # Mark Table 2 candidates
    for c in candidates_99:
        c["most_interesting"] = False
        c["table3"] = False

    # Combine all candidates
    all_candidates = candidates_99 + candidates_28_extra
    print(f"  Table 2: {len(candidates_99)} single-epoch red point sources")
    print(f"  Table 3: {len(candidates_28_extra)} highest-quality vanishing candidates")
    print(f"  Total: {len(all_candidates)} unique VASCO candidates")

    # ── Cross-match with Gaia DR3 ──
    print(f"\n[3/4] Cross-matching {len(all_candidates)} candidates with Gaia DR3 (10\" cone search) ...")
    print("  This may take a few minutes ...")

    gaia_matched = 0
    gaia_failed = 0
    nearby_count = 0  # within 200 pc
    n_total = len(all_candidates)

    for i, c in enumerate(all_candidates):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_total}] Gaia matched: {gaia_matched}, failed: {gaia_failed}")
        gaia = query_gaia_for_position(c["ra_usno"], c["dec_usno"], radius_arcsec=10.0)
        if gaia:
            c["gaia"] = gaia
            gaia_matched += 1
            if gaia["distance_pc"] and gaia["distance_pc"] < 200:
                nearby_count += 1
        else:
            c["gaia"] = None
            gaia_failed += 1
        time.sleep(0.3)  # Rate limit

    print(f"\n  Gaia matches: {gaia_matched}/{n_total}")
    print(f"  Within 200 pc: {nearby_count}")
    print(f"  No Gaia match: {gaia_failed} (may be genuinely vanished or very faint)")

    # ── Build target JSONs ──
    print("\n[4/4] Building target files ...")

    targets_all = []
    for c in all_candidates:
        # Use Gaia position if available, otherwise USNO position
        if c["gaia"]:
            ra = c["gaia"]["ra_gaia"]
            dec = c["gaia"]["dec_gaia"]
            dist = c["gaia"]["distance_pc"]
            gaia_id = c["gaia"]["gaia_source_id"]
            host = f"Gaia DR3 {gaia_id}"
        else:
            ra = c["ra_usno"]
            dec = c["dec_usno"]
            dist = None
            gaia_id = None
            host = f"USNO_{c['ra_str'].replace(' ', '')}_{c['dec_str'].replace(' ', '')}"

        target = {
            "target_id": f"VASCO_{c['index']:03d}",
            "host_star": host,
            "ra": ra,
            "dec": dec,
            "distance_pc": dist,
            "hz_flag": False,
            "source_paper": "Villarroel et al. 2020 (AJ 159:8)",
            "source_catalog": "VASCO (Vanishing & Appearing Sources)",
            "discovery_channel": "optical_vanishing",
            "discovery_reason": f"USNO B1.0 source missing in Pan-STARRS DR1 (rmag={c['rmag_usno']})" if c.get('rmag_usno') else "USNO B1.0 source missing in Pan-STARRS DR1 (Table 3 highest-quality)",
            "table": "Table 3 (highest quality)" if c.get("table3") else "Table 2 (single-epoch red sources)",
            "rmag_usno": c["rmag_usno"],
            "ra_usno": c["ra_usno"],
            "dec_usno": c["dec_usno"],
        }

        if c["gaia"]:
            target["gaia_dr3_source_id"] = gaia_id
            target["phot_g_mean_mag"] = c["gaia"]["phot_g_mean_mag"]
            target["bp_rp"] = c["gaia"]["bp_rp"]
            target["ruwe"] = c["gaia"]["ruwe"]

        targets_all.append(target)

    output_dir = Path("data/targets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write combined 127 (99 Table 2 + 28 Table 3)
    output_all = {
        "campaign": "vasco_vanishing_127",
        "phase": "novel_targets",
        "description": (
            "VASCO (Vanishing & Appearing Sources during a Century of Observations) "
            "project: 127 vanishing star candidates from Villarroel et al. (2020). "
            "99 from Table 2 (single-epoch red point sources) + 28 from Table 3 "
            "(highest-quality vanishing candidates). USNO B1.0 objects with no "
            "Pan-STARRS DR1 counterpart within 30\". These are stars that may have "
            "genuinely vanished — the exact physical signature of a star being "
            "enclosed by a megastructure. EXODUS checks for residual IR, radio, "
            "or other anomalies at these positions."
        ),
        "source": "https://arxiv.org/abs/1911.05068",
        "vizier_catalog": "J/AJ/159/8",
        "targets": targets_all,
    }

    with open(output_dir / "vasco_vanishing_127.json", "w") as f:
        json.dump(output_all, f, indent=2, default=str)
    print(f"  Wrote {len(targets_all)} targets → data/targets/vasco_vanishing_127.json")

    # Also write Table 3 only (28 highest quality — run these first)
    targets_t3 = [t for t in targets_all if t.get("table") and "Table 3" in t["table"]]
    output_t3 = {
        "campaign": "vasco_vanishing_t3_28",
        "phase": "novel_targets",
        "description": (
            "VASCO Table 3: 28 highest-quality vanishing star candidates. "
            "These passed the strictest quality checks in Villarroel et al. (2020). "
            "Highest priority for EXODUS analysis — run these first."
        ),
        "source": "https://arxiv.org/abs/1911.05068",
        "vizier_catalog": "J/AJ/159/8",
        "targets": targets_t3,
    }

    with open(output_dir / "vasco_vanishing_28.json", "w") as f:
        json.dump(output_t3, f, indent=2, default=str)
    print(f"  Wrote {len(targets_t3)} targets → data/targets/vasco_vanishing_28.json")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  VASCO TARGET BUILD COMPLETE")
    print("=" * 60)

    gaia_targets = [t for t in targets_all if t.get("gaia_dr3_source_id")]
    no_gaia = [t for t in targets_all if not t.get("gaia_dr3_source_id")]
    nearby = [t for t in gaia_targets if t["distance_pc"] and t["distance_pc"] < 200]
    n_t3 = len(targets_t3)

    print(f"\n  Total candidates: {len(targets_all)}")
    print(f"  Table 2 (red point sources): {len(targets_all) - n_t3}")
    print(f"  Table 3 (highest quality): {n_t3}")
    print(f"  Gaia-matched: {len(gaia_targets)} ({len(gaia_targets)/len(targets_all)*100:.0f}%)")
    print(f"  No Gaia match: {len(no_gaia)} (genuinely vanished or very faint)")
    print(f"  Within 200 pc: {len(nearby)}")

    if no_gaia:
        print(f"\n  ⚠️  {len(no_gaia)} targets have NO Gaia counterpart!")
        print("     These are the most intriguing — they may have truly vanished.")
        print("     EXODUS will check for residual IR/radio at these positions.")

    print(f"\n  Next steps:")
    print(f"  1. Quick run (28 highest quality): ./venv/bin/python scripts/run_quick.py --target-file data/targets/vasco_vanishing_28.json --tier 0")
    print(f"  2. Full run (all 127): ./venv/bin/python scripts/run_quick.py --target-file data/targets/vasco_vanishing_127.json --tier 0")
    print(f"  3. Any target with IR detection but no optical = IMMEDIATE ESCALATION")


if __name__ == "__main__":
    build_vasco_targets()
