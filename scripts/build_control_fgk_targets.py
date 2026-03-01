#!/usr/bin/env python
"""
build_control_fgk_targets.py — Build a control sample of ~100 FGK stars
matched to Contardo sample properties but WITHOUT any IR-excess preselection.

Selection criteria (to match Contardo sample):
  - Teff range: 4500-6800 K (FGK stars) via teff_gspphot
  - Distance range: 500-2000 pc (matches IR-selected sample distance range)
  - Quality cuts: parallax_over_error > 5, phot_g_mean_mag < 16
  - RUWE < 1.4 (matches Contardo RUWE range)
  - NO IR preselection — purely Gaia-based selection

Strategy:
  - Query Gaia DR3 via TAP in declination strips to avoid timeout
  - Randomly select 100 targets (seed=42) from the full result set
  - Exclude any stars within 5" of a Contardo 53 sample member

Output: data/targets/control_fgk_matched.json (standard EXODUS campaign format)
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from astroquery.gaia import Gaia

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT / "data" / "targets"
OUTPUT_FILE = TARGET_DIR / "control_fgk_matched.json"
CONTARDO_FILE = TARGET_DIR / "contardo_53.json"

# ── Parameters ─────────────────────────────────────────────────────────────
TEFF_MIN = 4500
TEFF_MAX = 6800
DIST_MIN = 500    # pc
DIST_MAX = 2000   # pc
PARALLAX_SNR_MIN = 5
GMAG_MAX = 16
RUWE_MAX = 1.4
N_TARGETS = 100
SEED = 42
MATCH_RADIUS_ARCSEC = 5.0  # exclusion radius for Contardo stars

# Declination strip boundaries (6 strips of 30 degrees each, -90 to +90)
DEC_STRIPS = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]

# Limit per strip to keep queries fast — we need ~100 total, but want a large
# pool to sample from. Each strip returns up to this many rows.
ROWS_PER_STRIP = 5000


def load_contardo_positions():
    """Load Contardo 53 target positions for exclusion."""
    with open(CONTARDO_FILE) as f:
        data = json.load(f)
    positions = []
    for t in data["targets"]:
        positions.append((t["ra"], t["dec"]))
    print(f"Loaded {len(positions)} Contardo targets for exclusion")
    return np.array(positions)


def angular_separation_deg(ra1, dec1, ra2_arr, dec2_arr):
    """Compute angular separation in arcsec between (ra1,dec1) and arrays."""
    ra1r, dec1r = np.radians(ra1), np.radians(dec1)
    ra2r, dec2r = np.radians(ra2_arr), np.radians(dec2_arr)
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = np.sin(ddec / 2) ** 2 + np.cos(dec1r) * np.cos(dec2r) * np.sin(dra / 2) ** 2
    sep_rad = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return np.degrees(sep_rad) * 3600  # arcsec


def query_strip(dec_lo, dec_hi, max_rows=ROWS_PER_STRIP):
    """Query Gaia DR3 for FGK stars in a declination strip."""
    # Convert distance to parallax: parallax (mas) = 1000 / distance (pc)
    plx_min = 1000.0 / DIST_MAX  # 0.5 mas for 2000 pc
    plx_max = 1000.0 / DIST_MIN  # 2.0 mas for 500 pc

    query = f"""
    SELECT TOP {max_rows}
        source_id, ra, dec, parallax, parallax_over_error,
        phot_g_mean_mag, ruwe,
        teff_gspphot, logg_gspphot,
        1000.0 / parallax AS distance_pc
    FROM gaiadr3.gaia_source
    WHERE teff_gspphot BETWEEN {TEFF_MIN} AND {TEFF_MAX}
        AND parallax BETWEEN {plx_min} AND {plx_max}
        AND parallax_over_error > {PARALLAX_SNR_MIN}
        AND phot_g_mean_mag < {GMAG_MAX}
        AND ruwe < {RUWE_MAX}
        AND dec BETWEEN {dec_lo} AND {dec_hi}
        AND teff_gspphot IS NOT NULL
        AND logg_gspphot IS NOT NULL
    ORDER BY random_index
    """

    print(f"  Querying dec [{dec_lo:+.0f}, {dec_hi:+.0f}] ...", end=" ", flush=True)
    t0 = time.time()

    try:
        job = Gaia.launch_job(query)
        result = job.get_results()
        dt = time.time() - t0
        print(f"got {len(result)} rows in {dt:.1f}s")
        return result
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def main():
    print("=" * 70)
    print("BUILD CONTROL FGK SAMPLE — matched to Contardo properties")
    print("  Teff: {}-{} K | Dist: {}-{} pc | RUWE < {} | G < {}".format(
        TEFF_MIN, TEFF_MAX, DIST_MIN, DIST_MAX, RUWE_MAX, GMAG_MAX))
    print("  NO IR preselection — purely Gaia-based")
    print("=" * 70)

    # Load Contardo positions for exclusion
    contardo_pos = load_contardo_positions()

    # Query Gaia in declination strips
    all_rows = []
    print(f"\nQuerying Gaia DR3 in {len(DEC_STRIPS)} declination strips:")

    for dec_lo, dec_hi in DEC_STRIPS:
        result = query_strip(dec_lo, dec_hi)
        if result is not None and len(result) > 0:
            for row in result:
                all_rows.append({
                    "source_id": int(row["source_id"]),
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "distance_pc": float(row["distance_pc"]),
                    "teff_k": float(row["teff_gspphot"]),
                    "logg": float(row["logg_gspphot"]),
                    "phot_g_mean_mag": float(row["phot_g_mean_mag"]),
                    "ruwe": float(row["ruwe"]),
                    "parallax_snr": float(row["parallax_over_error"]),
                })

    total_from_gaia = len(all_rows)
    print(f"\nTotal from Gaia: {total_from_gaia} stars")

    if total_from_gaia == 0:
        print("ERROR: No stars returned from Gaia. Exiting.")
        sys.exit(1)

    # Exclude Contardo 53 members (within 5" match radius)
    print(f"\nExcluding Contardo 53 members (match radius = {MATCH_RADIUS_ARCSEC}\")")
    filtered = []
    n_excluded = 0
    for row in all_rows:
        seps = angular_separation_deg(
            row["ra"], row["dec"],
            contardo_pos[:, 0], contardo_pos[:, 1]
        )
        if np.min(seps) < MATCH_RADIUS_ARCSEC:
            n_excluded += 1
        else:
            filtered.append(row)

    print(f"  Excluded {n_excluded} Contardo matches")
    print(f"  Remaining: {len(filtered)} stars")

    if len(filtered) < N_TARGETS:
        print(f"WARNING: Only {len(filtered)} stars available, need {N_TARGETS}")
        selected = filtered
    else:
        # Random selection with fixed seed
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(filtered), size=N_TARGETS, replace=False)
        selected = [filtered[i] for i in sorted(indices)]

    print(f"\nSelected {len(selected)} control targets (seed={SEED})")

    # ── Statistics ──────────────────────────────────────────────────────────
    teffs = np.array([s["teff_k"] for s in selected])
    dists = np.array([s["distance_pc"] for s in selected])
    ruwes = np.array([s["ruwe"] for s in selected])
    gmags = np.array([s["phot_g_mean_mag"] for s in selected])
    loggs = np.array([s["logg"] for s in selected])

    print("\n" + "=" * 70)
    print("SAMPLE STATISTICS")
    print("=" * 70)
    print(f"  Total Gaia results:    {total_from_gaia}")
    print(f"  After Contardo excl.:  {len(filtered)}")
    print(f"  Selected:              {len(selected)}")
    print(f"\n  Teff (K):     min={teffs.min():.0f}  median={np.median(teffs):.0f}  "
          f"max={teffs.max():.0f}  mean={teffs.mean():.0f}  std={teffs.std():.0f}")
    print(f"  Distance (pc): min={dists.min():.0f}  median={np.median(dists):.0f}  "
          f"max={dists.max():.0f}  mean={dists.mean():.0f}  std={dists.std():.0f}")
    print(f"  RUWE:          min={ruwes.min():.3f}  median={np.median(ruwes):.3f}  "
          f"max={ruwes.max():.3f}  mean={ruwes.mean():.3f}  std={ruwes.std():.3f}")
    print(f"  G mag:         min={gmags.min():.2f}  median={np.median(gmags):.2f}  "
          f"max={gmags.max():.2f}  mean={gmags.mean():.2f}  std={gmags.std():.2f}")
    print(f"  log(g):        min={loggs.min():.2f}  median={np.median(loggs):.2f}  "
          f"max={loggs.max():.2f}  mean={loggs.mean():.2f}  std={loggs.std():.2f}")

    # Teff distribution bins
    print(f"\n  Teff distribution:")
    for lo, hi, label in [(4500, 5200, "K-type (4500-5200)"),
                           (5200, 6000, "G-type (5200-6000)"),
                           (6000, 6800, "F-type (6000-6800)")]:
        n = np.sum((teffs >= lo) & (teffs < hi))
        print(f"    {label}: {n} ({100*n/len(teffs):.0f}%)")

    # Distance distribution bins
    print(f"\n  Distance distribution:")
    for lo, hi in [(500, 800), (800, 1200), (1200, 1600), (1600, 2000)]:
        n = np.sum((dists >= lo) & (dists < hi))
        print(f"    {lo}-{hi} pc: {n} ({100*n/len(dists):.0f}%)")

    # ── Build output JSON ──────────────────────────────────────────────────
    targets = []
    for s in selected:
        targets.append({
            "target_id": f"CTRL_FGK_{s['source_id']}",
            "host_star": f"Gaia DR3 {s['source_id']}",
            "ra": round(s["ra"], 6),
            "dec": round(s["dec"], 6),
            "distance_pc": round(s["distance_pc"], 1),
            "hz_flag": False,
            "notes": "Control FGK star — no IR preselection",
            "discovery_reason": "control_sample",
            "gaia_source_id": s["source_id"],
            "teff_k": round(s["teff_k"], 1),
            "logg": round(s["logg"], 3),
            "phot_g_mean_mag": round(s["phot_g_mean_mag"], 3),
        })

    campaign = {
        "campaign": "control_fgk_matched",
        "description": (
            "100 random FGK stars at 500-2000pc, Teff 4500-6800K, RUWE<1.4. "
            "Control sample matched to Contardo properties WITHOUT IR preselection."
        ),
        "phase": "verification",
        "selection_criteria": {
            "teff_range_k": [TEFF_MIN, TEFF_MAX],
            "distance_range_pc": [DIST_MIN, DIST_MAX],
            "parallax_snr_min": PARALLAX_SNR_MIN,
            "phot_g_mean_mag_max": GMAG_MAX,
            "ruwe_max": RUWE_MAX,
            "ir_preselection": False,
            "random_seed": SEED,
            "n_gaia_pool": total_from_gaia,
            "n_after_contardo_exclusion": len(filtered),
        },
        "targets": targets,
    }

    # Ensure output directory exists
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(campaign, f, indent=2)

    print(f"\nTarget file written: {OUTPUT_FILE}")
    print(f"  {len(targets)} targets in campaign '{campaign['campaign']}'")
    print("\nDone.")


if __name__ == "__main__":
    main()
