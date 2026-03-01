#!/usr/bin/env python3
"""
Project EXODUS — Build White Dwarf IR Survey Target List
==========================================================

Query Gaia DR3 for white dwarfs within 50 pc, cross-match with AllWISE
to find WDs with WISE detections, and prioritise those with possible
IR excess (W1-W2 > 0.1 or W3 detection).

Why WDs are a SETI blind spot
-----------------------------
The binary astrophysical template that explains EVERY current EXODUS
multi-channel detection CANNOT apply to white dwarfs — they are not on
the main sequence.  A WD with unexplained mid-IR excess cannot be
"unresolved MS companion shifting the photocentre" because the WD IS
the stellar endpoint.  Any WD hit survives the binary template **by
construction**.

WD + unexplained W3 excess = either disrupted planetary debris
(publishable astrophysics) or something weirder.

Selection criteria
------------------
- Gaia DR3 parallax > 20 mas (within 50 pc)
- parallax_over_error > 10 (reliable distance)
- Absolute G magnitude > 10 (WD locus on HR diagram)
- BP-RP < 1.0 (hot WDs preferred — cleaner IR baseline)
- AllWISE cross-match within 5 arcsec
- Priority tier: W1-W2 > 0.1 OR W3 detection (possible IR excess)
- Fallback: all WISE-detected WDs if <200 with excess
- Hard limit: 200 targets

Output: data/targets/wd_ir_survey.json

Usage
-----
    ./venv/bin/python scripts/build_wd_ir_targets.py
    ./venv/bin/python scripts/build_wd_ir_targets.py --max-distance 100
    ./venv/bin/python scripts/build_wd_ir_targets.py --bp-rp-max 1.5 --max-targets 300
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger

log = get_logger("build_wd_ir")


# =====================================================================
#  Configuration
# =====================================================================

# Declination strips to avoid Gaia TAP timeout on large queries
DEC_STRIPS = [
    (-90, -45), (-45, -15), (-15, 15), (15, 45), (45, 90),
]

# AllWISE cross-match radius (arcsec)
ALLWISE_MATCH_RADIUS = 5.0

# IR excess thresholds
W1_W2_EXCESS_THRESHOLD = 0.1   # W1-W2 color excess indicating non-stellar emission
W3_DETECTION_FLAG = True         # Any W3 detection counts as interesting


# =====================================================================
#  Helpers
# =====================================================================

def _isnan(val) -> bool:
    """Check if value is NaN or None."""
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return True


def _safe_float(val, default=None) -> Optional[float]:
    """Safely convert to float, returning default on failure."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if math.isnan(v) else v
    except (ValueError, TypeError):
        return default


# =====================================================================
#  Step 1: Query Gaia DR3 for White Dwarf candidates
# =====================================================================

def query_gaia_white_dwarfs(
    max_distance_pc: float = 50.0,
    bp_rp_max: float = 1.0,
) -> pd.DataFrame:
    """Query Gaia DR3 for white dwarfs using HR diagram locus.

    WD selection criteria:
    - parallax > 1000/max_distance_pc (within distance limit)
    - parallax_over_error > 10 (reliable distance)
    - Absolute G > 10 (faint for their temperature — WD locus)
    - bp_rp < bp_rp_max (hot WDs preferred)

    Returns a DataFrame with all Gaia columns needed.
    """
    from astroquery.gaia import Gaia

    min_parallax = 1000.0 / max_distance_pc

    all_rows = []

    for dec_lo, dec_hi in DEC_STRIPS:
        # Absolute G = phot_g_mean_mag + 5*log10(parallax) - 10
        # Want M_G > 10 => phot_g_mean_mag > 20 - 5*log10(parallax)
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
          AND bp_rp < {bp_rp_max}
          AND bp_rp IS NOT NULL
          AND dec >= {dec_lo} AND dec < {dec_hi}
        ORDER BY parallax DESC
        """

        log.info("Querying Gaia WDs: dec [%d, %d), parallax > %.1f mas, bp_rp < %.1f ...",
                 dec_lo, dec_hi, min_parallax, bp_rp_max)

        for attempt in range(3):
            try:
                job = Gaia.launch_job(adql)
                table = job.get_results()
                df = table.to_pandas()
                log.info("  -> %d WD candidates in dec strip [%d, %d)", len(df), dec_lo, dec_hi)
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
        log.error("No WD data returned from Gaia.")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["source_id"])

    # Compute absolute magnitude
    combined["abs_g"] = combined["phot_g_mean_mag"] + 5 * np.log10(combined["parallax"]) - 10
    combined["distance_pc"] = 1000.0 / combined["parallax"]

    log.info("Total WD candidates within %.0f pc (M_G > 10, bp_rp < %.1f): %d",
             max_distance_pc, bp_rp_max, len(combined))

    return combined


# =====================================================================
#  Step 2: Cross-match with AllWISE
# =====================================================================

def crossmatch_allwise(gaia_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-match Gaia WD candidates with AllWISE catalog.

    Uses CDS X-Match bulk service (IRSA-independent, fast).
    Uploads Gaia positions as CSV, cross-matches against AllWISE
    (II/328/allwise) on CDS, returns closest matches within radius.

    Falls back to VizieR per-target queries if CDS X-Match fails.

    Returns the Gaia DataFrame augmented with WISE columns.
    """
    import io
    import requests

    n_total = len(gaia_df)
    log.info("Cross-matching %d WD candidates with AllWISE via CDS X-Match (%.1f\" radius)...",
             n_total, ALLWISE_MATCH_RADIUS)

    # Initialize WISE columns
    wise_cols = ["w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro",
                 "w3mpro", "w3sigmpro", "w4mpro", "w4sigmpro",
                 "cc_flags", "ext_flg", "allwise_sep_arcsec"]
    for col in wise_cols:
        gaia_df[col] = np.nan if col not in ("cc_flags", "ext_flg") else ""
    gaia_df["wise_detected"] = False

    # Build CSV upload for CDS X-Match
    csv_buf = io.StringIO()
    csv_buf.write("source_id,ra,dec\n")
    for _, row in gaia_df.iterrows():
        csv_buf.write(f"{int(row['source_id'])},{row['ra']},{row['dec']}\n")
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    CDS_XMATCH_URL = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
    ALLWISE_CAT = "vizier:II/328/allwise"

    # CDS X-Match has a ~100k row limit; 1,707 is fine
    log.info("  Uploading %d positions to CDS X-Match...", n_total)
    batch_start = time.time()

    try:
        resp = requests.post(
            CDS_XMATCH_URL,
            data={
                "request": "xmatch",
                "distMaxArcsec": ALLWISE_MATCH_RADIUS,
                "RESPONSEFORMAT": "csv",
                "cat1": "user_table",
                "colRA1": "ra",
                "colDec1": "dec",
                "cat2": ALLWISE_CAT,
            },
            files={"user_table": ("upload.csv", csv_bytes, "text/csv")},
            timeout=300,
        )
        resp.raise_for_status()
    except Exception as e:
        log.error("CDS X-Match failed: %s — falling back to VizieR per-target", e)
        return _crossmatch_allwise_vizier(gaia_df)

    # Parse response CSV
    try:
        xmatch_df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        log.error("Failed to parse CDS X-Match response: %s", e)
        return _crossmatch_allwise_vizier(gaia_df)

    elapsed = time.time() - batch_start
    log.info("  CDS X-Match returned %d rows in %.1fs", len(xmatch_df), elapsed)

    if xmatch_df.empty:
        log.warning("  Zero matches returned from CDS X-Match")
        return gaia_df

    # Map CDS column names (they vary slightly)
    col_map = {}
    for col in xmatch_df.columns:
        cl = col.lower()
        if cl == "angdist":
            col_map["angDist"] = col
        elif cl == "w1mag" or cl == "w1mpro":
            col_map["w1mpro"] = col
        elif cl == "e_w1mag" or cl == "w1sigmpro" or cl == "e_w1mpro":
            col_map["w1sigmpro"] = col
        elif cl == "w2mag" or cl == "w2mpro":
            col_map["w2mpro"] = col
        elif cl == "e_w2mag" or cl == "w2sigmpro" or cl == "e_w2mpro":
            col_map["w2sigmpro"] = col
        elif cl == "w3mag" or cl == "w3mpro":
            col_map["w3mpro"] = col
        elif cl == "e_w3mag" or cl == "w3sigmpro" or cl == "e_w3mpro":
            col_map["w3sigmpro"] = col
        elif cl == "w4mag" or cl == "w4mpro":
            col_map["w4mpro"] = col
        elif cl == "e_w4mag" or cl == "w4sigmpro" or cl == "e_w4mpro":
            col_map["w4sigmpro"] = col
        elif cl == "ccf" or cl == "cc_flags":
            col_map["cc_flags"] = col
        elif cl == "ex" or cl == "ext_flg":
            col_map["ext_flg"] = col

    log.info("  Column mapping: %s", col_map)
    log.info("  Available columns: %s", list(xmatch_df.columns))

    # Keep only closest match per source_id
    dist_col = col_map.get("angDist", "angDist")
    if dist_col in xmatch_df.columns:
        xmatch_df = xmatch_df.sort_values(dist_col).drop_duplicates(subset=["source_id"], keep="first")

    # Merge into gaia_df
    n_matched = 0
    gaia_df = gaia_df.copy()
    gaia_df["source_id"] = gaia_df["source_id"].astype(np.int64)
    xmatch_df["source_id"] = xmatch_df["source_id"].astype(np.int64)

    for _, xrow in xmatch_df.iterrows():
        sid = int(xrow["source_id"])
        mask = gaia_df["source_id"] == sid
        if not mask.any():
            continue

        idx = gaia_df[mask].index[0]
        gaia_df.at[idx, "wise_detected"] = True
        n_matched += 1

        if dist_col in xrow.index:
            gaia_df.at[idx, "allwise_sep_arcsec"] = _safe_float(xrow[dist_col], np.nan)

        for target_col, src_col in col_map.items():
            if target_col in ("angDist",):
                continue
            if src_col in xrow.index:
                val = xrow[src_col]
                if target_col in ("cc_flags", "ext_flg"):
                    gaia_df.at[idx, target_col] = str(val) if val is not None and str(val) != "nan" else ""
                else:
                    gaia_df.at[idx, target_col] = _safe_float(val, np.nan)

    elapsed_total = time.time() - batch_start
    log.info("AllWISE cross-match complete via CDS: %d/%d matched (%.0fs total)", n_matched, n_total, elapsed_total)
    return gaia_df


def _crossmatch_allwise_vizier(gaia_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: per-target VizieR cone search for AllWISE (II/328/allwise)."""
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    Vizier.TIMEOUT = 30
    viz = Vizier(columns=["W1mag", "e_W1mag", "W2mag", "e_W2mag",
                          "W3mag", "e_W3mag", "W4mag", "e_W4mag",
                          "ccf", "ex", "RAJ2000", "DEJ2000"],
                 catalog="II/328/allwise", row_limit=5)

    n_total = len(gaia_df)
    n_matched = 0
    n_failed = 0
    batch_start = time.time()

    for i, (idx, row) in enumerate(gaia_df.iterrows()):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - batch_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            log.info("  VizieR AllWISE: %d/%d (%.1f/s) — %d matched", i + 1, n_total, rate, n_matched)

        try:
            coord = SkyCoord(ra=float(row["ra"]), dec=float(row["dec"]), unit="deg")
            tables = viz.query_region(coord, radius=ALLWISE_MATCH_RADIUS * u.arcsec)
            if tables and len(tables) > 0 and len(tables[0]) > 0:
                t = tables[0]
                cat = SkyCoord(ra=t["RAJ2000"], dec=t["DEJ2000"], unit="deg")
                seps = coord.separation(cat)
                best = seps.argmin()

                gaia_df.at[idx, "wise_detected"] = True
                gaia_df.at[idx, "allwise_sep_arcsec"] = float(seps[best].arcsec)

                for tgt, src in [("w1mpro", "W1mag"), ("w1sigmpro", "e_W1mag"),
                                 ("w2mpro", "W2mag"), ("w2sigmpro", "e_W2mag"),
                                 ("w3mpro", "W3mag"), ("w3sigmpro", "e_W3mag"),
                                 ("w4mpro", "W4mag"), ("w4sigmpro", "e_W4mag")]:
                    if src in t.colnames:
                        val = t[src][best]
                        gaia_df.at[idx, tgt] = _safe_float(val, np.nan)

                for tgt, src in [("cc_flags", "ccf"), ("ext_flg", "ex")]:
                    if src in t.colnames:
                        val = t[src][best]
                        gaia_df.at[idx, tgt] = str(val) if val is not None else ""

                n_matched += 1
        except Exception:
            n_failed += 1

        time.sleep(0.3)

    log.info("VizieR AllWISE fallback: %d/%d matched, %d failed (%.0fs)",
             n_matched, n_total, n_failed, time.time() - batch_start)
    return gaia_df


# =====================================================================
#  Step 3: Classify IR excess and build target list
# =====================================================================

def classify_ir_excess(df: pd.DataFrame) -> pd.DataFrame:
    """Classify WDs by IR excess indicators.

    Priority tiers:
    1. ir_excess_candidate: W1-W2 > threshold OR W3 detection with S/N > 3
    2. wise_detected: Has WISE detection but no clear excess
    3. gaia_only: No WISE detection (lowest priority)
    """
    df["w1_w2_color"] = np.nan
    df["has_w3"] = False
    df["ir_excess_flag"] = False
    df["priority_tier"] = "gaia_only"

    for idx in df.index:
        if not df.at[idx, "wise_detected"]:
            continue

        df.at[idx, "priority_tier"] = "wise_detected"

        w1 = _safe_float(df.at[idx, "w1mpro"])
        w2 = _safe_float(df.at[idx, "w2mpro"])
        w3 = _safe_float(df.at[idx, "w3mpro"])
        w3_err = _safe_float(df.at[idx, "w3sigmpro"])

        # Compute W1-W2 color
        if w1 is not None and w2 is not None:
            color = w1 - w2
            df.at[idx, "w1_w2_color"] = color

            if color > W1_W2_EXCESS_THRESHOLD:
                df.at[idx, "ir_excess_flag"] = True
                df.at[idx, "priority_tier"] = "ir_excess_candidate"

        # Check W3 detection (S/N > 3)
        if w3 is not None:
            if w3_err is not None and w3_err > 0 and (w3 / w3_err) > 3:
                df.at[idx, "has_w3"] = True
                df.at[idx, "ir_excess_flag"] = True
                df.at[idx, "priority_tier"] = "ir_excess_candidate"
            elif w3_err is None or w3_err == 0:
                # W3 detected but no error — still count it
                df.at[idx, "has_w3"] = True
                df.at[idx, "ir_excess_flag"] = True
                df.at[idx, "priority_tier"] = "ir_excess_candidate"

    # Log tier breakdown
    tier_counts = df["priority_tier"].value_counts()
    log.info("IR excess classification:")
    for tier, count in tier_counts.items():
        log.info("  %s: %d", tier, count)

    return df


def select_targets(
    df: pd.DataFrame,
    max_targets: int = 200,
) -> List[Dict[str, Any]]:
    """Select up to max_targets WDs, prioritising IR excess candidates.

    Selection order:
    1. All ir_excess_candidate (sorted by distance)
    2. Fill remaining with wise_detected (sorted by distance)
    3. If still room, fill with gaia_only (sorted by distance)
    """
    targets = []

    # Tier 1: IR excess candidates (highest priority)
    excess = df[df["priority_tier"] == "ir_excess_candidate"].sort_values("distance_pc")
    log.info("Tier 1 (IR excess candidates): %d", len(excess))

    for _, row in excess.iterrows():
        if len(targets) >= max_targets:
            break
        targets.append(_row_to_target(row, tier="ir_excess_candidate"))

    # Tier 2: WISE-detected but no clear excess
    if len(targets) < max_targets:
        wise_det = df[df["priority_tier"] == "wise_detected"].sort_values("distance_pc")
        log.info("Tier 2 (WISE detected, no excess): %d", len(wise_det))

        for _, row in wise_det.iterrows():
            if len(targets) >= max_targets:
                break
            targets.append(_row_to_target(row, tier="wise_detected"))

    # Tier 3: Gaia-only (lowest priority — fallback only)
    if len(targets) < max_targets:
        gaia_only = df[df["priority_tier"] == "gaia_only"].sort_values("distance_pc")
        log.info("Tier 3 (Gaia only, no WISE): %d", len(gaia_only))

        for _, row in gaia_only.iterrows():
            if len(targets) >= max_targets:
                break
            targets.append(_row_to_target(row, tier="gaia_only"))

    log.info("Selected %d targets (limit: %d)", len(targets), max_targets)
    return targets


def _row_to_target(row: pd.Series, tier: str) -> Dict[str, Any]:
    """Convert a DataFrame row to an EXODUS target dict."""
    sid = str(int(row["source_id"]))
    dist = round(float(row["distance_pc"]), 2)
    abs_g = round(float(row["abs_g"]), 3)

    # Build discovery reason
    reasons = [f"WD within {dist:.0f} pc (M_G={abs_g:.1f})"]
    if tier == "ir_excess_candidate":
        w1w2 = _safe_float(row.get("w1_w2_color"))
        has_w3 = bool(row.get("has_w3", False))
        parts = []
        if w1w2 is not None and w1w2 > W1_W2_EXCESS_THRESHOLD:
            parts.append(f"W1-W2={w1w2:.2f}")
        if has_w3:
            parts.append("W3 detected")
        if parts:
            reasons.append("IR excess: " + ", ".join(parts))
    elif tier == "wise_detected":
        reasons.append("WISE detected (no clear excess)")

    target = {
        "target_id": f"WD_IR_{sid}",
        "host_star": f"Gaia DR3 {sid}",
        "ra": round(float(row["ra"]), 6),
        "dec": round(float(row["dec"]), 6),
        "distance_pc": dist,
        "phot_g_mean_mag": round(float(row["phot_g_mean_mag"]), 3) if not _isnan(row.get("phot_g_mean_mag")) else None,
        "bp_rp": round(float(row["bp_rp"]), 3) if not _isnan(row.get("bp_rp")) else None,
        "abs_g": abs_g,
        "ruwe": round(float(row["ruwe"]), 3) if not _isnan(row.get("ruwe")) else None,
        "teff": round(float(row["teff_gspphot"]), 0) if not _isnan(row.get("teff_gspphot")) else None,
        "logg": round(float(row["logg_gspphot"]), 2) if not _isnan(row.get("logg_gspphot")) else None,
        # WISE photometry
        "W1": round(float(row["w1mpro"]), 3) if not _isnan(row.get("w1mpro")) else None,
        "W2": round(float(row["w2mpro"]), 3) if not _isnan(row.get("w2mpro")) else None,
        "W3": round(float(row["w3mpro"]), 3) if not _isnan(row.get("w3mpro")) else None,
        "W4": round(float(row["w4mpro"]), 3) if not _isnan(row.get("w4mpro")) else None,
        "w1_w2_color": round(float(row["w1_w2_color"]), 3) if not _isnan(row.get("w1_w2_color")) else None,
        "has_w3_detection": bool(row.get("has_w3", False)),
        "allwise_sep_arcsec": round(float(row["allwise_sep_arcsec"]), 2) if not _isnan(row.get("allwise_sep_arcsec")) else None,
        # Classification
        "ir_excess_flag": bool(row.get("ir_excess_flag", False)),
        "priority_tier": tier,
        "source_tier": "wd_ir_survey",
        "discovery_reason": "; ".join(reasons),
    }

    return target


# =====================================================================
#  Step 4: Save campaign file
# =====================================================================

def build_campaign_file(
    targets: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write target JSON file in EXODUS campaign format."""
    campaign = {
        "campaign": "wd_ir_survey",
        "description": (
            "White dwarf IR survey: Gaia DR3 WDs within 50 pc cross-matched "
            "with AllWISE. Prioritises WDs with IR excess (W1-W2 > 0.1 or W3 "
            "detection). Binary template cannot explain WD IR excess — any "
            "multi-channel hit survives by construction. Key population for "
            "SETI detection."
        ),
        "phase": "wd_ir_survey",
        "n_targets": len(targets),
        "selection_stats": stats,
        "targets": targets,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(campaign, f, indent=2, default=str)

    log.info("Wrote %d WD targets to %s", len(targets), output_path)


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXODUS White Dwarf IR Survey Target Builder"
    )
    parser.add_argument(
        "--max-distance", type=float, default=50.0,
        help="Maximum distance in parsecs (default: 50)"
    )
    parser.add_argument(
        "--bp-rp-max", type=float, default=1.0,
        help="Maximum BP-RP color (default: 1.0 for hot WDs)"
    )
    parser.add_argument(
        "--max-targets", type=int, default=200,
        help="Maximum number of targets (default: 200)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: data/targets/wd_ir_survey.json)"
    )
    parser.add_argument(
        "--skip-allwise", action="store_true",
        help="Skip AllWISE cross-match (Gaia-only, for testing)"
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "targets" / "wd_ir_survey.json"
    )

    # ------------------------------------------------------------------
    # Step 1: Gaia query
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  EXODUS — White Dwarf IR Survey Target Builder")
    print(f"{'='*70}")
    print(f"  Max distance:  {args.max_distance} pc")
    print(f"  BP-RP max:     {args.bp_rp_max}")
    print(f"  Max targets:   {args.max_targets}")
    print(f"  Output:        {output}")
    print(f"{'='*70}\n")

    print("[1/4] Querying Gaia DR3 for white dwarf candidates...")
    gaia_df = query_gaia_white_dwarfs(
        max_distance_pc=args.max_distance,
        bp_rp_max=args.bp_rp_max,
    )

    if gaia_df.empty:
        print("ERROR: No WD candidates found in Gaia. Aborting.")
        sys.exit(1)

    n_gaia = len(gaia_df)
    print(f"  Found {n_gaia} WD candidates in Gaia DR3")
    print(f"  Distance range: {gaia_df['distance_pc'].min():.1f} - {gaia_df['distance_pc'].max():.1f} pc")
    print(f"  Abs G range: {gaia_df['abs_g'].min():.1f} - {gaia_df['abs_g'].max():.1f}")
    if gaia_df["bp_rp"].notna().any():
        print(f"  BP-RP range: {gaia_df['bp_rp'].min():.2f} - {gaia_df['bp_rp'].max():.2f}")

    # ------------------------------------------------------------------
    # Step 2: AllWISE cross-match
    # ------------------------------------------------------------------
    if args.skip_allwise:
        print("\n[2/4] Skipping AllWISE cross-match (--skip-allwise)")
        gaia_df["wise_detected"] = False
        for col in ["w1mpro", "w2mpro", "w3mpro", "w4mpro",
                     "w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro",
                     "cc_flags", "ext_flg", "allwise_sep_arcsec"]:
            gaia_df[col] = np.nan if col not in ("cc_flags", "ext_flg") else ""
    else:
        print(f"\n[2/4] Cross-matching {n_gaia} WDs with AllWISE catalog...")
        print(f"  Match radius: {ALLWISE_MATCH_RADIUS}\"")
        print(f"  (This will take ~{n_gaia * 0.7:.0f}s at ~1.5 queries/sec)")
        gaia_df = crossmatch_allwise(gaia_df)

    n_wise = int(gaia_df["wise_detected"].sum())
    print(f"  WISE detections: {n_wise}/{n_gaia} ({100*n_wise/n_gaia:.1f}%)")

    # ------------------------------------------------------------------
    # Step 3: IR excess classification
    # ------------------------------------------------------------------
    print(f"\n[3/4] Classifying IR excess...")
    gaia_df = classify_ir_excess(gaia_df)

    n_excess = int(gaia_df["ir_excess_flag"].sum())
    n_w3 = int(gaia_df["has_w3"].sum())
    w1w2_vals = gaia_df.loc[gaia_df["w1_w2_color"].notna(), "w1_w2_color"]
    n_w1w2_excess = int((w1w2_vals > W1_W2_EXCESS_THRESHOLD).sum()) if len(w1w2_vals) > 0 else 0

    print(f"  IR excess candidates: {n_excess}")
    print(f"    W1-W2 > {W1_W2_EXCESS_THRESHOLD}: {n_w1w2_excess}")
    print(f"    W3 detected: {n_w3}")
    print(f"  WISE detected (no excess): {n_wise - n_excess}")
    print(f"  Gaia only (no WISE): {n_gaia - n_wise}")

    # ------------------------------------------------------------------
    # Step 4: Select targets and save
    # ------------------------------------------------------------------
    print(f"\n[4/4] Selecting targets (limit: {args.max_targets})...")
    targets = select_targets(gaia_df, max_targets=args.max_targets)

    if not targets:
        print("ERROR: No targets selected. Aborting.")
        sys.exit(1)

    # Compute statistics
    tier_counts = {}
    for t in targets:
        tier = t.get("priority_tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    distances = [t["distance_pc"] for t in targets]
    dist_bins = {
        "0-10pc": sum(1 for d in distances if d <= 10),
        "10-20pc": sum(1 for d in distances if 10 < d <= 20),
        "20-30pc": sum(1 for d in distances if 20 < d <= 30),
        "30-40pc": sum(1 for d in distances if 30 < d <= 40),
        "40-50pc": sum(1 for d in distances if 40 < d <= 50),
    }

    stats = {
        "n_gaia_candidates": n_gaia,
        "n_wise_detected": n_wise,
        "n_ir_excess": n_excess,
        "n_w1w2_excess": n_w1w2_excess,
        "n_w3_detected": n_w3,
        "n_selected": len(targets),
        "tier_breakdown": tier_counts,
        "distance_distribution": dist_bins,
        "max_distance_pc": args.max_distance,
        "bp_rp_max": args.bp_rp_max,
    }

    build_campaign_file(targets, stats, output)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  EXODUS — White Dwarf IR Survey: COMPLETE")
    print(f"{'='*70}")
    print(f"  Gaia WD candidates:       {n_gaia}")
    print(f"  WISE detections:          {n_wise} ({100*n_wise/max(n_gaia,1):.1f}%)")
    print(f"  IR excess candidates:     {n_excess}")
    print(f"    W1-W2 > {W1_W2_EXCESS_THRESHOLD}:            {n_w1w2_excess}")
    print(f"    W3 detected:            {n_w3}")
    print(f"  Targets selected:         {len(targets)}")
    print(f"")
    print(f"  Priority tier breakdown:")
    for tier, count in sorted(tier_counts.items()):
        print(f"    {tier:25s} {count:4d}")
    print(f"")
    print(f"  Distance distribution:")
    for bin_name, count in dist_bins.items():
        bar = "#" * (count // 2) if count > 0 else ""
        print(f"    {bin_name:10s} {count:4d} {bar}")
    print(f"")
    print(f"  Distance range: {min(distances):.1f} - {max(distances):.1f} pc")
    print(f"  Median distance: {np.median(distances):.1f} pc")
    print(f"")
    print(f"  Output: {output}")
    print(f"{'='*70}")

    # Print top 10 IR excess candidates
    excess_targets = [t for t in targets if t.get("ir_excess_flag")]
    if excess_targets:
        print(f"\n  Top IR excess candidates (up to 10):")
        print(f"  {'ID':>25s}  {'Dist':>6s}  {'M_G':>5s}  {'W1-W2':>6s}  {'W3':>5s}  Reason")
        print(f"  {'-'*25}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*30}")
        for t in excess_targets[:10]:
            w1w2_str = f"{t['w1_w2_color']:.2f}" if t.get("w1_w2_color") is not None else "  -  "
            w3_str = f"{t['W3']:.1f}" if t.get("W3") is not None else "  -  "
            dist_str = f"{t['distance_pc']:.1f}"
            mg_str = f"{t['abs_g']:.1f}"
            print(f"  {t['target_id']:>25s}  {dist_str:>6s}  {mg_str:>5s}  {w1w2_str:>6s}  {w3_str:>5s}  {t['discovery_reason'][:50]}")


if __name__ == "__main__":
    main()
