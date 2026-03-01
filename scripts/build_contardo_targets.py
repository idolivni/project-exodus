#!/usr/bin/env python3
"""
Project EXODUS — Build Contardo & Hogg 53 FGK IR-Excess Targets
=================================================================

Downloads the 53 FGK main-sequence stars with unexplained mid-IR
excess from Contardo & Hogg (2024, arXiv:2403.18941).

These are the ONLY known FGK stars with Dyson-sphere-level IR excess
(f_d = 0.005–0.10, i.e. up to 10% of stellar luminosity) that also
have RUWE < 1.4 (not binaries). Running them through EXODUS adds UV,
radio, HR anomaly, and proper motion channels.

The paper is literally titled "Not A Technosignature Search" — but
these are exactly the stars a technosignature search should examine.

Usage
-----
    python scripts/build_contardo_targets.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger

log = get_logger("build_contardo")

# GitHub raw URL for the candidate CSV
CSV_URL = "https://raw.githubusercontent.com/contardog/NotATechnosignatureSearch/main/53candidates_small.csv"
FALLBACK_CSV_URL = "https://raw.githubusercontent.com/contardog/NotATechnosignatureSearch/master/53candidates_small.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "targets" / "contardo_53.json"


def download_csv() -> str:
    """Download the 53-candidate CSV from GitHub."""
    import urllib.request

    for url in [CSV_URL, FALLBACK_CSV_URL]:
        try:
            log.info("Downloading from %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "EXODUS/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            log.warning("Failed to download from %s: %s", url, e)

    raise RuntimeError("Could not download Contardo & Hogg CSV from GitHub")


def parse_csv(csv_text: str) -> list:
    """Parse the CSV into EXODUS target format."""
    reader = csv.DictReader(io.StringIO(csv_text))

    targets = []
    for row in reader:
        try:
            ra = float(row.get("ra") or row.get("RA") or row.get("ra_deg", 0))
            dec = float(row.get("dec") or row.get("DEC") or row.get("dec_deg", 0))
            gaia_id = row.get("source_id") or row.get("Gaia_DR3_source_id") or row.get("gaia_source_id", "")

            # Try to get distance
            dist = None
            for key in ["distance_pc", "dist_pc", "r_med_geo", "parallax"]:
                if key in row and row[key]:
                    try:
                        val = float(row[key])
                        if key == "parallax" and val > 0:
                            dist = 1000.0 / val
                        else:
                            dist = val
                        break
                    except (ValueError, TypeError):
                        pass

            target_id = f"CONTARDO_{gaia_id}" if gaia_id else f"CONTARDO_RA{ra:.4f}_DEC{dec:.4f}"

            target = {
                "target_id": target_id,
                "host_star": target_id,
                "ra": ra,
                "dec": dec,
                "hz_flag": False,
                "notes": "Contardo & Hogg 2024 — unexplained FGK IR excess (f_d up to 10%), RUWE < 1.4",
                "source_paper": "arXiv:2403.18941",
                "discovery_reason": "unexplained_ir_excess",
            }

            if gaia_id:
                target["gaia_source_id"] = int(gaia_id)
            if dist and dist > 0:
                target["distance_pc"] = round(dist, 2)

            # Preserve any additional photometry
            for key in ["phot_g_mean_mag", "G", "bp_rp", "BP_RP"]:
                if key in row and row[key]:
                    try:
                        norm_key = key.lower().replace("bp_rp", "bp_rp").replace("g", "phot_g_mean_mag") if key == "G" else key
                        target[norm_key] = round(float(row[key]), 4)
                    except (ValueError, TypeError):
                        pass

            targets.append(target)

        except Exception as e:
            log.warning("Skipping row: %s", e)
            continue

    return targets


def main():
    log.info("Building Contardo & Hogg 53 FGK IR-excess target file")

    csv_text = download_csv()
    targets = parse_csv(csv_text)

    log.info("Parsed %d targets from CSV", len(targets))

    campaign = {
        "campaign": "contardo_53_fgk",
        "description": (
            "53 FGK main-sequence stars with unexplained mid-IR excess "
            "(f_d = 0.005-0.10) from Contardo & Hogg 2024 (arXiv:2403.18941). "
            "All have RUWE < 1.4 — not binaries. Fractional luminosities up to "
            "10% are at Dyson-sphere levels. Running through 10-channel EXODUS "
            "adds UV, radio, HR, PM channels for multi-channel convergence test."
        ),
        "targets": targets,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(campaign, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  EXODUS — Contardo & Hogg 53 FGK Targets")
    print(f"{'='*60}")
    print(f"  Targets: {len(targets)}")
    if targets:
        with_dist = [t for t in targets if t.get("distance_pc")]
        if with_dist:
            dists = [t["distance_pc"] for t in with_dist]
            print(f"  Distance range: {min(dists):.1f} — {max(dists):.1f} pc")
        with_gaia = [t for t in targets if t.get("gaia_source_id")]
        print(f"  With Gaia source ID: {len(with_gaia)}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
