#!/usr/bin/env python3
"""
EXODUS — VLASS→Gaia Blind Cross-Match Campaign Builder

Strategy: Reverse the standard pipeline logic.
Instead of starting from optical stars and looking for radio,
start from the VLASS 3 GHz radio catalog and cross-match against
Gaia DR3 FGK dwarfs. Any FGK dwarf that is a VLASS source
AND has RUWE < 1.4 (not a binary) is already extraordinary.

Approach (two-step server-side cross-match):
  1. Download filtered VLASS positions via VizieR ADQL
     (catalog: J/ApJS/255/30/comp, Gordon+2021 CIRADA QL)
  2. Upload to CDS X-Match service for positional cross-match
     against Gaia DR3 (I/355/gaiadr3) within 3"
  3. Filter matches for FGK dwarfs (Teff, logg, RUWE, Plx)

Usage:
    ./venv/bin/python scripts/build_vlass_gaia_targets.py [--chunk-size N]
"""

import json
import sys
import time
import logging
import argparse
import io
import csv
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VLASS-GAIA] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import requests
except ImportError:
    log.error("Missing: requests. pip install requests")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
CDS_XMATCH = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
VLASS_TABLE = "J/ApJS/255/30/comp"  # Gordon+2021 CIRADA VLASS QL catalog
GAIA_TABLE = "vizier:I/355/gaiadr3"

MATCH_RADIUS_ARCSEC = 3.0
MIN_FLUX_MJY = 0.8
MAX_DECONV_ARCSEC = 3.0  # unresolved in VLASS beam

# Gaia FGK dwarf cuts
TEFF_MIN, TEFF_MAX = 4000, 7000
LOGG_MIN = 4.0
RUWE_MAX = 1.4

OUTPUT_PATH = ROOT / "data" / "targets" / "vlass_gaia_crossmatch.json"

# RA chunk size for VizieR ADQL (degrees)
RA_STEP = 30


def step1_download_vlass(timeout: int = 120) -> list[dict]:
    """Download filtered VLASS positions via VizieR ADQL in RA chunks."""
    log.info("Step 1: Download filtered VLASS positions via VizieR ADQL")
    log.info("  Catalog: %s", VLASS_TABLE)
    log.info("  Filters: Ftot > %.1f mJy, QualFlag=0, DupFlag=0, DCMaj < %.1f\"",
             MIN_FLUX_MJY, MAX_DECONV_ARCSEC)

    all_sources: list[dict] = []

    for ra_start in range(0, 360, RA_STEP):
        ra_end = ra_start + RA_STEP
        query = f'''
        SELECT "CompName", "RAJ2000", "DEJ2000", "Ftot", "e_Ftot", "DCMaj"
        FROM "{VLASS_TABLE}"
        WHERE "Ftot" > {MIN_FLUX_MJY}
          AND "QualFlag" = 0
          AND "DupFlag" = 0
          AND "RAJ2000" BETWEEN {ra_start} AND {ra_end}
        '''

        t0 = time.time()
        try:
            r = requests.post(
                VIZIER_TAP,
                data={
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "csv",
                    "QUERY": query,
                    "MAXREC": "500000",
                },
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            log.warning("  RA [%3d,%3d): TIMEOUT", ra_start, ra_end)
            continue
        except requests.exceptions.RequestException as e:
            log.warning("  RA [%3d,%3d): %s", ra_start, ra_end, e)
            continue

        dt = time.time() - t0

        if r.status_code != 200 or not r.text.strip():
            log.warning("  RA [%3d,%3d): HTTP %d (%.1fs)", ra_start, ra_end, r.status_code, dt)
            continue

        lines = r.text.strip().split("\n")
        n_raw = len(lines) - 1

        # Filter for unresolved sources (DCMaj < MAX or masked/zero)
        reader = csv.DictReader(io.StringIO(r.text))
        chunk: list[dict] = []
        for row in reader:
            dcmaj_s = row.get("DCMaj", "")
            if dcmaj_s:
                try:
                    if float(dcmaj_s) > MAX_DECONV_ARCSEC:
                        continue
                except ValueError:
                    pass
            chunk.append(row)

        all_sources.extend(chunk)
        log.info("  RA [%3d,%3d): %d raw -> %d unresolved (%.1fs)",
                 ra_start, ra_end, n_raw, len(chunk), dt)

    log.info("  Total filtered VLASS sources: %d", len(all_sources))
    return all_sources


def step2_xmatch_gaia(
    vlass_sources: list[dict],
    chunk_size: int = 50000,
    timeout: int = 600,
) -> list[str]:
    """Upload VLASS positions to CDS X-Match for cross-match against Gaia DR3."""
    log.info("Step 2: CDS X-Match upload against Gaia DR3 within %.0f\"",
             MATCH_RADIUS_ARCSEC)
    log.info("  Chunks of %d sources, timeout %ds per chunk", chunk_size, timeout)

    all_csv_results: list[str] = []

    for i in range(0, len(vlass_sources), chunk_size):
        chunk = vlass_sources[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(vlass_sources) + chunk_size - 1) // chunk_size

        # Build CSV for upload
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["ra", "dec", "CompName", "Ftot", "e_Ftot", "DCMaj"])
        for src in chunk:
            writer.writerow([
                src.get("RAJ2000", ""),
                src.get("DEJ2000", ""),
                src.get("CompName", ""),
                src.get("Ftot", ""),
                src.get("e_Ftot", ""),
                src.get("DCMaj", ""),
            ])
        csv_data = csv_buf.getvalue()

        log.info("  Chunk %d/%d: %d sources (%.1f MB)...",
                 chunk_num, total_chunks, len(chunk), len(csv_data) / 1024 / 1024)

        t0 = time.time()
        try:
            r = requests.post(
                CDS_XMATCH,
                data={
                    "REQUEST": "xmatch",
                    "RESPONSEFORMAT": "csv",
                    "cat2": GAIA_TABLE,
                    "distMaxArcsec": str(MATCH_RADIUS_ARCSEC),
                    "MAXREC": "2000000",
                    "selection": "best",
                    "colRA1": "ra",
                    "colDec1": "dec",
                },
                files={
                    "cat1": ("vlass_chunk.csv", csv_data, "text/csv"),
                },
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            log.warning("    TIMEOUT after %ds", timeout)
            continue
        except requests.exceptions.RequestException as e:
            log.warning("    Request failed: %s", e)
            continue

        dt = time.time() - t0

        if r.status_code == 200 and "csv" in r.headers.get("content-type", ""):
            lines = r.text.strip().split("\n")
            n_matches = len(lines) - 1
            all_csv_results.append(r.text)
            log.info("    -> %d Gaia matches (%.1fs, %.1f MB)",
                     n_matches, dt, len(r.text) / 1024 / 1024)
        else:
            log.warning("    FAILED: HTTP %d (%.1fs)", r.status_code, dt)
            # Log error details
            if "xml" in r.text[:200].lower():
                import re
                msg = re.search(r"Message:\s*(.+?)(?:\n|Trace)", r.text)
                if msg:
                    log.warning("    Server: %s", msg.group(1).strip())

    total_matches = sum(
        len(t.strip().split("\n")) - 1 for t in all_csv_results
    )
    log.info("  Total raw Gaia matches: %d", total_matches)
    return all_csv_results


def step3_filter_fgk(match_csvs: list[str]) -> list[dict]:
    """Filter CDS X-Match results for FGK dwarfs."""
    log.info("Step 3: Filter for FGK dwarfs")
    log.info("  Teff %d-%dK, logg > %.1f, RUWE < %.1f, Plx > 0",
             TEFF_MIN, TEFF_MAX, LOGG_MIN, RUWE_MAX)

    targets: list[dict] = []

    for match_csv in match_csvs:
        reader = csv.DictReader(io.StringIO(match_csv))
        for row in reader:
            try:
                # Required stellar params
                teff_s = row.get("Teff", "")
                logg_s = row.get("logg", "")
                ruwe_s = row.get("RUWE", "")
                plx_s = row.get("Plx", "")
                ftot_s = row.get("Ftot", "")

                if not all([teff_s, logg_s, ruwe_s, plx_s, ftot_s]):
                    continue

                teff = float(teff_s)
                logg_val = float(logg_s)
                ruwe = float(ruwe_s)
                plx = float(plx_s)
                ftot = float(ftot_s)

                # Apply FGK dwarf cuts
                if not (TEFF_MIN <= teff <= TEFF_MAX):
                    continue
                if not (logg_val > LOGG_MIN):
                    continue
                if not (ruwe < RUWE_MAX):
                    continue
                if not (plx > 0):
                    continue

                # Derived quantities
                sep = float(row.get("angDist", "99"))
                dist_pc = 1000.0 / plx
                source_id = row.get("Source", "")
                comp = row.get("CompName", "")
                gmag = float(row.get("Gmag", "")) if row.get("Gmag") else None
                feh_s = row.get("[Fe/H]", "")
                feh = float(feh_s) if feh_s else None
                e_ftot = float(row.get("e_Ftot", "")) if row.get("e_Ftot") else None
                gaia_ra = float(row.get("RAdeg", 0))
                gaia_dec = float(row.get("DEdeg", 0))

                target = {
                    "target_id": f"VLASS_{comp.replace(' ', '_').replace(',', '')}",
                    "host_star": f"Gaia DR3 {source_id}",
                    "ra": gaia_ra,
                    "dec": gaia_dec,
                    "hz_flag": False,
                    "notes": (
                        f"VLASS-Gaia blind cross-match: "
                        f"F_3GHz={ftot:.2f}mJy at {sep:.2f}\" offset. "
                        f"Teff={teff:.0f}K, logg={logg_val:.2f}, "
                        f"[Fe/H]={'%.2f' % feh if feh is not None else 'N/A'}, "
                        f"RUWE={ruwe:.2f}, d={dist_pc:.0f}pc, "
                        f"G={'%.1f' % gmag if gmag else 'N/A'}"
                    ),
                    "source_paper": "Gordon+2021 (CIRADA VLASS QL)",
                    "discovery_reason": "radio_fgk_dwarf_crossmatch",
                    "gaia_source_id": source_id,
                    "vlass_flux_mJy": ftot,
                    "vlass_flux_err_mJy": e_ftot,
                    "vlass_sep_arcsec": round(sep, 3),
                    "vlass_component": comp,
                    "distance_pc": round(dist_pc, 1),
                    "teff": teff,
                    "logg": logg_val,
                    "mh": feh,
                    "ruwe": ruwe,
                    "gmag": gmag,
                }
                targets.append(target)

            except (ValueError, TypeError):
                pass

    log.info("  FGK dwarfs found: %d", len(targets))
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="VLASS->Gaia cross-match builder (CDS X-Match)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50000,
        help="Number of VLASS sources per CDS X-Match upload chunk",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Timeout per X-Match chunk (seconds)",
    )
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("VLASS -> Gaia Blind Cross-Match Campaign Builder")
    log.info("  Server-side cross-match via CDS X-Match")
    log.info("=" * 70)

    t_start = time.time()

    # Step 1: Download VLASS positions
    vlass_sources = step1_download_vlass()
    if not vlass_sources:
        log.error("No VLASS sources retrieved. Check VizieR connectivity.")
        sys.exit(1)

    # Step 2: Cross-match against Gaia DR3
    match_csvs = step2_xmatch_gaia(
        vlass_sources,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
    )

    # Step 3: Filter for FGK dwarfs
    fgk_targets = step3_filter_fgk(match_csvs)

    # Sort by flux (brightest radio first)
    fgk_targets.sort(key=lambda t: t.get("vlass_flux_mJy", 0), reverse=True)

    # De-duplicate by Gaia source ID
    seen: set[str] = set()
    unique: list[dict] = []
    for t in fgk_targets:
        gid = t.get("gaia_source_id", "")
        if gid not in seen:
            seen.add(gid)
            unique.append(t)

    total_matches = sum(
        len(t.strip().split("\n")) - 1 for t in match_csvs
    )
    t_total = time.time() - t_start

    log.info("")
    log.info("=" * 70)
    log.info("RESULT: %d VLASS sources scanned (full sky)", len(vlass_sources))
    log.info("        %d positional Gaia matches within %.0f\"",
             total_matches, MATCH_RADIUS_ARCSEC)
    log.info("        %d pass FGK dwarf cuts (%d unique)",
             len(fgk_targets), len(unique))
    log.info("        Total time: %.1f min", t_total / 60)
    log.info("=" * 70)

    for i, t in enumerate(unique):
        log.info("  [%d] %s", i + 1, t["vlass_component"])
        log.info("      Gaia %s — F=%.2f mJy, sep=%.2f\", "
                 "Teff=%.0f K, logg=%.2f, RUWE=%.2f, d=%.0f pc",
                 t["gaia_source_id"], t["vlass_flux_mJy"],
                 t["vlass_sep_arcsec"], t["teff"], t["logg"],
                 t["ruwe"], t["distance_pc"])

    # Write output
    campaign = {
        "campaign": "vlass_gaia_crossmatch",
        "description": (
            f"VLASS->Gaia blind cross-match: {len(unique)} radio-detected FGK dwarfs. "
            f"Unresolved VLASS 3GHz sources (F>{MIN_FLUX_MJY}mJy, Qual=0) "
            f"matched to Gaia DR3 FGK dwarfs (Teff {TEFF_MIN}-{TEFF_MAX}K, "
            f"logg>{LOGG_MIN}, RUWE<{RUWE_MAX}) within {MATCH_RADIUS_ARCSEC:.0f}\". "
            f"Full-sky CDS X-Match: {len(vlass_sources)} VLASS sources uploaded, "
            f"{total_matches} positional matches, {len(unique)} FGK dwarfs. "
            "Ref: Gordon+2021 (CIRADA VLASS QL catalog)."
        ),
        "targets": unique,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(campaign, f, indent=2)
    log.info("Wrote %s (%d targets)", OUTPUT_PATH, len(unique))


if __name__ == "__main__":
    main()
