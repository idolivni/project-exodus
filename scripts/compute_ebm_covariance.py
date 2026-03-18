#!/usr/bin/env python3
"""
Compute the EBM covariance matrix from matched controls.

Re-scores all matched controls through all 6 detection channels,
building a (n_controls, 6) score matrix.  Computes the covariance
of -2*log(p) statistics for use with Empirical Brown's Method.

Output: data/reports/ebm_covariance.json
    - covariance_matrix: 6x6 list-of-lists
    - correlation_matrix: 6x6 list-of-lists
    - channel_names: ordered list of channel names
    - n_controls: number of controls scored
    - n_valid_per_channel: number of non-NaN entries per channel
    - control_p_matrix: (n_controls, 6) list-of-lists (for verification)

Usage:
    ./venv/bin/python scripts/compute_ebm_covariance.py
    ./venv/bin/python scripts/compute_ebm_covariance.py --target-file data/targets/contardo_53.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
# ThreadPoolExecutor removed — queries run directly to avoid
# "signal only works in main thread" errors with IRSA/VizieR
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, safe_json_dump
from src.core.controls import select_matched_controls
from src.core.statistics import compute_ebm_covariance, calibrate_score_to_pvalue
from src.scoring.exodus_score import EXODUSScorer

log = get_logger("ebm_covariance")

# Channel order (must match scorer)
CHANNELS = [
    "ir_excess",
    "proper_motion_anomaly",
    "hr_anomaly",
    "uv_anomaly",
    "radio_emission",
    "ir_variability",
]

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def load_targets(target_file: Path) -> list:
    """Load target list from JSON."""
    from src.ingestion.target_loader import load_target_file
    ct = load_target_file(str(target_file))
    return ct.targets


def build_control_cohort(targets: list) -> list:
    """Select matched controls using same logic as run_quick.py."""
    from src.ingestion.gaia_query import (
        cone_search as gaia_cone_search,
    )
    try:
        from src.ingestion.gaia_query import batch_cone_search as gaia_batch_cone_search
    except ImportError:
        gaia_batch_cone_search = None

    log.info("Selecting matched controls for %d targets ...", len(targets))

    # Build Gaia field catalog (same as run_quick.py)
    ras = [t.get("ra", 0.0) for t in targets]
    decs = [t.get("dec", 0.0) for t in targets]
    spread_deg = max(np.ptp(ras), np.ptp(decs))

    catalog_dicts = []
    if spread_deg < 5.0:
        median_ra = float(np.median(ras))
        median_dec = float(np.median(decs))
        log.info("  Clustered targets (spread=%.1f°), single cone", spread_deg)
        field_df = gaia_cone_search(median_ra, median_dec, radius_arcsec=1800.0, top_n=500)
        if field_df is not None and not field_df.empty:
            for _, row in field_df.iterrows():
                catalog_dicts.append(row.to_dict())
    else:
        log.info("  Spread targets (%.1f°), per-target cones", spread_deg)
        seen = set()
        if gaia_batch_cone_search is not None:
            positions = [(t["ra"], t["dec"]) for t in targets]
            batch_results = gaia_batch_cone_search(
                positions, radius_arcsec=600.0, top_n_per_position=100, batch_size=10,
            )
            for i, _t in enumerate(targets):
                field_df = batch_results.get(i)
                if field_df is not None and not field_df.empty:
                    for _, row in field_df.iterrows():
                        sid = row.get("source_id")
                        if sid not in seen:
                            seen.add(sid)
                            catalog_dicts.append(row.to_dict())
        else:
            for t in targets:
                field_df = gaia_cone_search(t["ra"], t["dec"], radius_arcsec=600.0, top_n=100)
                if field_df is not None and not field_df.empty:
                    for _, row in field_df.iterrows():
                        sid = row.get("source_id")
                        if sid not in seen:
                            seen.add(sid)
                            catalog_dicts.append(row.to_dict())

    log.info("  Field catalog: %d stars", len(catalog_dicts))

    # Compute distance_pc and b_gal for matching
    for d in catalog_dicts:
        plx = d.get("parallax")
        d["distance_pc"] = 1000.0 / float(plx) if plx and float(plx) > 0 else None
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(ra=d["ra"] * u.deg, dec=d["dec"] * u.deg)
            d["b_gal"] = float(c.galactic.b.deg)
        except Exception:
            d["b_gal"] = 0.0

    # Augment targets with Gaia data (needed for matching)
    target_match = []
    log.info("  Augmenting %d targets with Gaia data ...", len(targets))
    for t in targets:
        td = {"ra": t["ra"], "dec": t["dec"]}
        # Query Gaia to get photometry + astrometry
        gaia_df = gaia_cone_search(t["ra"], t["dec"], radius_arcsec=5.0, top_n=1)
        if gaia_df is not None and not gaia_df.empty:
            row = gaia_df.iloc[0]
            for key in ("phot_g_mean_mag", "bp_rp", "parallax", "source_id",
                         "ruwe", "phot_bp_mean_mag", "phot_rp_mean_mag",
                         "pmra", "pmdec"):
                # Handle case-insensitive column names
                for col in gaia_df.columns:
                    if col.lower() == key.lower():
                        val = row[col]
                        if val is not None and np.isfinite(float(val)):
                            td[key] = float(val)
                        break
        # Compute derived fields
        plx = td.get("parallax")
        td["distance_pc"] = 1000.0 / plx if plx and plx > 0 else None
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg)
            td["b_gal"] = float(c.galactic.b.deg)
        except Exception:
            td["b_gal"] = 0.0
        target_match.append(td)
    n_complete = sum(1 for td in target_match
                     if all(td.get(k) is not None for k in ["phot_g_mean_mag", "bp_rp", "distance_pc", "b_gal"]))
    log.info("  %d/%d targets with complete matching features", n_complete, len(targets))

    cohort = select_matched_controls(
        target_match, catalog_dicts,
        n_per_target=10,
        match_on=["phot_g_mean_mag", "bp_rp", "distance_pc", "b_gal"],
        target_id_key="source_id",
    )
    log.info("Selected %d matched controls", len(cohort.controls))
    return cohort.controls


def score_control(ctrl: dict, channel: str, timeout: int = 10) -> float:
    """Score a single control star on a single channel. Returns raw score."""

    ctrl_ra = ctrl.get("ra")
    ctrl_dec = ctrl.get("dec")
    if ctrl_ra is None or ctrl_dec is None:
        return float("nan")

    try:
        if channel == "ir_excess":
            from src.ingestion.ir_surveys import get_2mass, get_wise
            from src.processing.ir_excess import compute_ir_excess

            ir = {"ra": ctrl_ra, "dec": ctrl_dec}
            _2m = get_2mass(ctrl_ra, ctrl_dec, radius_arcsec=5.0)
            if _2m:
                for b in ("J", "H", "Ks", "J_err", "H_err", "Ks_err"):
                    if b in _2m:
                        ir[b] = _2m[b]
            _wi = get_wise(ctrl_ra, ctrl_dec, radius_arcsec=5.0)
            if _wi:
                for b in ("W1", "W2", "W3", "W4",
                          "W1_err", "W2_err", "W3_err", "W4_err"):
                    if b in _wi:
                        ir[b] = _wi[b]
            # Add Gaia optical bands
            for gk, ok in [("phot_g_mean_mag", "G"),
                           ("phot_bp_mean_mag", "BP"),
                           ("phot_rp_mean_mag", "RP")]:
                v = ctrl.get(gk)
                if v is not None and np.isfinite(v):
                    ir[ok] = float(v)
                    ir[ok + "_err"] = 0.01
            ir_result = compute_ir_excess(ir)
            return EXODUSScorer._get_ir_excess_score({
                "sigma_W3": ir_result.sigma_W3,
                "sigma_W4": ir_result.sigma_W4,
                "is_candidate": ir_result.is_candidate,
                "excess_W3": ir_result.excess_W3,
                "excess_W4": ir_result.excess_W4,
            })

        elif channel == "proper_motion_anomaly":
            ruwe = ctrl.get("ruwe", 1.0)
            if ruwe is not None and np.isfinite(ruwe):
                return EXODUSScorer._get_astrometric_score({
                    "ruwe": float(ruwe),
                    "astrometric_excess_noise_sig": 0.0,
                })
            return float("nan")

        elif channel == "hr_anomaly":
            from src.detection.hr_anomaly import compute_hr_anomaly
            ctrl_gaia = {
                "teff_gspphot": ctrl.get("teff_gspphot"),
                "logg_gspphot": ctrl.get("logg_gspphot"),
                "mh_gspphot": ctrl.get("mh_gspphot"),
                "bp_rp": ctrl.get("bp_rp"),
                "phot_g_mean_mag": ctrl.get("phot_g_mean_mag"),
            }
            ctrl_astro = {
                "parallax": ctrl.get("parallax"),
                "ruwe": ctrl.get("ruwe", 1.0),
            }
            plx = ctrl.get("parallax")
            dist = 1000.0 / plx if plx and plx > 0 else None
            hr_res = compute_hr_anomaly(
                gaia_params=ctrl_gaia,
                astrometry=ctrl_astro,
                distance_pc=dist,
            )
            return EXODUSScorer._get_hr_anomaly_score(hr_res.to_dict())

        elif channel == "uv_anomaly":
            from src.detection.uv_anomaly import compute_uv_anomaly
            from src.ingestion.galex_catalog import query_galex_cone, compute_uv_metrics

            galex_raw = query_galex_cone(ctrl_ra, ctrl_dec, 30.0)
            if not galex_raw:
                return float("nan")  # No GALEX coverage
            ctrl_gaia = {
                "teff_gspphot": ctrl.get("teff_gspphot"),
                "bp_rp": ctrl.get("bp_rp"),
                "phot_g_mean_mag": ctrl.get("phot_g_mean_mag"),
            }
            uv_met = compute_uv_metrics(galex_raw, gaia_params=ctrl_gaia)
            uv_res = compute_uv_anomaly(
                uv_metrics=uv_met, galex_raw=galex_raw, ir_excess_data=None,
            )
            return EXODUSScorer._get_uv_anomaly_score(uv_res.to_dict())

        elif channel == "radio_emission":
            from src.detection.radio_emission import compute_radio_emission
            from src.ingestion.vlass_catalog import query_radio_continuum

            radio_raw = query_radio_continuum(ctrl_ra, ctrl_dec, 15.0)
            if not radio_raw:
                return 0.0  # No radio source = valid null
            plx = ctrl.get("parallax")
            dist = 1000.0 / plx if plx and plx > 0 else None
            re_res = compute_radio_emission(
                radio_continuum=radio_raw, distance_pc=dist,
            )
            return EXODUSScorer._get_radio_emission_score(re_res.to_dict())

        elif channel == "ir_variability":
            # NEOWISE TAP is down — skip entirely for now
            return float("nan")

    except Exception as exc:
        log.debug("Channel %s failed for control: %s", channel, exc)
        return float("nan")

    return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Compute EBM covariance matrix")
    parser.add_argument(
        "--target-file",
        default=str(PROJECT_ROOT / "data" / "targets" / "contardo_53.json"),
        help="Target file to match controls against",
    )
    parser.add_argument(
        "--existing-controls",
        default=None,
        help="Path to JSON with pre-saved controls (skip Gaia query)",
    )
    args = parser.parse_args()

    t0 = time.time()
    log.info("=== EBM Covariance Computation ===")

    # Load or build control cohort
    if args.existing_controls and Path(args.existing_controls).exists():
        log.info("Loading existing controls from %s", args.existing_controls)
        with open(args.existing_controls) as f:
            controls = json.load(f)
    else:
        targets = load_targets(Path(args.target_file))
        controls = build_control_cohort(targets)
        # Save controls for future re-use
        controls_path = REPORTS_DIR / "ebm_controls.json"
        with open(controls_path, "w") as fp:
            safe_json_dump(controls, fp)
        log.info("Saved %d controls to %s", len(controls), controls_path)

    n_controls = len(controls)
    n_channels = len(CHANNELS)
    log.info("Scoring %d controls × %d channels ...", n_controls, n_channels)

    # Load existing control scores for calibration
    # (needed to convert raw scores → p-values)
    cal_report = REPORTS_DIR / "uv_irvar_calibration.json"
    existing_control_scores = {}
    if cal_report.exists():
        with open(cal_report) as f:
            cal_data = json.load(f)
        existing_control_scores = cal_data.get("control_scores", {})
    quick_report = REPORTS_DIR / "quick_run_20260228_214340.json"
    if quick_report.exists():
        with open(quick_report) as f:
            qr = json.load(f)
        # The quick_run doesn't store control_scores in the report directly;
        # they're passed at runtime. We'll use the UV/IRvar calibration ones.

    # Build raw score matrix
    raw_scores = np.full((n_controls, n_channels), np.nan, dtype=np.float64)

    for ci, ctrl in enumerate(controls):
        if (ci + 1) % 50 == 0 or ci == 0:
            elapsed = time.time() - t0
            log.info(
                "  Control %d/%d (%.1f min elapsed) ...",
                ci + 1, n_controls, elapsed / 60,
            )

        for chi, ch_name in enumerate(CHANNELS):
            raw_scores[ci, chi] = score_control(ctrl, ch_name, timeout=10)

    # Convert raw scores to calibrated p-values using existing control distributions
    p_matrix = np.ones((n_controls, n_channels), dtype=np.float64)
    for chi, ch_name in enumerate(CHANNELS):
        col = raw_scores[:, chi]
        valid = ~np.isnan(col)
        n_valid = np.sum(valid)
        log.info(
            "  Channel %s: %d/%d controls with data",
            ch_name, n_valid, n_controls,
        )

        if n_valid < 10:
            log.warning("  %s: too few controls (%d), using p=1.0", ch_name, n_valid)
            continue

        # Use the valid scores themselves as the null distribution
        # (leave-one-out calibration to avoid circularity)
        valid_scores = col[valid]
        for ci in range(n_controls):
            if np.isnan(col[ci]):
                p_matrix[ci, chi] = 1.0  # no data → null
            else:
                # Leave-one-out: compare to all OTHER controls
                others = np.concatenate([valid_scores[:ci], valid_scores[ci + 1:]]) \
                    if ci < len(valid_scores) else valid_scores
                p_val = calibrate_score_to_pvalue(col[ci], others.tolist())
                p_matrix[ci, chi] = max(p_val, 1e-300)

    # Compute covariance using the statistics module
    cov = compute_ebm_covariance(p_matrix)

    # Also compute correlation for reporting
    t_mat = -2.0 * np.log(np.clip(p_matrix, 1e-300, 1.0))
    corr = np.corrcoef(t_mat, rowvar=False)

    elapsed = time.time() - t0
    log.info("Covariance computed in %.1f min", elapsed / 60)
    log.info("Channel order: %s", CHANNELS)
    log.info("Covariance diagonal: %s", [f"{cov[i,i]:.3f}" for i in range(n_channels)])
    log.info("Off-diagonal correlations:")
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            log.info("  %s ↔ %s: r=%.3f", CHANNELS[i], CHANNELS[j], corr[i, j])

    # Save results
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_controls": n_controls,
        "channel_names": CHANNELS,
        "n_valid_per_channel": {
            ch: int(np.sum(~np.isnan(raw_scores[:, i])))
            for i, ch in enumerate(CHANNELS)
        },
        "covariance_matrix": cov.tolist(),
        "correlation_matrix": corr.tolist(),
        "elapsed_sec": elapsed,
        # Store p-matrix for verification (can be large)
        "control_p_matrix": p_matrix.tolist(),
    }

    out_path = REPORTS_DIR / "ebm_covariance.json"
    with open(out_path, "w") as fp:
        safe_json_dump(result, fp)
    log.info("Saved to %s", out_path)

    # Print summary table
    print("\n=== EBM Covariance Matrix ===")
    print(f"{'':20s}", end="")
    for ch in CHANNELS:
        print(f"{ch[:12]:>13s}", end="")
    print()
    for i, ch_i in enumerate(CHANNELS):
        print(f"{ch_i:20s}", end="")
        for j in range(n_channels):
            print(f"{cov[i,j]:13.3f}", end="")
        print()

    print("\n=== Correlation Matrix ===")
    print(f"{'':20s}", end="")
    for ch in CHANNELS:
        print(f"{ch[:12]:>13s}", end="")
    print()
    for i, ch_i in enumerate(CHANNELS):
        print(f"{ch_i:20s}", end="")
        for j in range(n_channels):
            print(f"{corr[i,j]:13.3f}", end="")
        print()


if __name__ == "__main__":
    main()
