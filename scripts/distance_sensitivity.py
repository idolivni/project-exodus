#!/usr/bin/env python3
"""
EXODUS Distance-Scaled Sensitivity Tables for Paper 1
=====================================================

Takes the injection-recovery results (tested at a reference distance) and
scales them to 10, 20, 30, and 50 pc using channel-appropriate physics:

  - Flux-based channels (ir_excess, radio_anomaly):
      SNR_obs ~ F_signal / sigma_noise ~ 1/d^2 (flux drops as 1/d^2)
      => sigma_50%(d) = sigma_50%_ref * (d / d_ref)
      i.e., the injected SNR required to achieve the same recovery
      fraction grows linearly with distance.

  - Astrometric channel (proper_motion_anomaly):
      PM_signal ~ v_perp / d => angular signal drops as 1/d
      PM_error is approximately constant at Gaia's precision
      => sigma_50%(d) = sigma_50%_ref * (d / d_ref)
      Same linear scaling, but the physical mechanism is different.

  - Transit (transit_anomaly):
      Transit depth is distance-independent (delta_F/F is geometric).
      But noise increases as sqrt(flux), so SNR ~ sqrt(F) ~ 1/d.
      => sigma_50%(d) = sigma_50%_ref * (d / d_ref)
      However, at large d the star may fall below magnitude limits entirely.

Physical quantity translations:
  - IR excess sigma -> fractional dust luminosity f_d and W3/W4 mag excess
  - PM anomaly sigma -> PM discrepancy in mas/yr
  - Radio sigma -> flux density excess in mJy
  - Transit sigma -> transit depth (percent)

Output:
  - data/reports/sensitivity_by_distance.json
  - Formatted table printed to stdout

Usage:
    ./venv/bin/python scripts/distance_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Configuration ────────────────────────────────────────────────────

DISTANCES_PC = [10, 20, 30, 50]       # target distances
D_REF_PC = 10.0                        # reference distance of injection tests

# Physical calibration constants
# These translate pipeline SNR units into physical quantities.
# Derived from WISE photometric uncertainties (AllWISE), Gaia DR3 PM
# precision, VLASS survey depth, and TESS photometric precision.

PHYSICAL_CALIBRATION = {
    "ir_excess": {
        "description": "IR excess (W3/W4)",
        "snr_to_mag_excess_W3": 0.10,     # 1 sigma ~ 0.10 mag in W3 (AllWISE typical)
        "snr_to_mag_excess_W4": 0.20,     # 1 sigma ~ 0.20 mag in W4
        "mag_excess_to_fd": 0.03,         # 1 mag W3 excess ~ 3% fractional luminosity
        "survey_depth_mag_W3": 16.6,      # AllWISE 5sigma point source W3 depth
        "survey_depth_mag_W4": 15.0,      # AllWISE 5sigma point source W4 depth
    },
    "proper_motion_anomaly": {
        "description": "Astrometric (PM/RUWE)",
        "snr_to_pm_masyr": 1.0,           # 1 sigma ~ 1.0 mas/yr PM discrepancy (Gaia DR3 floor)
        "catwise_floor_masyr": 3.0,       # CatWISE systematic floor (Marocco+2021)
        "gaia_pm_precision_masyr": 0.02,  # Gaia DR3 bright-star PM precision
    },
    "radio_anomaly": {
        "description": "Radio emission",
        "snr_to_flux_mJy": 0.12,          # 1 sigma ~ 0.12 mJy (VLASS 3 GHz rms)
        "vlass_5sigma_mJy": 0.60,         # VLASS 5sigma detection threshold
    },
    "transit_anomaly": {
        "description": "Transit/dimming",
        "snr_to_depth_pct": 0.20,         # 1 sigma ~ 0.20% transit depth (TESS bright stars)
        "tess_mag_limit": 16.0,           # TESS practical magnitude limit
    },
    "multi_channel": {
        "description": "Multi-channel convergence (2+ channels)",
        "note": "Combined; inherits scaling from dominant channel",
    },
}

# Channel scaling types
SCALING_TYPE = {
    "ir_excess": "flux",
    "radio_anomaly": "flux",
    "proper_motion_anomaly": "astrometric",
    "transit_anomaly": "transit",
    "multi_channel": "flux",  # dominated by IR in practice
}


# ── Core Functions ───────────────────────────────────────────────────

def load_injection_results() -> dict:
    """Load the injection-recovery results JSON."""
    path = PROJECT_ROOT / "data" / "reports" / "injection_recovery.json"
    with open(path) as f:
        return json.load(f)


def interpolate_recovery_threshold(
    snr_values: list[float],
    recovery_rates: list[float],
    threshold: float = 0.50,
) -> float | None:
    """
    Linearly interpolate the SNR at which recovery rate crosses `threshold`.
    Returns None if the curve never reaches the threshold.
    """
    snr_arr = np.array(snr_values)
    rate_arr = np.array(recovery_rates)

    # Find crossing points
    for i in range(len(rate_arr) - 1):
        if rate_arr[i] < threshold <= rate_arr[i + 1]:
            # Linear interpolation
            frac = (threshold - rate_arr[i]) / (rate_arr[i + 1] - rate_arr[i])
            return float(snr_arr[i] + frac * (snr_arr[i + 1] - snr_arr[i]))

    # Check if first point is already above threshold
    if len(rate_arr) > 0 and rate_arr[0] >= threshold:
        return float(snr_arr[0])

    return None


def scale_snr_threshold(snr_ref: float, d_pc: float, d_ref: float = D_REF_PC) -> float:
    """
    Scale an SNR threshold from reference distance to target distance.

    For all channel types in our pipeline, the required injected SNR to
    achieve the same recovery fraction scales linearly with distance:
      sigma_required(d) = sigma_required(d_ref) * (d / d_ref)

    Physics:
      - Flux channels: F ~ 1/d^2, noise ~ const => SNR ~ 1/d^2,
        so to maintain fixed *observed* SNR, injected signal must be
        stronger by factor (d/d_ref)^2 in flux, which is (d/d_ref)
        in the SNR-sigma space of the pipeline.
      - PM: angular signal ~ 1/d, PM error ~ const => same linear scaling
      - Transit: depth is geometric (d-independent) but photometric noise
        increases => effective SNR drops linearly with d for fixed depth
    """
    return snr_ref * (d_pc / d_ref)


def compute_physical_quantity(channel: str, snr_value: float, d_pc: float) -> dict:
    """
    Convert a pipeline SNR threshold into physical observables at distance d.

    Returns a dict of physical quantities relevant to the channel.
    """
    cal = PHYSICAL_CALIBRATION.get(channel, {})
    result = {}

    if channel == "ir_excess":
        # W3 magnitude excess
        mag_ex_W3 = snr_value * cal["snr_to_mag_excess_W3"]
        mag_ex_W4 = snr_value * cal["snr_to_mag_excess_W4"]
        # Fractional luminosity (f_d)
        fd = mag_ex_W3 * cal["mag_excess_to_fd"]
        # Covering fraction (approximate: f_d for blackbody re-emission)
        covering_frac = fd  # first-order approximation
        result = {
            "W3_mag_excess": round(mag_ex_W3, 3),
            "W4_mag_excess": round(mag_ex_W4, 3),
            "fractional_luminosity_fd": round(fd, 4),
            "covering_fraction_approx": round(covering_frac, 4),
            "unit_W3": "mag",
            "unit_fd": "L_dust/L_star",
        }

    elif channel == "proper_motion_anomaly":
        # PM discrepancy
        pm_disc = snr_value * cal["snr_to_pm_masyr"]
        # Tangential velocity (v_t = 4.74 * PM[arcsec/yr] * d[pc])
        v_t_kms = 4.74 * (pm_disc / 1000.0) * d_pc
        result = {
            "pm_discrepancy_masyr": round(pm_disc, 2),
            "tangential_velocity_kms": round(v_t_kms, 3),
            "catwise_floor_masyr": cal["catwise_floor_masyr"],
            "unit_pm": "mas/yr",
            "unit_vt": "km/s",
        }

    elif channel == "radio_anomaly":
        # Flux density at reference (scales as 1/d^2)
        flux_ref = snr_value * cal["snr_to_flux_mJy"]
        # At distance d, the intrinsic source must be stronger
        flux_at_d = flux_ref  # This is the observed flux needed at the survey
        # Luminosity at distance d
        d_cm = d_pc * 3.086e18  # pc to cm
        lum_cgs = flux_at_d * 1e-26 * 4 * np.pi * d_cm**2  # erg/s/Hz
        result = {
            "observed_flux_mJy": round(flux_at_d, 3),
            "radio_luminosity_erg_s_Hz": f"{lum_cgs:.2e}",
            "vlass_5sigma_mJy": cal["vlass_5sigma_mJy"],
            "unit_flux": "mJy",
        }

    elif channel == "transit_anomaly":
        depth_pct = snr_value * cal["snr_to_depth_pct"]
        # Approximate apparent mag of a Sun-like star at distance d
        M_V_sun = 4.83
        m_V = M_V_sun + 5 * np.log10(d_pc / 10.0)
        detectable = m_V < cal["tess_mag_limit"]
        result = {
            "transit_depth_pct": round(depth_pct, 2),
            "transit_depth_ppm": round(depth_pct * 1e4, 0),
            "approx_V_mag_solar": round(m_V, 1),
            "within_TESS_limit": detectable,
            "unit": "percent",
        }

    elif channel == "multi_channel":
        # Composite: dominated by IR channel in our pipeline
        ir_phys = compute_physical_quantity("ir_excess", snr_value, d_pc)
        result = {
            "dominant_channel": "ir_excess",
            "ir_physical": ir_phys,
            "note": "Multi-channel threshold dominated by IR; "
                    "requires 2+ channels above individual thresholds",
        }

    return result


def build_sensitivity_table(data: dict) -> dict:
    """
    Build the full distance-scaled sensitivity table.

    For each channel and each distance, computes:
      - 50% recovery SNR threshold
      - 90% recovery SNR threshold
      - Minimum detectable signal (5-sigma equivalent)
      - Physical quantity translations
    """
    curves = data["recovery_curves"]
    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_distance_pc": D_REF_PC,
            "target_distances_pc": DISTANCES_PC,
            "scaling_notes": {
                "flux_channels": "SNR_required(d) = SNR_required(d_ref) * (d / d_ref)",
                "astrometric": "PM_signal ~ 1/d, error ~ const => same linear scaling",
                "transit": "Depth is d-independent but noise grows => linear SNR scaling",
            },
            "source": "data/reports/injection_recovery.json",
            "n_trials_per_snr": data["config"]["n_trials"],
        },
        "channels": {},
    }

    for ch_name, curve in curves.items():
        snr_vals = curve["snr_values"]
        rec_rates = curve["recovery_rates"]

        # Compute reference thresholds via interpolation
        snr_50_ref = interpolate_recovery_threshold(snr_vals, rec_rates, 0.50)
        snr_90_ref = interpolate_recovery_threshold(snr_vals, rec_rates, 0.90)

        # Also find 100% threshold
        snr_100_ref = None
        for s, r in zip(snr_vals, rec_rates):
            if r >= 1.0:
                snr_100_ref = s
                break

        ch_result = {
            "channel": ch_name,
            "scaling_type": SCALING_TYPE.get(ch_name, "flux"),
            "description": PHYSICAL_CALIBRATION.get(ch_name, {}).get("description", ch_name),
            "reference": {
                "distance_pc": D_REF_PC,
                "snr_50pct": snr_50_ref,
                "snr_90pct": snr_90_ref,
                "snr_100pct": snr_100_ref,
                "false_positive_rate": data.get("false_positive_rates", {}).get(ch_name, None),
            },
            "distance_scaling": [],
        }

        for d_pc in DISTANCES_PC:
            entry = {"distance_pc": d_pc}

            # Scale thresholds
            if snr_50_ref is not None:
                snr_50_d = scale_snr_threshold(snr_50_ref, d_pc)
                entry["snr_50pct"] = round(snr_50_d, 2)
                entry["physical_50pct"] = compute_physical_quantity(ch_name, snr_50_d, d_pc)
            else:
                entry["snr_50pct"] = None
                entry["physical_50pct"] = {"note": "Channel never reaches 50% recovery"}

            if snr_90_ref is not None:
                snr_90_d = scale_snr_threshold(snr_90_ref, d_pc)
                entry["snr_90pct"] = round(snr_90_d, 2)
                entry["physical_90pct"] = compute_physical_quantity(ch_name, snr_90_d, d_pc)
            else:
                entry["snr_90pct"] = None
                entry["physical_90pct"] = {"note": "Channel never reaches 90% recovery"}

            # Minimum detectable signal at 5-sigma
            min_det_snr = 5.0 * (d_pc / D_REF_PC)
            entry["min_detectable_5sigma"] = {
                "snr": round(min_det_snr, 2),
                "physical": compute_physical_quantity(ch_name, min_det_snr, d_pc),
            }

            # Survey-specific reachability
            entry["reachable"] = assess_reachability(ch_name, d_pc, snr_50_ref)

            ch_result["distance_scaling"].append(entry)

        results["channels"][ch_name] = ch_result

    return results


def assess_reachability(channel: str, d_pc: float, snr_50_ref: float | None) -> dict:
    """
    Assess whether a channel can realistically detect signals at distance d.

    Considers survey depth limits, systematic floors, and magnitude limits.
    """
    cal = PHYSICAL_CALIBRATION.get(channel, {})
    reachable = True
    limiting_factor = None

    if channel == "ir_excess":
        # Check if a Sun-like star is above AllWISE depth at distance d
        # M_W3(Sun) ~ 3.3 (Vega), apparent W3 at d
        M_W3_sun = 3.3
        m_W3 = M_W3_sun + 5 * np.log10(d_pc / 10.0)
        if m_W3 > cal.get("survey_depth_mag_W3", 16.6) - 2.0:
            # Within 2 mag of depth limit: degraded sensitivity
            reachable = False
            limiting_factor = f"Star W3={m_W3:.1f} near AllWISE depth {cal['survey_depth_mag_W3']}"

    elif channel == "proper_motion_anomaly":
        # PM precision degrades with distance; Gaia PM errors grow
        # for fainter stars. At 50 pc a solar-type star is G~8.3
        # and PM precision is ~0.02 mas/yr (still excellent).
        # At >500 pc, systematics dominate.
        if d_pc > 200:
            reachable = False
            limiting_factor = "Gaia PM precision degrades significantly beyond 200 pc"

    elif channel == "transit_anomaly":
        M_V_sun = 4.83
        m_V = M_V_sun + 5 * np.log10(d_pc / 10.0)
        if m_V > cal.get("tess_mag_limit", 16.0):
            reachable = False
            limiting_factor = f"Star V={m_V:.1f} below TESS mag limit {cal['tess_mag_limit']}"

    elif channel == "radio_anomaly":
        # VLASS covers dec > -40 with rms ~ 0.12 mJy
        # At large distances, only very luminous sources detectable
        if snr_50_ref is not None:
            scaled_snr = snr_50_ref * (d_pc / D_REF_PC)
            flux_needed = scaled_snr * cal.get("snr_to_flux_mJy", 0.12)
            if flux_needed > 10.0:  # 10 mJy is unrealistically bright for stellar radio
                reachable = False
                limiting_factor = f"Required flux {flux_needed:.1f} mJy exceeds stellar radio range"

    return {
        "reachable": reachable,
        "limiting_factor": limiting_factor,
    }


# ── Formatted Output ────────────────────────────────────────────────

def print_summary_table(results: dict):
    """Print a formatted summary table suitable for paper inclusion."""

    print("=" * 90)
    print("  EXODUS DISTANCE-SCALED SENSITIVITY TABLE")
    print("  Reference injection-recovery at d_ref = {:.0f} pc".format(
        results["metadata"]["reference_distance_pc"]))
    print("=" * 90)

    channels_order = ["ir_excess", "proper_motion_anomaly", "transit_anomaly",
                      "radio_anomaly", "multi_channel"]

    # ── Table 1: SNR Thresholds by Distance ──
    print("\n  TABLE 1: 50% Recovery SNR Threshold by Distance")
    print("  " + "-" * 75)
    header = f"  {'Channel':<28}"
    for d in DISTANCES_PC:
        header += f" {'d=' + str(d) + 'pc':>10}"
    print(header)
    print("  " + "-" * 75)

    for ch in channels_order:
        if ch not in results["channels"]:
            continue
        ch_data = results["channels"][ch]
        desc = ch_data["description"]
        row = f"  {desc:<28}"
        for entry in ch_data["distance_scaling"]:
            val = entry.get("snr_50pct")
            if val is not None:
                row += f" {val:>10.1f}"
            else:
                row += f" {'N/R':>10}"
        print(row)

    # ── Table 2: Physical Quantities at 50% Recovery ──
    print("\n\n  TABLE 2: Physical Quantities at 50% Recovery Threshold")
    print("  " + "-" * 80)

    for ch in channels_order:
        if ch not in results["channels"]:
            continue
        ch_data = results["channels"][ch]
        print(f"\n  {ch_data['description']}:")

        if ch == "ir_excess":
            header2 = f"    {'Distance':>10} {'SNR_50%':>8} {'W3 excess':>10} {'W4 excess':>10} {'f_d':>10} {'Covering':>10}"
            print(header2)
            print("    " + "-" * 65)
            for entry in ch_data["distance_scaling"]:
                d = entry["distance_pc"]
                snr = entry.get("snr_50pct")
                if snr is None:
                    print(f"    {d:>7} pc {'N/R':>8} {'---':>10} {'---':>10} {'---':>10} {'---':>10}")
                    continue
                phys = entry["physical_50pct"]
                print(f"    {d:>7} pc {snr:>8.1f} {phys['W3_mag_excess']:>9.2f}m {phys['W4_mag_excess']:>9.2f}m"
                      f" {phys['fractional_luminosity_fd']:>9.3f} {phys['covering_fraction_approx']:>9.3f}")

        elif ch == "proper_motion_anomaly":
            header2 = f"    {'Distance':>10} {'SNR_50%':>8} {'PM disc.':>10} {'v_t':>10}"
            print(header2)
            print("    " + "-" * 45)
            for entry in ch_data["distance_scaling"]:
                d = entry["distance_pc"]
                snr = entry.get("snr_50pct")
                if snr is None:
                    print(f"    {d:>7} pc {'N/R':>8} {'---':>10} {'---':>10}")
                    continue
                phys = entry["physical_50pct"]
                print(f"    {d:>7} pc {snr:>8.1f} {phys['pm_discrepancy_masyr']:>8.1f} mas/yr"
                      f" {phys['tangential_velocity_kms']:>8.2f} km/s")

        elif ch == "radio_anomaly":
            header2 = f"    {'Distance':>10} {'SNR_50%':>8} {'Flux':>10} {'L_radio':>18}"
            print(header2)
            print("    " + "-" * 50)
            for entry in ch_data["distance_scaling"]:
                d = entry["distance_pc"]
                snr = entry.get("snr_50pct")
                if snr is None:
                    print(f"    {d:>7} pc {'N/R':>8} {'---':>10} {'---':>18}")
                    continue
                phys = entry["physical_50pct"]
                print(f"    {d:>7} pc {snr:>8.1f} {phys['observed_flux_mJy']:>8.2f} mJy"
                      f" {phys['radio_luminosity_erg_s_Hz']:>18} erg/s/Hz")

        elif ch == "transit_anomaly":
            header2 = f"    {'Distance':>10} {'SNR_50%':>8} {'Depth':>10} {'V_mag':>8} {'TESS OK':>8}"
            print(header2)
            print("    " + "-" * 50)
            for entry in ch_data["distance_scaling"]:
                d = entry["distance_pc"]
                snr = entry.get("snr_50pct")
                if snr is None:
                    print(f"    {d:>7} pc {'N/R':>8} {'---':>10} {'---':>8} {'---':>8}")
                    continue
                phys = entry["physical_50pct"]
                ok = "Yes" if phys.get("within_TESS_limit", False) else "No"
                print(f"    {d:>7} pc {snr:>8.1f} {phys['transit_depth_pct']:>8.1f}%"
                      f" {phys['approx_V_mag_solar']:>8.1f} {ok:>8}")

        elif ch == "multi_channel":
            print("    (Inherits IR excess scaling; requires 2+ channels above threshold)")
            header2 = f"    {'Distance':>10} {'SNR_50%':>8} {'IR f_d':>10}"
            print(header2)
            print("    " + "-" * 35)
            for entry in ch_data["distance_scaling"]:
                d = entry["distance_pc"]
                snr = entry.get("snr_50pct")
                if snr is None:
                    print(f"    {d:>7} pc {'N/R':>8} {'---':>10}")
                    continue
                phys = entry.get("physical_50pct", {})
                ir_phys = phys.get("ir_physical", {})
                fd = ir_phys.get("fractional_luminosity_fd", "---")
                if isinstance(fd, float):
                    print(f"    {d:>7} pc {snr:>8.1f} {fd:>9.3f}")
                else:
                    print(f"    {d:>7} pc {snr:>8.1f} {'---':>10}")

    # ── Table 3: Minimum Detectable Signal at 5sigma ──
    print("\n\n  TABLE 3: Minimum Detectable Signal at 5-sigma by Distance")
    print("  " + "-" * 80)

    for ch in ["ir_excess", "proper_motion_anomaly", "radio_anomaly", "transit_anomaly"]:
        if ch not in results["channels"]:
            continue
        ch_data = results["channels"][ch]
        print(f"\n  {ch_data['description']}:")

        for entry in ch_data["distance_scaling"]:
            d = entry["distance_pc"]
            det = entry["min_detectable_5sigma"]
            snr = det["snr"]
            phys = det["physical"]

            if ch == "ir_excess":
                print(f"    d={d:>3} pc: 5sigma at SNR={snr:>5.1f} => "
                      f"W3 excess={phys['W3_mag_excess']:.2f} mag, "
                      f"f_d={phys['fractional_luminosity_fd']:.4f}")
            elif ch == "proper_motion_anomaly":
                print(f"    d={d:>3} pc: 5sigma at SNR={snr:>5.1f} => "
                      f"PM={phys['pm_discrepancy_masyr']:.1f} mas/yr, "
                      f"v_t={phys['tangential_velocity_kms']:.2f} km/s")
            elif ch == "radio_anomaly":
                print(f"    d={d:>3} pc: 5sigma at SNR={snr:>5.1f} => "
                      f"flux={phys['observed_flux_mJy']:.2f} mJy, "
                      f"L_radio={phys['radio_luminosity_erg_s_Hz']} erg/s/Hz")
            elif ch == "transit_anomaly":
                ok = "Yes" if phys.get("within_TESS_limit", False) else "No"
                print(f"    d={d:>3} pc: 5sigma at SNR={snr:>5.1f} => "
                      f"depth={phys['transit_depth_pct']:.1f}%, "
                      f"V={phys['approx_V_mag_solar']:.1f} mag, TESS={ok}")

    # ── Table 4: Reachability Summary ──
    print("\n\n  TABLE 4: Reachability Summary")
    print("  " + "-" * 75)
    header3 = f"  {'Channel':<28}"
    for d in DISTANCES_PC:
        header3 += f" {'d=' + str(d) + 'pc':>10}"
    print(header3)
    print("  " + "-" * 75)

    for ch in channels_order:
        if ch not in results["channels"]:
            continue
        ch_data = results["channels"][ch]
        desc = ch_data["description"]
        row = f"  {desc:<28}"
        for entry in ch_data["distance_scaling"]:
            r = entry["reachable"]
            status = "Yes" if r["reachable"] else "No"
            row += f" {status:>10}"
        print(row)

    # ── LaTeX Table ──
    print_latex_table(results)


def print_latex_table(results: dict):
    """Print LaTeX-ready table for Paper 1."""
    print("\n\n  === LaTeX TABLE (Paper 1) ===\n")
    print(r"  \begin{table*}")
    print(r"  \centering")
    print(r"  \caption{Distance-scaled detection sensitivity for EXODUS channels.")
    print(r"  The 50\% recovery SNR threshold from injection-recovery tests is")
    print(r"  scaled linearly with distance. Physical quantities are computed at")
    print(r"  each distance using survey-calibrated noise properties.}")
    print(r"  \label{tab:distance_sensitivity}")
    print(r"  \begin{tabular}{lccccl}")
    print(r"  \hline")
    print(r"  Channel & \multicolumn{4}{c}{50\% Recovery SNR} & Key Physical Threshold \\")
    header_dists = " & ".join([f"$d={d}$ pc" for d in DISTANCES_PC])
    print(f"   & {header_dists} & \\\\")
    print(r"  \hline")

    channel_order = ["ir_excess", "proper_motion_anomaly",
                     "transit_anomaly", "radio_anomaly", "multi_channel"]
    latex_names = {
        "ir_excess": "IR excess (W3/W4)",
        "proper_motion_anomaly": "Astrometric (PM)",
        "transit_anomaly": "Transit/dimming",
        "radio_anomaly": "Radio emission",
        "multi_channel": "Multi-channel ($\\geq$2 ch)",
    }

    for ch in channel_order:
        if ch not in results["channels"]:
            continue
        ch_data = results["channels"][ch]
        name = latex_names.get(ch, ch)

        vals = []
        for entry in ch_data["distance_scaling"]:
            snr = entry.get("snr_50pct")
            if snr is not None:
                vals.append(f"${snr:.1f}$")
            else:
                vals.append("N/R")

        # Key physical threshold at 20 pc
        phys_str = ""
        entry_20 = None
        for entry in ch_data["distance_scaling"]:
            if entry["distance_pc"] == 20:
                entry_20 = entry
                break

        if entry_20 and entry_20.get("snr_50pct") is not None:
            p50 = entry_20.get("physical_50pct", {})
            if ch == "ir_excess":
                phys_str = f"$f_d = {p50.get('fractional_luminosity_fd', 0):.3f}$ at 20 pc"
            elif ch == "proper_motion_anomaly":
                phys_str = f"$\\Delta\\mu = {p50.get('pm_discrepancy_masyr', 0):.1f}$ mas/yr"
            elif ch == "transit_anomaly":
                phys_str = f"depth $= {p50.get('transit_depth_pct', 0):.1f}$\\%"
            elif ch == "radio_anomaly":
                phys_str = f"$F_\\nu = {p50.get('observed_flux_mJy', 0):.2f}$ mJy"
            elif ch == "multi_channel":
                ir_p = p50.get("ir_physical", {})
                phys_str = f"$f_d = {ir_p.get('fractional_luminosity_fd', 0):.3f}$ (IR dom.)"
        else:
            phys_str = "not reached"

        val_str = " & ".join(vals)
        print(f"  {name} & {val_str} & {phys_str} \\\\")

    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table*}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  EXODUS Distance Sensitivity Analysis")
    print("  Scaling injection-recovery results to multiple distances")
    print("=" * 90)

    # Load
    data = load_injection_results()
    n_channels = len(data["recovery_curves"])
    n_trials = data["total_trials"]
    print(f"\n  Loaded injection-recovery results: {n_channels} channels, {n_trials} total trials")

    # Build
    results = build_sensitivity_table(data)

    # Save
    outpath = PROJECT_ROOT / "data" / "reports" / "sensitivity_by_distance.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Convert any non-serializable types
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=sanitize)
    print(f"\n  Saved: {outpath}")

    # Print tables
    print_summary_table(results)

    print("\n" + "=" * 90)
    print("  Distance sensitivity analysis complete.")
    print(f"  Output: {outpath}")
    print("=" * 90)


if __name__ == "__main__":
    main()
