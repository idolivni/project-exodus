"""
Project EXODUS — Background Galaxy Contamination Check
======================================================

Mid-IR excess is the primary Dyson sphere detection channel, but the W3/W4
bands (12/22 µm) have large beams (6.5"/12") that can pick up emission
from background galaxies projected near the target star.  This module
quantifies the contamination probability using:

1. **WISE colour diagnostics**: Background galaxies have characteristic
   W1-W2, W2-W3, W3-W4 colours (Stern et al. 2012; Wright et al. 2010).
   Stellar photospheres do not.

2. **Source density estimation**: Higher background galaxy density at
   low galactic latitudes and near galaxy clusters increases contamination
   probability.

3. **W4 beam confusion**: The 12" W4 beam can contain multiple WISE
   sources.  We flag cases where the AllWISE catalog shows neighbors
   within 12" that could contribute flux.

4. **Extended source flag**: AllWISE ext_flg indicates the source is
   resolved / extended — consistent with a galaxy.

This module operates on data already gathered by the pipeline (no new
network queries).  It's designed to plug into both the RedTeamEngine
and standalone scripts.

Usage
-----
    from src.vetting.galaxy_contamination import check_galaxy_contamination

    result = check_galaxy_contamination(target_data)
    if result.contamination_likely:
        print(f"IR excess likely from background galaxy: {result.explanation}")
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("vetting.galaxy_contamination")


# ── WISE colour cuts for galaxy identification ──────────────────────
# Based on Stern+2012, Wright+2010, Mateos+2012

# AGN/galaxy locus in WISE color space
W1_W2_AGN_THRESHOLD = 0.8     # W1-W2 > 0.8 → likely AGN (Stern+2012)
W2_W3_GALAXY_MIN = 2.0        # W2-W3 > 2.0 typical of star-forming galaxies
W3_W4_GALAXY_MIN = 1.5        # W3-W4 > 1.5 typical of dusty galaxies

# Stellar photosphere colors (main sequence)
W1_W2_STELLAR_MAX = 0.3       # Normal stars have W1-W2 < 0.3
W2_W3_STELLAR_MAX = 0.5       # Normal stars have W2-W3 < 0.5

# Beam sizes
W3_BEAM_ARCSEC = 6.5
W4_BEAM_ARCSEC = 12.0


@dataclass
class GalaxyContaminationResult:
    """Result of background galaxy contamination check."""

    has_data: bool = False

    # Input measurements
    w1_mag: Optional[float] = None
    w2_mag: Optional[float] = None
    w3_mag: Optional[float] = None
    w4_mag: Optional[float] = None

    # WISE colours
    w1_w2: Optional[float] = None
    w2_w3: Optional[float] = None
    w3_w4: Optional[float] = None

    # Galaxy indicators
    agn_colour: bool = False          # W1-W2 > 0.8
    galaxy_w2w3_colour: bool = False  # W2-W3 > 2.0
    galaxy_w3w4_colour: bool = False  # W3-W4 > 1.5
    non_stellar_colours: bool = False # Any colour outside stellar locus

    # Confusion indicators
    extended_source: bool = False     # AllWISE ext_flg > 0
    w4_beam_neighbors: int = 0        # AllWISE sources within 12"
    w3_beam_neighbors: int = 0        # AllWISE sources within 6.5"
    low_galactic_latitude: bool = False  # |b| < 10°
    galactic_latitude: Optional[float] = None

    # Scoring
    contamination_score: float = 0.0  # [0, 1]: probability of galaxy contamination
    contamination_likely: bool = False
    explanation: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_data": self.has_data,
            "w1_w2": _round_opt(self.w1_w2, 3),
            "w2_w3": _round_opt(self.w2_w3, 3),
            "w3_w4": _round_opt(self.w3_w4, 3),
            "agn_colour": self.agn_colour,
            "galaxy_w2w3_colour": self.galaxy_w2w3_colour,
            "galaxy_w3w4_colour": self.galaxy_w3w4_colour,
            "non_stellar_colours": self.non_stellar_colours,
            "extended_source": self.extended_source,
            "w4_beam_neighbors": self.w4_beam_neighbors,
            "w3_beam_neighbors": self.w3_beam_neighbors,
            "low_galactic_latitude": self.low_galactic_latitude,
            "galactic_latitude": _round_opt(self.galactic_latitude, 1),
            "contamination_score": round(self.contamination_score, 3),
            "contamination_likely": self.contamination_likely,
            "explanation": self.explanation,
            "flags": self.flags,
        }


def _round_opt(val: Optional[float], n: int) -> Optional[float]:
    return round(val, n) if val is not None else None


def _safe_float(val) -> Optional[float]:
    """Convert to float safely."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if not math.isfinite(f) else f
    except (ValueError, TypeError):
        return None


def _galactic_latitude(ra_deg: float, dec_deg: float) -> float:
    """Approximate galactic latitude from equatorial coords."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    # North galactic pole: (RA, Dec) = (192.8595, 27.1284) deg
    ra_ngp = math.radians(192.8595)
    dec_ngp = math.radians(27.1284)

    sin_b = (
        math.sin(dec) * math.sin(dec_ngp)
        + math.cos(dec) * math.cos(dec_ngp) * math.cos(ra - ra_ngp)
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_b))))


def check_galaxy_contamination(
    target_data: Dict[str, Any],
    context: Optional[str] = None,
) -> GalaxyContaminationResult:
    """Check whether a target's IR excess could be from a background galaxy.

    Parameters
    ----------
    target_data : dict
        Target data dict from the EXODUS pipeline. Expected keys:
        - ir_photometry: dict with w1mpro, w2mpro, w3mpro, w4mpro, ext_flg
        - ra, dec: target coordinates
        - allwise_neighbors (optional): list of neighbor dicts

    Returns
    -------
    GalaxyContaminationResult
    """
    result = GalaxyContaminationResult()

    # Extract IR photometry
    ir = target_data.get("ir_photometry", {})
    if not ir:
        return result

    w1 = _safe_float(ir.get("w1mpro") or ir.get("W1mag") or ir.get("w1_mag") or ir.get("W1"))
    w2 = _safe_float(ir.get("w2mpro") or ir.get("W2mag") or ir.get("w2_mag") or ir.get("W2"))
    w3 = _safe_float(ir.get("w3mpro") or ir.get("W3mag") or ir.get("w3_mag") or ir.get("W3"))
    w4 = _safe_float(ir.get("w4mpro") or ir.get("W4mag") or ir.get("w4_mag") or ir.get("W4"))

    if w1 is None and w2 is None:
        return result

    result.has_data = True
    result.w1_mag = w1
    result.w2_mag = w2
    result.w3_mag = w3
    result.w4_mag = w4

    # ── Compute WISE colours ─────────────────────────────────────
    flags = []
    score_components = []

    if w1 is not None and w2 is not None:
        result.w1_w2 = w1 - w2
        if result.w1_w2 > W1_W2_AGN_THRESHOLD:
            result.agn_colour = True
            flags.append(f"AGN-like W1-W2={result.w1_w2:.2f} (>{W1_W2_AGN_THRESHOLD})")
            score_components.append(0.8)
        elif result.w1_w2 > W1_W2_STELLAR_MAX:
            result.non_stellar_colours = True
            flags.append(f"Non-stellar W1-W2={result.w1_w2:.2f} (>{W1_W2_STELLAR_MAX})")
            score_components.append(0.3)

    if w2 is not None and w3 is not None:
        result.w2_w3 = w2 - w3
        if result.w2_w3 > W2_W3_GALAXY_MIN:
            result.galaxy_w2w3_colour = True
            flags.append(f"Galaxy-like W2-W3={result.w2_w3:.2f} (>{W2_W3_GALAXY_MIN})")
            score_components.append(0.5)
        elif result.w2_w3 > W2_W3_STELLAR_MAX:
            result.non_stellar_colours = True

    if w3 is not None and w4 is not None:
        result.w3_w4 = w3 - w4
        if result.w3_w4 > W3_W4_GALAXY_MIN:
            result.galaxy_w3w4_colour = True
            # Audit fix F4: for IR-selected samples,
            # W3-W4 excess is WHY the target was selected — the check is
            # circular.  Only flag galaxy if there's ADDITIONAL evidence.
            if context == "ir_selected":
                flags.append(
                    f"W3-W4={result.w3_w4:.2f} (>{W3_W4_GALAXY_MIN}) "
                    f"[NOTE: circular for IR-selected sample — needs "
                    f"independent evidence (ext_flg, W1-W2) to confirm galaxy]"
                )
                # Do NOT add to score_components — avoid circular reasoning
            else:
                flags.append(f"Galaxy-like W3-W4={result.w3_w4:.2f} (>{W3_W4_GALAXY_MIN})")
                score_components.append(0.4)

    # ── Extended source flag ─────────────────────────────────────
    ext_flg = ir.get("ext_flg") or ir.get("ext_flag") or ir.get("wise_ext_flg")
    if ext_flg is not None:
        try:
            ext_val = int(ext_flg)
            if ext_val > 0:
                result.extended_source = True
                flags.append(f"Extended source (ext_flg={ext_val})")
                score_components.append(0.7)
        except (ValueError, TypeError):
            pass

    # ── W4 beam confusion ────────────────────────────────────────
    # Check for AllWISE neighbors within the W4 beam
    neighbors = target_data.get("allwise_neighbors", [])
    if isinstance(neighbors, list):
        for n in neighbors:
            sep = _safe_float(n.get("separation_arcsec") or n.get("sep"))
            if sep is not None:
                if sep <= W4_BEAM_ARCSEC:
                    result.w4_beam_neighbors += 1
                if sep <= W3_BEAM_ARCSEC:
                    result.w3_beam_neighbors += 1

        if result.w4_beam_neighbors > 0:
            flags.append(f"{result.w4_beam_neighbors} source(s) in W4 beam (12\")")
            score_components.append(min(0.3 * result.w4_beam_neighbors, 0.6))

    # ── Galactic latitude ────────────────────────────────────────
    ra = _safe_float(target_data.get("ra"))
    dec = _safe_float(target_data.get("dec"))
    if ra is not None and dec is not None:
        glat = _galactic_latitude(ra, dec)
        result.galactic_latitude = glat
        if abs(glat) < 10:
            result.low_galactic_latitude = True
            flags.append(f"Low galactic latitude b={glat:.1f}°")
            score_components.append(0.3)

    # ── W4-only excess pattern ───────────────────────────────────
    # W4-only excess is the most common galaxy-confusion false positive
    ir_excess = target_data.get("ir_excess", {})
    if isinstance(ir_excess, dict):
        sigma_w3 = _safe_float(ir_excess.get("sigma_W3"))
        sigma_w4 = _safe_float(ir_excess.get("sigma_W4"))
        if sigma_w4 is not None and sigma_w4 > 3.0:
            if sigma_w3 is None or sigma_w3 < 2.0:
                flags.append(f"W4-only excess (W4={sigma_w4:.1f}σ, W3={sigma_w3 or 0:.1f}σ) — confusion likely")
                score_components.append(0.6)

    # ── Combine into contamination score ─────────────────────────
    if score_components:
        # Use maximum + diminishing contributions from additional flags
        score_components.sort(reverse=True)
        score = score_components[0]
        for additional in score_components[1:]:
            score += additional * 0.3  # diminishing weight
        score = min(score, 1.0)
    else:
        score = 0.0

    result.contamination_score = score
    result.contamination_likely = score >= 0.5
    result.flags = flags

    # Generate explanation
    if not flags:
        result.explanation = "No galaxy contamination indicators found"
    elif result.contamination_likely:
        result.explanation = (
            f"Background galaxy contamination LIKELY (score={score:.2f}): "
            + "; ".join(flags[:3])
        )
    else:
        result.explanation = (
            f"Minor contamination risk (score={score:.2f}): "
            + "; ".join(flags[:3])
        )

    if result.has_data:
        log.debug(
            "Galaxy contamination check: score=%.2f  flags=%s",
            score, flags,
        )

    return result


# ── PM-IR Correlation Check ─────────────────────────────────────────

def check_pm_ir_correlation(
    target_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Check if PM anomaly and IR excess are likely correlated.

    When a background source is blended in the WISE beam, it creates:
      1. IR excess (extra flux at W3/W4)
      2. WISE photocentre shift → WISE-Gaia PM discrepancy

    These are NOT independent detections but a single source of contamination.
    This function flags such cases and computes a correlation penalty.

    Returns dict with:
        pm_ir_correlated : bool
        correlation_score : float (0-1, higher = more likely correlated)
        effective_pm_weight : float (0-1, multiply PM channel by this)
        explanation : str
    """
    result = {
        "pm_ir_correlated": False,
        "correlation_score": 0.0,
        "effective_pm_weight": 1.0,
        "explanation": "No PM-IR correlation detected",
    }

    # Check if both PM and IR channels fired
    channel_scores = target_data.get("channel_scores", {})

    ir_ch = channel_scores.get("ir_excess", {})
    pm_ch = channel_scores.get("proper_motion_anomaly", {})

    if not isinstance(ir_ch, dict) or not isinstance(pm_ch, dict):
        return result

    ir_score = ir_ch.get("score", 0)
    pm_score = pm_ch.get("score", 0)

    # Only applies if both channels are active
    if ir_score < 0.15 or pm_score < 0.15:
        return result

    # Identify the SOURCE of PM anomaly
    pm_details = pm_ch.get("details", {})
    ruwe = _safe_float(pm_details.get("ruwe"))
    aen_sig = _safe_float(pm_details.get("astrometric_excess_noise_sig"))
    wise_pm = pm_details.get("wise_gaia_pm", {})
    wise_pm_sigma = _safe_float(wise_pm.get("pm_discrepancy_sigma"))
    is_wise_pm_discrepant = wise_pm.get("is_discrepant", False)

    # RUWE and AEN-based PM anomalies are INDEPENDENT of IR
    # (they come from optical-only Gaia astrometry)
    ruwe_anomaly = ruwe is not None and ruwe > 1.4
    aen_anomaly = aen_sig is not None and aen_sig > 5.0

    if ruwe_anomaly or aen_anomaly:
        # PM anomaly from Gaia-internal metrics → independent of IR
        result["explanation"] = (
            "PM anomaly from Gaia astrometry (RUWE/AEN), independent of IR excess"
        )
        return result

    # WISE-Gaia PM discrepancy IS potentially correlated with IR
    if is_wise_pm_discrepant or (wise_pm_sigma is not None and wise_pm_sigma > 2.0):
        # Compute correlation score based on how much PM comes from WISE-Gaia offset
        ir_details = ir_ch.get("details", {})
        sigma_w3 = _safe_float(ir_details.get("sigma_W3"))
        sigma_w4 = _safe_float(ir_details.get("sigma_W4"))

        # Higher IR excess → more photocentre shift → higher correlation
        max_ir_sigma = max(sigma_w3 or 0, sigma_w4 or 0)

        if max_ir_sigma > 5 and wise_pm_sigma is not None and wise_pm_sigma > 2:
            # Both are strong: likely correlated
            corr = min(1.0, 0.3 + 0.1 * (max_ir_sigma / 10.0) + 0.1 * (wise_pm_sigma / 5.0))

            # W4-dominant excess is more suspicious (larger beam → easier blending)
            if sigma_w4 is not None and sigma_w3 is not None and sigma_w4 > 3 * sigma_w3:
                corr = min(1.0, corr + 0.2)

            result["pm_ir_correlated"] = True
            result["correlation_score"] = round(corr, 3)
            result["effective_pm_weight"] = round(max(0.1, 1.0 - corr), 3)
            result["explanation"] = (
                f"PM anomaly from WISE-Gaia offset ({wise_pm_sigma:.1f}σ) "
                f"likely correlated with IR excess ({max_ir_sigma:.0f}σ). "
                f"Effective PM weight: {result['effective_pm_weight']:.2f}"
            )

            log.debug(
                "PM-IR correlation for %s: corr=%.2f, weight=%.2f",
                target_data.get("target_id", "?"),
                corr, result["effective_pm_weight"],
            )

    return result
