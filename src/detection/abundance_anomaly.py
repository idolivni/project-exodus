"""
Stellar Abundance Anomaly Detection for Project EXODUS.

Cross-matches targets against APOGEE DR17 and GALAH DR4 spectroscopic
surveys via VizieR to check for chemical abundance anomalies.

A star with anomalous element ratios — especially heavy r-process elements
(Eu, Ba), phosphorus, or lithium — could indicate:
  1. Industrial pollution of the stellar photosphere
  2. Accretion of technologically processed material
  3. An unresolved circumstellar structure with anomalous emission lines

Diagnostic thresholds (from Perplexity research brief, Prompt 7):

  | Ratio       | Natural Range (thin disk) | Flag if        |
  |-------------|--------------------------|----------------|
  | [Eu/Fe]     | -0.2 to +0.4            | > +0.8         |
  | [Ni/Fe]     | -0.2 to +0.2            | > +0.4         |
  | [P/Fe]      | -0.1 to +0.4            | > +1.0         |
  | [Co/Ni]     | -0.1 to +0.1            | > +0.3 or < -0.3 |
  | A(Li)       | 0.5 to 2.5 (solar-like) | > 3.5          |
  | [Ce/Fe]     | -0.2 to +0.3            | > +0.6         |

This is MAXIMALLY INDEPENDENT of all other EXODUS channels:
  - IR excess = thermal emission from physical structure
  - PM anomaly = gravitational/astrometric
  - UV/radio = non-thermal emission
  - Abundance = CHEMICAL COMPOSITION of the star itself

Public API
----------
compute_abundance_anomaly(ra, dec, **kwargs)
    Cross-match a target against spectroscopic surveys for abundance anomalies.
    Returns AbundanceAnomalyResult with anomaly_score, flagged ratios.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, load_cache, save_cache

log = get_logger("detection.abundance_anomaly")

# ── Constants ────────────────────────────────────────────────────────
SEARCH_RADIUS_ARCSEC = 3.0  # Tight match for spectroscopic surveys

# Diagnostic abundance thresholds (thin-disk stars)
# Each entry: (key_in_catalog, natural_low, natural_high, flag_low, flag_high, weight)
# weight reflects how diagnostic the ratio is for technosignature search
ABUNDANCE_THRESHOLDS = {
    "Eu_Fe": {
        "natural_range": (-0.2, 0.4),
        "flag_above": 0.8,
        "flag_below": None,
        "weight": 1.0,
        "description": "Europium excess — r-process element, industrial indicator",
    },
    "Ni_Fe": {
        "natural_range": (-0.2, 0.2),
        "flag_above": 0.4,
        "flag_below": None,
        "weight": 0.6,
        "description": "Nickel excess — siderophile, metallic processing indicator",
    },
    "P_Fe": {
        "natural_range": (-0.1, 0.4),
        "flag_above": 1.0,
        "flag_below": None,
        "weight": 0.8,
        "description": "Phosphorus excess — volatile, rare in stellar photospheres",
    },
    "Co_Ni": {
        "natural_range": (-0.1, 0.1),
        "flag_above": 0.3,
        "flag_below": -0.3,
        "weight": 0.5,
        "description": "Cobalt/Nickel ratio — non-standard nucleosynthesis indicator",
    },
    "Li_A": {
        "natural_range": (0.5, 2.5),
        "flag_above": 3.5,
        "flag_below": None,
        "weight": 0.7,
        "description": "Lithium abundance — fragile element, super-Li giant anomaly",
    },
    "Ce_Fe": {
        "natural_range": (-0.2, 0.3),
        "flag_above": 0.6,
        "flag_below": None,
        "weight": 0.8,
        "description": "Cerium excess — s-process element, mass transfer indicator",
    },
}


# ── Result dataclass ─────────────────────────────────────────────────

@dataclass
class AbundanceAnomalyResult:
    """Result of stellar abundance anomaly analysis."""
    # Data quality
    has_apogee: bool = False
    has_galah: bool = False
    n_surveys: int = 0
    n_abundances_checked: int = 0
    data_source: str = "none"  # "apogee", "galah", "both", "none"

    # Anomaly metrics
    n_anomalous_ratios: int = 0
    anomalous_ratios: List[str] = field(default_factory=list)
    most_anomalous_ratio: Optional[str] = None
    most_anomalous_sigma: float = 0.0  # How far outside natural range
    abundance_details: Dict[str, Any] = field(default_factory=dict)

    # Stellar parameters from spectroscopy
    teff: Optional[float] = None
    logg: Optional[float] = None
    fe_h: Optional[float] = None

    # Combined score (0-1)
    anomaly_score: float = 0.0
    is_anomalous: bool = False

    # For channel integration
    p_value: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_apogee": self.has_apogee,
            "has_galah": self.has_galah,
            "n_surveys": self.n_surveys,
            "n_abundances_checked": self.n_abundances_checked,
            "data_source": self.data_source,
            "n_anomalous_ratios": self.n_anomalous_ratios,
            "anomalous_ratios": self.anomalous_ratios,
            "most_anomalous_ratio": self.most_anomalous_ratio,
            "most_anomalous_sigma": round(self.most_anomalous_sigma, 3),
            "abundance_details": self.abundance_details,
            "teff": self.teff,
            "logg": self.logg,
            "fe_h": self.fe_h,
            "anomaly_score": round(self.anomaly_score, 4),
            "is_anomalous": self.is_anomalous,
            "p_value": self.p_value,
        }


# ── Main computation ─────────────────────────────────────────────────

def compute_abundance_anomaly(
    ra: float,
    dec: float,
    use_cache: bool = True,
) -> AbundanceAnomalyResult:
    """Cross-match a target against spectroscopic surveys for abundance anomalies.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in degrees (ICRS).
    use_cache : bool
        Whether to use local cache.

    Returns
    -------
    AbundanceAnomalyResult
    """
    cache_key = f"abundance_anomaly_{ra:.6f}_{dec:.6f}"
    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            log.debug("Abundance anomaly loaded from cache for (%.4f, %.4f)", ra, dec)
            return _dict_to_result(cached)

    result = AbundanceAnomalyResult()
    all_abundances: Dict[str, float] = {}

    # Query APOGEE DR17 (near-IR, best for detailed abundances)
    apogee_data = _query_apogee_abundances(ra, dec)
    if apogee_data:
        result.has_apogee = True
        all_abundances.update(apogee_data.get("abundances", {}))
        result.teff = apogee_data.get("teff")
        result.logg = apogee_data.get("logg")
        result.fe_h = apogee_data.get("fe_h")

    # Query GALAH DR4 (optical, complementary elements)
    galah_data = _query_galah_abundances(ra, dec)
    if galah_data:
        result.has_galah = True
        # Merge — GALAH may have elements APOGEE doesn't
        for key, val in galah_data.get("abundances", {}).items():
            if key not in all_abundances:
                all_abundances[key] = val
        # Fill in stellar params if APOGEE didn't have them
        if result.teff is None:
            result.teff = galah_data.get("teff")
        if result.logg is None:
            result.logg = galah_data.get("logg")
        if result.fe_h is None:
            result.fe_h = galah_data.get("fe_h")

    result.n_surveys = sum([result.has_apogee, result.has_galah])

    if result.n_surveys == 0:
        result.data_source = "none"
        if use_cache:
            save_cache(cache_key, result.to_dict())
        return result

    # Determine data source label
    if result.has_apogee and result.has_galah:
        result.data_source = "both"
    elif result.has_apogee:
        result.data_source = "apogee"
    else:
        result.data_source = "galah"

    # ── Check each diagnostic ratio against thresholds ────────────
    max_sigma = 0.0
    max_ratio_name = None
    anomalous_ratios = []

    for ratio_name, threshold_info in ABUNDANCE_THRESHOLDS.items():
        if ratio_name not in all_abundances:
            continue

        result.n_abundances_checked += 1
        value = all_abundances[ratio_name]
        nat_lo, nat_hi = threshold_info["natural_range"]
        flag_above = threshold_info["flag_above"]
        flag_below = threshold_info["flag_below"]
        weight = threshold_info["weight"]

        # Compute deviation in "natural range widths"
        nat_width = nat_hi - nat_lo
        is_flagged = False

        if flag_above is not None and value > flag_above:
            deviation = (value - flag_above) / max(nat_width, 0.01)
            is_flagged = True
        elif flag_below is not None and value < flag_below:
            deviation = (flag_below - value) / max(nat_width, 0.01)
            is_flagged = True
        else:
            deviation = 0.0

        detail = {
            "value": round(value, 4),
            "natural_range": [nat_lo, nat_hi],
            "is_flagged": is_flagged,
            "deviation_sigma": round(deviation * weight, 3),
            "description": threshold_info["description"],
        }
        result.abundance_details[ratio_name] = detail

        if is_flagged:
            anomalous_ratios.append(ratio_name)
            weighted_sigma = deviation * weight
            if weighted_sigma > max_sigma:
                max_sigma = weighted_sigma
                max_ratio_name = ratio_name

    result.n_anomalous_ratios = len(anomalous_ratios)
    result.anomalous_ratios = anomalous_ratios
    result.most_anomalous_ratio = max_ratio_name
    result.most_anomalous_sigma = max_sigma

    # ── Compute anomaly score ────────────────────────────────────
    if result.n_anomalous_ratios == 0:
        result.anomaly_score = 0.0
    elif result.n_anomalous_ratios == 1:
        # Single anomalous ratio: score based on deviation
        result.anomaly_score = min(1.0, 0.3 + 0.1 * max_sigma)
    else:
        # Multiple anomalous ratios: much more suspicious
        # Base score from worst ratio, boosted by count
        base = min(1.0, 0.3 + 0.1 * max_sigma)
        multi_bonus = 1.0 + 0.2 * (result.n_anomalous_ratios - 1)
        result.anomaly_score = min(1.0, base * multi_bonus)

    result.is_anomalous = result.anomaly_score > 0.3

    # Approximate p-value from number of anomalous ratios
    # Very rough: each ratio has ~5% chance of being flagged, so
    # 2+ simultaneous flags is ~0.25%, 3+ is ~0.01%
    if result.n_anomalous_ratios == 0:
        result.p_value = 1.0
    elif result.n_anomalous_ratios == 1:
        result.p_value = 0.05
    elif result.n_anomalous_ratios == 2:
        result.p_value = 0.0025
    else:
        result.p_value = 0.001 ** min(result.n_anomalous_ratios - 2, 3)

    if result.is_anomalous:
        log.info(
            "Abundance ANOMALY at (%.4f, %.4f): score=%.3f, %d anomalous ratios (%s), "
            "worst=%s (%.2f sigma), from %s",
            ra, dec, result.anomaly_score, result.n_anomalous_ratios,
            ", ".join(anomalous_ratios), max_ratio_name, max_sigma,
            result.data_source,
        )

    if use_cache:
        save_cache(cache_key, result.to_dict())

    return result


# ── Survey query functions ───────────────────────────────────────────

def _query_apogee_abundances(ra: float, dec: float) -> Optional[Dict]:
    """Query APOGEE DR17 via VizieR for detailed element abundances.

    APOGEE (near-IR H-band) provides the best individual element
    abundances: C, N, O, Na, Mg, Al, Si, P, S, K, Ca, Ti, V, Cr, Mn,
    Fe, Co, Ni, Ce, and sometimes Eu.
    """
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        v = Vizier(columns=["*"], row_limit=1)
        # III/284 = APOGEE DR17 allStar
        tables = v.query_region(
            coord, radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
            catalog="III/284",
        )
        if not tables or len(tables) == 0 or len(tables[0]) == 0:
            return None

        row = tables[0][0]
        colnames = [c.lower() for c in tables[0].colnames]

        result = {"survey": "APOGEE", "abundances": {}}

        # Stellar parameters
        for key, col_options in [
            ("teff", ["teff"]),
            ("logg", ["logg"]),
            ("fe_h", ["[m/h]", "[fe/h]", "m_h", "fe_h"]),
        ]:
            for col in col_options:
                if col in colnames:
                    try:
                        val = float(row[col])
                        if np.isfinite(val):
                            result[key] = val
                            break
                    except (ValueError, TypeError):
                        pass

        # Extract abundance ratios
        # APOGEE column naming: [X/Fe] or X_FE or [X/M]
        fe_h_val = result.get("fe_h", 0.0)

        # Mapping from our diagnostic ratio names to possible APOGEE columns
        abundance_map = {
            "Ni_Fe": ["ni_fe", "[ni/fe]", "ni_h"],
            "P_Fe": ["p_fe", "[p/fe]", "p_h"],
            "Co_Ni": [],  # Computed from Co and Ni
            "Ce_Fe": ["ce_fe", "[ce/fe]", "ce_h"],
            "Eu_Fe": ["eu_fe", "[eu/fe]"],
        }

        for ratio_name, col_options in abundance_map.items():
            for col in col_options:
                if col in colnames:
                    try:
                        val = float(row[col])
                        if np.isfinite(val) and abs(val) < 9:  # APOGEE uses 99.99 for missing
                            # If column is [X/H], convert to [X/Fe]
                            if col.endswith("_h") and not col.endswith("fe_h"):
                                val = val - fe_h_val
                            result["abundances"][ratio_name] = val
                            break
                    except (ValueError, TypeError):
                        pass

        # Compute Co/Ni ratio if both available
        co_val = None
        ni_val = None
        for col in ["co_fe", "[co/fe]", "co_h"]:
            if col in colnames:
                try:
                    v_ = float(row[col])
                    if np.isfinite(v_) and abs(v_) < 9:
                        co_val = v_ if not col.endswith("_h") else v_ - fe_h_val
                        break
                except (ValueError, TypeError):
                    pass
        for col in ["ni_fe", "[ni/fe]", "ni_h"]:
            if col in colnames:
                try:
                    v_ = float(row[col])
                    if np.isfinite(v_) and abs(v_) < 9:
                        ni_val = v_ if not col.endswith("_h") else v_ - fe_h_val
                        break
                except (ValueError, TypeError):
                    pass
        if co_val is not None and ni_val is not None:
            result["abundances"]["Co_Ni"] = co_val - ni_val

        if result["abundances"]:
            log.debug(
                "APOGEE abundances at (%.4f, %.4f): %d ratios measured",
                ra, dec, len(result["abundances"]),
            )

        return result if result["abundances"] else None

    except Exception as exc:
        log.debug("APOGEE query failed for (%.4f, %.4f): %s", ra, dec, exc)
        return None


def _query_galah_abundances(ra: float, dec: float) -> Optional[Dict]:
    """Query GALAH DR4 via VizieR for element abundances.

    GALAH (optical) provides: Li, C, O, Na, Mg, Al, Si, K, Ca, Sc,
    Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Rb, Sr, Y, Zr, Mo, Ru, Ba,
    La, Ce, Nd, Sm, Eu.
    """
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        v = Vizier(columns=["*"], row_limit=1)
        # III/290 = GALAH DR4
        tables = v.query_region(
            coord, radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
            catalog="III/290",
        )
        if not tables or len(tables) == 0 or len(tables[0]) == 0:
            return None

        row = tables[0][0]
        colnames = [c.lower() for c in tables[0].colnames]

        result = {"survey": "GALAH", "abundances": {}}

        # Stellar parameters
        for key, col_options in [
            ("teff", ["teff"]),
            ("logg", ["logg"]),
            ("fe_h", ["fe_h", "[fe/h]"]),
        ]:
            for col in col_options:
                if col in colnames:
                    try:
                        val = float(row[col])
                        if np.isfinite(val):
                            result[key] = val
                            break
                    except (ValueError, TypeError):
                        pass

        fe_h_val = result.get("fe_h", 0.0)

        # GALAH abundance columns: element_fe or [element/fe]
        abundance_map = {
            "Eu_Fe": ["eu_fe", "[eu/fe]"],
            "Ni_Fe": ["ni_fe", "[ni/fe]"],
            "Ce_Fe": ["ce_fe", "[ce/fe]"],
            "Li_A": ["a_li", "li_fe"],  # Lithium absolute abundance A(Li)
        }

        for ratio_name, col_options in abundance_map.items():
            for col in col_options:
                if col in colnames:
                    try:
                        val = float(row[col])
                        if np.isfinite(val) and abs(val) < 9:
                            # Li uses absolute abundance A(Li), not [Li/Fe]
                            if ratio_name == "Li_A" and col == "li_fe":
                                # Convert [Li/Fe] to approximate A(Li)
                                val = val + fe_h_val + 1.05  # A(Li)_sun ≈ 1.05
                            result["abundances"][ratio_name] = val
                            break
                    except (ValueError, TypeError):
                        pass

        # Compute Co/Ni from individual abundances
        co_val = None
        ni_val = None
        for col in ["co_fe", "[co/fe]"]:
            if col in colnames:
                try:
                    v_ = float(row[col])
                    if np.isfinite(v_) and abs(v_) < 9:
                        co_val = v_
                        break
                except (ValueError, TypeError):
                    pass
        for col in ["ni_fe", "[ni/fe]"]:
            if col in colnames:
                try:
                    v_ = float(row[col])
                    if np.isfinite(v_) and abs(v_) < 9:
                        ni_val = v_
                        break
                except (ValueError, TypeError):
                    pass
        if co_val is not None and ni_val is not None:
            result["abundances"]["Co_Ni"] = co_val - ni_val

        if result["abundances"]:
            log.debug(
                "GALAH abundances at (%.4f, %.4f): %d ratios measured",
                ra, dec, len(result["abundances"]),
            )

        return result if result["abundances"] else None

    except Exception as exc:
        log.debug("GALAH query failed for (%.4f, %.4f): %s", ra, dec, exc)
        return None


# ── Helpers ──────────────────────────────────────────────────────────

def _dict_to_result(data: Dict[str, Any]) -> AbundanceAnomalyResult:
    """Reconstruct an AbundanceAnomalyResult from a cached dict."""
    return AbundanceAnomalyResult(
        has_apogee=data.get("has_apogee", False),
        has_galah=data.get("has_galah", False),
        n_surveys=data.get("n_surveys", 0),
        n_abundances_checked=data.get("n_abundances_checked", 0),
        data_source=data.get("data_source", "none"),
        n_anomalous_ratios=data.get("n_anomalous_ratios", 0),
        anomalous_ratios=data.get("anomalous_ratios", []),
        most_anomalous_ratio=data.get("most_anomalous_ratio"),
        most_anomalous_sigma=data.get("most_anomalous_sigma", 0.0),
        abundance_details=data.get("abundance_details", {}),
        teff=data.get("teff"),
        logg=data.get("logg"),
        fe_h=data.get("fe_h"),
        anomaly_score=data.get("anomaly_score", 0.0),
        is_anomalous=data.get("is_anomalous", False),
        p_value=data.get("p_value", 1.0),
    )


# ── CLI demo ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Stellar Abundance Anomaly Detection")
    print("=" * 70)

    # Przybylski's Star — most chemically anomalous star known
    ra, dec = 184.098, -47.028
    print(f"\n  Checking Przybylski's Star (RA={ra}, Dec={dec})")
    result = compute_abundance_anomaly(ra, dec)
    print(f"  Surveys: APOGEE={result.has_apogee}, GALAH={result.has_galah}")
    print(f"  Abundances checked: {result.n_abundances_checked}")
    print(f"  Anomalous ratios: {result.n_anomalous_ratios}")
    if result.anomalous_ratios:
        print(f"  Flagged: {', '.join(result.anomalous_ratios)}")
    print(f"  Anomaly score: {result.anomaly_score:.4f}")
    print(f"  Is anomalous: {result.is_anomalous}")

    # Sun-like star — should be normal
    ra2, dec2 = 279.234, 38.784  # near Vega
    print(f"\n  Checking near-Vega star (RA={ra2}, Dec={dec2})")
    result2 = compute_abundance_anomaly(ra2, dec2)
    print(f"  Surveys: APOGEE={result2.has_apogee}, GALAH={result2.has_galah}")
    print(f"  Anomaly score: {result2.anomaly_score:.4f}")

    print("\n" + "=" * 70)
