"""
Campaign target file loader for Project EXODUS.

Loads curated target lists from JSON files, validates their format,
and enriches missing stellar metadata via Gaia coordinate lookup.
This enables the pipeline to run on custom target sets (calibration,
golden sample, Hephaistos replication) instead of only querying the
NASA Exoplanet Archive.

Public API
----------
load_target_file(path)
    Load and validate a campaign target JSON file.

enrich_target_metadata(targets)
    Fill missing Gaia fields (mag, color, galactic latitude) by
    coordinate crossmatch.  Needed for matched control selection.

validate_target_format(targets)
    Check that all required fields are present and correctly typed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("ingestion.target_loader")


# =====================================================================
#  Constants
# =====================================================================

# Required fields in every target entry
REQUIRED_FIELDS = {"target_id", "host_star", "ra", "dec"}

# Optional but recommended fields
RECOMMENDED_FIELDS = {"distance_pc", "hz_flag"}

# Canonical channel names (must match src/scoring/exodus_score.py)
CANONICAL_CHANNELS = {
    "ir_excess",
    "transit_anomaly",
    "radio_anomaly",
    "gaia_photometric_anomaly",
    "habitable_zone_planet",
    "proper_motion_anomaly",
}


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class CampaignTargets:
    """Result of loading a campaign target file.

    Attributes
    ----------
    campaign : str
        Campaign identifier (e.g. ``"calibration"``, ``"golden_sample"``).
    description : str
        Human-readable description of the campaign.
    targets : list of dict
        Target dicts in the same format as ``get_hz_planets()`` returns.
    phase : str
        Campaign phase identifier.
    raw_metadata : dict
        Any extra top-level fields from the JSON file.
    """

    campaign: str = ""
    description: str = ""
    targets: List[Dict[str, Any]] = field(default_factory=list)
    phase: str = ""
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def positive_controls(self) -> List[Dict[str, Any]]:
        return [t for t in self.targets if t.get("is_positive_control", False)]

    @property
    def negative_controls(self) -> List[Dict[str, Any]]:
        return [t for t in self.targets if t.get("is_negative_control", False)]

    def summary(self) -> str:
        lines = [
            f"Campaign: {self.campaign}",
            f"  Phase:       {self.phase or 'unspecified'}",
            f"  Description: {self.description}",
            f"  Targets:     {self.n_targets}",
        ]
        n_pos = len(self.positive_controls)
        n_neg = len(self.negative_controls)
        if n_pos or n_neg:
            lines.append(f"  Controls:    {n_pos} positive, {n_neg} negative")

        for t in self.targets:
            ctrl = ""
            if t.get("is_positive_control"):
                ctrl = " [+ctrl]"
            elif t.get("is_negative_control"):
                ctrl = " [-ctrl]"
            dist = t.get("distance_pc")
            if isinstance(dist, (int, float)):
                lines.append(
                    f"    {t['target_id']:>25s}  "
                    f"RA={t['ra']:>8.4f}  Dec={t['dec']:>+8.4f}  "
                    f"d={dist:>6.1f} pc{ctrl}"
                )
            else:
                lines.append(
                    f"    {t['target_id']:>25s}  "
                    f"RA={t['ra']:>8.4f}  Dec={t['dec']:>+8.4f}{ctrl}"
                )

        return "\n".join(lines)


# =====================================================================
#  Core functions
# =====================================================================

def load_target_file(path: str | Path) -> CampaignTargets:
    """Load and validate a campaign target JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON target file.

    Returns
    -------
    CampaignTargets
        Validated campaign targets ready for pipeline processing.

    Raises
    ------
    FileNotFoundError
        If the target file does not exist.
    ValueError
        If the file is not valid JSON or has missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")

    log.info("Loading campaign target file: %s", path)

    with open(path, "r") as f:
        data = json.load(f)

    # Validate top-level structure
    if "targets" not in data:
        raise ValueError(f"Target file must contain a 'targets' array: {path}")

    if not isinstance(data["targets"], list):
        raise ValueError(f"'targets' must be a list: {path}")

    campaign = data.get("campaign", path.stem)
    description = data.get("description", "")
    phase = data.get("phase", "")

    # Validate and normalise each target
    targets = []
    warnings = validate_target_format(data["targets"])
    if warnings:
        for w in warnings:
            log.warning("  Target validation: %s", w)

    for i, raw_target in enumerate(data["targets"]):
        target = _normalise_target(raw_target, i)
        targets.append(target)

    # Extra metadata (anything beyond campaign/description/phase/targets)
    raw_metadata = {
        k: v for k, v in data.items()
        if k not in {"campaign", "description", "phase", "targets"}
    }

    result = CampaignTargets(
        campaign=campaign,
        description=description,
        targets=targets,
        phase=phase,
        raw_metadata=raw_metadata,
    )

    log.info(
        "Loaded %d targets for campaign '%s' (%s)",
        result.n_targets, campaign, description,
    )
    return result


def _normalise_target(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Normalise a raw target dict to the pipeline-expected format.

    Ensures all required fields exist and have the right types.
    Adds defaults for optional fields.
    """
    target = {}

    # Required fields
    target["target_id"] = str(raw.get("target_id", f"custom_target_{index:03d}"))
    target["host_star"] = str(raw.get("host_star", target["target_id"]))
    target["ra"] = float(raw["ra"])
    target["dec"] = float(raw["dec"])

    # Standard optional fields
    dist = raw.get("distance_pc")
    target["distance_pc"] = float(dist) if dist is not None else None
    target["hz_flag"] = bool(raw.get("hz_flag", False))

    # Campaign-specific fields (passed through)
    target["expected_channels"] = raw.get("expected_channels", {})
    target["expected_behavior"] = raw.get("expected_behavior", "")
    target["is_positive_control"] = bool(raw.get("is_positive_control", False))
    target["is_negative_control"] = bool(raw.get("is_negative_control", False))
    target["notes"] = raw.get("notes", "")

    # Provenance
    target["source"] = "custom_target_file"

    # Pre-populated stellar params (if provided in JSON)
    for stellar_field in ["phot_g_mean_mag", "bp_rp", "b_gal", "spectral_type"]:
        if stellar_field in raw:
            target[stellar_field] = raw[stellar_field]

    # Pre-populated proper motions (Gaia DR3, mas/yr)
    for pm_field in ["pmra_mas", "pmdec_mas"]:
        if pm_field in raw:
            target[pm_field] = float(raw[pm_field])

    return target


def validate_target_format(
    targets: List[Dict[str, Any]],
) -> List[str]:
    """Check that all targets have required fields and correct types.

    Parameters
    ----------
    targets : list of dict
        Raw target entries from the JSON file.

    Returns
    -------
    list of str
        Validation warnings.  Empty if everything is correct.
    """
    warnings = []

    for i, t in enumerate(targets):
        tid = t.get("target_id", f"target_{i}")

        # Check required fields
        for field_name in REQUIRED_FIELDS:
            if field_name not in t:
                warnings.append(
                    f"Target {tid}: missing required field '{field_name}'"
                )

        # Check RA/Dec ranges
        ra = t.get("ra")
        dec = t.get("dec")
        if ra is not None and not (0 <= float(ra) <= 360):
            warnings.append(f"Target {tid}: RA={ra} outside [0, 360]")
        if dec is not None and not (-90 <= float(dec) <= 90):
            warnings.append(f"Target {tid}: Dec={dec} outside [-90, 90]")

        # Check distance
        dist = t.get("distance_pc")
        if dist is not None and float(dist) <= 0:
            warnings.append(f"Target {tid}: distance_pc={dist} should be positive")

        # Validate expected_channels keys
        expected = t.get("expected_channels", {})
        for channel_name in expected:
            if channel_name not in CANONICAL_CHANNELS:
                warnings.append(
                    f"Target {tid}: unknown channel '{channel_name}' "
                    f"in expected_channels (valid: {CANONICAL_CHANNELS})"
                )

        # Validate expected values
        valid_expectations = {"positive", "negative", "neutral"}
        for ch_name, ch_val in expected.items():
            if ch_val not in valid_expectations:
                warnings.append(
                    f"Target {tid}: expected_channels['{ch_name}'] = "
                    f"'{ch_val}' not in {valid_expectations}"
                )

    return warnings


def enrich_target_metadata(
    targets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fill missing stellar metadata for custom targets.

    Attempts to query Gaia DR3 by coordinates to fill:
    - ``phot_g_mean_mag`` (G-band apparent magnitude)
    - ``bp_rp`` (BP-RP colour index)
    - ``b_gal`` (galactic latitude)

    These fields are required by the matched control selector
    (``src/core/controls.py``).  If Gaia lookup fails, the fields
    remain ``None`` and the target will be excluded from matching
    (which is acceptable for calibration runs).

    Parameters
    ----------
    targets : list of dict
        Target dicts from :func:`load_target_file`.

    Returns
    -------
    list of dict
        The same targets with missing fields filled where possible.
    """
    # Try to import Gaia query module
    try:
        from src.ingestion.gaia_query import get_stellar_params
    except ImportError:
        log.warning(
            "Gaia query module unavailable; skipping metadata enrichment. "
            "Control matching may not work for custom targets."
        )
        # Still compute galactic latitude from RA/Dec (no Gaia needed)
        for t in targets:
            if "b_gal" not in t or t["b_gal"] is None:
                t["b_gal"] = _compute_galactic_latitude(t["ra"], t["dec"])
        return targets

    n_enriched = 0
    for t in targets:
        needs_enrichment = (
            "phot_g_mean_mag" not in t
            or "bp_rp" not in t
            or t.get("phot_g_mean_mag") is None
            or t.get("bp_rp") is None
        )

        if needs_enrichment:
            try:
                gaia_df = get_stellar_params(
                    t["ra"], t["dec"], radius_arcsec=5.0
                )
                if gaia_df is not None and not gaia_df.empty:
                    row = gaia_df.iloc[0]
                    if "phot_g_mean_mag" not in t or t.get("phot_g_mean_mag") is None:
                        val = row.get("phot_g_mean_mag")
                        if val is not None and np.isfinite(val):
                            t["phot_g_mean_mag"] = float(val)
                    if "bp_rp" not in t or t.get("bp_rp") is None:
                        val = row.get("bp_rp")
                        if val is not None and np.isfinite(val):
                            t["bp_rp"] = float(val)
                    n_enriched += 1
            except Exception as exc:
                log.debug(
                    "Gaia enrichment failed for %s: %s",
                    t["target_id"], exc,
                )

        # Always compute galactic latitude from coordinates
        if "b_gal" not in t or t["b_gal"] is None:
            t["b_gal"] = _compute_galactic_latitude(t["ra"], t["dec"])

    log.info("Enriched metadata for %d/%d targets", n_enriched, len(targets))
    return targets


def _compute_galactic_latitude(ra_deg: float, dec_deg: float) -> float:
    """Compute galactic latitude from equatorial coordinates.

    Uses the standard transformation from J2000 equatorial to
    galactic coordinates.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Right ascension and declination in degrees (J2000).

    Returns
    -------
    float
        Galactic latitude in degrees.
    """
    # North galactic pole (J2000): RA=192.8595, Dec=+27.1284
    # Galactic centre longitude: l_NCP = 122.932
    ra_ngp = np.radians(192.8595)
    dec_ngp = np.radians(27.1284)

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    sin_b = (
        np.sin(dec) * np.sin(dec_ngp)
        + np.cos(dec) * np.cos(dec_ngp) * np.cos(ra - ra_ngp)
    )
    b_gal = np.degrees(np.arcsin(np.clip(sin_b, -1.0, 1.0)))

    return float(b_gal)


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("  Project EXODUS -- Target Loader Demo")
    print("=" * 70)

    # Create a test target file
    test_data = {
        "campaign": "test_campaign",
        "description": "Unit test for target loader",
        "phase": "calibration",
        "targets": [
            {
                "target_id": "Vega",
                "host_star": "alpha_Lyr",
                "ra": 279.2347,
                "dec": 38.7837,
                "distance_pc": 7.7,
                "hz_flag": False,
                "is_positive_control": True,
                "expected_channels": {"ir_excess": "positive"},
                "notes": "Known debris disk",
            },
            {
                "target_id": "51_Peg",
                "host_star": "51_Peg",
                "ra": 344.3667,
                "dec": 20.7689,
                "distance_pc": 15.5,
                "hz_flag": False,
                "is_negative_control": True,
                "expected_channels": {
                    "ir_excess": "negative",
                    "transit_anomaly": "negative",
                },
                "notes": "First exoplanet host, boring",
            },
            {
                "target_id": "Proxima_Centauri_b",
                "host_star": "Proxima_Centauri",
                "ra": 217.4290,
                "dec": -62.6794,
                "distance_pc": 1.3,
                "hz_flag": True,
                "notes": "Closest HZ planet",
            },
        ],
    }

    # Write test file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(test_data, f, indent=2)
        test_path = f.name

    # ------------------------------------------------------------------
    # Test 1: Load target file
    # ------------------------------------------------------------------
    print("\n[1] Loading target file")
    print("-" * 50)

    campaign = load_target_file(test_path)
    print(campaign.summary())
    assert campaign.n_targets == 3, f"Expected 3 targets, got {campaign.n_targets}"
    assert campaign.campaign == "test_campaign"
    assert len(campaign.positive_controls) == 1
    assert len(campaign.negative_controls) == 1
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 2: Validate format
    # ------------------------------------------------------------------
    print("\n[2] Validating target format")
    print("-" * 50)

    warnings = validate_target_format(test_data["targets"])
    assert len(warnings) == 0, f"Unexpected warnings: {warnings}"
    print(f"  No warnings: PASS")

    # Test with bad data
    bad_targets = [
        {"target_id": "bad1", "ra": 400, "dec": 0},  # RA out of range
        {"ra": 100, "dec": 100},  # missing fields, Dec out of range
    ]
    bad_warnings = validate_target_format(bad_targets)
    assert len(bad_warnings) >= 2, f"Expected warnings, got {bad_warnings}"
    print(f"  Bad data caught ({len(bad_warnings)} warnings): PASS")

    # ------------------------------------------------------------------
    # Test 3: Galactic latitude computation
    # ------------------------------------------------------------------
    print("\n[3] Galactic latitude computation")
    print("-" * 50)

    # Galactic centre: RA~266.4, Dec~-28.9 -> b~0
    b_gc = _compute_galactic_latitude(266.4, -28.9)
    print(f"  Galactic centre: b = {b_gc:.1f} deg (should be ~0)")
    assert abs(b_gc) < 5, f"Galactic centre should have b~0, got {b_gc}"

    # North galactic pole: RA=192.86, Dec=+27.13 -> b~90
    b_ngp = _compute_galactic_latitude(192.86, 27.13)
    print(f"  North galactic pole: b = {b_ngp:.1f} deg (should be ~90)")
    assert abs(b_ngp - 90) < 2, f"NGP should have b~90, got {b_ngp}"
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 4: Enrichment (without Gaia -- just galactic latitude)
    # ------------------------------------------------------------------
    print("\n[4] Target metadata enrichment")
    print("-" * 50)

    enriched = enrich_target_metadata(campaign.targets)
    for t in enriched:
        assert "b_gal" in t and t["b_gal"] is not None, \
            f"Missing b_gal for {t['target_id']}"
        print(f"  {t['target_id']:>25s}  b_gal = {t['b_gal']:>+6.1f} deg")
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 5: Target normalisation
    # ------------------------------------------------------------------
    print("\n[5] Target normalisation")
    print("-" * 50)

    for t in campaign.targets:
        assert t["source"] == "custom_target_file"
        assert isinstance(t["expected_channels"], dict)
        assert isinstance(t["ra"], float)
        assert isinstance(t["dec"], float)
    print("  All targets correctly normalised: PASS")

    # Cleanup
    Path(test_path).unlink()

    print("\n" + "=" * 70)
    print("  All tests passed.")
    print("=" * 70)
