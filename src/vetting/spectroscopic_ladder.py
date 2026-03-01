"""
Spectroscopic Follow-up Ladder for Project EXODUS.

Before requesting telescope time for a candidate, check archival
spectroscopic databases for existing observations. This ladder queries
(in order of accessibility):

  1. SDSS DR18 (optical spectra, ~5M objects)
  2. LAMOST DR10 (optical spectra, ~20M objects, strong in northern sky)
  3. GALAH DR4 (high-res optical, ~0.9M southern FGK stars)
  4. APOGEE DR17 (near-IR H-band, ~0.7M objects)

For each match, we extract:
  - Spectral classification (star type, luminosity class)
  - Radial velocity (binary indicator)
  - Chemical abundances (if available)
  - Any spectral peculiarities (emission lines, broad features)

The output is a FollowUpDecision:
  - SKIP: Spectrum explains the anomaly (e.g. SB2 binary explains PM)
  - ARCHIVE_SUFFICIENT: Good spectrum exists, no new observation needed
  - NEEDS_FOLLOWUP: No archival spectrum, or spectrum shows something interesting
  - HIGH_PRIORITY: Archival spectrum shows something genuinely anomalous

Public API
----------
    check_archival_spectra(ra, dec, target_id) -> SpectroscopicResult
    batch_check(targets) -> dict[str, SpectroscopicResult]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("vetting.spectroscopic_ladder")

SEARCH_RADIUS_ARCSEC = 3.0  # Tight match for spectroscopic targets


@dataclass
class SpectroscopicResult:
    """Result of archival spectroscopic search."""
    target_id: str = ""

    # Which surveys have spectra
    has_sdss: bool = False
    has_lamost: bool = False
    has_galah: bool = False
    has_apogee: bool = False
    n_surveys: int = 0

    # Best spectral classification
    spec_type: Optional[str] = None        # e.g. "F5V", "K2III", "M3"
    spec_source: Optional[str] = None      # Which survey provided it

    # Radial velocity
    rv_km_s: Optional[float] = None
    rv_err_km_s: Optional[float] = None
    rv_n_obs: int = 0
    rv_variable: bool = False              # Multiple RV measurements differ

    # Chemical abundances
    fe_h: Optional[float] = None           # [Fe/H]
    alpha_fe: Optional[float] = None       # [alpha/Fe]

    # Spectral flags
    emission_lines: bool = False           # H-alpha, Ca II emission
    broad_lines: bool = False              # Indicates rapid rotation or AGN
    peculiar: bool = False                 # Any unusual spectral features

    # Decision
    decision: str = "NEEDS_FOLLOWUP"
    # SKIP, ARCHIVE_SUFFICIENT, NEEDS_FOLLOWUP, HIGH_PRIORITY
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "has_sdss": self.has_sdss,
            "has_lamost": self.has_lamost,
            "has_galah": self.has_galah,
            "has_apogee": self.has_apogee,
            "n_surveys": self.n_surveys,
            "spec_type": self.spec_type,
            "spec_source": self.spec_source,
            "rv_km_s": self.rv_km_s,
            "rv_err_km_s": self.rv_err_km_s,
            "rv_n_obs": self.rv_n_obs,
            "rv_variable": self.rv_variable,
            "fe_h": self.fe_h,
            "alpha_fe": self.alpha_fe,
            "emission_lines": self.emission_lines,
            "broad_lines": self.broad_lines,
            "peculiar": self.peculiar,
            "decision": self.decision,
            "reason": self.reason,
        }


def check_archival_spectra(
    ra: float,
    dec: float,
    target_id: str = "unknown",
    pipeline_data: Optional[Dict] = None,
) -> SpectroscopicResult:
    """Check archival spectroscopic databases for a target.

    Parameters
    ----------
    ra, dec : float
        Position in degrees.
    target_id : str
        Target identifier.
    pipeline_data : dict, optional
        EXODUS pipeline data for this target (used to cross-reference
        anomaly channels with spectral findings).
    """
    result = SpectroscopicResult(target_id=target_id)

    # Query each survey in order of size/accessibility
    sdss = _query_sdss(ra, dec)
    if sdss:
        result.has_sdss = True
        _merge_spectral_info(result, sdss, "SDSS")

    lamost = _query_lamost(ra, dec)
    if lamost:
        result.has_lamost = True
        _merge_spectral_info(result, lamost, "LAMOST")

    galah = _query_galah(ra, dec)
    if galah:
        result.has_galah = True
        _merge_spectral_info(result, galah, "GALAH")

    apogee = _query_apogee(ra, dec)
    if apogee:
        result.has_apogee = True
        _merge_spectral_info(result, apogee, "APOGEE")

    result.n_surveys = sum([
        result.has_sdss, result.has_lamost,
        result.has_galah, result.has_apogee,
    ])

    # Make follow-up decision
    _decide(result, pipeline_data)

    return result


def batch_check(
    targets: List[Dict[str, Any]],
) -> Dict[str, SpectroscopicResult]:
    """Check archival spectra for a batch of targets."""
    results = {}
    for i, t in enumerate(targets):
        tid = t.get("target_id", f"target_{i}")
        ra, dec = t["ra"], t["dec"]

        if (i + 1) % 20 == 0 or i == 0:
            log.info("Spectroscopic ladder: %d/%d", i + 1, len(targets))

        results[tid] = check_archival_spectra(
            ra, dec, target_id=tid, pipeline_data=t
        )

    # Summary
    n_with = sum(1 for r in results.values() if r.n_surveys > 0)
    n_skip = sum(1 for r in results.values() if r.decision == "SKIP")
    n_high = sum(1 for r in results.values() if r.decision == "HIGH_PRIORITY")
    log.info(
        "Spectroscopic ladder: %d/%d have archival spectra, %d SKIP, %d HIGH_PRIORITY",
        n_with, len(results), n_skip, n_high,
    )
    return results


# ── Survey query functions ───────────────────────────────────────────

def _query_sdss(ra: float, dec: float) -> Optional[Dict]:
    """Query SDSS DR18 via VizieR for spectroscopic data."""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        v = Vizier(columns=["*"], row_limit=1)
        # V/154 = SDSS DR18 SpecObj
        tables = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
                                catalog="V/154/sdss18")
        if tables and len(tables) > 0 and len(tables[0]) > 0:
            row = tables[0][0]
            result = {"survey": "SDSS"}
            for col in ("spCl", "subCl", "zsp", "e_zsp", "snMedian"):
                if col in tables[0].colnames:
                    val = row[col]
                    if val is not None:
                        try:
                            result[col] = float(val) if col != "spCl" and col != "subCl" else str(val)
                        except (ValueError, TypeError):
                            result[col] = str(val)
            return result
        return None
    except Exception as exc:
        log.debug("SDSS query failed: %s", exc)
        return None


def _query_lamost(ra: float, dec: float) -> Optional[Dict]:
    """Query LAMOST DR10 via VizieR."""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        v = Vizier(columns=["*"], row_limit=1)
        # V/164 = LAMOST DR10 general catalog
        tables = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
                                catalog="V/164")
        if tables and len(tables) > 0 and len(tables[0]) > 0:
            row = tables[0][0]
            result = {"survey": "LAMOST"}
            for col in ("SpType", "RV", "e_RV", "Teff", "logg", "[Fe/H]"):
                if col in tables[0].colnames:
                    val = row[col]
                    if val is not None:
                        try:
                            result[col] = float(val) if col != "SpType" else str(val)
                        except (ValueError, TypeError):
                            result[col] = str(val)
            return result
        return None
    except Exception as exc:
        log.debug("LAMOST query failed: %s", exc)
        return None


def _query_galah(ra: float, dec: float) -> Optional[Dict]:
    """Query GALAH DR4 via VizieR."""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        v = Vizier(columns=["*"], row_limit=1)
        # III/290 = GALAH DR4
        tables = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
                                catalog="III/290")
        if tables and len(tables) > 0 and len(tables[0]) > 0:
            row = tables[0][0]
            result = {"survey": "GALAH"}
            for col in ("rv_galah", "e_rv_galah", "teff", "logg",
                        "fe_h", "alpha_fe", "flag_sp"):
                if col in tables[0].colnames:
                    val = row[col]
                    if val is not None:
                        try:
                            result[col] = float(val)
                        except (ValueError, TypeError):
                            result[col] = str(val)
            return result
        return None
    except Exception as exc:
        log.debug("GALAH query failed: %s", exc)
        return None


def _query_apogee(ra: float, dec: float) -> Optional[Dict]:
    """Query APOGEE DR17 via VizieR."""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        v = Vizier(columns=["*"], row_limit=1)
        # III/284 = APOGEE DR17
        tables = v.query_region(coord, radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
                                catalog="III/284")
        if tables and len(tables) > 0 and len(tables[0]) > 0:
            row = tables[0][0]
            result = {"survey": "APOGEE"}
            for col in ("Vhelio", "e_Vhelio", "Teff", "logg",
                        "[M/H]", "[a/M]", "ASPCAP"):
                if col in tables[0].colnames:
                    val = row[col]
                    if val is not None:
                        try:
                            result[col] = float(val) if col != "ASPCAP" else str(val)
                        except (ValueError, TypeError):
                            result[col] = str(val)
            return result
        return None
    except Exception as exc:
        log.debug("APOGEE query failed: %s", exc)
        return None


# ── Merge and decision logic ─────────────────────────────────────────

def _merge_spectral_info(result: SpectroscopicResult, data: Dict, source: str):
    """Merge spectral info from one survey into the result."""
    # Spectral type (prefer first available)
    if result.spec_type is None:
        sp = data.get("SpType") or data.get("spCl")
        if sp and str(sp).strip() and str(sp).strip() != "--":
            result.spec_type = str(sp).strip()
            result.spec_source = source

    # Radial velocity
    rv_keys = {
        "SDSS": None,  # SDSS uses redshift, not RV
        "LAMOST": "RV",
        "GALAH": "rv_galah",
        "APOGEE": "Vhelio",
    }
    rv_key = rv_keys.get(source)
    if rv_key and rv_key in data:
        rv = data[rv_key]
        if rv is not None and np.isfinite(rv):
            if result.rv_km_s is not None:
                # Multiple RV measurements — check for variability
                delta = abs(rv - result.rv_km_s)
                if delta > 10.0:  # >10 km/s difference = likely binary
                    result.rv_variable = True
            result.rv_km_s = float(rv)
            result.rv_n_obs += 1

            rv_err_key = f"e_{rv_key}" if f"e_{rv_key}" in data else None
            if rv_err_key:
                result.rv_err_km_s = float(data[rv_err_key])

    # Metallicity
    fe_h_keys = {"LAMOST": "[Fe/H]", "GALAH": "fe_h", "APOGEE": "[M/H]"}
    feh_key = fe_h_keys.get(source)
    if feh_key and feh_key in data:
        val = data[feh_key]
        if val is not None and np.isfinite(val):
            result.fe_h = float(val)

    # Alpha enhancement
    if source == "GALAH" and "alpha_fe" in data:
        val = data["alpha_fe"]
        if val is not None and np.isfinite(val):
            result.alpha_fe = float(val)
    elif source == "APOGEE" and "[a/M]" in data:
        val = data["[a/M]"]
        if val is not None and np.isfinite(val):
            result.alpha_fe = float(val)


def _decide(result: SpectroscopicResult, pipeline_data: Optional[Dict]):
    """Make follow-up decision based on spectroscopic findings."""
    reasons = []

    # No archival spectra → needs follow-up
    if result.n_surveys == 0:
        result.decision = "NEEDS_FOLLOWUP"
        result.reason = "No archival spectra found"
        return

    # RV variability = likely binary → explains PM anomaly
    if result.rv_variable:
        reasons.append("RV variable (likely SB binary)")
        # If the pipeline anomaly is PM-only, this explains it
        if pipeline_data:
            channels = _get_active_channels(pipeline_data)
            if channels == {"proper_motion_anomaly"} or channels == {"proper_motion_anomaly", "ir_excess"}:
                result.decision = "SKIP"
                result.reason = "SB binary (RV variable) explains PM anomaly"
                return

    # Known spectral type that explains anomaly
    if result.spec_type:
        sp_lower = result.spec_type.lower().strip()
        # Giant/subgiant with IR excess → normal circumstellar dust
        if any(lc in sp_lower for lc in ("iii", "iv", "ii")):
            reasons.append(f"Luminosity class {result.spec_type}")
            if pipeline_data and "ir_excess" in _get_active_channels(pipeline_data):
                result.decision = "SKIP"
                result.reason = f"Giant/subgiant ({result.spec_type}) — IR excess is normal"
                return

        # Emission-line star → active chromosphere
        if "e" in sp_lower:
            result.emission_lines = True
            reasons.append("Emission-line star")

    # Peculiar metallicity (very metal-poor or metal-rich)
    if result.fe_h is not None and abs(result.fe_h) > 1.0:
        reasons.append(f"Extreme metallicity [Fe/H]={result.fe_h:.2f}")
        result.peculiar = True

    # High alpha enhancement → old thick disk/halo population
    if result.alpha_fe is not None and result.alpha_fe > 0.3:
        reasons.append(f"High [alpha/Fe]={result.alpha_fe:.2f}")

    # Decision
    if result.peculiar or result.emission_lines:
        result.decision = "HIGH_PRIORITY"
        result.reason = "Spectral peculiarities: " + "; ".join(reasons)
    elif reasons:
        result.decision = "ARCHIVE_SUFFICIENT"
        result.reason = "; ".join(reasons)
    else:
        result.decision = "ARCHIVE_SUFFICIENT"
        result.reason = f"Normal spectrum ({result.spec_type or 'untyped'}) from {result.n_surveys} survey(s)"


def _get_active_channels(pipeline_data: Dict) -> set:
    """Extract active channel names from pipeline data."""
    score_data = pipeline_data.get("exodus_score", {})
    channels = score_data.get("channel_scores", {})
    active = set()
    for ch_name, ch_data in channels.items():
        if isinstance(ch_data, dict) and ch_data.get("is_active"):
            active.add(ch_name)
    return active


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Spectroscopic Follow-up Ladder — check archival spectra"
    )
    parser.add_argument("--targets", help="Target JSON file")
    parser.add_argument("--report", help="EXODUS report JSON (uses top_targets)")
    parser.add_argument("--ra", type=float, help="Single target RA (deg)")
    parser.add_argument("--dec", type=float, help="Single target Dec (deg)")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    if args.ra is not None and args.dec is not None:
        # Single target mode
        r = check_archival_spectra(args.ra, args.dec, target_id="manual")
        print(f"  Spec type: {r.spec_type} ({r.spec_source})")
        print(f"  Surveys:   {r.n_surveys} ({', '.join(s for s, h in [('SDSS', r.has_sdss), ('LAMOST', r.has_lamost), ('GALAH', r.has_galah), ('APOGEE', r.has_apogee)] if h)})")
        print(f"  RV:        {r.rv_km_s} km/s (variable={r.rv_variable})")
        print(f"  [Fe/H]:    {r.fe_h}")
        print(f"  Decision:  {r.decision}")
        print(f"  Reason:    {r.reason}")

    elif args.report:
        # Report mode — check top targets from a pipeline report
        with open(args.report) as f:
            report = json.load(f)
        top = report.get("top_targets", [])
        if not top:
            top = report.get("all_scored", [])[:20]
        print(f"Checking {len(top)} targets from report...")
        results = batch_check(top)

        for tid, r in results.items():
            icon = {"SKIP": "X", "ARCHIVE_SUFFICIENT": "~",
                    "NEEDS_FOLLOWUP": "?", "HIGH_PRIORITY": "!"}[r.decision]
            print(f"  [{icon}] {tid}: {r.decision} — {r.reason}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump({tid: r.to_dict() for tid, r in results.items()},
                          f, indent=2, default=str)

    elif args.targets:
        with open(args.targets) as f:
            data = json.load(f)
        targets = data.get("targets", data) if isinstance(data, dict) else data
        results = batch_check(targets)

        for tid, r in results.items():
            if r.decision in ("SKIP", "HIGH_PRIORITY"):
                print(f"  {r.decision}: {tid} — {r.reason}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump({tid: r.to_dict() for tid, r in results.items()},
                          f, indent=2, default=str)
    else:
        parser.print_help()
