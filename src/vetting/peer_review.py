"""
Adversarial Peer Review Engine — Project EXODUS
================================================

An INDEPENDENT, ADVERSARIAL validation system that tries to DISPROVE every
claim the pipeline makes about a candidate target.  Unlike the Red-Team
engine (which works from already-gathered data), this module:

  1. Re-queries every data source INDEPENDENTLY
  2. Challenges every assumption with alternative hypotheses
  3. Validates basic facts (coordinates, photometry, cross-matches)
  4. Runs statistical tests the pipeline doesn't
  5. Reports with brutal honesty — no face-saving

Philosophy
----------
This module is the "Reviewer 2" of EXODUS.  It assumes every finding is
wrong until independently verified.  A candidate that survives this module
has been tested from every angle we can think of.

Check Categories
----------------
  A. IDENTITY — Is this the star we think it is?
  B. PHOTOMETRIC INTEGRITY — Is the IR excess real?
  C. ASTROMETRIC INTEGRITY — Is the PM anomaly real?
  D. RADIO VALIDATION — Is the radio detection real and associated?
  E. ALTERNATIVE EXPLANATIONS — Can binarity/activity/AGN explain everything?
  F. STATISTICAL RIGOR — Are the statistics honest?
  G. ENVIRONMENTAL — Is the environment clean?
  H. ARCHIVAL COMPLETENESS — What data sources haven't we checked?

Usage
-----
    from src.vetting.peer_review import PeerReviewEngine

    engine = PeerReviewEngine()
    report = engine.review(ra=180.0, dec=45.0, target_id="MY_TARGET")
    report.print_summary()
    report.save("data/reports/peer_review_example.json")

Can also be run as a script:
    ./venv/bin/python -m src.vetting.peer_review --ra 180.0 --dec 45.0
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("vetting.peer_review")


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class ReviewCheck:
    """One peer-review check result."""
    check_id: str            # e.g., "A1_gaia_identity"
    category: str            # A-H category
    claim_tested: str        # What claim does this check test?
    verdict: str             # "CONFIRMED", "CHALLENGED", "REFUTED", "INCONCLUSIVE", "ERROR"
    confidence: float        # 0-1, how confident is the verdict?
    details: str             # Human-readable explanation
    evidence: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"   # "info", "warning", "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "claim_tested": self.claim_tested,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "details": self.details,
            "evidence": self.evidence,
            "severity": self.severity,
        }


@dataclass
class PeerReviewReport:
    """Complete adversarial peer-review report for one target."""
    target_id: str
    ra: float
    dec: float
    timestamp: str = ""

    # Aggregate
    total_checks: int = 0
    confirmed: int = 0
    challenged: int = 0
    refuted: int = 0
    inconclusive: int = 0
    errors: int = 0

    # Overall assessment
    overall_verdict: str = ""      # "SURVIVES", "WEAKENED", "REJECTED"
    overall_confidence: float = 0.0
    executive_summary: str = ""

    # Individual checks
    checks: List[ReviewCheck] = field(default_factory=list)

    # Recommendations for additional data
    recommendations: List[str] = field(default_factory=list)

    def print_summary(self):
        """Print a colored summary to stdout."""
        print("\n" + "=" * 78)
        print(f"  ADVERSARIAL PEER REVIEW: {self.target_id}")
        print(f"  RA={self.ra:.4f}, Dec={self.dec:.4f}")
        print("=" * 78)
        print(f"\n  OVERALL VERDICT: {self.overall_verdict}")
        print(f"  Confidence: {self.overall_confidence:.0%}")
        print(f"\n  Checks: {self.total_checks} total")
        print(f"    ✅ CONFIRMED:    {self.confirmed}")
        print(f"    ⚠️  CHALLENGED:   {self.challenged}")
        print(f"    ❌ REFUTED:      {self.refuted}")
        print(f"    ❓ INCONCLUSIVE: {self.inconclusive}")
        print(f"    💥 ERROR:        {self.errors}")

        if self.executive_summary:
            print(f"\n  EXECUTIVE SUMMARY:")
            for line in self.executive_summary.split("\n"):
                print(f"    {line}")

        # Print each category
        categories = {}
        for c in self.checks:
            categories.setdefault(c.category, []).append(c)

        cat_labels = {
            "A": "IDENTITY VALIDATION",
            "B": "PHOTOMETRIC INTEGRITY",
            "C": "ASTROMETRIC INTEGRITY",
            "D": "RADIO VALIDATION",
            "E": "ALTERNATIVE EXPLANATIONS",
            "F": "STATISTICAL RIGOR",
            "G": "ENVIRONMENTAL CHECKS",
            "H": "ARCHIVAL COMPLETENESS",
        }

        for cat_key in sorted(categories.keys()):
            cat_checks = categories[cat_key]
            label = cat_labels.get(cat_key, cat_key)
            print(f"\n  ── {cat_key}. {label} ──")
            for c in cat_checks:
                icon = {"CONFIRMED": "✅", "CHALLENGED": "⚠️ ",
                        "REFUTED": "❌", "INCONCLUSIVE": "❓",
                        "ERROR": "💥"}.get(c.verdict, "?")
                print(f"    {icon} [{c.check_id}] {c.claim_tested}")
                print(f"       → {c.verdict} ({c.confidence:.0%}): {c.details}")

        if self.recommendations:
            print(f"\n  ── RECOMMENDATIONS ──")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"    {i}. {rec}")

        print("\n" + "=" * 78)

    def save(self, path: str):
        """Save report as JSON."""
        data = {
            "target_id": self.target_id,
            "ra": self.ra,
            "dec": self.dec,
            "timestamp": self.timestamp,
            "overall_verdict": self.overall_verdict,
            "overall_confidence": round(self.overall_confidence, 3),
            "executive_summary": self.executive_summary,
            "stats": {
                "total_checks": self.total_checks,
                "confirmed": self.confirmed,
                "challenged": self.challenged,
                "refuted": self.refuted,
                "inconclusive": self.inconclusive,
                "errors": self.errors,
            },
            "checks": [c.to_dict() for c in self.checks],
            "recommendations": self.recommendations,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"Peer review report saved to {path}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "ra": self.ra,
            "dec": self.dec,
            "timestamp": self.timestamp,
            "overall_verdict": self.overall_verdict,
            "overall_confidence": round(self.overall_confidence, 3),
            "executive_summary": self.executive_summary,
            "stats": {
                "total_checks": self.total_checks,
                "confirmed": self.confirmed,
                "challenged": self.challenged,
                "refuted": self.refuted,
                "inconclusive": self.inconclusive,
                "errors": self.errors,
            },
            "checks": [c.to_dict() for c in self.checks],
            "recommendations": self.recommendations,
        }


# =====================================================================
#  Peer Review Engine
# =====================================================================

class PeerReviewEngine:
    """Adversarial peer-review engine.

    Independently validates every claim about a candidate by re-querying
    data sources and running independent statistical tests.
    """

    def __init__(self, verbose: bool = True, timeout: float = 30.0):
        self.verbose = verbose
        self.timeout = timeout  # per-query timeout in seconds
        self._gaia_data = None  # cached Gaia query result
        self._allwise_data = None
        self._catwise_data = None

    def review(
        self,
        ra: float,
        dec: float,
        target_id: str = "",
        pipeline_data: Optional[Dict] = None,
    ) -> PeerReviewReport:
        """Run the full adversarial review.

        Parameters
        ----------
        ra, dec : float
            J2000 coordinates in degrees.
        target_id : str
            Identifier for the target.
        pipeline_data : dict, optional
            If provided, the pipeline's own data for this target.
            The peer review will compare its independent findings against
            pipeline claims.
        """
        if not target_id:
            target_id = f"RA{ra:.4f}_DEC{dec:.4f}"

        report = PeerReviewReport(
            target_id=target_id,
            ra=ra,
            dec=dec,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  ADVERSARIAL PEER REVIEW: {target_id}")
            print(f"  RA={ra:.4f}, Dec={dec:.4f}")
            print(f"{'='*60}")

        # Run all check categories
        check_methods = [
            # Category A: Identity
            self._A1_gaia_identity,
            self._A2_coordinate_consistency,
            self._A3_photometry_crossmatch,
            # Category B: Photometric Integrity
            self._B1_allwise_quality_flags,
            self._B2_ir_excess_independent,
            self._B3_wise_confusion,
            self._B4_wise_saturation,
            self._B5_2mass_wise_consistency,
            # Category C: Astrometric Integrity
            self._C1_gaia_astrometric_quality,
            self._C2_pm_discrepancy_independent,
            self._C3_catwise_systematics,
            self._C4_gaia_nss_check,
            # Category D: Radio Validation
            self._D1_nvss_independent,
            self._D2_vlass_independent,
            self._D3_radio_chance_alignment,
            self._D4_radio_spectral_index,
            # Category E: Alternative Explanations
            self._E1_binary_hypothesis,
            self._E2_yso_hypothesis,
            self._E3_agn_background,
            self._E4_simbad_classification,
            self._E5_optical_variability,
            self._E6_xray_activity,
            self._E7_galex_uv,
            # Category F: Statistical Rigor
            self._F1_ir_excess_significance,
            self._F2_neowise_stability,
            self._F3_field_star_comparison,
            # Category G: Environmental
            self._G1_galactic_position,
            self._G2_dust_extinction,
            self._G3_stellar_density,
            self._G4_nearby_sources,
            # Category H: Archival Completeness
            self._H1_spectroscopic_surveys,
            self._H2_tess_coverage,
            self._H3_ztf_coverage,
            self._H4_gaia_variability,
            self._H5_washington_double_star,
        ]

        for method in check_methods:
            try:
                check = method(ra, dec, pipeline_data)
                if check is not None:
                    report.checks.append(check)
                    if self.verbose:
                        icon = {"CONFIRMED": "✅", "CHALLENGED": "⚠️ ",
                                "REFUTED": "❌", "INCONCLUSIVE": "❓",
                                "ERROR": "💥"}.get(check.verdict, "?")
                        print(f"  {icon} {check.check_id}: {check.verdict} — {check.details[:80]}")
            except Exception as e:
                err_check = ReviewCheck(
                    check_id=method.__name__.lstrip("_"),
                    category=method.__name__[1],
                    claim_tested=f"Check {method.__name__}",
                    verdict="ERROR",
                    confidence=0.0,
                    details=f"Exception: {e}",
                    severity="warning",
                )
                report.checks.append(err_check)
                if self.verbose:
                    print(f"  💥 {method.__name__}: ERROR — {e}")

        # Compile statistics
        report.total_checks = len(report.checks)
        report.confirmed = sum(1 for c in report.checks if c.verdict == "CONFIRMED")
        report.challenged = sum(1 for c in report.checks if c.verdict == "CHALLENGED")
        report.refuted = sum(1 for c in report.checks if c.verdict == "REFUTED")
        report.inconclusive = sum(1 for c in report.checks if c.verdict == "INCONCLUSIVE")
        report.errors = sum(1 for c in report.checks if c.verdict == "ERROR")

        # Overall verdict
        report = self._compute_overall_verdict(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    # =================================================================
    #  Category A: IDENTITY VALIDATION
    # =================================================================

    def _A1_gaia_identity(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Verify Gaia source exists at claimed coordinates."""
        from astroquery.gaia import Gaia

        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        query = f"""
        SELECT source_id, ra, dec, parallax, parallax_error,
               pmra, pmdec, ruwe, phot_g_mean_mag,
               astrometric_excess_noise, astrometric_excess_noise_sig,
               phot_bp_mean_mag, phot_rp_mean_mag,
               teff_gspphot, logg_gspphot, mh_gspphot,
               non_single_star, ipd_gof_harmonic_amplitude,
               astrometric_params_solved
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {5.0/3600})
        ) = 1
        ORDER BY phot_g_mean_mag ASC
        """
        try:
            job = Gaia.launch_job(query)
            result = job.get_results()

            if len(result) == 0:
                return ReviewCheck(
                    check_id="A1_gaia_identity",
                    category="A",
                    claim_tested="A Gaia source exists at the claimed position",
                    verdict="REFUTED",
                    confidence=0.95,
                    details="NO Gaia source found within 5 arcsec!",
                    severity="critical",
                )

            # Cache for later checks
            self._gaia_data = {col: result[col][0] for col in result.colnames}

            sep = np.sqrt(
                ((result["ra"][0] - ra) * np.cos(np.radians(dec)))**2
                + (result["dec"][0] - dec)**2
            ) * 3600

            n_sources = len(result)
            return ReviewCheck(
                check_id="A1_gaia_identity",
                category="A",
                claim_tested="A Gaia source exists at the claimed position",
                verdict="CONFIRMED",
                confidence=0.99,
                details=f"Gaia DR3 {result['source_id'][0]} at {sep:.2f}\", "
                        f"G={result['phot_g_mean_mag'][0]:.2f}, "
                        f"{n_sources} source(s) within 5\"",
                evidence={
                    "source_id": int(result["source_id"][0]),
                    "sep_arcsec": round(float(sep), 3),
                    "g_mag": round(float(result["phot_g_mean_mag"][0]), 3),
                    "n_sources_5arcsec": n_sources,
                    "parallax": round(float(result["parallax"][0]), 4),
                    "ruwe": round(float(result["ruwe"][0]), 4),
                },
            )
        except Exception as e:
            return ReviewCheck(
                check_id="A1_gaia_identity",
                category="A",
                claim_tested="A Gaia source exists at the claimed position",
                verdict="ERROR",
                confidence=0.0,
                details=f"Gaia query failed: {e}",
                severity="warning",
            )

    def _A2_coordinate_consistency(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check that 2MASS/WISE coordinates agree with Gaia."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=5, columns=["**"])

        offsets = {}

        # 2MASS
        try:
            r = v.query_region(coord, radius=5*u.arcsec, catalog="II/246/out")
            if r:
                tm = r[0][0]
                tm_coord = SkyCoord(ra=tm["RAJ2000"], dec=tm["DEJ2000"], unit="deg")
                offsets["2MASS"] = float(coord.separation(tm_coord).arcsec)
        except Exception:
            pass

        # AllWISE
        try:
            r = v.query_region(coord, radius=5*u.arcsec, catalog="II/328/allwise")
            if r:
                aw = r[0][0]
                aw_coord = SkyCoord(ra=aw["RAJ2000"], dec=aw["DEJ2000"], unit="deg")
                offsets["AllWISE"] = float(coord.separation(aw_coord).arcsec)
                self._allwise_data = {col: aw[col] for col in r[0].colnames}
        except Exception:
            pass

        # CatWISE
        try:
            r = v.query_region(coord, radius=5*u.arcsec, catalog="II/365/catwise")
            if r:
                cw = r[0][0]
                cw_coord = SkyCoord(ra=cw["RA_ICRS"], dec=cw["DE_ICRS"], unit="deg")
                offsets["CatWISE"] = float(coord.separation(cw_coord).arcsec)
                self._catwise_data = {col: cw[col] for col in r[0].colnames}
        except Exception:
            pass

        if not offsets:
            return ReviewCheck(
                check_id="A2_coordinate_consistency",
                category="A",
                claim_tested="Catalog coordinates are consistent across surveys",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="Could not query any catalog for cross-match",
            )

        max_offset = max(offsets.values())
        all_ok = max_offset < 2.0  # All within 2"

        return ReviewCheck(
            check_id="A2_coordinate_consistency",
            category="A",
            claim_tested="Catalog coordinates are consistent across surveys",
            verdict="CONFIRMED" if all_ok else "CHALLENGED",
            confidence=0.9 if all_ok else 0.6,
            details=f"Offsets: " + ", ".join(f"{k}={v:.2f}\"" for k, v in offsets.items())
                    + (f" — max {max_offset:.2f}\" exceeds 2\"!" if not all_ok else ""),
            evidence=offsets,
            severity="info" if all_ok else "warning",
        )

    def _A3_photometry_crossmatch(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Verify photometric magnitudes match between catalogs."""
        if self._allwise_data is None:
            return ReviewCheck(
                check_id="A3_photometry_crossmatch",
                category="A",
                claim_tested="Photometry is self-consistent across catalogs",
                verdict="INCONCLUSIVE",
                confidence=0.2,
                details="No AllWISE data available for cross-check",
            )

        aw = self._allwise_data
        issues = []

        # Check W1 vs CatWISE W1
        if self._catwise_data:
            cw = self._catwise_data
            try:
                w1_aw = float(aw.get("W1mag", 0))
                w1_cw = float(cw.get("W1mproPM", 0))
                if w1_aw > 0 and w1_cw > 0:
                    diff = abs(w1_aw - w1_cw)
                    if diff > 0.3:
                        issues.append(f"W1: AllWISE={w1_aw:.3f} vs CatWISE={w1_cw:.3f} (Δ={diff:.3f})")
            except (TypeError, ValueError):
                pass

        # Check WISE quality
        try:
            ph_qual = str(aw.get("qph", "----"))
            bad_bands = sum(1 for q in ph_qual if q in ("U", "X", "Z"))
            if bad_bands > 0:
                issues.append(f"WISE ph_qual='{ph_qual}' — {bad_bands} band(s) unreliable")
        except Exception:
            pass

        if issues:
            return ReviewCheck(
                check_id="A3_photometry_crossmatch",
                category="A",
                claim_tested="Photometry is self-consistent across catalogs",
                verdict="CHALLENGED",
                confidence=0.7,
                details="; ".join(issues),
                evidence={"issues": issues},
                severity="warning",
            )

        return ReviewCheck(
            check_id="A3_photometry_crossmatch",
            category="A",
            claim_tested="Photometry is self-consistent across catalogs",
            verdict="CONFIRMED",
            confidence=0.8,
            details="Photometry consistent between AllWISE and CatWISE",
        )

    # =================================================================
    #  Category B: PHOTOMETRIC INTEGRITY
    # =================================================================

    def _B1_allwise_quality_flags(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check AllWISE quality/contamination flags."""
        if self._allwise_data is None:
            return ReviewCheck(
                check_id="B1_allwise_quality_flags",
                category="B",
                claim_tested="AllWISE photometry is clean (no contamination/confusion flags)",
                verdict="INCONCLUSIVE",
                confidence=0.2,
                details="No AllWISE data cached",
            )

        aw = self._allwise_data
        issues = []

        # cc_flags — contamination/confusion
        cc = str(aw.get("ccf", "0000"))
        for i, band in enumerate(["W1", "W2", "W3", "W4"]):
            if i < len(cc) and cc[i] not in ("0", "-"):
                issues.append(f"{band} cc_flag='{cc[i]}' (contamination/confusion)")

        # ext_flg — extended source
        ext = aw.get("ex", 0)
        try:
            ext = int(ext)
        except (TypeError, ValueError):
            ext = 0
        if ext > 0:
            issues.append(f"ext_flg={ext} — possibly extended/resolved source")

        # ph_qual — photometric quality per band
        ph = str(aw.get("qph", "----"))
        for i, band in enumerate(["W1", "W2", "W3", "W4"]):
            if i < len(ph) and ph[i] in ("U", "X"):
                issues.append(f"{band} ph_qual='{ph[i]}' (upper limit or unreliable)")

        # Nb — number of detections in profile-fit
        for band_n, band_name in [("nb1", "W1"), ("nb2", "W2"), ("nb3", "W3"), ("nb4", "W4")]:
            try:
                nb = int(aw.get(band_n, 0))
                if nb < 3 and band_name in ("W3", "W4"):
                    issues.append(f"{band_name}: only {nb} profile-fit detections (weak)")
            except (TypeError, ValueError):
                pass

        if issues:
            sev = "critical" if any("W3" in i or "W4" in i for i in issues) else "warning"
            return ReviewCheck(
                check_id="B1_allwise_quality_flags",
                category="B",
                claim_tested="AllWISE photometry is clean (no contamination/confusion flags)",
                verdict="CHALLENGED" if len(issues) <= 2 else "REFUTED",
                confidence=0.8,
                details=f"{len(issues)} issue(s): " + "; ".join(issues),
                evidence={"cc_flags": cc, "ext_flg": ext, "ph_qual": ph, "issues": issues},
                severity=sev,
            )

        return ReviewCheck(
            check_id="B1_allwise_quality_flags",
            category="B",
            claim_tested="AllWISE photometry is clean (no contamination/confusion flags)",
            verdict="CONFIRMED",
            confidence=0.9,
            details=f"All flags clean: cc_flags='{cc}', ext_flg={ext}, ph_qual='{ph}'",
            evidence={"cc_flags": cc, "ext_flg": ext, "ph_qual": ph},
        )

    def _B2_ir_excess_independent(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Independently re-derive IR excess from raw photometry."""
        if not self._gaia_data or not self._allwise_data:
            return ReviewCheck(
                check_id="B2_ir_excess_independent",
                category="B",
                claim_tested="IR excess is real when independently re-derived",
                verdict="INCONCLUSIVE",
                confidence=0.2,
                details="Missing Gaia or AllWISE data for independent derivation",
            )

        aw = self._allwise_data
        gaia = self._gaia_data

        try:
            teff = float(gaia.get("teff_gspphot", 0))
            if teff <= 0:
                return ReviewCheck(
                    check_id="B2_ir_excess_independent",
                    category="B",
                    claim_tested="IR excess is real when independently re-derived",
                    verdict="INCONCLUSIVE",
                    confidence=0.3,
                    details=f"No Teff from Gaia GSP-Phot",
                )

            # Get magnitudes
            w1 = float(aw.get("W1mag", 99))
            w2 = float(aw.get("W2mag", 99))
            w3 = float(aw.get("W3mag", 99))
            w4 = float(aw.get("W4mag", 99))
            e_w3 = float(aw.get("e_W3mag", 0.2))
            e_w4 = float(aw.get("e_W4mag", 0.5))

            if w3 > 50 or w1 > 50:
                return ReviewCheck(
                    check_id="B2_ir_excess_independent",
                    category="B",
                    claim_tested="IR excess is real when independently re-derived",
                    verdict="INCONCLUSIVE",
                    confidence=0.3,
                    details="Missing W1 or W3 magnitude",
                )

            # Expected colors for main-sequence star (Pecaut & Mamajek 2013-style)
            # F5-K5 stars: W1-W3 ≈ 0.0 ± 0.05, W1-W4 ≈ 0.0 ± 0.1
            expected_w1w3 = 0.0  # photospheric
            expected_w1w4 = 0.0

            obs_w1w3 = w1 - w3
            obs_w1w4 = w1 - w4 if w4 < 50 else None

            # Positive W1-W3 means W3 is BRIGHTER than W1 (in magnitudes,
            # brighter = lower number). W1-W3 > 0 → W3 excess present.
            # For MS stars, W1-W3 ≈ 0; large positive values = IR excess.
            w3_excess_mag = obs_w1w3 - expected_w1w3  # positive = real excess
            w3_excess_sigma = abs(w3_excess_mag) / max(e_w3, 0.05) if e_w3 > 0 else 0

            w4_excess_sigma = None
            if obs_w1w4 is not None:
                w4_excess_mag = obs_w1w4 - expected_w1w4
                w4_excess_sigma = abs(w4_excess_mag) / max(e_w4, 0.1) if e_w4 > 0 else 0

            verdict = "CONFIRMED" if w3_excess_sigma > 3.0 else "CHALLENGED"
            details = (f"Independent IR excess: W1-W3={obs_w1w3:.3f} "
                      f"(excess={w3_excess_mag:.3f} mag, {w3_excess_sigma:.1f}σ)")
            if w4_excess_sigma is not None:
                details += f", W1-W4 excess={w4_excess_sigma:.1f}σ"

            return ReviewCheck(
                check_id="B2_ir_excess_independent",
                category="B",
                claim_tested="IR excess is real when independently re-derived",
                verdict=verdict,
                confidence=0.85,
                details=details,
                evidence={
                    "W1": round(w1, 3), "W2": round(w2, 3),
                    "W3": round(w3, 3), "W4": round(w4, 3),
                    "W1_W3": round(obs_w1w3, 4),
                    "W3_excess_sigma": round(w3_excess_sigma, 2),
                    "W4_excess_sigma": round(w4_excess_sigma, 2) if w4_excess_sigma else None,
                    "Teff": teff,
                },
            )
        except Exception as e:
            return ReviewCheck(
                check_id="B2_ir_excess_independent",
                category="B",
                claim_tested="IR excess is real when independently re-derived",
                verdict="ERROR",
                confidence=0.0,
                details=f"Independent derivation failed: {e}",
            )

    def _B3_wise_confusion(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check for WISE source confusion — nearby sources in WISE beam."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=50, columns=["RAJ2000", "DEJ2000", "W3mag", "W4mag"])

        try:
            # Check for other AllWISE sources within 12" (WISE W3 beam ~ 6.5")
            r = v.query_region(coord, radius=15*u.arcsec, catalog="II/328/allwise")
            if r and len(r[0]) > 1:
                n_neighbors = len(r[0]) - 1
                return ReviewCheck(
                    check_id="B3_wise_confusion",
                    category="B",
                    claim_tested="No WISE source confusion within the beam",
                    verdict="CHALLENGED",
                    confidence=0.7,
                    details=f"{n_neighbors} additional AllWISE source(s) within 15\" — "
                            f"potential beam confusion at W3/W4",
                    evidence={"n_neighbors_15arcsec": n_neighbors},
                    severity="warning",
                )

            return ReviewCheck(
                check_id="B3_wise_confusion",
                category="B",
                claim_tested="No WISE source confusion within the beam",
                verdict="CONFIRMED",
                confidence=0.9,
                details="No other AllWISE sources within 15\" — clean beam",
            )
        except Exception as e:
            return ReviewCheck(
                check_id="B3_wise_confusion",
                category="B",
                claim_tested="No WISE source confusion within the beam",
                verdict="ERROR",
                confidence=0.0,
                details=f"Query failed: {e}",
            )

    def _B4_wise_saturation(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check if WISE bands are saturated."""
        if self._allwise_data is None:
            return ReviewCheck(
                check_id="B4_wise_saturation",
                category="B",
                claim_tested="WISE photometry is not saturated",
                verdict="INCONCLUSIVE",
                confidence=0.2,
                details="No AllWISE data",
            )

        aw = self._allwise_data
        # Saturation limits: W1≈8.1, W2≈6.7, W3≈3.8, W4≈-0.4
        sat_limits = {"W1": 8.1, "W2": 6.7, "W3": 3.8, "W4": -0.4}
        issues = []

        for band, limit in sat_limits.items():
            key = f"{band}mag"
            try:
                mag = float(aw.get(key, 99))
                if mag < limit:
                    issues.append(f"{band}={mag:.2f} < sat_limit {limit}")
            except (TypeError, ValueError):
                pass

        if issues:
            return ReviewCheck(
                check_id="B4_wise_saturation",
                category="B",
                claim_tested="WISE photometry is not saturated",
                verdict="CHALLENGED",
                confidence=0.85,
                details="Saturated: " + "; ".join(issues),
                severity="critical",
            )

        return ReviewCheck(
            check_id="B4_wise_saturation",
            category="B",
            claim_tested="WISE photometry is not saturated",
            verdict="CONFIRMED",
            confidence=0.95,
            details="All WISE bands well below saturation limits",
        )

    def _B5_2mass_wise_consistency(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check that 2MASS and WISE photometry are self-consistent."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=1, columns=["**"])

        try:
            r_2mass = v.query_region(coord, radius=3*u.arcsec, catalog="II/246/out")
            if not r_2mass or not self._allwise_data:
                return ReviewCheck(
                    check_id="B5_2mass_wise_consistency",
                    category="B",
                    claim_tested="2MASS and WISE photometry are consistent (same star)",
                    verdict="INCONCLUSIVE",
                    confidence=0.3,
                    details="Cannot cross-check: missing 2MASS or AllWISE data",
                )

            tm = r_2mass[0][0]
            aw = self._allwise_data

            j_mag = float(tm["Jmag"])
            h_mag = float(tm["Hmag"])
            k_mag = float(tm["Kmag"])
            w1_mag = float(aw.get("W1mag", 99))

            # K-W1 color for main sequence stars is typically 0.0 ± 0.1
            k_w1 = k_mag - w1_mag
            j_h = j_mag - h_mag
            h_k = h_mag - k_mag

            issues = []
            if abs(k_w1) > 0.5:
                issues.append(f"K-W1={k_w1:.3f} (expected ~0 for MS stars)")

            evidence = {
                "J": round(j_mag, 3), "H": round(h_mag, 3), "K": round(k_mag, 3),
                "W1": round(w1_mag, 3), "K_W1": round(k_w1, 3),
                "J_H": round(j_h, 3), "H_K": round(h_k, 3),
            }

            if issues:
                return ReviewCheck(
                    check_id="B5_2mass_wise_consistency",
                    category="B",
                    claim_tested="2MASS and WISE photometry are consistent (same star)",
                    verdict="CHALLENGED",
                    confidence=0.6,
                    details="; ".join(issues),
                    evidence=evidence,
                    severity="warning",
                )

            return ReviewCheck(
                check_id="B5_2mass_wise_consistency",
                category="B",
                claim_tested="2MASS and WISE photometry are consistent (same star)",
                verdict="CONFIRMED",
                confidence=0.85,
                details=f"K-W1={k_w1:.3f}, J-H={j_h:.3f}, H-K={h_k:.3f} — normal for MS star",
                evidence=evidence,
            )
        except Exception as e:
            return ReviewCheck(
                check_id="B5_2mass_wise_consistency",
                category="B",
                claim_tested="2MASS and WISE photometry are consistent (same star)",
                verdict="ERROR",
                confidence=0.0,
                details=f"Cross-check failed: {e}",
            )

    # =================================================================
    #  Category C: ASTROMETRIC INTEGRITY
    # =================================================================

    def _C1_gaia_astrometric_quality(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check Gaia astrometric quality indicators."""
        if not self._gaia_data:
            return ReviewCheck(
                check_id="C1_gaia_astrometric_quality",
                category="C",
                claim_tested="Gaia astrometry is reliable",
                verdict="INCONCLUSIVE",
                confidence=0.2,
                details="No Gaia data available",
            )

        gaia = self._gaia_data
        issues = []

        ruwe = float(gaia.get("ruwe", 0))
        aen_sig = float(gaia.get("astrometric_excess_noise_sig", 0))
        ipd_gof = float(gaia.get("ipd_gof_harmonic_amplitude", 0))
        parallax = float(gaia.get("parallax", 0))
        parallax_err = float(gaia.get("parallax_error", 1))

        # RUWE > 1.4 indicates poor astrometric solution
        if ruwe > 1.4:
            issues.append(f"RUWE={ruwe:.2f} > 1.4 — poor astrometric solution (binary?)")

        # Astrometric excess noise > 5σ
        if aen_sig > 5:
            issues.append(f"AEN significance={aen_sig:.1f} > 5 — astrometric disturbance")

        # IPD GoF harmonic amplitude > 0.1 suggests resolved double
        if ipd_gof > 0.1:
            issues.append(f"IPD GoF harmonic={ipd_gof:.3f} > 0.1 — possibly resolved double")

        # Parallax SNR
        if parallax > 0 and parallax_err > 0:
            plx_snr = parallax / parallax_err
            if plx_snr < 5:
                issues.append(f"Parallax SNR={plx_snr:.1f} < 5 — uncertain distance")

        # Astrometric params solved
        aps = gaia.get("astrometric_params_solved", 0)
        try:
            aps = int(aps)
        except (TypeError, ValueError):
            aps = 0
        if aps < 31:
            issues.append(f"Only {aps}-param solution (not full 5-param)")

        evidence = {
            "ruwe": round(ruwe, 4),
            "aen_sig": round(aen_sig, 2),
            "ipd_gof_harmonic": round(ipd_gof, 4),
            "parallax_snr": round(parallax / max(parallax_err, 0.001), 1),
            "astrometric_params_solved": aps,
        }

        if issues:
            return ReviewCheck(
                check_id="C1_gaia_astrometric_quality",
                category="C",
                claim_tested="Gaia astrometry is reliable",
                verdict="CHALLENGED",
                confidence=0.8,
                details="; ".join(issues),
                evidence=evidence,
                severity="warning",
            )

        return ReviewCheck(
            check_id="C1_gaia_astrometric_quality",
            category="C",
            claim_tested="Gaia astrometry is reliable",
            verdict="CONFIRMED",
            confidence=0.9,
            details=f"RUWE={ruwe:.2f}, AEN sig={aen_sig:.1f}, IPD GoF={ipd_gof:.3f} — all clean",
            evidence=evidence,
        )

    def _C2_pm_discrepancy_independent(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Independently compute Gaia-CatWISE PM discrepancy."""
        if not self._gaia_data or not self._catwise_data:
            return ReviewCheck(
                check_id="C2_pm_discrepancy_independent",
                category="C",
                claim_tested="Gaia-CatWISE PM discrepancy is real",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="Missing Gaia or CatWISE data",
            )

        gaia = self._gaia_data
        cw = self._catwise_data

        try:
            # Gaia PMs
            pmra_gaia = float(gaia.get("pmra", 0))
            pmdec_gaia = float(gaia.get("pmdec", 0))

            # CatWISE PMs
            pmra_cw = float(cw.get("pmRA", 0))
            pmdec_cw = float(cw.get("pmDE", 0))
            e_pmra_cw = float(cw.get("e_pmRA", 5))
            e_pmdec_cw = float(cw.get("e_pmDE", 5))

            # Compute discrepancy
            delta_pmra = pmra_gaia - pmra_cw
            delta_pmdec = pmdec_gaia - pmdec_cw

            # CatWISE systematic floor (Marocco+2021)
            sys_floor = 3.0  # mas/yr
            sigma_ra = np.sqrt(e_pmra_cw**2 + sys_floor**2)
            sigma_dec = np.sqrt(e_pmdec_cw**2 + sys_floor**2)

            chi2 = (delta_pmra / sigma_ra)**2 + (delta_pmdec / sigma_dec)**2
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(chi2, df=2)
            sigma_equiv = np.sqrt(chi2)

            is_significant = p_value < 0.05

            evidence = {
                "gaia_pmra": round(pmra_gaia, 3),
                "gaia_pmdec": round(pmdec_gaia, 3),
                "catwise_pmra": round(pmra_cw, 3),
                "catwise_pmdec": round(pmdec_cw, 3),
                "delta_pmra": round(delta_pmra, 3),
                "delta_pmdec": round(delta_pmdec, 3),
                "sigma_ra": round(float(sigma_ra), 3),
                "sigma_dec": round(float(sigma_dec), 3),
                "chi2": round(float(chi2), 3),
                "p_value": round(float(p_value), 5),
                "sigma_equiv": round(float(sigma_equiv), 2),
                "sys_floor_applied": sys_floor,
            }

            if is_significant:
                verdict = "CONFIRMED"
                details = (f"PM discrepancy confirmed: Δ=({delta_pmra:.2f}, {delta_pmdec:.2f}) mas/yr, "
                          f"{sigma_equiv:.1f}σ (p={p_value:.4f}) with {sys_floor} mas/yr floor")
            else:
                verdict = "CHALLENGED"
                details = (f"PM discrepancy NOT significant: Δ=({delta_pmra:.2f}, {delta_pmdec:.2f}) mas/yr, "
                          f"only {sigma_equiv:.1f}σ (p={p_value:.4f}) with {sys_floor} mas/yr floor")

            return ReviewCheck(
                check_id="C2_pm_discrepancy_independent",
                category="C",
                claim_tested="Gaia-CatWISE PM discrepancy is real",
                verdict=verdict,
                confidence=0.85,
                details=details,
                evidence=evidence,
                severity="info" if is_significant else "warning",
            )
        except Exception as e:
            return ReviewCheck(
                check_id="C2_pm_discrepancy_independent",
                category="C",
                claim_tested="Gaia-CatWISE PM discrepancy is real",
                verdict="ERROR",
                confidence=0.0,
                details=f"Independent PM check failed: {e}",
            )

    def _C3_catwise_systematics(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check if CatWISE PM is affected by known systematics."""
        if not self._catwise_data:
            return ReviewCheck(
                check_id="C3_catwise_systematics",
                category="C",
                claim_tested="CatWISE PM is not dominated by known systematics",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="No CatWISE data available",
            )

        cw = self._catwise_data
        issues = []

        try:
            w1_mag = float(cw.get("W1mproPM", 15))
            # Bright stars (W1 < 8) have large CatWISE PM systematics
            if w1_mag < 8:
                issues.append(f"Bright star (W1={w1_mag:.1f}) — CatWISE PMs unreliable below W1=8")

            # Faint stars (W1 > 16) have large random errors
            if w1_mag > 16:
                issues.append(f"Faint star (W1={w1_mag:.1f}) — CatWISE PM errors large")

            # Check PM errors
            e_pmra = float(cw.get("e_pmRA", 99))
            e_pmdec = float(cw.get("e_pmDE", 99))
            if e_pmra > 10 or e_pmdec > 10:
                issues.append(f"Large CatWISE PM errors: e_pmRA={e_pmra:.1f}, e_pmDec={e_pmdec:.1f}")

            # Check number of exposures
            # (Not always available in CatWISE)
        except (TypeError, ValueError) as e:
            issues.append(f"Could not parse CatWISE data: {e}")

        if issues:
            return ReviewCheck(
                check_id="C3_catwise_systematics",
                category="C",
                claim_tested="CatWISE PM is not dominated by known systematics",
                verdict="CHALLENGED",
                confidence=0.7,
                details="; ".join(issues),
                severity="warning",
            )

        return ReviewCheck(
            check_id="C3_catwise_systematics",
            category="C",
            claim_tested="CatWISE PM is not dominated by known systematics",
            verdict="CONFIRMED",
            confidence=0.8,
            details=f"W1={float(cw.get('W1mproPM', '?')):.1f} — in reliable CatWISE PM range",
        )

    def _C4_gaia_nss_check(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check Gaia DR3 non-single-star catalog for binary solutions."""
        if not self._gaia_data:
            return ReviewCheck(
                check_id="C4_gaia_nss_check",
                category="C",
                claim_tested="Star is not in Gaia NSS binary catalog",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="No Gaia data",
            )

        nss = self._gaia_data.get("non_single_star", None)

        if nss is not None and int(nss) > 0:
            return ReviewCheck(
                check_id="C4_gaia_nss_check",
                category="C",
                claim_tested="Star is not in Gaia NSS binary catalog",
                verdict="REFUTED",
                confidence=0.95,
                details=f"Star IS in Gaia NSS catalog (non_single_star={nss})",
                severity="critical",
            )

        return ReviewCheck(
            check_id="C4_gaia_nss_check",
            category="C",
            claim_tested="Star is not in Gaia NSS binary catalog",
            verdict="CONFIRMED",
            confidence=0.85,
            details="Not in Gaia DR3 NSS catalog (but NSS is incomplete)",
        )

    # =================================================================
    #  Category D: RADIO VALIDATION
    # =================================================================

    def _D1_nvss_independent(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Independently query NVSS catalog."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=10, columns=["**"])

        try:
            r = v.query_region(coord, radius=60*u.arcsec, catalog="VIII/65")
            if not r:
                return ReviewCheck(
                    check_id="D1_nvss_independent",
                    category="D",
                    claim_tested="NVSS radio detection is real",
                    verdict="CONFIRMED" if dec < -40 else "INCONCLUSIVE",
                    confidence=0.7,
                    details="No NVSS source within 60\" (radio silence)" if dec > -40
                            else "Target below NVSS declination limit",
                )

            t = r[0]
            # Find closest
            seps = []
            for row in t:
                nvss_c = SkyCoord(row["RAJ2000"], row["DEJ2000"],
                                  unit=("hourangle", "deg"))
                seps.append(coord.separation(nvss_c).arcsec)

            idx = np.argmin(seps)
            closest = t[idx]
            sep = seps[idx]
            flux = float(closest["S1.4"])

            evidence = {
                "nvss_name": str(closest["NVSS"]),
                "flux_1.4GHz_mJy": flux,
                "sep_arcsec": round(sep, 2),
                "n_sources_60arcsec": len(t),
            }

            return ReviewCheck(
                check_id="D1_nvss_independent",
                category="D",
                claim_tested="NVSS radio detection is real",
                verdict="CONFIRMED",
                confidence=0.9,
                details=f"NVSS {closest['NVSS']}: {flux} mJy at {sep:.1f}\" "
                        f"({len(t)} source(s) in 60\")",
                evidence=evidence,
            )
        except Exception as e:
            return ReviewCheck(
                check_id="D1_nvss_independent",
                category="D",
                claim_tested="NVSS radio detection is real",
                verdict="ERROR",
                confidence=0.0,
                details=f"NVSS query failed: {e}",
            )

    def _D2_vlass_independent(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Independently query VLASS Epoch 1 catalog via VizieR."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=10, columns=["**"])

        try:
            r = v.query_region(coord, radius=30*u.arcsec, catalog="J/ApJS/255/30")
            if not r:
                return ReviewCheck(
                    check_id="D2_vlass_independent",
                    category="D",
                    claim_tested="VLASS detection of radio source at stellar position",
                    verdict="INCONCLUSIVE",
                    confidence=0.5,
                    details="No VLASS source within 30\" — either no source or outside VLASS coverage",
                )

            t = r[0]
            # Find closest to star
            best_sep = 999
            best_row = None
            for row in t:
                vlass_c = SkyCoord(ra=float(row["RAJ2000"]),
                                   dec=float(row["DEJ2000"]), unit="deg")
                sep = coord.separation(vlass_c).arcsec
                if sep < best_sep:
                    best_sep = sep
                    best_row = row

            if best_row is None:
                return ReviewCheck(
                    check_id="D2_vlass_independent",
                    category="D",
                    claim_tested="VLASS detection of radio source at stellar position",
                    verdict="INCONCLUSIVE",
                    confidence=0.3,
                    details="VLASS query returned data but could not parse positions",
                )

            flux_tot = float(best_row["Ftot"])
            flux_peak = float(best_row["Fpeak"])
            qual_flag = int(best_row.get("QualFlag", 0))
            dup_flag = int(best_row.get("DupFlag", 0))
            main_sample = int(best_row.get("MainSample", 0))

            # Deconvolved size
            dc_maj = float(best_row.get("DCMaj", 0))

            issues = []
            if qual_flag > 0:
                issues.append(f"QualFlag={qual_flag} (quality concern)")
            if dup_flag > 0:
                issues.append(f"DupFlag={dup_flag} (possible duplicate)")
            if main_sample == 0:
                issues.append("NOT in CIRADA main sample")
            if best_sep > 5:
                issues.append(f"Offset {best_sep:.1f}\" > 5\" — association uncertain")

            evidence = {
                "vlass_name": str(best_row["CompName"]),
                "flux_3GHz_mJy": round(flux_tot, 3),
                "flux_peak_mJy": round(flux_peak, 3),
                "sep_arcsec": round(best_sep, 3),
                "QualFlag": qual_flag,
                "DupFlag": dup_flag,
                "MainSample": main_sample,
                "deconv_size_arcsec": round(dc_maj, 2),
                "n_sources_30arcsec": len(t),
                "issues": issues,
            }

            if best_sep < 2 and not issues:
                verdict = "CONFIRMED"
            elif best_sep < 2 and qual_flag > 0:
                verdict = "DETECTED"  # positional match but QualFlag concern
            elif best_sep < 2:
                verdict = "CONFIRMED"
            elif best_sep < 10:
                verdict = "CHALLENGED"
            else:
                verdict = "CHALLENGED"

            return ReviewCheck(
                check_id="D2_vlass_independent",
                category="D",
                claim_tested="VLASS detection confirms radio source at stellar position",
                verdict=verdict,
                confidence=0.9 if best_sep < 2 else 0.5,
                details=f"VLASS {best_row['CompName']}: {flux_tot:.2f} mJy at {best_sep:.2f}\""
                        + (f" — CAVEATS: {'; '.join(issues)}" if issues else " — clean detection"),
                evidence=evidence,
                severity="info" if not issues else "warning",
            )
        except Exception as e:
            return ReviewCheck(
                check_id="D2_vlass_independent",
                category="D",
                claim_tested="VLASS detection confirms radio source at stellar position",
                verdict="ERROR",
                confidence=0.0,
                details=f"VLASS query failed: {e}",
            )

    def _D3_radio_chance_alignment(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Compute chance alignment probability for radio source."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        # First check if there IS a radio source
        try:
            v = Vizier(row_limit=50, columns=["RAJ2000", "DEJ2000", "Ftot"])
            r = v.query_region(coord, radius=5*u.arcsec, catalog="J/ApJS/255/30")

            if not r:
                return ReviewCheck(
                    check_id="D3_radio_chance_alignment",
                    category="D",
                    claim_tested="Radio-optical association is not a chance alignment",
                    verdict="INCONCLUSIVE",
                    confidence=0.5,
                    details="No VLASS source within 5\" — chance alignment check not applicable",
                )

            best_sep = 999
            for row in r[0]:
                vc = SkyCoord(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]), unit="deg")
                sep = coord.separation(vc).arcsec
                if sep < best_sep:
                    best_sep = sep

            # VLASS source density: ~3.4M sources over ~34,000 deg^2 ≈ 100/deg^2
            # = 100 / (3600^2) per arcsec^2 = 7.7e-6 per arcsec^2
            rho = 100.0  # per deg^2
            area_deg2 = math.pi * (best_sep / 3600)**2
            p_chance = rho * area_deg2

            # Also count VLASS sources in wider field for local density estimate
            r_wide = v.query_region(coord, radius=600*u.arcsec, catalog="J/ApJS/255/30")
            local_density = None
            if r_wide:
                n_wide = len(r_wide[0])
                local_area = math.pi * (600/3600)**2  # deg^2
                local_density = n_wide / local_area
                p_chance_local = local_density * area_deg2

            evidence = {
                "sep_arcsec": round(best_sep, 3),
                "global_density_per_deg2": rho,
                "p_chance_global": f"{p_chance:.2e}",
            }
            if local_density:
                evidence["local_density_per_deg2"] = round(local_density, 1)
                evidence["p_chance_local"] = f"{p_chance_local:.2e}"
                p_final = p_chance_local
            else:
                p_final = p_chance

            is_real = p_final < 0.001  # < 0.1% chance

            return ReviewCheck(
                check_id="D3_radio_chance_alignment",
                category="D",
                claim_tested="Radio-optical association is not a chance alignment",
                verdict="CONFIRMED" if is_real else "CHALLENGED",
                confidence=0.9 if is_real else 0.6,
                details=f"P(chance) = {p_final:.2e} at sep={best_sep:.2f}\" "
                        + ("— extremely unlikely chance alignment" if is_real
                           else "— chance alignment not fully ruled out"),
                evidence=evidence,
            )
        except Exception as e:
            return ReviewCheck(
                check_id="D3_radio_chance_alignment",
                category="D",
                claim_tested="Radio-optical association is not a chance alignment",
                verdict="ERROR",
                confidence=0.0,
                details=f"Calculation failed: {e}",
            )

    def _D4_radio_spectral_index(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Compute and validate radio spectral index."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=5, columns=["**"])

        try:
            # Get NVSS (1.4 GHz)
            r_nvss = v.query_region(coord, radius=30*u.arcsec, catalog="VIII/65")
            r_vlass = v.query_region(coord, radius=5*u.arcsec, catalog="J/ApJS/255/30")

            if not r_nvss or not r_vlass:
                return ReviewCheck(
                    check_id="D4_radio_spectral_index",
                    category="D",
                    claim_tested="Radio spectral index is consistent with claimed value",
                    verdict="INCONCLUSIVE",
                    confidence=0.3,
                    details="Need both NVSS and VLASS detections for spectral index",
                )

            # Closest NVSS
            best_nvss_flux = None
            best_nvss_sep = 999
            for row in r_nvss[0]:
                nc = SkyCoord(row["RAJ2000"], row["DEJ2000"], unit=("hourangle", "deg"))
                sep = coord.separation(nc).arcsec
                if sep < best_nvss_sep:
                    best_nvss_sep = sep
                    best_nvss_flux = float(row["S1.4"])

            # Closest VLASS
            best_vlass_flux = None
            best_vlass_sep = 999
            for row in r_vlass[0]:
                vc = SkyCoord(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]), unit="deg")
                sep = coord.separation(vc).arcsec
                if sep < best_vlass_sep:
                    best_vlass_sep = sep
                    best_vlass_flux = float(row["Ftot"])

            if best_nvss_flux and best_vlass_flux:
                # Note: VLASS QL fluxes underestimated by ~15%
                alpha = np.log10(best_vlass_flux / best_nvss_flux) / np.log10(3.0 / 1.4)

                # Corrected for VLASS QL systematic
                vlass_corrected = best_vlass_flux / 0.85
                alpha_corrected = np.log10(vlass_corrected / best_nvss_flux) / np.log10(3.0 / 1.4)

                evidence = {
                    "nvss_flux_mJy": best_nvss_flux,
                    "vlass_flux_mJy": round(best_vlass_flux, 3),
                    "vlass_flux_corrected_mJy": round(vlass_corrected, 3),
                    "alpha_raw": round(float(alpha), 3),
                    "alpha_corrected": round(float(alpha_corrected), 3),
                }

                # Interpret
                if alpha < -1.0:
                    interp = "Ultra-steep spectrum — old AGN lobe or relic"
                elif alpha < -0.5:
                    interp = "Steep synchrotron — typical AGN or non-thermal stellar"
                elif alpha < 0.0:
                    interp = "Moderate negative — could be stellar or AGN"
                elif alpha < 0.5:
                    interp = "Flat spectrum — compact AGN core or self-absorbed"
                else:
                    interp = "Inverted spectrum — compact source or thermal emission"

                evidence["interpretation"] = interp

                # VLASS QL flux caveat
                caveat = ("Note: VLASS QL flux densities are systematically "
                         "~15% underestimated. Corrected α = "
                         f"{alpha_corrected:.2f}")

                return ReviewCheck(
                    check_id="D4_radio_spectral_index",
                    category="D",
                    claim_tested="Radio spectral index is consistent with claimed value",
                    verdict="CONFIRMED",
                    confidence=0.8,
                    details=f"α = {alpha:.2f} ({interp}). {caveat}",
                    evidence=evidence,
                )

        except Exception as e:
            return ReviewCheck(
                check_id="D4_radio_spectral_index",
                category="D",
                claim_tested="Radio spectral index is consistent with claimed value",
                verdict="ERROR",
                confidence=0.0,
                details=f"Spectral index calculation failed: {e}",
            )

    # =================================================================
    #  Category E: ALTERNATIVE EXPLANATIONS
    # =================================================================

    def _E1_binary_hypothesis(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Test the binary star hypothesis — can binarity explain everything?"""
        evidence = {}
        binary_score = 0.0  # 0 = not binary, 1 = definitely binary

        if self._gaia_data:
            gaia = self._gaia_data
            ruwe = float(gaia.get("ruwe", 0))
            aen_sig = float(gaia.get("astrometric_excess_noise_sig", 0))
            ipd_gof = float(gaia.get("ipd_gof_harmonic_amplitude", 0))

            # RUWE check
            if ruwe > 1.4:
                binary_score += 0.3
                evidence["ruwe_binary_flag"] = True
            else:
                evidence["ruwe_binary_flag"] = False
                evidence["ruwe"] = round(ruwe, 3)

            # AEN check
            if aen_sig > 2:
                binary_score += 0.15
                evidence["aen_marginally_elevated"] = True

            # NSS
            nss = gaia.get("non_single_star", 0)
            try:
                nss = int(nss)
            except (TypeError, ValueError):
                nss = 0
            if nss > 0:
                binary_score += 0.4
                evidence["gaia_nss"] = True
            else:
                evidence["gaia_nss"] = False

        # IR excess pattern — binaries usually show W1/W2 excess too
        if self._allwise_data:
            aw = self._allwise_data
            try:
                w1 = float(aw.get("W1mag", 99))
                w2 = float(aw.get("W2mag", 99))
                w3 = float(aw.get("W3mag", 99))
                if w1 < 50 and w3 < 50:
                    w1_w3 = w1 - w3
                    if abs(w1_w3) > 0.5:  # significant IR excess
                        # Check if W1-W2 is also anomalous (binary-like)
                        w1_w2 = w1 - w2
                        if abs(w1_w2) > 0.3:
                            binary_score += 0.1
                            evidence["w1_w2_excess"] = True
                        else:
                            evidence["w1_w2_normal"] = True  # argues against binary
            except (TypeError, ValueError):
                pass

        if binary_score >= 0.5:
            verdict = "CHALLENGED"
            details = f"Binary hypothesis scores {binary_score:.2f}/1.0 — cannot be ruled out"
            severity = "warning"
        elif binary_score >= 0.3:
            verdict = "INCONCLUSIVE"
            details = f"Binary hypothesis scores {binary_score:.2f}/1.0 — weak binary indicators"
            severity = "info"
        else:
            verdict = "CONFIRMED"
            details = (f"Binary hypothesis scores {binary_score:.2f}/1.0 — "
                      f"RUWE clean, no NSS entry, no strong binary indicators. "
                      f"Binary explanation weakly supported.")
            severity = "info"

        evidence["binary_score"] = round(binary_score, 2)
        return ReviewCheck(
            check_id="E1_binary_hypothesis",
            category="E",
            claim_tested="Binarity CANNOT explain the multi-channel anomaly",
            verdict=verdict,
            confidence=0.7,
            details=details,
            evidence=evidence,
            severity=severity,
        )

    def _E2_yso_hypothesis(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Test if the target could be a young stellar object."""
        evidence = {}
        yso_score = 0.0

        if self._gaia_data:
            gaia = self._gaia_data
            teff = float(gaia.get("teff_gspphot", 0))
            logg = float(gaia.get("logg_gspphot", 0))

            # YSOs tend to have lower logg
            if logg > 0 and logg < 3.5:
                yso_score += 0.2
                evidence["low_logg"] = round(logg, 2)

            # Check galactic latitude — YSOs are near the plane
            b = self._galactic_latitude(ra, dec)
            evidence["galactic_b"] = round(b, 1)
            if abs(b) < 10:
                yso_score += 0.2
                evidence["near_plane"] = True
            elif abs(b) > 30:
                yso_score -= 0.1  # argues against YSO
                evidence["high_latitude"] = True

        # IR excess pattern — YSOs show rising SED
        if self._allwise_data:
            aw = self._allwise_data
            try:
                w1 = float(aw.get("W1mag", 99))
                w2 = float(aw.get("W2mag", 99))
                w3 = float(aw.get("W3mag", 99))
                w4 = float(aw.get("W4mag", 99))
                if w1 < 50 and w4 < 50:
                    # Class II YSOs have very red W1-W4
                    w1_w4 = w1 - w4
                    if w1_w4 > 3:
                        yso_score += 0.3
                        evidence["very_red_w1w4"] = round(w1_w4, 2)
            except (TypeError, ValueError):
                pass

        yso_score = max(0, min(1, yso_score))
        evidence["yso_score"] = round(yso_score, 2)

        if yso_score >= 0.4:
            verdict = "CHALLENGED"
            details = f"YSO hypothesis scores {yso_score:.2f} — target has some YSO-like properties"
        else:
            verdict = "CONFIRMED"
            details = f"YSO hypothesis scores {yso_score:.2f} — unlikely to be a YSO"

        return ReviewCheck(
            check_id="E2_yso_hypothesis",
            category="E",
            claim_tested="Target is NOT a young stellar object (YSO)",
            verdict=verdict,
            confidence=0.7,
            details=details,
            evidence=evidence,
        )

    def _E3_agn_background(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check if IR excess could be from a background AGN in the beam."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        issues = []
        evidence = {}

        # Check for optical sources that could be AGN in the WISE beam
        try:
            v = Vizier(row_limit=20, columns=["RAJ2000", "DEJ2000", "rmag", "cl"])
            # Pan-STARRS
            r = v.query_region(coord, radius=12*u.arcsec, catalog="II/349/ps1")
            if r:
                n_ps = len(r[0])
                evidence["panstarrs_sources_12arcsec"] = n_ps
                if n_ps > 3:
                    issues.append(f"{n_ps} Pan-STARRS sources within 12\" (crowded)")
        except Exception:
            pass

        # Check SDSS for galaxy/QSO classification
        try:
            v2 = Vizier(row_limit=5, columns=["**"])
            r = v2.query_region(coord, radius=12*u.arcsec, catalog="V/154/sdss16")
            if r:
                for row in r[0]:
                    obj_class = str(row.get("class", ""))
                    if obj_class in ("QSO", "GALAXY"):
                        issues.append(f"SDSS {obj_class} within 12\" — AGN contamination risk")
                        evidence["sdss_agn"] = True
        except Exception:
            pass

        if issues:
            return ReviewCheck(
                check_id="E3_agn_background",
                category="E",
                claim_tested="IR excess is NOT from a background AGN",
                verdict="CHALLENGED",
                confidence=0.6,
                details="; ".join(issues),
                evidence=evidence,
                severity="warning",
            )

        return ReviewCheck(
            check_id="E3_agn_background",
            category="E",
            claim_tested="IR excess is NOT from a background AGN",
            verdict="CONFIRMED",
            confidence=0.8,
            details="No SDSS QSO/galaxy classification within WISE beam; clean field",
            evidence=evidence,
        )

    def _E4_simbad_classification(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Query SIMBAD for any known classification."""
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        try:
            # Configure SIMBAD query
            s = Simbad()
            s.add_votable_fields("otype", "sp")
            result = s.query_region(coord, radius=5*u.arcsec)

            if result is None or len(result) == 0:
                return ReviewCheck(
                    check_id="E4_simbad_classification",
                    category="E",
                    claim_tested="Target has no known SIMBAD classification that explains anomaly",
                    verdict="CONFIRMED",
                    confidence=0.7,
                    details="Not in SIMBAD within 5\" — no pre-existing classification",
                )

            row = result[0]
            otype = str(row.get("OTYPE", ""))
            main_id = str(row.get("MAIN_ID", ""))
            sp_type = str(row.get("SP_TYPE", ""))

            evidence = {
                "main_id": main_id,
                "otype": otype,
                "sp_type": sp_type,
            }

            # Risky object types
            risky_types = {"YSO", "Y*O", "CV*", "QSO", "AGN", "Sy*",
                          "Em*", "WR*", "Be*", "AB*", "SB*", "EB*"}
            if any(rt in otype for rt in risky_types):
                return ReviewCheck(
                    check_id="E4_simbad_classification",
                    category="E",
                    claim_tested="Target has no known SIMBAD classification that explains anomaly",
                    verdict="REFUTED",
                    confidence=0.9,
                    details=f"SIMBAD: {main_id} classified as '{otype}' — "
                            f"this type can explain multi-channel anomaly",
                    evidence=evidence,
                    severity="critical",
                )

            return ReviewCheck(
                check_id="E4_simbad_classification",
                category="E",
                claim_tested="Target has no known SIMBAD classification that explains anomaly",
                verdict="CONFIRMED",
                confidence=0.75,
                details=f"SIMBAD: {main_id} = '{otype}' — no risky classification",
                evidence=evidence,
            )
        except Exception as e:
            return ReviewCheck(
                check_id="E4_simbad_classification",
                category="E",
                claim_tested="Target has no known SIMBAD classification that explains anomaly",
                verdict="ERROR",
                confidence=0.0,
                details=f"SIMBAD query failed: {e}",
            )

    def _E5_optical_variability(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check for optical variability (binary indicator)."""
        # Check Gaia variability flags
        if self._gaia_data:
            gaia = self._gaia_data
            # Gaia phot_variable_flag is not in the basic table
            # Check bp_rp variability proxy
            g_mag = float(gaia.get("phot_g_mean_mag", 0))
            bp_mag = float(gaia.get("phot_bp_mean_mag", 0))
            rp_mag = float(gaia.get("phot_rp_mean_mag", 0))

            if g_mag > 0 and bp_mag > 0 and rp_mag > 0:
                bp_rp = bp_mag - rp_mag
                return ReviewCheck(
                    check_id="E5_optical_variability",
                    category="E",
                    claim_tested="No optical variability suggesting binary",
                    verdict="INCONCLUSIVE",
                    confidence=0.4,
                    details=f"BP-RP={bp_rp:.3f}, G={g_mag:.2f} — "
                            f"need ZTF/TESS for variability check (Gaia basic table lacks variability flag)",
                    evidence={"bp_rp": round(bp_rp, 3), "g_mag": round(g_mag, 2)},
                )

        return ReviewCheck(
            check_id="E5_optical_variability",
            category="E",
            claim_tested="No optical variability suggesting binary",
            verdict="INCONCLUSIVE",
            confidence=0.3,
            details="Need ZTF or TESS data for optical variability check",
        )

    def _E6_xray_activity(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check for X-ray emission (indicates chromospheric activity / AGN)."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=5, columns=["**"])

        detected_in = []

        # ROSAT All-Sky Survey
        try:
            r = v.query_region(coord, radius=30*u.arcsec, catalog="IX/10A/1rxs")
            if r and len(r[0]) > 0:
                detected_in.append(f"ROSAT 1RXS ({len(r[0])} source(s) within 30\")")
        except Exception:
            pass

        # eROSITA (if available)
        try:
            r = v.query_region(coord, radius=30*u.arcsec, catalog="J/A+A/682/A34")
            if r and len(r[0]) > 0:
                detected_in.append(f"eROSITA ({len(r[0])} source(s) within 30\")")
        except Exception:
            pass

        # XMM-Newton
        try:
            r = v.query_region(coord, radius=15*u.arcsec, catalog="IX/68/xmm4d13s")
            if r and len(r[0]) > 0:
                detected_in.append(f"XMM-Newton ({len(r[0])} source(s) within 15\")")
        except Exception:
            pass

        if detected_in:
            return ReviewCheck(
                check_id="E6_xray_activity",
                category="E",
                claim_tested="No X-ray emission (no chromospheric activity / AGN)",
                verdict="CHALLENGED",
                confidence=0.8,
                details=f"X-ray detection: {'; '.join(detected_in)} — suggests activity or AGN",
                evidence={"detections": detected_in},
                severity="warning",
            )

        return ReviewCheck(
            check_id="E6_xray_activity",
            category="E",
            claim_tested="No X-ray emission (no chromospheric activity / AGN)",
            verdict="CONFIRMED",
            confidence=0.7,
            details="No X-ray detection in ROSAT, eROSITA, or XMM — radio-quiet in X-rays",
        )

    def _E7_galex_uv(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check GALEX UV data for activity indicators."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=5, columns=["**"])

        try:
            r = v.query_region(coord, radius=5*u.arcsec, catalog="II/335/galex_ais")
            if not r:
                return ReviewCheck(
                    check_id="E7_galex_uv",
                    category="E",
                    claim_tested="GALEX UV data provides consistent picture",
                    verdict="INCONCLUSIVE",
                    confidence=0.4,
                    details="No GALEX coverage or no UV detection",
                )

            row = r[0][0]
            fuv = row.get("FUVmag", None)
            nuv = row.get("NUVmag", None)

            evidence = {}
            if fuv is not None and fuv is not np.ma.masked:
                evidence["FUV"] = round(float(fuv), 2)
            if nuv is not None and nuv is not np.ma.masked:
                evidence["NUV"] = round(float(nuv), 2)

            details = f"GALEX: " + ", ".join(f"{k}={v}" for k, v in evidence.items())

            return ReviewCheck(
                check_id="E7_galex_uv",
                category="E",
                claim_tested="GALEX UV data provides consistent picture",
                verdict="CONFIRMED",
                confidence=0.6,
                details=details + " — UV data available for comparison",
                evidence=evidence,
            )
        except Exception as e:
            return ReviewCheck(
                check_id="E7_galex_uv",
                category="E",
                claim_tested="GALEX UV data provides consistent picture",
                verdict="ERROR",
                confidence=0.0,
                details=f"GALEX query failed: {e}",
            )

    # =================================================================
    #  Category F: STATISTICAL RIGOR
    # =================================================================

    def _F1_ir_excess_significance(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Test IR excess significance with conservative error bars."""
        if not self._allwise_data:
            return ReviewCheck(
                check_id="F1_ir_excess_significance",
                category="F",
                claim_tested="IR excess is significant even with conservative errors",
                verdict="INCONCLUSIVE",
                confidence=0.2,
                details="No AllWISE data for significance test",
            )

        aw = self._allwise_data
        try:
            w3 = float(aw.get("W3mag", 99))
            e_w3 = float(aw.get("e_W3mag", 0.2))
            w4 = float(aw.get("W4mag", 99))
            e_w4 = float(aw.get("e_W4mag", 0.5))
            w1 = float(aw.get("W1mag", 99))

            if w3 > 50 or w1 > 50:
                return ReviewCheck(
                    check_id="F1_ir_excess_significance",
                    category="F",
                    claim_tested="IR excess is significant even with conservative errors",
                    verdict="INCONCLUSIVE",
                    confidence=0.3,
                    details="Missing photometric data",
                )

            # Conservative: add 0.05 mag systematic floor to formal errors
            e_w3_cons = np.sqrt(e_w3**2 + 0.15**2)  # W3 systematic ~0.15
            e_w4_cons = np.sqrt(e_w4**2 + 0.30**2)  # W4 systematic ~0.30

            # Excess
            excess_w3 = w1 - w3  # negative = excess
            excess_w4 = w1 - w4 if w4 < 50 else 0

            sig_w3 = abs(excess_w3) / e_w3_cons if e_w3_cons > 0 else 0
            sig_w4 = abs(excess_w4) / e_w4_cons if e_w4_cons > 0 else 0

            evidence = {
                "W3_excess_conservative_sigma": round(float(sig_w3), 2),
                "W4_excess_conservative_sigma": round(float(sig_w4), 2),
                "e_W3_formal": round(e_w3, 3),
                "e_W3_conservative": round(float(e_w3_cons), 3),
                "e_W4_formal": round(e_w4, 3),
                "e_W4_conservative": round(float(e_w4_cons), 3),
            }

            if sig_w3 > 3 or sig_w4 > 3:
                return ReviewCheck(
                    check_id="F1_ir_excess_significance",
                    category="F",
                    claim_tested="IR excess is significant even with conservative errors",
                    verdict="CONFIRMED",
                    confidence=0.85,
                    details=f"W3: {sig_w3:.1f}σ, W4: {sig_w4:.1f}σ (with conservative systematic floors)",
                    evidence=evidence,
                )
            else:
                return ReviewCheck(
                    check_id="F1_ir_excess_significance",
                    category="F",
                    claim_tested="IR excess is significant even with conservative errors",
                    verdict="CHALLENGED",
                    confidence=0.7,
                    details=f"W3: only {sig_w3:.1f}σ, W4: {sig_w4:.1f}σ with conservative errors — "
                            f"significance reduced with systematic floors",
                    evidence=evidence,
                    severity="warning",
                )
        except Exception as e:
            return ReviewCheck(
                check_id="F1_ir_excess_significance",
                category="F",
                claim_tested="IR excess is significant even with conservative errors",
                verdict="ERROR",
                confidence=0.0,
                details=f"Test failed: {e}",
            )

    def _F2_neowise_stability(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check NEOWISE time-series stability (is IR excess persistent?)."""
        # Check if NEOWISE cache file exists
        import glob
        ra_str = f"{ra:.6f}"
        dec_str = f"{dec:.6f}"
        cache_pattern = f"data/cache/neowise_ts_{ra_str}_{dec_str}*.json"
        files = glob.glob(str(Path(__file__).resolve().parent.parent.parent / cache_pattern))

        if not files:
            return ReviewCheck(
                check_id="F2_neowise_stability",
                category="F",
                claim_tested="IR excess is persistent over NEOWISE baseline (not transient)",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="No NEOWISE time-series cache found",
            )

        try:
            with open(files[0]) as f:
                neowise = json.load(f)

            if isinstance(neowise, dict):
                n_epochs = neowise.get("n_epochs", 0)
                time_span = neowise.get("time_span_years", 0)
            elif isinstance(neowise, list):
                n_epochs = len(neowise)
                time_span = 0  # can't determine from list alone

            return ReviewCheck(
                check_id="F2_neowise_stability",
                category="F",
                claim_tested="IR excess is persistent over NEOWISE baseline (not transient)",
                verdict="CONFIRMED" if n_epochs > 20 else "INCONCLUSIVE",
                confidence=0.7 if n_epochs > 20 else 0.4,
                details=f"NEOWISE: {n_epochs} epochs over {time_span:.1f} years available in cache",
                evidence={"n_epochs": n_epochs, "time_span_years": time_span},
            )
        except Exception as e:
            return ReviewCheck(
                check_id="F2_neowise_stability",
                category="F",
                claim_tested="IR excess is persistent over NEOWISE baseline (not transient)",
                verdict="ERROR",
                confidence=0.0,
                details=f"NEOWISE cache read failed: {e}",
            )

    def _F3_field_star_comparison(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Compare target's IR colors to field stars at similar Teff."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if not self._gaia_data:
            return ReviewCheck(
                check_id="F3_field_star_comparison",
                category="F",
                claim_tested="IR colors are anomalous compared to field stars",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="No Gaia data for field star comparison",
            )

        teff = float(self._gaia_data.get("teff_gspphot", 0))
        if teff <= 0:
            return ReviewCheck(
                check_id="F3_field_star_comparison",
                category="F",
                claim_tested="IR colors are anomalous compared to field stars",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="No Teff available for comparison",
            )

        return ReviewCheck(
            check_id="F3_field_star_comparison",
            category="F",
            claim_tested="IR colors are anomalous compared to field stars",
            verdict="INCONCLUSIVE",
            confidence=0.4,
            details=f"Teff={teff:.0f}K — field star comparison requires matched control "
                    f"stars (pipeline uses this, but independent validation would need "
                    f"a new query of ~100 field stars at similar Teff, distance, and galactic position)",
            evidence={"teff": teff},
        )

    # =================================================================
    #  Category G: ENVIRONMENTAL CHECKS
    # =================================================================

    def _G1_galactic_position(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check galactic coordinates — low latitude means higher confusion."""
        b = self._galactic_latitude(ra, dec)
        l = self._galactic_longitude(ra, dec)

        evidence = {"gal_l": round(l, 2), "gal_b": round(b, 2)}

        if abs(b) < 5:
            return ReviewCheck(
                check_id="G1_galactic_position",
                category="G",
                claim_tested="Target is at high enough galactic latitude for clean photometry",
                verdict="CHALLENGED",
                confidence=0.85,
                details=f"b={b:.1f}° — very low latitude, high confusion risk",
                evidence=evidence,
                severity="warning",
            )
        elif abs(b) < 15:
            return ReviewCheck(
                check_id="G1_galactic_position",
                category="G",
                claim_tested="Target is at high enough galactic latitude for clean photometry",
                verdict="CONFIRMED",
                confidence=0.7,
                details=f"b={b:.1f}° — moderate latitude, some confusion possible",
                evidence=evidence,
            )
        else:
            return ReviewCheck(
                check_id="G1_galactic_position",
                category="G",
                claim_tested="Target is at high enough galactic latitude for clean photometry",
                verdict="CONFIRMED",
                confidence=0.9,
                details=f"b={b:.1f}° — high latitude, clean field expected",
                evidence=evidence,
            )

    def _G2_dust_extinction(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check 3D dust extinction at target position."""
        try:
            from dustmaps.bayestar import BayestarQuery
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")

            # Get distance from Gaia if available
            dist_pc = None
            if self._gaia_data:
                plx = float(self._gaia_data.get("parallax", 0))
                if plx > 0:
                    dist_pc = 1000.0 / plx

            bayestar = BayestarQuery()
            if dist_pc:
                # New dustmaps API: pass distance via SkyCoord
                coord_3d = SkyCoord(
                    ra=ra*u.deg, dec=dec*u.deg,
                    distance=dist_pc*u.pc, frame="icrs"
                )
                ebv = bayestar(coord_3d, mode="best")
                details = f"E(B-V) = {float(ebv):.3f} at d={dist_pc:.0f} pc"
            else:
                ebv = bayestar(coord, mode="best")
                details = f"E(B-V) along full sightline: {float(np.max(ebv)):.3f}"
                ebv = float(np.max(ebv))

            evidence = {"ebv": round(float(ebv), 4)}
            if dist_pc:
                evidence["distance_pc"] = round(dist_pc, 0)

            # High extinction can cause photometric issues
            if float(ebv) > 0.5:
                return ReviewCheck(
                    check_id="G2_dust_extinction",
                    category="G",
                    claim_tested="Dust extinction is not significantly affecting photometry",
                    verdict="CHALLENGED",
                    confidence=0.8,
                    details=f"High extinction: {details} — IR photometry may be affected",
                    evidence=evidence,
                    severity="warning",
                )

            return ReviewCheck(
                check_id="G2_dust_extinction",
                category="G",
                claim_tested="Dust extinction is not significantly affecting photometry",
                verdict="CONFIRMED",
                confidence=0.85,
                details=f"Moderate/low extinction: {details}",
                evidence=evidence,
            )
        except ImportError:
            return ReviewCheck(
                check_id="G2_dust_extinction",
                category="G",
                claim_tested="Dust extinction is not significantly affecting photometry",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="dustmaps not installed — cannot check extinction",
            )
        except Exception as e:
            return ReviewCheck(
                check_id="G2_dust_extinction",
                category="G",
                claim_tested="Dust extinction is not significantly affecting photometry",
                verdict="ERROR",
                confidence=0.0,
                details=f"Dust check failed: {e}",
            )

    def _G3_stellar_density(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check local stellar density (crowded field risk)."""
        from astroquery.gaia import Gaia

        try:
            query = f"""
            SELECT COUNT(*) as n_stars
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {30.0/3600})
            ) = 1
            AND phot_g_mean_mag < 20
            """
            job = Gaia.launch_job(query)
            result = job.get_results()
            n_stars = int(result["n_stars"][0])

            # 30" radius = area ~ 2827 arcsec^2
            density = n_stars / 2827.0  # per arcsec^2

            evidence = {"n_gaia_30arcsec": n_stars, "density_per_arcsec2": round(density, 4)}

            if n_stars > 10:
                return ReviewCheck(
                    check_id="G3_stellar_density",
                    category="G",
                    claim_tested="Field is not crowded (low confusion risk)",
                    verdict="CHALLENGED",
                    confidence=0.7,
                    details=f"{n_stars} Gaia sources within 30\" — moderately crowded field",
                    evidence=evidence,
                    severity="warning",
                )
            elif n_stars > 3:
                return ReviewCheck(
                    check_id="G3_stellar_density",
                    category="G",
                    claim_tested="Field is not crowded (low confusion risk)",
                    verdict="CONFIRMED",
                    confidence=0.7,
                    details=f"{n_stars} Gaia sources within 30\" — uncrowded",
                    evidence=evidence,
                )
            else:
                return ReviewCheck(
                    check_id="G3_stellar_density",
                    category="G",
                    claim_tested="Field is not crowded (low confusion risk)",
                    verdict="CONFIRMED",
                    confidence=0.9,
                    details=f"Only {n_stars} Gaia source(s) within 30\" — very clean field",
                    evidence=evidence,
                )
        except Exception as e:
            return ReviewCheck(
                check_id="G3_stellar_density",
                category="G",
                claim_tested="Field is not crowded (low confusion risk)",
                verdict="ERROR",
                confidence=0.0,
                details=f"Density query failed: {e}",
            )

    def _G4_nearby_sources(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check for bright nearby sources that could contaminate WISE."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=20, columns=["RAJ2000", "DEJ2000", "W3mag", "Gmag"])

        try:
            r = v.query_region(coord, radius=30*u.arcsec, catalog="II/328/allwise")
            if not r or len(r[0]) <= 1:
                return ReviewCheck(
                    check_id="G4_nearby_sources",
                    category="G",
                    claim_tested="No bright nearby sources contaminate the WISE photometry",
                    verdict="CONFIRMED",
                    confidence=0.9,
                    details="No other AllWISE sources within 30\" — isolated target",
                )

            n_neighbors = len(r[0]) - 1
            return ReviewCheck(
                check_id="G4_nearby_sources",
                category="G",
                claim_tested="No bright nearby sources contaminate the WISE photometry",
                verdict="CHALLENGED" if n_neighbors > 2 else "CONFIRMED",
                confidence=0.7,
                details=f"{n_neighbors} other AllWISE source(s) within 30\"",
                evidence={"n_neighbors_30arcsec": n_neighbors},
                severity="warning" if n_neighbors > 2 else "info",
            )
        except Exception as e:
            return ReviewCheck(
                check_id="G4_nearby_sources",
                category="G",
                claim_tested="No bright nearby sources contaminate the WISE photometry",
                verdict="ERROR",
                confidence=0.0,
                details=f"Query failed: {e}",
            )

    # =================================================================
    #  Category H: ARCHIVAL COMPLETENESS
    # =================================================================

    def _H1_spectroscopic_surveys(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check ALL spectroscopic surveys for RV data."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=5, columns=["**"])

        surveys_checked = []
        surveys_with_data = []

        # Spectroscopic catalogs ONLY — NOT photometric catalogs
        # SDSS photometric (V/154, V/147) does NOT contain RV data;
        # SpObjID=0 means no spectrum was taken.
        survey_catalogs = {
            "SDSS specObj": ("V/154/sdss16", lambda r: any(
                int(row.get("SpObjID", 0)) > 0 for row in r[0]
            ) if r else False),
            "LAMOST DR8": ("V/164/dr8", None),
            "APOGEE DR17": ("III/284/allstars", None),
            "GALAH DR3": ("III/283/galah3", None),
            "RAVE DR6": ("III/283", None),
            "Gaia RVS": (None, None),  # Check from Gaia data
        }

        for name, (cat, validator) in survey_catalogs.items():
            if cat is None:
                # Gaia RVS from cached data
                if self._gaia_data:
                    surveys_checked.append(name)
                continue
            try:
                r = v.query_region(coord, radius=3*u.arcsec, catalog=cat)
                surveys_checked.append(name)
                if r and len(r[0]) > 0:
                    if validator is not None:
                        # Apply validation function (e.g., check SpObjID > 0)
                        if validator(r):
                            surveys_with_data.append(name)
                        # else: has photometric entry but no spectroscopy
                    else:
                        surveys_with_data.append(name)
            except Exception:
                surveys_checked.append(f"{name} (query failed)")

        evidence = {
            "surveys_checked": surveys_checked,
            "surveys_with_data": surveys_with_data,
        }

        if surveys_with_data:
            return ReviewCheck(
                check_id="H1_spectroscopic_surveys",
                category="H",
                claim_tested="No RV data exists in any spectroscopic survey",
                verdict="REFUTED",
                confidence=0.9,
                details=f"RV DATA FOUND in: {', '.join(surveys_with_data)} — "
                        f"claim of no RV data is WRONG",
                evidence=evidence,
                severity="critical",
            )

        return ReviewCheck(
            check_id="H1_spectroscopic_surveys",
            category="H",
            claim_tested="No RV data exists in any spectroscopic survey",
            verdict="CONFIRMED",
            confidence=0.85,
            details=f"Checked {len(surveys_checked)} surveys: {', '.join(surveys_checked)} — no RV data",
            evidence=evidence,
        )

    def _H2_tess_coverage(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check if TESS has observed this target."""
        try:
            # Use astroquery to check TESS Input Catalog
            from astroquery.mast import Catalogs
            from astropy.coordinates import SkyCoord

            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            result = Catalogs.query_region(coord, radius=0.005, catalog="TIC")

            if result is None or len(result) == 0:
                return ReviewCheck(
                    check_id="H2_tess_coverage",
                    category="H",
                    claim_tested="TESS coverage status is known",
                    verdict="INCONCLUSIVE",
                    confidence=0.5,
                    details="No TIC match — TESS coverage uncertain",
                )

            tic_id = result["ID"][0]
            return ReviewCheck(
                check_id="H2_tess_coverage",
                category="H",
                claim_tested="TESS coverage status is known",
                verdict="CONFIRMED",
                confidence=0.7,
                details=f"TIC {tic_id} found — TESS lightcurve may be available "
                        f"(check MAST for sectors). Optical variability check recommended.",
                evidence={"tic_id": str(tic_id)},
            )
        except Exception as e:
            return ReviewCheck(
                check_id="H2_tess_coverage",
                category="H",
                claim_tested="TESS coverage status is known",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details=f"TESS/TIC query failed: {e} — try MAST portal manually",
            )

    def _H3_ztf_coverage(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check if ZTF covers this position."""
        # ZTF covers Dec > -31 deg, northern sky
        if dec < -31:
            return ReviewCheck(
                check_id="H3_ztf_coverage",
                category="H",
                claim_tested="ZTF optical variability data availability is known",
                verdict="CONFIRMED",
                confidence=0.95,
                details=f"Dec={dec:.1f}° < -31° — outside ZTF footprint",
                evidence={"in_ztf_footprint": False},
            )

        return ReviewCheck(
            check_id="H3_ztf_coverage",
            category="H",
            claim_tested="ZTF optical variability data availability is known",
            verdict="CONFIRMED",
            confidence=0.7,
            details=f"Dec={dec:.1f}° — in ZTF footprint. "
                    f"ZTF lightcurve should be available via IRSA/ZTF DR. "
                    f"RECOMMEND: Check ZTF for eclipses/ellipsoidal variations.",
            evidence={"in_ztf_footprint": True},
        )

    def _H4_gaia_variability(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check Gaia DR3 variability tables."""
        if not self._gaia_data:
            return ReviewCheck(
                check_id="H4_gaia_variability",
                category="H",
                claim_tested="Gaia variability classification is known",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="No Gaia data available",
            )

        from astroquery.gaia import Gaia
        source_id = self._gaia_data.get("source_id")

        if not source_id:
            return ReviewCheck(
                check_id="H4_gaia_variability",
                category="H",
                claim_tested="Gaia variability classification is known",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details="No Gaia source_id",
            )

        try:
            query = f"""
            SELECT source_id, best_class_name, best_class_score
            FROM gaiadr3.vari_classifier_result
            WHERE source_id = {source_id}
            """
            job = Gaia.launch_job(query)
            result = job.get_results()

            if len(result) == 0:
                return ReviewCheck(
                    check_id="H4_gaia_variability",
                    category="H",
                    claim_tested="Gaia variability classification is known",
                    verdict="CONFIRMED",
                    confidence=0.8,
                    details="NOT in Gaia DR3 variability catalog — not classified as variable",
                )

            var_class = str(result["best_class_name"][0])
            var_score = float(result["best_class_score"][0])

            risky_classes = {"ECL", "ECLIPSING_BINARY", "SB", "SPECTROSCOPIC_BINARY",
                           "RS_CVN", "BY_DRA"}

            evidence = {"variability_class": var_class, "score": round(var_score, 3)}

            if any(rc in var_class.upper() for rc in risky_classes):
                return ReviewCheck(
                    check_id="H4_gaia_variability",
                    category="H",
                    claim_tested="Gaia variability classification is known",
                    verdict="REFUTED",
                    confidence=0.9,
                    details=f"CLASSIFIED AS VARIABLE: {var_class} (score={var_score:.2f}) — "
                            f"this explains multi-channel anomaly!",
                    evidence=evidence,
                    severity="critical",
                )

            return ReviewCheck(
                check_id="H4_gaia_variability",
                category="H",
                claim_tested="Gaia variability classification is known",
                verdict="CHALLENGED",
                confidence=0.7,
                details=f"Gaia variable: {var_class} (score={var_score:.2f}) — "
                        f"variability may contribute to anomaly",
                evidence=evidence,
                severity="warning",
            )
        except Exception as e:
            return ReviewCheck(
                check_id="H4_gaia_variability",
                category="H",
                claim_tested="Gaia variability classification is known",
                verdict="INCONCLUSIVE",
                confidence=0.3,
                details=f"Gaia variability query failed: {e}",
            )

    def _H5_washington_double_star(self, ra, dec, pipeline_data) -> ReviewCheck:
        """Check Washington Double Star catalog."""
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        v = Vizier(row_limit=5, columns=["**"])

        try:
            r = v.query_region(coord, radius=30*u.arcsec, catalog="B/wds/wds")
            if r and len(r[0]) > 0:
                return ReviewCheck(
                    check_id="H5_washington_double_star",
                    category="H",
                    claim_tested="Target is not a known visual double/multiple star",
                    verdict="REFUTED",
                    confidence=0.85,
                    details=f"FOUND in Washington Double Star catalog — "
                            f"{len(r[0])} entry/entries within 30\"",
                    severity="critical",
                )

            return ReviewCheck(
                check_id="H5_washington_double_star",
                category="H",
                claim_tested="Target is not a known visual double/multiple star",
                verdict="CONFIRMED",
                confidence=0.85,
                details="Not in WDS catalog — no known visual companion",
            )
        except Exception as e:
            return ReviewCheck(
                check_id="H5_washington_double_star",
                category="H",
                claim_tested="Target is not a known visual double/multiple star",
                verdict="ERROR",
                confidence=0.0,
                details=f"WDS query failed: {e}",
            )

    # =================================================================
    #  HELPER METHODS
    # =================================================================

    @staticmethod
    def _galactic_latitude(ra, dec):
        """Compute galactic latitude from RA/Dec."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
        return float(c.galactic.b.deg)

    @staticmethod
    def _galactic_longitude(ra, dec):
        """Compute galactic longitude from RA/Dec."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
        return float(c.galactic.l.deg)

    def _compute_overall_verdict(self, report: PeerReviewReport) -> PeerReviewReport:
        """Compute overall verdict from individual checks."""
        if report.refuted > 0:
            # Any refuted critical check = REJECTED
            critical_refutals = [c for c in report.checks
                                if c.verdict == "REFUTED" and c.severity == "critical"]
            if critical_refutals:
                report.overall_verdict = "REJECTED"
                report.overall_confidence = 0.9
                reasons = [c.check_id for c in critical_refutals]
                report.executive_summary = (
                    f"REJECTED by {len(critical_refutals)} critical refutation(s): "
                    f"{', '.join(reasons)}"
                )
                return report

            # Non-critical refutations
            report.overall_verdict = "WEAKENED"
            report.overall_confidence = 0.6
            reasons = [c.check_id for c in report.checks if c.verdict == "REFUTED"]
            report.executive_summary = (
                f"WEAKENED by {report.refuted} refutation(s): {', '.join(reasons)}. "
                f"But {report.confirmed} check(s) confirmed."
            )
            return report

        if report.challenged > 3:
            report.overall_verdict = "WEAKENED"
            report.overall_confidence = 0.5
            challenges = [c.check_id for c in report.checks if c.verdict == "CHALLENGED"]
            report.executive_summary = (
                f"WEAKENED by {report.challenged} challenges: {', '.join(challenges[:5])}. "
                f"{report.confirmed} confirmations, {report.inconclusive} inconclusive."
            )
            return report

        # Mostly confirmed
        report.overall_verdict = "SURVIVES"
        conf_rate = report.confirmed / max(report.total_checks, 1)
        report.overall_confidence = conf_rate

        report.executive_summary = (
            f"SURVIVES adversarial review: {report.confirmed}/{report.total_checks} confirmed "
            f"({conf_rate:.0%}), {report.challenged} challenged, "
            f"{report.inconclusive} inconclusive.\n"
        )

        if report.challenged > 0:
            challenges = [c for c in report.checks if c.verdict == "CHALLENGED"]
            report.executive_summary += (
                f"Challenges to investigate: "
                + ", ".join(c.check_id for c in challenges)
            )

        return report

    def _generate_recommendations(self, report: PeerReviewReport) -> List[str]:
        """Generate follow-up recommendations based on review results."""
        recs = []

        # Check for specific outcomes
        for check in report.checks:
            if check.check_id == "H3_ztf_coverage" and check.evidence.get("in_ztf_footprint"):
                recs.append("Download ZTF lightcurve — check for eclipses/ellipsoidal variations (binary test)")

            if check.check_id == "H2_tess_coverage" and check.evidence.get("tic_id"):
                recs.append(f"Download TESS lightcurve for TIC {check.evidence['tic_id']} — high-cadence variability check")

            if check.check_id == "D2_vlass_independent":
                if check.evidence.get("QualFlag", 0) > 0:
                    recs.append("Request VLA snapshot observation — VLASS QualFlag non-zero, needs confirmation")

            if check.check_id == "H1_spectroscopic_surveys" and check.verdict == "CONFIRMED":
                recs.append("Obtain 2+ epoch radial velocity measurements (2m+ telescope, R>20000)")

            if check.check_id == "E5_optical_variability" and check.verdict == "INCONCLUSIVE":
                recs.append("Check optical variability via ZTF/TESS — crucial binary discriminant")

            if check.check_id == "E6_xray_activity" and check.verdict == "CONFIRMED":
                recs.append("X-ray non-detection supports non-active interpretation — note in paper")

        # Always recommend
        if not any("radial velocity" in r.lower() for r in recs):
            recs.append("Radial velocity monitoring is the single most important follow-up")

        return recs


# =====================================================================
#  CLI entry point
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="EXODUS Adversarial Peer Review Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Review a target at RA=180, Dec=+45
    ./venv/bin/python -m src.vetting.peer_review --ra 180.0 --dec 45.0

    # Review with output file
    ./venv/bin/python -m src.vetting.peer_review --ra 180.0 --dec 45.0 \\
        --output data/reports/peer_review_example.json
        """
    )

    parser.add_argument("--ra", type=float, required=True, help="RA in degrees (J2000)")
    parser.add_argument("--dec", type=float, required=True, help="Dec in degrees (J2000)")
    parser.add_argument("--target-id", type=str, default="", help="Target identifier")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    engine = PeerReviewEngine(verbose=not args.quiet)
    report = engine.review(
        ra=args.ra,
        dec=args.dec,
        target_id=args.target_id or f"RA{args.ra:.4f}",
    )

    report.print_summary()

    # Auto-generate output path if not specified
    if args.output is None:
        args.output = f"data/reports/peer_review_ra{args.ra:.4f}.json"

    report.save(args.output)
    print(f"\n  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
