#!/usr/bin/env python3
"""
Project EXODUS -- Full Research Loop Runner
============================================

Orchestrates the complete EXODUS pipeline in an iterative research loop:

    Load targets -> Ingest data -> Process (IR excess, transit anomaly, radio)
    -> Score -> Hypothesis Generator -> Analyst validation
    -> Breakthrough Engine (for UNEXPLAINED) -> Evolver self-improvement
    -> Generate report -> Loop

Each iteration builds on the last.  Full state is checkpointed to disk after
every iteration so that runs can be resumed with ``--resume``.

Usage
-----
    python scripts/run_exodus.py --max-iterations 5 --tier tier1
    python scripts/run_exodus.py --resume                       # pick up where you left off
    python scripts/run_exodus.py --quick                        # fast laptop run

See ``python scripts/run_exodus.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Ensure project root is on sys.path ──────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Project utilities (always required) ─────────────────────────────
from src.utils import get_config, get_logger, save_result, safe_json_dump, PROJECT_ROOT as _PR

log = get_logger("runner")

# ── Graceful imports: each module is optional ────────────────────────
# Ingestion
try:
    from src.ingestion.exoplanet_archive import (
        get_hz_planets,
        get_all_hosts,
        query_exoplanet_archive,
    )
except ImportError as e:
    log.warning("exoplanet_archive unavailable: %s", e)
    get_hz_planets = get_all_hosts = query_exoplanet_archive = None

try:
    from src.ingestion.gaia_query import (
        get_stellar_params,
        get_astrometry,
        get_epoch_photometry,
        query_target_list,
    )
except ImportError as e:
    log.warning("gaia_query unavailable: %s", e)
    get_stellar_params = get_astrometry = get_epoch_photometry = None
    query_target_list = None

try:
    from src.ingestion.ir_surveys import (
        get_ir_photometry,
        get_ir_photometry_batch,
    )
except ImportError as e:
    log.warning("ir_surveys unavailable: %s", e)
    get_ir_photometry = get_ir_photometry_batch = None

try:
    from src.ingestion.lightcurves import get_lightcurve, stitch_lightcurves
except ImportError as e:
    log.warning("lightcurves unavailable: %s", e)
    get_lightcurve = stitch_lightcurves = None

try:
    from src.ingestion.breakthrough_listen import (
        get_observation,
        get_spectrogram,
        list_available_targets,
    )
except ImportError as e:
    log.warning("breakthrough_listen unavailable: %s", e)
    get_observation = get_spectrogram = list_available_targets = None

# Processing
try:
    from src.processing.ir_excess import compute_ir_excess, compute_ir_excess_batch
except ImportError as e:
    log.warning("ir_excess unavailable: %s", e)
    compute_ir_excess = compute_ir_excess_batch = None

try:
    from src.processing.transit_anomaly import (
        detect_transit_anomaly,
        detect_irregular_dimming,
    )
except ImportError as e:
    log.warning("transit_anomaly unavailable: %s", e)
    detect_transit_anomaly = detect_irregular_dimming = None

try:
    from src.processing.radio_processor import process_spectrogram
except ImportError as e:
    log.warning("radio_processor unavailable: %s", e)
    process_spectrogram = None

# Correlation
try:
    from src.correlation.multi_modal import MultiModalCorrelator
except ImportError as e:
    log.warning("multi_modal correlator unavailable: %s", e)
    MultiModalCorrelator = None

# Scoring
try:
    from src.scoring.exodus_score import EXODUSScorer
except ImportError as e:
    log.warning("exodus_score unavailable: %s", e)
    EXODUSScorer = None

try:
    from src.core.controls import select_matched_controls
except ImportError as e:
    log.warning("controls unavailable: %s", e)
    select_matched_controls = None

try:
    from src.ingestion.gaia_query import cone_search as gaia_cone_search
except ImportError as e:
    log.warning("gaia cone_search unavailable: %s", e)
    gaia_cone_search = None

# Multi-messenger detection
try:
    from src.detection.gamma_exoplanet_crossmatch import crossmatch_fermi_exoplanets
except ImportError:
    crossmatch_fermi_exoplanets = None

try:
    from src.detection.neutrino_exoplanet_crossmatch import crossmatch_neutrino_exoplanets
except ImportError:
    crossmatch_neutrino_exoplanets = None

try:
    from src.detection.gw_exoplanet_crossmatch import crossmatch_gw_exoplanets
except ImportError:
    crossmatch_gw_exoplanets = None

try:
    from src.detection.pulsar_structure_search import search_pulsar_los
except ImportError:
    search_pulsar_los = None

try:
    from src.detection.frb_orbital_correlation import correlate_frb_orbits
except ImportError:
    correlate_frb_orbits = None

try:
    from src.detection.temporal_archaeology import TemporalArchaeology
except ImportError:
    TemporalArchaeology = None

try:
    from src.detection.stellar_anomaly import compute_pm_consistency
except ImportError:
    compute_pm_consistency = None

try:
    from src.ingestion.ir_surveys import get_catwise
except ImportError:
    get_catwise = None

try:
    from src.vetting.astrophysical_templates import UnexplainabilityScorer
except ImportError as e:
    log.warning("astrophysical_templates unavailable: %s", e)
    UnexplainabilityScorer = None

try:
    from src.vetting.red_team import RedTeamEngine
except ImportError as e:
    log.warning("red_team unavailable: %s", e)
    RedTeamEngine = None

# Engines
try:
    from src.engines.generator import HypothesisGenerator
except ImportError as e:
    log.warning("generator unavailable: %s", e)
    HypothesisGenerator = None

try:
    from src.engines.analyst import AnalystEngine, ValidationStatus
except ImportError as e:
    log.warning("analyst unavailable: %s", e)
    AnalystEngine = ValidationStatus = None

try:
    from src.engines.breakthrough import BreakthroughEngine
except ImportError as e:
    log.warning("breakthrough unavailable: %s", e)
    BreakthroughEngine = None

try:
    from src.engines.evolver import EvolverEngine, ResearchState
except ImportError as e:
    log.warning("evolver unavailable: %s", e)
    EvolverEngine = ResearchState = None


# =====================================================================
#  ExodusRunner
# =====================================================================

class ExodusRunner:
    """Full research loop orchestrator for Project EXODUS.

    Parameters
    ----------
    max_iterations : int
        Number of research-loop iterations to run.
    tier : str
        Target selection tier (``"tier1"`` or ``"tier2"``).
    quick : bool
        If True, run a cut-down version suitable for a laptop
        (20 targets, 1 iteration, IR + Gaia only).
    resume : bool
        If True, attempt to reload state from the last checkpoint.
    """

    # Paths for persistence
    STATE_DIR = PROJECT_ROOT / "data" / "state"
    STATE_FILE = STATE_DIR / "runner_state.json"
    REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

    def __init__(
        self,
        max_iterations: int = 10,
        tier: str = "tier1",
        quick: bool = False,
        resume: bool = False,
        target_file: Optional[str] = None,
        campaign_report: bool = False,
    ):
        self.max_iterations = max_iterations
        self.tier = tier
        self.quick = quick
        self.resume = resume
        self.target_file = target_file
        self.campaign_report = campaign_report
        self.campaign_metadata = None  # set by _load_targets if using target file

        # Config from settings.yaml
        self.cfg = get_config()

        # Current iteration counter (may be overridden by _load_state)
        self.iteration = 0

        # Accumulated results across iterations
        self.all_results: List[Dict[str, Any]] = []
        self.all_scores: List[Dict[str, Any]] = []
        self.hypotheses_tested: List[Dict[str, Any]] = []
        self.anomalies_found: int = 0
        self.false_positives: int = 0

        # Thresholds (evolve over time)
        self.thresholds: Dict[str, float] = {
            "anomaly_sigma": self.cfg.get("search", {}).get("anomaly_sigma", 3.0),
        }

        # Initialise engines (only those that imported successfully)
        self.generator = HypothesisGenerator() if HypothesisGenerator else None
        self.analyst = AnalystEngine() if AnalystEngine else None
        self.breakthrough = BreakthroughEngine() if BreakthroughEngine else None
        self.evolver = EvolverEngine() if EvolverEngine else None
        self.correlator = MultiModalCorrelator() if MultiModalCorrelator else None
        self.scorer = EXODUSScorer() if EXODUSScorer else None

        # Ensure output directories exist
        self.STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        if self.resume:
            self._load_state()

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the full research loop.

        Returns
        -------
        dict
            Summary of the entire run, including top targets and
            recommendations from the Evolver.
        """
        log.info(
            "=== Project EXODUS Research Loop ===  "
            "iterations=%d  tier=%s  quick=%s  resume=%s",
            self.max_iterations, self.tier, self.quick, self.resume,
        )

        start_time = time.time()
        start_iter = self.iteration

        for i in range(start_iter, self.max_iterations):
            self.iteration = i
            iter_start = time.time()
            log.info(
                "────────── Iteration %d / %d ──────────",
                i + 1, self.max_iterations,
            )

            try:
                # Step 1: Load targets
                targets = self._load_targets()
                if not targets:
                    log.error("No targets loaded -- aborting iteration.")
                    break

                log.info("Loaded %d targets.", len(targets))

                # Step 2: Ingest / gather observational data
                targets_data = self._gather_data(targets)

                # Step 3: Process data (IR excess, transit anomaly, radio)
                results = self._process_data(targets_data)

                # Step 3a: Multi-messenger cross-matching
                mm_summary = self._run_multi_messenger(results)
                self.mm_summary = mm_summary

                # Step 3b: Select matched controls and calibrate scorer
                control_scores = self._select_controls(results)
                self._control_scores = control_scores or {}
                if control_scores:
                    self.scorer = EXODUSScorer(control_scores=control_scores) if EXODUSScorer else None

                # Step 4: Score targets (now calibrated if controls available)
                scored = self._score_targets(results)

                # Step 4a: Unexplainability evaluation
                self._evaluate_unexplainability(scored)

                # Step 4b: Red-Team falsification
                self._run_red_team(scored)

                # Step 5-7: Hypothesis cycle
                #   Generator -> Analyst -> Breakthrough (for UNEXPLAINED)
                hyp_results = self._run_hypothesis_cycle(results)

                # Step 8: Evolve (self-improve thresholds & strategies)
                evolution = self._evolve_iteration(results, hyp_results)

                # Accumulate results for campaign report generation
                self.all_results = results

                # Step 9: Generate iteration report
                iter_summary = {
                    "iteration": i,
                    "n_targets": len(targets),
                    "n_results": len(results),
                    "n_scored": len(scored),
                    "multi_messenger": mm_summary,
                    "hypothesis_results": hyp_results,
                    "evolution": evolution,
                    "elapsed_sec": round(time.time() - iter_start, 1),
                }

                self._generate_report(i, iter_summary)

                # Step 10: Checkpoint state
                self._save_state()

                log.info(
                    "Iteration %d complete in %.1f s.",
                    i + 1, time.time() - iter_start,
                )

            except KeyboardInterrupt:
                log.warning("Interrupted by user at iteration %d.", i + 1)
                self._save_state()
                break

            except Exception:
                log.error("Error in iteration %d:\n%s", i + 1, traceback.format_exc())
                self._save_state()
                continue

        elapsed = time.time() - start_time
        summary = self._build_run_summary(elapsed)
        self._generate_report("final", summary)

        # Campaign report generation (if --campaign-report flag set)
        if self.campaign_report and self.campaign_metadata is not None:
            try:
                from src.output.campaign_report import (
                    generate_calibration_report,
                    generate_campaign_report,
                )
                campaign_name = getattr(self.campaign_metadata, "campaign", "")
                results = self.all_results if self.all_results else []

                if campaign_name == "calibration":
                    cal_report = generate_calibration_report(
                        results, self.campaign_metadata,
                    )
                    log.info(
                        "Calibration report: %s (all_passed=%s)",
                        cal_report.campaign, cal_report.all_passed,
                    )
                else:
                    camp_report = generate_campaign_report(
                        results, self.campaign_metadata,
                    )
                    log.info(
                        "Campaign report: %s (%d convergent, %d FDR-sig)",
                        camp_report.campaign,
                        len(camp_report.convergent_targets),
                        len(camp_report.fdr_targets),
                    )
            except Exception as exc:
                log.warning("Campaign report generation failed: %s", exc)

        log.info(
            "=== EXODUS run complete.  %d iterations in %.1f min ===",
            self.iteration - start_iter + 1, elapsed / 60.0,
        )
        return summary

    # ── Step 1: Load targets ─────────────────────────────────────────

    def _load_targets(self) -> List[Dict[str, Any]]:
        """Load targets according to the selected tier or custom target file.

        Returns a list of dicts, each with at minimum:
            ``target_id``, ``ra``, ``dec``
        Plus any planet / host metadata from the archive or target file.
        """
        # --- Custom target file path ---
        if self.target_file is not None:
            try:
                from src.ingestion.target_loader import (
                    load_target_file,
                    enrich_target_metadata,
                )
                campaign = load_target_file(self.target_file)
                self.campaign_metadata = campaign
                targets = enrich_target_metadata(campaign.targets)
                log.info(
                    "Loaded %d targets from campaign '%s': %s",
                    len(targets), campaign.campaign, campaign.description,
                )
                return targets
            except Exception:
                log.error(
                    "Failed to load target file '%s':\n%s",
                    self.target_file, traceback.format_exc(),
                )
                return []

        # --- Standard NASA Exoplanet Archive path ---
        targets: List[Dict[str, Any]] = []

        if get_hz_planets is None:
            log.warning("Exoplanet archive unavailable; using empty target list.")
            return targets

        try:
            if self.tier == "tier1":
                max_dist = 100  # pc
                df = get_hz_planets(max_distance_pc=max_dist)
            else:
                max_dist = 500
                df = get_hz_planets(max_distance_pc=max_dist)

            if df is None or df.empty:
                log.warning("No targets returned from exoplanet archive.")
                return targets

            # Respect tier limits from config
            tier_key = f"{self.tier}_max"
            tier_max = self.cfg.get("targets", {}).get(tier_key, len(df))
            df = df.head(tier_max)

            # Quick mode: cap at 20
            if self.quick:
                df = df.head(20)

            for _, row in df.iterrows():
                targets.append({
                    "target_id": str(row.get("planet_name", row.get("host_star", ""))),
                    "host_star": str(row.get("host_star", "")),
                    "ra": float(row["ra_deg"]),
                    "dec": float(row["dec_deg"]),
                    "distance_pc": float(row.get("distance_pc", 0)) if row.get("distance_pc") else None,
                    "hz_flag": bool(row.get("hz_flag", False)),
                })

        except Exception:
            log.error("Failed to load targets:\n%s", traceback.format_exc())

        return targets

    # ── PM-aware Gaia source selection helpers ──────────────────────

    @staticmethod
    def _search_radius_for_target(target: Dict[str, Any], catalog_epoch: float = 2010.0) -> float:
        """Compute a proper-motion-aware search radius for a target."""
        from src.utils import get_config
        try:
            cfg = get_config()
            base_radius = float(cfg["search"]["crossmatch_radius_arcsec"])
        except Exception:
            base_radius = 5.0

        dist = target.get("distance_pc")
        if dist is None or dist <= 0:
            return base_radius

        known_pmra = target.get("pmra_mas")
        known_pmdec = target.get("pmdec_mas")
        if known_pmra is not None and known_pmdec is not None:
            import numpy as np
            total_pm_mas = np.sqrt(known_pmra**2 + known_pmdec**2)
            estimated_pm = total_pm_mas / 1000.0
        else:
            estimated_pm = min(5.0 / dist, 10.0)

        dt = abs(catalog_epoch - 2000.0)
        required = base_radius + estimated_pm * dt * 1.5
        return max(base_radius, min(required, 120.0))

    @staticmethod
    def _pick_target_from_gaia(
        gaia_df: "pd.DataFrame",
        target_ra_j2000: float,
        target_dec_j2000: float,
        gaia_epoch: float = 2016.0,
    ) -> "pd.Series":
        """Select the correct target by back-propagating Gaia positions to J2000."""
        import numpy as np

        if len(gaia_df) == 1:
            return gaia_df.iloc[0]

        dt = 2000.0 - gaia_epoch
        best_idx = 0
        best_sep = float("inf")

        for i in range(len(gaia_df)):
            row = gaia_df.iloc[i]
            gaia_ra = float(row.get("ra", 0))
            gaia_dec = float(row.get("dec", 0))
            pmra = row.get("pmra", 0)
            pmdec = row.get("pmdec", 0)

            if pmra is None or pmdec is None:
                pmra, pmdec = 0.0, 0.0
            elif not (np.isfinite(pmra) and np.isfinite(pmdec)):
                pmra, pmdec = 0.0, 0.0

            dra_deg = (float(pmra) / 1000.0 / 3600.0) / np.cos(
                np.radians(gaia_dec)
            )
            ddec_deg = float(pmdec) / 1000.0 / 3600.0
            ra_at_j2000 = gaia_ra + dra_deg * dt
            dec_at_j2000 = gaia_dec + ddec_deg * dt

            cos_dec = np.cos(np.radians(target_dec_j2000))
            sep = np.sqrt(
                ((ra_at_j2000 - target_ra_j2000) * cos_dec) ** 2
                + (dec_at_j2000 - target_dec_j2000) ** 2
            ) * 3600.0

            if sep < best_sep:
                best_sep = sep
                best_idx = i

        log.info(
            "    Gaia source selection: %d candidates, best=%.1f\" from J2000",
            len(gaia_df), best_sep,
        )
        return gaia_df.iloc[best_idx]

    @staticmethod
    def _lightcurve_query_name(target: Dict[str, Any]) -> str:
        """Convert a target dict to a lightkurve-friendly query name.

        lightkurve resolves names like "HD 139139", "TOI-700", "Kepler-442".
        Our target_ids use underscores and include planet suffixes (e.g.,
        "Kepler-442_b").  This method strips the planet suffix and replaces
        underscores with spaces where appropriate.
        """
        import re

        # Prefer host_star over target_id (avoids planet suffix)
        name = target.get("host_star") or target.get("target_id", "")

        # Replace underscores with spaces for multi-word names
        # but preserve hyphens in catalog names (Kepler-442, TOI-700, TRAPPIST-1)
        name = name.replace("_", " ")

        # Strip trailing planet designators: " b", " c", " d", " e", " f", " g"
        name = re.sub(r'\s+[b-h]$', '', name)

        return name.strip()

    # ── Step 2: Ingest / gather data ─────────────────────────────────

    def _gather_data(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ingest observational data for each target.

        Pulls from Gaia, IR surveys, light curves, and Breakthrough Listen.
        Each target dict is enriched in-place with the raw data layers.
        """
        log.info("Gathering data for %d targets ...", len(targets))

        for idx, t in enumerate(targets):
            ra, dec = t["ra"], t["dec"]
            tid = t["target_id"]
            log.info(
                "  [%d/%d] %s  (RA=%.4f, Dec=%.4f)",
                idx + 1, len(targets), tid, ra, dec,
            )

            # --- Use known PMs from target file if available ---
            _known_pmra = t.get("pmra_mas")
            _known_pmdec = t.get("pmdec_mas")
            pmra_mas = float(_known_pmra) if _known_pmra is not None else 0.0
            pmdec_mas = float(_known_pmdec) if _known_pmdec is not None else 0.0
            matched_source_id = None

            # Dynamic Gaia search radius for nearby/high-PM targets
            gaia_radius = self._search_radius_for_target(t, catalog_epoch=2016.0)

            # --- Step 1: Gaia astrometry (PM-aware source selection) ---
            if get_astrometry is not None:
                try:
                    astro_df = get_astrometry(ra, dec, radius_arcsec=gaia_radius)
                    if astro_df is not None and not astro_df.empty:
                        best_row = self._pick_target_from_gaia(
                            astro_df, ra, dec, gaia_epoch=2016.0,
                        )
                        t["gaia_astrometry"] = best_row.to_dict()
                        matched_source_id = best_row.get("source_id")
                        # Use Gaia PM if no known PM
                        if abs(pmra_mas) < 0.1 and abs(pmdec_mas) < 0.1:
                            import numpy as np
                            _pmra = best_row.get("pmra")
                            _pmdec = best_row.get("pmdec")
                            if (_pmra is not None and _pmdec is not None
                                    and np.isfinite(_pmra) and np.isfinite(_pmdec)):
                                pmra_mas = float(_pmra)
                                pmdec_mas = float(_pmdec)
                except Exception as exc:
                    log.debug("Gaia astrometry failed for %s: %s", tid, exc)

            # --- Step 1b: Gaia stellar parameters (match by source_id) ---
            if get_stellar_params is not None:
                try:
                    gaia_df = get_stellar_params(ra, dec, radius_arcsec=gaia_radius)
                    if gaia_df is not None and not gaia_df.empty:
                        if (matched_source_id is not None
                                and "source_id" in gaia_df.columns):
                            match = gaia_df[
                                gaia_df["source_id"] == matched_source_id
                            ]
                            if not match.empty:
                                t["gaia_params"] = match.iloc[0].to_dict()
                                log.info("    Stellar params matched by source_id")
                            else:
                                best_row = self._pick_target_from_gaia(
                                    gaia_df, ra, dec, gaia_epoch=2016.0,
                                )
                                t["gaia_params"] = best_row.to_dict()
                        else:
                            best_row = self._pick_target_from_gaia(
                                gaia_df, ra, dec, gaia_epoch=2016.0,
                            )
                            t["gaia_params"] = best_row.to_dict()
                except Exception as exc:
                    log.debug("Gaia params failed for %s: %s", tid, exc)

            # --- Gaia epoch photometry ---
            if get_epoch_photometry is not None:
                try:
                    source_id = (t.get("gaia_params") or {}).get("source_id")
                    if source_id:
                        epoch_df = get_epoch_photometry(int(source_id))
                        if epoch_df is not None and not epoch_df.empty:
                            t["gaia_epoch_photometry"] = {
                                "time": epoch_df["g_obs_time"].tolist()
                                    if "g_obs_time" in epoch_df.columns else [],
                                "mag": epoch_df["g_transit_mag"].tolist()
                                    if "g_transit_mag" in epoch_df.columns else [],
                            }
                except Exception as exc:
                    log.debug("Gaia epoch phot failed for %s: %s", tid, exc)

            # --- Step 2: Epoch-propagated coordinates for IR queries ---
            # High-PM stars can be 15-40" from J2000 positions in 2MASS/WISE
            has_pm = abs(pmra_mas) > 0.1 or abs(pmdec_mas) > 0.1
            gaia_astro = t.get("gaia_astrometry", {})
            gaia_ra_2016 = gaia_astro.get("ra")
            gaia_dec_2016 = gaia_astro.get("dec")

            if has_pm and gaia_ra_2016 is not None and gaia_dec_2016 is not None:
                import numpy as np
                gaia_ra_2016 = float(gaia_ra_2016)
                gaia_dec_2016 = float(gaia_dec_2016)
                dra_per_yr = (pmra_mas / 1000.0 / 3600.0) / np.cos(
                    np.radians(gaia_dec_2016))
                ddec_per_yr = pmdec_mas / 1000.0 / 3600.0
                # 2MASS: J2000 epoch
                ra_2mass = gaia_ra_2016 + dra_per_yr * (2000.0 - 2016.0)
                dec_2mass = gaia_dec_2016 + ddec_per_yr * (2000.0 - 2016.0)
                # AllWISE: ~2010 epoch
                ra_wise = gaia_ra_2016 + dra_per_yr * (2010.0 - 2016.0)
                dec_wise = gaia_dec_2016 + ddec_per_yr * (2010.0 - 2016.0)
                log.info("    PM-corrected coords: 2MASS(%.4f,%+.4f) WISE(%.4f,%+.4f)",
                         ra_2mass, dec_2mass, ra_wise, dec_wise)
            else:
                ra_2mass, dec_2mass = ra, dec
                ra_wise, dec_wise = ra, dec

            # --- IR photometry at epoch-corrected positions ---
            if get_ir_photometry is not None:
                try:
                    from src.utils import get_config as _gc
                    try:
                        _base_r = float(_gc()["search"]["crossmatch_radius_arcsec"])
                    except Exception:
                        _base_r = 5.0
                    from src.ingestion.ir_surveys import get_2mass, get_wise

                    ir_merged = {"ra": ra, "dec": dec}

                    # 2MASS at J2000 coords
                    twomass = get_2mass(ra_2mass, dec_2mass, radius_arcsec=_base_r)
                    if twomass:
                        for band in ("J", "H", "Ks", "J_err", "H_err", "Ks_err"):
                            if band in twomass:
                                ir_merged[band] = twomass[band]
                        ir_merged["twomass_designation"] = twomass.get("designation")

                    # WISE at ~2010 epoch coords
                    wise = get_wise(ra_wise, dec_wise, radius_arcsec=_base_r)
                    if wise:
                        for band in ("W1", "W2", "W3", "W4",
                                     "W1_err", "W2_err", "W3_err", "W4_err"):
                            if band in wise:
                                ir_merged[band] = wise[band]
                        ir_merged["wise_designation"] = wise.get("designation")
                        # WISE quality flags for contamination veto
                        if wise.get("cc_flags") is not None:
                            ir_merged["cc_flags"] = wise["cc_flags"]
                        if wise.get("ext_flg") is not None:
                            ir_merged["ext_flg"] = wise["ext_flg"]

                    # CatWISE2020 (PM-corrected W1/W2 + WISE proper motions)
                    if get_catwise is not None:
                        try:
                            catwise = get_catwise(ra_wise, dec_wise,
                                                  radius_arcsec=_base_r)
                            if catwise:
                                for ck in ("W1_catwise", "W2_catwise",
                                           "pmra_wise", "e_pmra_wise",
                                           "pmdec_wise", "e_pmdec_wise"):
                                    if ck in catwise:
                                        ir_merged[ck] = catwise[ck]
                                ir_merged["catwise_designation"] = catwise.get("designation")
                        except Exception as exc:
                            log.debug("CatWISE query failed for %s: %s", tid, exc)

                    # Merge Gaia G/BP/RP for full SED fit
                    gaia_p = t.get("gaia_params", {})
                    if gaia_p:
                        import numpy as np
                        for gaia_key, band, err_band in [
                            ("phot_g_mean_mag", "G", "G_err"),
                            ("phot_bp_mean_mag", "BP", "BP_err"),
                            ("phot_rp_mean_mag", "RP", "RP_err"),
                        ]:
                            val = gaia_p.get(gaia_key)
                            if val is not None and np.isfinite(val):
                                ir_merged[band] = float(val)
                                ir_merged[err_band] = 0.01
                    t["ir_photometry"] = ir_merged
                except Exception as exc:
                    log.debug("IR photometry failed for %s: %s", tid, exc)

            # --- Light curves (Kepler / TESS) ---
            if get_lightcurve is not None and not self.quick:
                try:
                    lk_name = self._lightcurve_query_name(t)
                    lc = get_lightcurve(lk_name)
                    if lc is not None:
                        t["lightcurve"] = {
                            "time": lc.time.value.tolist(),
                            "flux": lc.flux.value.tolist(),
                            "flux_err": lc.flux_err.value.tolist()
                                if lc.flux_err is not None else None,
                            "mission": (lc.meta.get("MISSION", "unknown")
                                        if hasattr(lc, "meta") else "unknown"),
                        }
                        log.info("    Lightcurve: %d points (%s)",
                                 len(lc.flux), t["lightcurve"]["mission"])
                except Exception as exc:
                    log.debug("Light curve failed for %s: %s", tid, exc)

            # --- Breakthrough Listen radio ---
            if get_spectrogram is not None and not self.quick:
                try:
                    spec, freqs, times, radio_source = get_spectrogram(tid)
                    if spec is not None:
                        t["radio_spectrogram"] = {
                            "spectrogram": spec,
                            "frequencies_mhz": freqs,
                            "timestamps_sec": times,
                            "data_source": radio_source,
                        }
                except Exception as exc:
                    log.debug("Radio data failed for %s: %s", tid, exc)

        return targets

    # ── Step 3: Process data ─────────────────────────────────────────

    def _process_data(
        self, targets_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run analysis pipelines on ingested data.

        Enriches each target dict with processing results:
            - ``ir_excess``: IRExcessResult (as dict)
            - ``transit_anomaly``: TransitAnomalyResult (as dict)
            - ``radio_anomaly``: RadioProcessorResult (as dict)
            - ``gaia_photometric_anomaly``: variability metrics
            - ``habitable_zone_planet``: HZ flag data
            - ``proper_motion_anomaly``: astrometric excess data
        """
        log.info("Processing data for %d targets ...", len(targets_data))
        anomaly_count = 0

        for t in targets_data:
            tid = t["target_id"]

            # --- IR excess ---
            if compute_ir_excess is not None and t.get("ir_photometry"):
                try:
                    ir_result = compute_ir_excess(t["ir_photometry"])
                    # Only record IR excess if blackbody fit had enough bands.
                    # With < 2 bands the fit is meaningless and calibration
                    # must NOT count this as "channel tested".
                    if ir_result.fit_bands_used >= 2:
                        t["ir_excess"] = {
                            "sigma_W3": ir_result.sigma_W3,
                            "sigma_W4": ir_result.sigma_W4,
                            "is_candidate": ir_result.is_candidate,
                            "fitted_teff": ir_result.fitted_teff,
                            "excess_W3": ir_result.excess_W3,
                            "excess_W4": ir_result.excess_W4,
                            "fit_chi2_reduced": ir_result.fit_chi2_reduced,
                            "fit_bands_used": ir_result.fit_bands_used,
                            "contamination_flag": ir_result.contamination_flag,
                            "data_source": "real",
                        }
                        if ir_result.is_candidate:
                            anomaly_count += 1
                    else:
                        log.info("IR excess for %s: only %d bands, insufficient for scoring",
                                 tid, ir_result.fit_bands_used)
                except Exception as exc:
                    log.debug("IR excess processing failed for %s: %s", tid, exc)

            # --- Transit anomaly (with downsampling + irregular dimming + starspot filter) ---
            if (
                detect_transit_anomaly is not None
                and t.get("lightcurve")
                and not self.quick
            ):
                try:
                    lc_data = t["lightcurve"]
                    time_arr = lc_data["time"]
                    flux_arr = lc_data["flux"]
                    flux_err_arr = lc_data.get("flux_err")

                    # Downsample large lightcurves to cap BLS runtime
                    _MAX_LC_POINTS = 30_000
                    n_orig = len(time_arr)
                    if n_orig > _MAX_LC_POINTS:
                        import numpy as np
                        step = max(2, -(-n_orig // _MAX_LC_POINTS))
                        time_arr = time_arr[::step]
                        flux_arr = flux_arr[::step]
                        if flux_err_arr is not None:
                            flux_err_arr = flux_err_arr[::step]
                        log.info("  Lightcurve downsampled: %d -> %d points (step=%d)",
                                 n_orig, len(time_arr), step)

                    if len(time_arr) >= 50:
                        ta_result = detect_transit_anomaly(
                            time=time_arr,
                            flux=flux_arr,
                            flux_err=flux_err_arr,
                        )

                        # Irregular dimming (Tabby's Star-type events)
                        irreg_score = 0.0
                        irreg_events = 0
                        if detect_irregular_dimming is not None and len(time_arr) >= 100:
                            try:
                                irreg_result = detect_irregular_dimming(
                                    time=time_arr, flux=flux_arr,
                                )
                                irreg_score = irreg_result.anomaly_score
                                irreg_events = irreg_result.n_events
                            except Exception:
                                pass

                        combined_score = max(ta_result.anomaly_score, irreg_score)

                        # Starspot rotation filter
                        starspot_flag = False
                        if (ta_result.anomaly_score > 0.2
                                and ta_result.shape_residual > 0.5
                                and ta_result.symmetry_score < 0.3
                                and ta_result.depth_variability < 0.3):
                            starspot_flag = True
                            log.info("  Starspot filter: %s BLS period=%.2fd, shape_res=%.3f"
                                     " >> suppressing periodic score",
                                     tid, ta_result.period, ta_result.shape_residual)
                            combined_score = irreg_score

                        t["transit_anomaly"] = {
                            "anomaly_score": combined_score,
                            "is_anomalous": (not starspot_flag and ta_result.is_anomalous)
                                            or combined_score > 0.3,
                            "period": ta_result.period,
                            "depth": ta_result.depth,
                            "symmetry_score": ta_result.symmetry_score,
                            "depth_variability": ta_result.depth_variability,
                            "shape_residual": ta_result.shape_residual,
                            "irregular_events": irreg_events,
                            "irregular_score": irreg_score,
                            "starspot_suppressed": starspot_flag,
                            "n_datapoints": n_orig,
                            "n_analyzed": len(time_arr),
                            "mission": lc_data.get("mission", "unknown"),
                            "data_source": "real",
                        }
                        if combined_score > 0.3:
                            anomaly_count += 1
                            log.info("  Transit anomaly for %s: score=%.3f "
                                     "(periodic=%.3f, irregular=%.3f)",
                                     tid, combined_score,
                                     ta_result.anomaly_score, irreg_score)
                    else:
                        log.info("  Lightcurve for %s too short (%d pts)",
                                 tid, len(time_arr))
                except Exception as exc:
                    log.debug("Transit anomaly failed for %s: %s", tid, exc)

            # --- Radio processing ---
            if (
                process_spectrogram is not None
                and t.get("radio_spectrogram")
                and not self.quick
            ):
                try:
                    rs = t["radio_spectrogram"]
                    radio_result = process_spectrogram(
                        spectrogram=rs["spectrogram"],
                        freqs=rs["frequencies_mhz"],
                        times=rs["timestamps_sec"],
                    )
                    non_rfi = [c for c in radio_result.candidates if not c.is_rfi]
                    max_snr = max((c.snr for c in non_rfi), default=0.0)
                    t["radio_anomaly"] = {
                        "n_candidates": len(non_rfi),
                        "max_snr": max_snr,
                        "n_rfi_flagged": radio_result.n_rfi_flagged,
                        "noise_floor": radio_result.noise_floor,
                        "data_source": rs.get("data_source", "unknown"),
                        "candidates": [
                            {
                                "frequency_hz": c.frequency_hz,
                                "drift_rate_hz_per_s": c.drift_rate_hz_per_s,
                                "snr": c.snr,
                                "is_rfi": c.is_rfi,
                            }
                            for c in radio_result.candidates
                        ],
                    }
                    if len(non_rfi) > 0:
                        anomaly_count += 1
                except Exception as exc:
                    log.debug("Radio processing failed for %s: %s", tid, exc)

            # --- Gaia photometric anomaly ---
            if t.get("gaia_epoch_photometry"):
                epoch = t["gaia_epoch_photometry"]
                mags = epoch.get("mag", [])
                if len(mags) > 2:
                    import numpy as np
                    mags_arr = np.array(mags, dtype=float)
                    mags_arr = mags_arr[np.isfinite(mags_arr)]
                    if len(mags_arr) > 2:
                        # Convert magnitudes to flux for proper variability
                        # calculation (magnitudes are logarithmic).
                        flux_arr = 10.0 ** (-0.4 * mags_arr)
                        variability = float(np.std(flux_arr) / np.mean(flux_arr)) \
                            if np.mean(flux_arr) != 0 else 0.0
                        t["gaia_photometric_anomaly"] = {
                            "phot_g_variability": variability,
                            "n_epochs": len(mags_arr),
                            "data_source": "real",
                        }

            # --- Habitable zone planet flag ---
            t["habitable_zone_planet"] = {
                "has_hz_planet": t.get("hz_flag", False),
                "n_hz_planets": 1 if t.get("hz_flag", False) else 0,
                "data_source": "real",
            }

            # --- Proper motion anomaly ---
            astro = t.get("gaia_astrometry", {})
            if astro:
                pm_anomaly = {
                    "ruwe": float(astro.get("ruwe") or 1.0),
                    "astrometric_excess_noise_sig": float(
                        astro.get("astrometric_excess_noise_sig") or 0.0
                    ),
                    "data_source": "real",
                }

                # WISE-Gaia PM consistency check
                ir = t.get("ir_photometry", {})
                if (compute_pm_consistency is not None
                        and ir.get("pmra_wise") is not None
                        and ir.get("pmdec_wise") is not None
                        and astro.get("pmra") is not None
                        and astro.get("pmdec") is not None):
                    try:
                        pm_check = compute_pm_consistency(
                            pmra_gaia=float(astro["pmra"]),
                            pmdec_gaia=float(astro["pmdec"]),
                            pmra_err_gaia=float(astro.get("pmra_error") or 0.1),
                            pmdec_err_gaia=float(astro.get("pmdec_error") or 0.1),
                            pmra_wise=float(ir["pmra_wise"]),
                            pmdec_wise=float(ir["pmdec_wise"]),
                            pmra_err_wise=float(ir.get("e_pmra_wise") or 5.0),
                            pmdec_err_wise=float(ir.get("e_pmdec_wise") or 5.0),
                            phot_g_mean_mag=float(astro["phot_g_mean_mag"])
                            if astro.get("phot_g_mean_mag") is not None else None,
                        )
                        pm_anomaly["wise_gaia_pm"] = pm_check
                        if pm_check.get("catwise_systematic_flag"):
                            log.info("  CATWISE SYSTEMATIC for %s: "
                                     "PM offset suppressed (floor=%.1f mas/yr)",
                                     tid, pm_check.get("wise_sys_floor", 0))
                        elif pm_check["is_discrepant"]:
                            log.info("  PM discrepancy for %s: %.1f sigma",
                                     tid, pm_check["pm_discrepancy_sigma"])
                    except Exception as exc:
                        log.debug("PM consistency failed for %s: %s", tid, exc)

                t["proper_motion_anomaly"] = pm_anomaly

        self.anomalies_found += anomaly_count
        log.info("Processing complete. %d new anomalies flagged.", anomaly_count)
        return targets_data

    # ── Step 3a: Multi-messenger cross-matching ─────────────────────

    def _run_multi_messenger(
        self, targets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run multi-messenger cross-match modules against the target list.

        Annotates each target dict with any cross-match hits.
        """
        log.info("=== Running multi-messenger cross-matching ===")
        summary: Dict[str, Any] = {}

        # Build exoplanet_hosts list in the dict format crossmatch modules expect
        exoplanet_hosts = []
        for t in targets:
            exoplanet_hosts.append({
                "host_star": t.get("host_star", t.get("target_id", "")),
                "hostname": t.get("host_star", t.get("target_id", "")),
                "name": t.get("target_id", ""),
                "ra": t.get("ra", 0.0),
                "dec": t.get("dec", 0.0),
                "distance_pc": t.get("distance_pc"),
            })

        # Gamma-ray (Fermi 4FGL)
        if crossmatch_fermi_exoplanets is not None:
            try:
                from src.ingestion.fermi_catalog import get_unidentified
                fermi_raw = get_unidentified()
                fermi = [s.to_dict() for s in fermi_raw]
                result = crossmatch_fermi_exoplanets(fermi, exoplanet_hosts)
                summary["gamma"] = {
                    "n_matches": result.n_matches,
                    "n_escalations": result.n_escalations,
                    "data_source": "simulation" if (fermi and fermi[0].get("simulated")) else "real",
                }
                gamma_ds = summary["gamma"]["data_source"]
                for m in result.matches:
                    if not m.escalation:
                        continue  # Only attach significance-qualified matches
                    host = m.host_name
                    for t in targets:
                        if t.get("target_id") == host or t.get("host_star") == host:
                            mm_dict = m.to_dict()
                            mm_dict["data_source"] = gamma_ds  # F-03: per-hit provenance
                            t.setdefault("multi_messenger", {})["gamma"] = mm_dict
                log.info("  Gamma-ray: %d matches (%d escalations)",
                         result.n_matches, result.n_escalations)
            except Exception as exc:
                log.warning("  Gamma-ray failed: %s", exc)

        # Neutrino (IceCube)
        if crossmatch_neutrino_exoplanets is not None:
            try:
                from src.ingestion.icecube_catalog import get_all_events
                nu_raw = get_all_events()
                neutrinos = [e.to_dict() for e in nu_raw]
                result = crossmatch_neutrino_exoplanets(neutrinos, exoplanet_hosts)
                summary["neutrino"] = {
                    "n_significant": result.n_significant,
                    "n_hosts_with_excess": len(result.hosts_with_excess),
                    "data_source": "simulation" if (neutrinos and neutrinos[0].get("source") == "simulated") else "real",
                }
                nu_ds = summary["neutrino"]["data_source"]
                for host_excess in result.hosts_with_excess:
                    if host_excess.p_corrected is not None and host_excess.p_corrected >= 0.0027:
                        continue  # Only attach Bonferroni-corrected significance
                    elif host_excess.p_corrected is None and host_excess.poisson_sigma < 3.0:
                        continue  # Fallback: raw sigma check
                    host = host_excess.host_name
                    for t in targets:
                        if t.get("target_id") == host or t.get("host_star") == host:
                            mm_dict = host_excess.to_dict()
                            mm_dict["data_source"] = nu_ds  # F-03: per-hit provenance
                            t.setdefault("multi_messenger", {})["neutrino"] = mm_dict
                log.info("  Neutrino: %d significant hosts", result.n_significant)
            except Exception as exc:
                log.warning("  Neutrino failed: %s", exc)

        # Gravitational waves
        if crossmatch_gw_exoplanets is not None:
            try:
                from src.ingestion.gw_events import get_all_events as get_gw_events
                gw_raw = get_gw_events()
                gw_events = [e.to_dict() for e in gw_raw]
                result = crossmatch_gw_exoplanets(gw_events, exoplanet_hosts)
                summary["gw"] = {
                    "n_coincidences": result.n_coincidences,
                    "data_source": "simulation" if (gw_events and gw_events[0].get("source") == "simulated") else "real",
                }
                gw_ds = summary["gw"]["data_source"]
                for c in result.coincidences:
                    if not c.is_low_fa:
                        continue  # Only attach low-false-alarm coincidences
                    host = c.host_name
                    for t in targets:
                        if t.get("target_id") == host or t.get("host_star") == host:
                            mm_dict = c.to_dict()
                            mm_dict["data_source"] = gw_ds  # F-03: per-hit provenance
                            t.setdefault("multi_messenger", {})["gw"] = mm_dict
                log.info("  GW: %d coincidences (%d low-FA)",
                         result.n_coincidences, len(result.low_fa_coincidences))
            except Exception as exc:
                log.warning("  GW failed: %s", exc)

        # Pulsar Shapiro delay
        if search_pulsar_los is not None:
            try:
                from src.ingestion.nanograv import get_all_pulsars
                pulsar_raw = get_all_pulsars()
                pulsars = [p.to_dict() for p in pulsar_raw]
                result = search_pulsar_los(pulsars, exoplanet_hosts)
                summary["pulsar"] = {
                    "n_candidates": result.n_candidates,
                    "n_los_matches": result.n_los_matches_total,
                    "data_source": "simulation" if (pulsars and pulsars[0].get("source") == "simulated") else "real",
                }
                pulsar_ds = summary["pulsar"]["data_source"]
                for pr in result.candidates:  # Only attach flagged candidates
                    for los_match in pr.los_matches:
                        host = los_match.host_star
                        for t in targets:
                            if t.get("target_id") == host or t.get("host_star") == host:
                                mm_dict = pr.to_dict()
                                mm_dict["data_source"] = pulsar_ds  # F-03: per-hit provenance
                                t.setdefault("multi_messenger", {})["pulsar"] = mm_dict
                log.info("  Pulsar: %d candidates, %d LOS matches",
                         result.n_candidates, result.n_los_matches_total)
            except Exception as exc:
                log.warning("  Pulsar failed: %s", exc)

        # FRB orbital correlation
        if correlate_frb_orbits is not None:
            try:
                from src.ingestion.frb_catalog import get_repeaters
                repeaters = get_repeaters()
                frb_result = correlate_frb_orbits(repeaters, exoplanet_hosts)
                summary["frb"] = {
                    "n_repeaters_tested": frb_result.n_repeaters_tested,
                    "n_spatial_matches": frb_result.n_spatial_matches,
                    "n_period_matches": frb_result.n_period_matches,
                    "significance_sigma": frb_result.significance_sigma,
                    "data_source": "real",  # FRB catalog is always real (cached CHIME/FRB data)
                }
                frb_ds = summary["frb"]["data_source"]
                for pm in frb_result.matches:
                    host = pm.host_name
                    for t in targets:
                        if t.get("target_id") == host or t.get("host_star") == host:
                            mm_dict = pm.__dict__.copy() if hasattr(pm, '__dict__') else {}
                            mm_dict["data_source"] = frb_ds  # F-03: per-hit provenance
                            t.setdefault("multi_messenger", {})["frb"] = mm_dict
                log.info("  FRB: %d spatial, %d period matches (%.1fσ)",
                         frb_result.n_spatial_matches, frb_result.n_period_matches,
                         frb_result.significance_sigma)
            except Exception as exc:
                log.warning("  FRB correlation failed: %s", exc)

        # Temporal Archaeology (NVSS vs VLASS radio survey comparison)
        if TemporalArchaeology is not None:
            try:
                ta = TemporalArchaeology()
                ta_targets = [{"ra": t["ra"], "dec": t["dec"],
                               "name": t.get("target_id", t.get("host_star", ""))}
                              for t in targets]
                ta_result = ta.scan_target_list(
                    ta_targets, search_radius_deg=0.05,
                    exoplanet_hosts=exoplanet_hosts,
                )
                summary["temporal_archaeology"] = {
                    "n_appeared": ta_result.n_appeared,
                    "n_disappeared": ta_result.n_disappeared,
                    "n_changed": ta_result.n_changed,
                    "n_high_priority": len(ta_result.high_priority),
                }
                for hp in ta_result.high_priority:
                    host = hp.exoplanet_host_name
                    for t in targets:
                        if t.get("target_id") == host or t.get("host_star") == host:
                            t.setdefault("multi_messenger", {})["temporal_radio"] = hp.to_dict()
                log.info("  Temporal Archaeology: %d appeared, %d disappeared, "
                         "%d changed, %d high-priority",
                         ta_result.n_appeared, ta_result.n_disappeared,
                         ta_result.n_changed, len(ta_result.high_priority))
            except Exception as exc:
                log.warning("  Temporal Archaeology failed: %s", exc)

        log.info("=== Multi-messenger complete: %d modules ran ===", len(summary))
        return summary

    # ── Step 3b: Matched control selection ──────────────────────────

    def _select_controls(
        self, targets: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Query field stars, select matched controls, score them.

        Returns control_scores dict for EXODUSScorer calibration.
        """
        import numpy as np
        control_scores: Dict[str, List[float]] = {}

        if select_matched_controls is None or gaia_cone_search is None:
            log.warning(
                "Matched controls unavailable (missing imports). "
                "Scoring will remain UNCALIBRATED."
            )
            return control_scores

        if compute_ir_excess is None:
            log.warning("IR excess unavailable; cannot score controls.")
            return control_scores

        log.info("=== Selecting matched controls for calibration ===")

        # Build field catalog from Gaia cone searches around targets
        ras = [t["ra"] for t in targets]
        decs = [t["dec"] for t in targets]
        spread_deg = max(float(np.ptp(ras)), float(np.ptp(decs)))

        catalog_dicts: List[Dict[str, Any]] = []
        if spread_deg < 5.0:
            median_ra = float(np.median(ras))
            median_dec = float(np.median(decs))
            field_df = gaia_cone_search(
                median_ra, median_dec, radius_arcsec=1800.0, top_n=500,
            )
            if field_df is not None and not field_df.empty:
                for _, row in field_df.iterrows():
                    catalog_dicts.append(row.to_dict())
        else:
            seen_source_ids = set()
            for t in targets:
                field_df = gaia_cone_search(
                    t["ra"], t["dec"], radius_arcsec=600.0, top_n=100,
                )
                if field_df is not None and not field_df.empty:
                    for _, row in field_df.iterrows():
                        sid = row.get("source_id")
                        if sid not in seen_source_ids:
                            seen_source_ids.add(sid)
                            catalog_dicts.append(row.to_dict())

        if not catalog_dicts:
            log.warning("No field stars found for control selection.")
            return control_scores

        log.info("  Field catalog: %d stars", len(catalog_dicts))

        # Add distance_pc and b_gal for matching
        for d in catalog_dicts:
            plx = d.get("parallax")
            if plx is not None and plx > 0:
                d["distance_pc"] = 1000.0 / plx
            else:
                d["distance_pc"] = None
            try:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                c = SkyCoord(ra=d["ra"] * u.deg, dec=d["dec"] * u.deg)
                d["b_gal"] = float(c.galactic.b.deg)
            except Exception:
                d["b_gal"] = 0.0

        # Prepare target matching dicts
        target_match_dicts = []
        for t in targets:
            td = {"target_id": t["target_id"]}
            gp = t.get("gaia_params", {})
            ga = t.get("gaia_astrometry", {})
            td["phot_g_mean_mag"] = gp.get("phot_g_mean_mag")
            td["bp_rp"] = gp.get("bp_rp")
            td["distance_pc"] = t.get("distance_pc")
            # Carry source_id for control-pool exclusion
            td["source_id"] = ga.get("source_id") or gp.get("source_id")
            td["ra"] = t.get("ra", 0.0)
            td["dec"] = t.get("dec", 0.0)
            try:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                c = SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg)
                td["b_gal"] = float(c.galactic.b.deg)
            except Exception:
                td["b_gal"] = 0.0
            target_match_dicts.append(td)

        # Select matched controls
        try:
            cohort = select_matched_controls(
                target_match_dicts, catalog_dicts,
                n_per_target=10,
                match_on=["phot_g_mean_mag", "bp_rp", "distance_pc", "b_gal"],
                target_id_key="source_id",
            )
            self._matching_caveats = cohort.matching_caveats
            log.info("  %s", cohort.summary())
        except Exception as exc:
            log.warning("Control selection failed: %s", exc)
            return control_scores

        if cohort.n_controls == 0:
            log.warning("No controls selected.")
            return control_scores

        # F-04 fix: calibrate all 6 channels (matching run_quick.py)
        log.info("  Scoring %d control stars (6-channel calibration) ...", cohort.n_controls)
        ir_scores: List[float] = []
        astro_scores: List[float] = []
        hr_scores: List[float] = []
        uv_scores: List[float] = []
        radio_scores: List[float] = []
        irvar_scores: List[float] = []

        # Lazy imports for additional channels
        try:
            from src.detection.hr_anomaly import compute_hr_anomaly
        except ImportError:
            compute_hr_anomaly = None
        try:
            from src.detection.uv_anomaly import compute_uv_anomaly, compute_uv_metrics
            from src.ingestion.galex_catalog import query_galex_cone
        except ImportError:
            compute_uv_anomaly = query_galex_cone = compute_uv_metrics = None
        try:
            from src.detection.radio_emission import compute_radio_emission
            from src.ingestion.vlass_catalog import query_radio_continuum
        except ImportError:
            compute_radio_emission = query_radio_continuum = None

        for ctrl in cohort.controls:
            ctrl_ra, ctrl_dec = ctrl.get("ra"), ctrl.get("dec")
            if ctrl_ra is None or ctrl_dec is None:
                continue

            # ── IR excess ──
            try:
                ir_data = get_ir_photometry(ctrl_ra, ctrl_dec, radius_arcsec=5.0)
                # Add Gaia bands
                for gaia_key, band, err_band in [
                    ("phot_g_mean_mag", "G", "G_err"),
                    ("phot_bp_mean_mag", "BP", "BP_err"),
                    ("phot_rp_mean_mag", "RP", "RP_err"),
                ]:
                    val = ctrl.get(gaia_key)
                    if val is not None and np.isfinite(val):
                        ir_data[band] = float(val)
                        ir_data[err_band] = 0.01
                ir_result = compute_ir_excess(ir_data)
                ir_dict = {
                    "sigma_W3": ir_result.sigma_W3,
                    "sigma_W4": ir_result.sigma_W4,
                    "is_candidate": ir_result.is_candidate,
                    "excess_W3": ir_result.excess_W3,
                    "excess_W4": ir_result.excess_W4,
                }
                ir_scores.append(EXODUSScorer._get_ir_excess_score(ir_dict))
            except Exception as exc:
                log.debug("IR excess control failed: %s", exc)

            # ── Proper motion anomaly ──
            ruwe = ctrl.get("ruwe", 1.0)
            if ruwe is not None and np.isfinite(ruwe):
                astro_scores.append(EXODUSScorer._get_astrometric_score({
                    "ruwe": float(ruwe),
                    "astrometric_excess_noise_sig": 0.0,
                }))

            # ── HR anomaly (Gaia data only — no network query needed) ──
            if compute_hr_anomaly is not None:
                try:
                    ctrl_gaia_params = {
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
                    ctrl_plx = ctrl.get("parallax")
                    ctrl_dist = (
                        1000.0 / ctrl_plx
                        if ctrl_plx and ctrl_plx > 0 else None
                    )
                    hr_res = compute_hr_anomaly(
                        gaia_params=ctrl_gaia_params,
                        astrometry=ctrl_astro,
                        distance_pc=ctrl_dist,
                    )
                    hr_score = EXODUSScorer._get_hr_anomaly_score(
                        hr_res.to_dict()
                    )
                    hr_scores.append(hr_score)
                except Exception as exc:
                    log.debug("HR anomaly control failed: %s", exc)

            # ── UV anomaly (GALEX VizieR query — 10s timeout) ──
            if (compute_uv_anomaly is not None
                    and query_galex_cone is not None
                    and compute_uv_metrics is not None):
                try:
                    from concurrent.futures import ThreadPoolExecutor as _TPE
                    with _TPE(max_workers=1) as _pool:
                        _fut = _pool.submit(
                            query_galex_cone, ctrl_ra, ctrl_dec, 30.0,
                        )
                        galex_raw = _fut.result(timeout=10)
                    if galex_raw:
                        ctrl_gaia_p = {
                            "teff_gspphot": ctrl.get("teff_gspphot"),
                            "bp_rp": ctrl.get("bp_rp"),
                            "phot_g_mean_mag": ctrl.get("phot_g_mean_mag"),
                        }
                        uv_met = compute_uv_metrics(
                            galex_raw, gaia_params=ctrl_gaia_p,
                        )
                        uv_res = compute_uv_anomaly(
                            uv_metrics=uv_met,
                            galex_raw=galex_raw,
                            ir_excess_data=None,
                        )
                        uv_score = EXODUSScorer._get_uv_anomaly_score(
                            uv_res.to_dict()
                        )
                        uv_scores.append(uv_score)
                except Exception as exc:
                    log.debug("UV anomaly control failed: %s", exc)

            # ── Radio emission (FIRST/NVSS/VLASS query — 10s timeout) ──
            if (compute_radio_emission is not None
                    and query_radio_continuum is not None):
                try:
                    from concurrent.futures import ThreadPoolExecutor as _TPE
                    with _TPE(max_workers=1) as _pool:
                        _fut = _pool.submit(
                            query_radio_continuum, ctrl_ra, ctrl_dec, 15.0,
                        )
                        radio_raw = _fut.result(timeout=10)
                    if radio_raw:
                        ctrl_plx = ctrl.get("parallax")
                        ctrl_dist = (
                            1000.0 / ctrl_plx
                            if ctrl_plx and ctrl_plx > 0 else None
                        )
                        re_res = compute_radio_emission(
                            radio_continuum=radio_raw,
                            distance_pc=ctrl_dist,
                        )
                        re_score = EXODUSScorer._get_radio_emission_score(
                            re_res.to_dict()
                        )
                    else:
                        # No radio source at position — valid null observation
                        re_score = 0.0
                    radio_scores.append(re_score)
                except Exception as exc:
                    log.debug("Radio emission control failed: %s", exc)

        # Build control_scores dict
        if ir_scores:
            control_scores["ir_excess"] = ir_scores
            log.info("  IR controls: %d, median=%.4f",
                     len(ir_scores), float(np.median(ir_scores)))
        if astro_scores:
            control_scores["proper_motion_anomaly"] = astro_scores
            log.info("  PM controls: %d, median=%.4f",
                     len(astro_scores), float(np.median(astro_scores)))
        if hr_scores:
            control_scores["hr_anomaly"] = hr_scores
            log.info("  HR controls: %d, median=%.4f",
                     len(hr_scores), float(np.median(hr_scores)))
        if uv_scores:
            control_scores["uv_anomaly"] = uv_scores
            log.info("  UV controls: %d, median=%.4f",
                     len(uv_scores), float(np.median(uv_scores)))
        if radio_scores:
            control_scores["radio_emission"] = radio_scores
            log.info("  Radio controls: %d, median=%.4f",
                     len(radio_scores), float(np.median(radio_scores)))
        # Note: IR variability not calibrated in exodus runner (requires
        # NEOWISE time-series queries averaging ~90s/star — impractical for
        # the research-loop runner). See run_quick.py for full calibration.

        log.info("=== Control calibration: %d channels ===", len(control_scores))
        return control_scores

    # ── Step 4: Score targets ────────────────────────────────────────

    def _score_targets(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score and rank all targets using score_all() for FDR/ranking.

        Uses :meth:`EXODUSScorer.score_all` instead of per-target
        ``score_target()`` so that:
        - Targets are ranked globally (``rank`` field populated).
        - FDR correction (Benjamini-Hochberg) is applied across all targets.
        - ``get_top_targets()`` is available downstream.

        Returns scored target dicts (with ``exodus_score`` key added).
        """
        if self.scorer is None:
            log.warning("EXODUSScorer not available; skipping scoring.")
            return results

        log.info("Scoring %d targets with score_all() (FDR + ranking) ...", len(results))

        try:
            scored_results = self.scorer.score_all(results)
            # Attach scored results back to target dicts
            for exodus_score in scored_results:
                for t in results:
                    if str(t.get("target_id", "")) == exodus_score.target_id:
                        t["exodus_score"] = exodus_score.to_dict()
                        break
        except Exception as exc:
            log.warning("score_all() failed, falling back to per-target: %s", exc)
            for t in results:
                try:
                    score = self.scorer.score_target(t)
                    t["exodus_score"] = score.to_dict()
                except Exception as exc2:
                    log.debug(
                        "Scoring failed for %s: %s", t.get("target_id"), exc2
                    )

        # Get top targets (now works because score_all populates _results)
        try:
            all_results = self.scorer.get_top_targets(
                n=len(self.scorer._results)
            )
            self.all_scores = [s.to_dict() for s in all_results]
            top = all_results[:20]
            log.info(
                "Top target: %s (score=%.4f, channels=%d)",
                top[0].target_id if top else "N/A",
                top[0].total_score if top else 0.0,
                top[0].n_active_channels if top else 0,
            )
        except RuntimeError:
            log.warning("get_top_targets() failed: score_all() was not called.")
        except Exception:
            pass

        return results

    # ── Step 4a: Unexplainability evaluation ────────────────────────

    def _evaluate_unexplainability(
        self, targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate Unexplainability Score for targets with active channels.

        Runs astrophysical template matching against each scored target
        to classify anomalies as EXPLAINED, PARTIALLY_EXPLAINED, or
        UNEXPLAINED.  The full runner always uses min_channels=1.
        """
        if UnexplainabilityScorer is None:
            log.debug("UnexplainabilityScorer not available; skipping.")
            return targets

        min_channels = 1
        scorer = UnexplainabilityScorer()
        n_evaluated = 0

        for t in targets:
            exodus = t.get("exodus_score")
            if not exodus:
                continue

            n_active = exodus.get("n_active_channels", 0)
            if n_active < min_channels:
                continue

            try:
                channel_scores = {}
                channel_details = {}
                for ch_name, ch_data in exodus.get("channel_scores", {}).items():
                    if isinstance(ch_data, dict):
                        channel_scores[ch_name] = ch_data.get("score", 0.0)
                        channel_details[ch_name] = ch_data.get("details", {})

                result = scorer.evaluate(
                    t.get("target_id", "unknown"),
                    channel_scores,
                    channel_details,
                )
                t["unexplainability"] = result.to_dict()
                n_evaluated += 1
            except Exception as exc:
                log.debug("Unexplainability evaluation failed for %s: %s",
                          t.get("target_id"), exc)

        if n_evaluated > 0:
            log.info("Unexplainability evaluation: %d targets assessed", n_evaluated)
            for t in targets:
                unex = t.get("unexplainability", {})
                if unex.get("classification") == "UNEXPLAINED":
                    log.info(
                        "  *** UNEXPLAINED: %s — score=%.3f, "
                        "residual=%s, best_template=%s (fit=%.3f)",
                        t.get("target_id"),
                        unex.get("unexplainability_score", 0),
                        unex.get("residual_channels", []),
                        unex.get("best_template", "?"),
                        unex.get("best_template_fit", 0),
                    )

        return targets

    # ── Step 4b: Red-Team falsification ──────────────────────────────

    def _run_red_team(
        self, targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run Red-Team falsification engine on scored targets.

        Adds ``red_team`` dict to each target that has >= 1 active channel.
        The Red-Team aggressively seeks natural explanations — works entirely
        from existing pipeline data (no new network queries).
        """
        if RedTeamEngine is None:
            return targets

        engine = RedTeamEngine()
        n_evaluated = 0
        n_demote = 0
        n_escalate = 0

        for t in targets:
            try:
                score = t.get("exodus_score", {})
                n_active = score.get("n_active_channels", 0)
                if n_active < 1:
                    continue

                verdict = engine.evaluate(t)
                t["red_team"] = verdict.to_dict()
                n_evaluated += 1

                if verdict.recommendation == "DEMOTE":
                    n_demote += 1
                elif verdict.recommendation == "ESCALATE":
                    n_escalate += 1

            except Exception as exc:
                log.debug(
                    "Red-team evaluation failed for %s: %s",
                    t.get("target_id"), exc,
                )

        if n_evaluated > 0:
            log.info(
                "Red-team falsification: %d targets assessed — "
                "%d ESCALATE, %d DEMOTE, %d MONITOR",
                n_evaluated, n_escalate, n_demote,
                n_evaluated - n_escalate - n_demote,
            )

            for t in targets:
                rt = t.get("red_team", {})
                if rt.get("recommendation") == "DEMOTE":
                    log.warning(
                        "  RED-TEAM DEMOTE: %s — risk=%.2f (%s), "
                        "top_concern: %s",
                        t.get("target_id"),
                        rt.get("overall_risk", 0),
                        rt.get("risk_level", "?"),
                        rt.get("top_risk", "none"),
                    )

        return targets

    # ── Steps 5-7: Hypothesis cycle ──────────────────────────────────

    def _run_hypothesis_cycle(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run the hypothesis generation -> validation -> escalation pipeline.

        1. Pull pending hypotheses from the Generator.
        2. Validate each with the Analyst.
        3. Escalate UNEXPLAINED results to the Breakthrough Engine.
        4. Generate follow-up hypotheses from interesting results.

        Returns a summary dict of hypothesis cycle results.
        """
        cycle_summary: Dict[str, Any] = {
            "tested": 0,
            "confirmed_natural": 0,
            "confirmed_artifact": 0,
            "unexplained": 0,
            "inconclusive": 0,
            "escalated": 0,
            "followups_generated": 0,
        }

        if self.generator is None:
            log.warning("HypothesisGenerator not available; skipping hypothesis cycle.")
            return cycle_summary

        # Inject creative strategies on first iteration
        if self.iteration == 0:
            try:
                new_strategies = self.generator.inject_creative_strategies()
                log.info("Injected %d creative strategies.", len(new_strategies))
            except Exception:
                pass

        # Get pending hypotheses
        try:
            pending = self.generator.get_pending()
        except Exception:
            log.error("Failed to get pending hypotheses.")
            pending = []

        if not pending:
            log.info("No pending hypotheses to test.")
            # Try generating follow-ups from current results
            self._generate_followups_from_results(results, cycle_summary)
            return cycle_summary

        log.info("Testing %d pending hypotheses ...", len(pending))

        # Build a map from method -> relevant data for validation
        # This aggregates results across all targets for each test method
        aggregated_data = self._aggregate_data_for_validation(results)

        for hyp in pending:
            hyp_id = hyp["hypothesis_id"]
            method = hyp.get("method", "")

            # Build the hypothesis dict the Analyst expects
            analyst_hyp = {
                "hypothesis_id": hyp_id,
                "test_method": method,
                "test_params": {},
            }

            # Get relevant data for this method
            data = aggregated_data.get(method, {})
            if not data:
                log.debug(
                    "No relevant data for hypothesis %s (method=%s).",
                    hyp_id, method,
                )
                continue

            # Validate
            if self.analyst is not None:
                try:
                    vresult = self.analyst.validate(analyst_hyp, data)
                    cycle_summary["tested"] += 1

                    # Map ValidationStatus to string
                    status_str = vresult.status.value if hasattr(vresult.status, "value") else str(vresult.status)

                    # Track results
                    hyp_tested = {
                        "id": hyp_id,
                        "text": hyp.get("claim", ""),
                        "method": method,
                        "status": self._map_validation_status(vresult.status),
                        "scores": {
                            "detection": vresult.detection_score,
                            "natural_explanation": vresult.natural_explanation_score,
                            "instrumental": vresult.instrumental_score,
                            "significance": vresult.statistical_significance,
                        },
                        "properties": {},
                    }
                    self.hypotheses_tested.append(hyp_tested)

                    # Update generator status
                    gen_status = "tested"
                    if vresult.status == ValidationStatus.UNEXPLAINED:
                        gen_status = "confirmed"  # promote for follow-up
                        cycle_summary["unexplained"] += 1
                    elif vresult.status == ValidationStatus.CONFIRMED_NATURAL:
                        gen_status = "rejected"
                        cycle_summary["confirmed_natural"] += 1
                        self.false_positives += 1
                    elif vresult.status == ValidationStatus.CONFIRMED_ARTIFACT:
                        gen_status = "rejected"
                        cycle_summary["confirmed_artifact"] += 1
                        self.false_positives += 1
                    elif vresult.status == ValidationStatus.INCONCLUSIVE:
                        cycle_summary["inconclusive"] += 1

                    self.generator.update_status(
                        hyp_id, gen_status, results=vresult.to_dict()
                    )

                    log.info(
                        "  Hypothesis %s -> %s (detection=%.3f)",
                        hyp_id, status_str, vresult.detection_score,
                    )

                    # Escalate UNEXPLAINED to Breakthrough Engine --
                    # BUT only if a specific target has FDR-qualified evidence.
                    # Per Core Directive (01_CORE_DIRECTIVES.md line 11):
                    # "No candidate escalates to breakthrough without
                    #  FDR-corrected threshold being met."
                    #
                    # The escalation threshold is p < 0.01 (stricter than
                    # the general FDR alpha=0.05) because breakthrough
                    # escalation triggers expensive verification cascades.
                    if (
                        vresult.status == ValidationStatus.UNEXPLAINED
                        and self.breakthrough is not None
                    ):
                        # Find the BEST target using CONSERVATIVE Stouffer p-value.
                        # Per Core Directive §4 (01_CORE_DIRECTIVES.md line 11):
                        # "No candidate advances to Breakthrough Level 3+
                        #  without FDR-corrected p < 0.01"
                        #
                        # We gate on stouffer_p < 0.01 (conservative, all-channel
                        # combination including p=1.0 for inactive channels) rather
                        # than Fisher-derived q_value which is anti-conservative
                        # due to post-activation-selection (Audit #5 fix).
                        # If stouffer_p is unavailable, fall back to q_value.
                        best_target = None
                        best_p = 1.0
                        for t in results:
                            es = t.get("exodus_score", {})
                            # Prefer stouffer_p (conservative) over q_value (post-selection)
                            sp = es.get("stouffer_p")
                            qv = es.get("q_value")
                            gate_p = sp if sp is not None else qv
                            if gate_p is not None and gate_p < 0.01:
                                if gate_p < best_p:
                                    best_p = gate_p
                                    best_target = t
                                    # Log when Fisher and Stouffer disagree
                                    if sp is not None and qv is not None:
                                        if qv < 0.01 and sp >= 0.01:
                                            log.warning(
                                                "  %s: Fisher q=%.4f < 0.01 but Stouffer p=%.4f >= 0.01 "
                                                "(post-selection bias suspected)",
                                                t.get("target_id"), qv, sp,
                                            )

                        if best_target is not None:
                            self._escalate_to_breakthrough(
                                vresult, results, cycle_summary,
                                escalation_target=best_target,
                            )
                        else:
                            # Check if p-values are even available
                            has_any_q = any(
                                (t.get("exodus_score", {}).get("stouffer_p") is not None
                                 or t.get("exodus_score", {}).get("q_value") is not None)
                                for t in results
                            )
                            if has_any_q:
                                log.info(
                                    "  Hypothesis %s is UNEXPLAINED but no target "
                                    "meets escalation threshold (stouffer_p < 0.01) "
                                    "-- escalation blocked per Core Directive §4.",
                                    hyp_id,
                                )
                            else:
                                log.warning(
                                    "  Hypothesis %s is UNEXPLAINED but escalation "
                                    "is IMPOSSIBLE: no FDR q-values available "
                                    "(matched controls not wired into scorer). "
                                    "Wire control_scores to enable breakthrough path.",
                                    hyp_id,
                                )

                except Exception as exc:
                    log.warning(
                        "Validation failed for hypothesis %s: %s", hyp_id, exc
                    )

        # Generate follow-ups
        self._generate_followups_from_results(results, cycle_summary)

        return cycle_summary

    def _aggregate_data_for_validation(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate target results into method-appropriate data dicts.

        The Analyst's validate methods expect specific data keys depending
        on the test method.  This builds those data bundles from the
        per-target processing results.
        """
        data_by_method: Dict[str, Dict[str, Any]] = {}

        # --- IR excess comparison ---
        target_excess = []
        control_excess = []
        for t in results:
            ir = t.get("ir_excess", {})
            if ir.get("sigma_W4") is not None:
                if ir.get("is_candidate"):
                    target_excess.append(float(ir["sigma_W4"]))
                else:
                    control_excess.append(float(ir["sigma_W4"]))
        if target_excess or control_excess:
            data_by_method["ir_excess_comparison"] = {
                "target_excess": target_excess or control_excess[:5],
                "control_excess": control_excess or [0.0] * 5,
            }

        # --- Light curve anomaly ---
        # Use the first target with a light curve
        for t in results:
            lc = t.get("lightcurve")
            if lc and len(lc.get("time", [])) >= 20:
                data_by_method["anomaly_detection_lightcurve"] = {
                    "time": lc["time"],
                    "flux": lc["flux"],
                    "flux_err": lc.get("flux_err"),
                    "n_sectors": 1,
                }
                break

        # --- Radio search ---
        for t in results:
            radio = t.get("radio_anomaly", {})
            if radio.get("candidates"):
                data_by_method["radio_search_targeted"] = {
                    "candidates": radio["candidates"],
                    "noise_floor": radio.get("noise_floor", 1.0),
                    "n_observations": 1,
                }
                break

        # --- Transit-IR correlation ---
        transit_depths = []
        ir_sigmas = []
        for t in results:
            ta = t.get("transit_anomaly", {})
            ir = t.get("ir_excess", {})
            if ta.get("depth") is not None and ir.get("sigma_W4") is not None:
                transit_depths.append(float(ta["depth"]))
                ir_sigmas.append(float(ir["sigma_W4"]))
        if len(transit_depths) >= 3:
            data_by_method["single_transit_ir_correlation"] = {
                "transit_depths": transit_depths,
                "ir_excess_sigmas": ir_sigmas,
                "control_ir": [0.0] * len(ir_sigmas),
                "n_matches": len(transit_depths),
            }

        return data_by_method

    def _escalate_to_breakthrough(
        self,
        vresult: Any,
        results: List[Dict[str, Any]],
        cycle_summary: Dict[str, Any],
        escalation_target: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Escalate an UNEXPLAINED validation result through the Breakthrough Engine.

        Parameters
        ----------
        escalation_target : dict, optional
            The specific FDR-qualified target to escalate.  If None, falls
            back to the first scored target (legacy behavior).
        """
        try:
            # Use the specific FDR-qualified target if provided.
            # Pass the full target dict so breakthrough levels have access
            # to photometry, Gaia params, IR excess results, etc.
            if escalation_target is not None:
                target_data = dict(escalation_target)
                target_data.setdefault("source_id", target_data.get("target_id", "unknown"))
            else:
                target_data = {"ra": 0.0, "dec": 0.0, "source_id": "unknown"}
                for t in results:
                    if t.get("exodus_score"):
                        target_data = dict(t)
                        target_data.setdefault("source_id", t.get("target_id", "unknown"))
                        break

            validation_dict = {
                "anomaly_type": "multi_channel",
                "confidence": vresult.detection_score,
                "details": vresult.explanation,
            }

            candidate = self.breakthrough.escalate(validation_dict, target_data)
            cycle_summary["escalated"] += 1

            log.info(
                "  -> Escalated to Breakthrough Engine: %s  level=%s  status=%s",
                candidate.candidate_id,
                candidate.current_level,
                candidate.status,
            )
        except Exception as exc:
            log.warning("Breakthrough escalation failed: %s", exc)

    def _generate_followups_from_results(
        self,
        results: List[Dict[str, Any]],
        cycle_summary: Dict[str, Any],
    ) -> None:
        """Generate follow-up hypotheses from interesting targets."""
        if self.generator is None:
            return

        # Find interesting targets (high EXODUS score or flagged anomalies)
        interesting = []
        for t in results:
            score = t.get("exodus_score", {})
            if isinstance(score, dict) and score.get("n_active_channels", 0) >= 2:
                interesting.append(t.get("target_id", "unknown"))

        if interesting:
            try:
                followup_results = {
                    "interesting_targets": interesting,
                    "n_interesting": len(interesting),
                }
                # Use the first tested hypothesis as parent, or a stats-based approach
                stats = self.generator.get_stats()
                all_hyps = self.generator.get_all()
                if all_hyps:
                    parent_id = all_hyps[0]["hypothesis_id"]
                    new_ids = self.generator.generate_followups(
                        parent_id, followup_results
                    )
                    cycle_summary["followups_generated"] += len(new_ids)
                    log.info("Generated %d follow-up hypotheses.", len(new_ids))
            except Exception as exc:
                log.debug("Follow-up generation failed: %s", exc)

    def _map_validation_status(self, status: Any) -> str:
        """Map a ValidationStatus enum to the Evolver's status constants."""
        if ValidationStatus is None:
            return "pending"

        mapping = {
            ValidationStatus.CONFIRMED_NATURAL: "natural",
            ValidationStatus.CONFIRMED_ARTIFACT: "artifact",
            ValidationStatus.UNEXPLAINED: "unexplained",
            ValidationStatus.INCONCLUSIVE: "pending",
        }
        return mapping.get(status, "pending")

    # ── Step 8: Evolve ───────────────────────────────────────────────

    def _evolve_iteration(
        self,
        results: List[Dict[str, Any]],
        hyp_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Feed the iteration state to the Evolver for self-improvement."""
        if self.evolver is None or ResearchState is None:
            log.warning("Evolver not available; skipping self-improvement.")
            return {}

        try:
            state = ResearchState(
                iteration=self.iteration,
                hypotheses_tested=self.hypotheses_tested,
                targets_processed=len(results),
                anomalies_found=self.anomalies_found,
                false_positives=self.false_positives,
                current_thresholds=dict(self.thresholds),
            )

            record = self.evolver.evolve(state)

            # Apply threshold changes
            for name, change in record.threshold_changes.items():
                new_val = change.get("new", self.thresholds.get(name))
                self.thresholds[name] = new_val
                log.info(
                    "  Threshold '%s': %.3f -> %.3f (%s)",
                    name, change.get("old", 0), new_val, change.get("reason", ""),
                )

            # Log recommendations
            for rec in record.recommendations:
                log.info("  Evolver recommends: %s", rec)

            return {
                "threshold_changes": record.threshold_changes,
                "new_hypotheses": record.new_hypotheses_generated,
                "strategies_promoted": record.strategies_promoted,
                "strategies_deprioritized": record.strategies_deprioritized,
                "false_positive_rate": record.false_positive_rate,
                "true_positive_rate": record.true_positive_rate,
                "recommendations": record.recommendations,
            }

        except Exception:
            log.error("Evolver failed:\n%s", traceback.format_exc())
            return {}

    # ── CX-7: Strip simulated MM data ──────────────────────────────

    @staticmethod
    def _strip_simulated_mm(summary: Dict[str, Any]) -> int:
        """Remove simulated multi-messenger entries from export data (CX-7).

        Returns the number of simulated entries stripped.
        """
        stripped = 0
        # Strip from top-level multi_messenger summary
        mm_top = summary.get("multi_messenger", {})
        for key in list(mm_top.keys()):
            if isinstance(mm_top[key], dict) and mm_top[key].get("data_source") == "simulation":
                del mm_top[key]
                stripped += 1
        # Strip from per-target multi_messenger data
        for list_key in ("top_targets", "all_scored"):
            for target in summary.get(list_key, []):
                mm = target.get("multi_messenger", {})
                for key in list(mm.keys()):
                    if isinstance(mm[key], dict) and mm[key].get("data_source") == "simulation":
                        del mm[key]
                        stripped += 1
        return stripped

    # ── Step 9: Generate report ──────────────────────────────────────

    def _generate_report(
        self, iteration: Any, summary: Dict[str, Any]
    ) -> Path:
        """Write a human-readable report for the iteration.

        Also saves the raw summary as JSON.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_name = f"iteration_{iteration}_{ts}"

        # CX-7: strip simulated MM data before saving publication-ready report
        n_stripped = self._strip_simulated_mm(summary)
        if n_stripped:
            log.info("Stripped %d simulated multi-messenger entries from report", n_stripped)

        # Save JSON
        json_path = self.REPORTS_DIR / f"{report_name}.json"
        with open(json_path, "w") as f:
            safe_json_dump(summary, f, indent=2)

        # Save human-readable text report
        txt_path = self.REPORTS_DIR / f"{report_name}.txt"
        lines = [
            "=" * 70,
            f"  Project EXODUS -- Iteration {iteration} Report",
            f"  Generated: {ts}",
            "=" * 70,
            "",
        ]

        if isinstance(summary.get("n_targets"), int):
            lines.append(f"Targets processed:    {summary['n_targets']}")
        if isinstance(summary.get("n_results"), int):
            lines.append(f"Results generated:    {summary['n_results']}")
        if isinstance(summary.get("elapsed_sec"), (int, float)):
            lines.append(f"Elapsed time:         {summary['elapsed_sec']:.1f} s")

        # Hypothesis cycle results
        hyp = summary.get("hypothesis_results", {})
        if hyp:
            lines.append("")
            lines.append("--- Hypothesis Cycle ---")
            lines.append(f"  Tested:             {hyp.get('tested', 0)}")
            lines.append(f"  Unexplained:        {hyp.get('unexplained', 0)}")
            lines.append(f"  Confirmed natural:  {hyp.get('confirmed_natural', 0)}")
            lines.append(f"  Confirmed artifact: {hyp.get('confirmed_artifact', 0)}")
            lines.append(f"  Inconclusive:       {hyp.get('inconclusive', 0)}")
            lines.append(f"  Escalated:          {hyp.get('escalated', 0)}")
            lines.append(f"  Follow-ups:         {hyp.get('followups_generated', 0)}")

        # Evolution
        evo = summary.get("evolution", {})
        if evo:
            lines.append("")
            lines.append("--- Evolver ---")
            if evo.get("false_positive_rate") is not None:
                lines.append(f"  FP rate:  {evo['false_positive_rate']:.3f}")
            if evo.get("true_positive_rate") is not None:
                lines.append(f"  TP rate:  {evo['true_positive_rate']:.3f}")
            for rec in evo.get("recommendations", []):
                lines.append(f"  Rec: {rec}")

        # Top targets (for final report)
        if self.all_scores:
            lines.append("")
            lines.append("--- Top EXODUS Targets ---")
            for i, s in enumerate(self.all_scores[:10]):
                lines.append(
                    f"  {i+1:>2}. {s.get('target_id', 'N/A'):>30s}  "
                    f"score={s.get('total_score', 0):.4f}  "
                    f"channels={s.get('n_active_channels', 0)}"
                )

        lines.append("")
        lines.append("=" * 70)

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        log.info("Report saved: %s", txt_path)
        return txt_path

    # ── State persistence ────────────────────────────────────────────

    def _save_state(self) -> None:
        """Checkpoint the full runner state to disk."""
        state = {
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "tier": self.tier,
            "quick": self.quick,
            "thresholds": self.thresholds,
            "anomalies_found": self.anomalies_found,
            "false_positives": self.false_positives,
            "all_scores": self.all_scores,
            "hypotheses_tested": self.hypotheses_tested,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.STATE_FILE, "w") as f:
            safe_json_dump(state, f, indent=2)

        log.info("State checkpointed to %s", self.STATE_FILE)

    def _load_state(self) -> None:
        """Restore runner state from the last checkpoint."""
        if not self.STATE_FILE.exists():
            log.info("No saved state found; starting fresh.")
            return

        try:
            with open(self.STATE_FILE) as f:
                state = json.load(f)

            self.iteration = state.get("iteration", 0) + 1  # resume NEXT iter
            self.thresholds = state.get("thresholds", self.thresholds)
            self.anomalies_found = state.get("anomalies_found", 0)
            self.false_positives = state.get("false_positives", 0)
            self.all_scores = state.get("all_scores", [])
            self.hypotheses_tested = state.get("hypotheses_tested", [])

            log.info(
                "Restored state from checkpoint.  Resuming at iteration %d.",
                self.iteration,
            )
        except Exception:
            log.error(
                "Failed to load state:\n%s", traceback.format_exc()
            )

    # ── Summary ──────────────────────────────────────────────────────

    def _build_run_summary(self, elapsed_sec: float) -> Dict[str, Any]:
        """Build the final run summary dict."""
        summary = {
            "project": "EXODUS",
            "completed_iterations": self.iteration + 1,
            "max_iterations": self.max_iterations,
            "tier": self.tier,
            "quick": self.quick,
            "elapsed_sec": round(elapsed_sec, 1),
            "elapsed_min": round(elapsed_sec / 60.0, 2),
            "thresholds": self.thresholds,
            "anomalies_found": self.anomalies_found,
            "false_positives": self.false_positives,
            "n_hypotheses_tested": len(self.hypotheses_tested),
            "top_targets": self.all_scores[:20],
            "all_scored": self.all_scores,
            "n_scored": len(self.all_scores),
            "channels_calibrated": list(getattr(self, "_control_scores", {}).keys()),
            "calibration_note": (
                "FDR significance is based only on calibrated channels; "
                "uncalibrated channels are excluded from Stouffer/Fisher combination"
            ) if getattr(self, "_control_scores", {}) else "No controls obtained — results are UNCALIBRATED",
            "matching_caveats": getattr(self, "_matching_caveats", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Multi-messenger summary (persisted from last iteration)
        if hasattr(self, "mm_summary") and self.mm_summary:
            summary["multi_messenger"] = self.mm_summary

        # Generator stats
        if self.generator:
            try:
                summary["generator_stats"] = self.generator.get_stats()
            except Exception:
                pass

        # Evolver recommendations
        if self.evolver:
            try:
                summary["evolver_recommendations"] = self.evolver.get_recommendations()
                summary["strategy_weights"] = self.evolver.get_strategy_weights()
            except Exception:
                pass

        # Breakthrough log
        if self.breakthrough:
            try:
                bt_log = self.breakthrough.get_log()
                summary["breakthrough_candidates"] = len(bt_log) if bt_log else 0
            except Exception:
                pass

        # Unexplainability summary
        if self.all_results:
            unexplained = []
            for t in self.all_results:
                unex = t.get("unexplainability")
                if unex:
                    unexplained.append({
                        "target_id": t.get("target_id"),
                        "classification": unex.get("classification"),
                        "unexplainability_score": unex.get("unexplainability_score"),
                        "best_template": unex.get("best_template"),
                        "best_template_fit": unex.get("best_template_fit"),
                        "residual_channels": unex.get("residual_channels"),
                    })
            if unexplained:
                summary["unexplained_targets"] = [
                    u for u in unexplained
                    if u["classification"] in ("UNEXPLAINED", "PARTIALLY_EXPLAINED")
                ]

        return summary


# =====================================================================
#  CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project EXODUS -- Full Research Loop Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_exodus.py                          # default: 10 iterations, tier1
  python scripts/run_exodus.py --max-iterations 5       # 5 iterations
  python scripts/run_exodus.py --tier tier2             # tier2 targets (up to 5000)
  python scripts/run_exodus.py --resume                 # resume from last checkpoint
  python scripts/run_exodus.py --quick                  # quick laptop run (20 targets, 1 iter)
        """,
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=10,
        help="Number of research loop iterations (default: 10).",
    )
    parser.add_argument(
        "--tier", "-t",
        type=str,
        default="tier1",
        choices=["tier1", "tier2"],
        help="Target selection tier (default: tier1).",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from the last saved checkpoint.",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: 20 targets, 1 iteration, IR + Gaia only.",
    )
    parser.add_argument(
        "--target-file",
        type=str,
        default=None,
        help="Path to a custom campaign target JSON file.  "
             "When provided, targets are loaded from this file "
             "instead of querying the NASA Exoplanet Archive.",
    )
    parser.add_argument(
        "--campaign-report",
        action="store_true",
        default=False,
        help="Generate enhanced campaign comparison report "
             "(expected vs observed channel behavior).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Quick mode overrides
    max_iter = args.max_iterations
    if args.quick:
        max_iter = 1

    runner = ExodusRunner(
        max_iterations=max_iter,
        tier=args.tier,
        quick=args.quick,
        resume=args.resume,
        target_file=args.target_file,
        campaign_report=args.campaign_report,
    )

    summary = runner.run()

    # Print top targets to stdout
    print("\n" + "=" * 70)
    print("  EXODUS RUN COMPLETE")
    print("=" * 70)
    print(f"  Iterations:     {summary.get('completed_iterations', 0)}")
    print(f"  Elapsed:        {summary.get('elapsed_min', 0):.1f} min")
    print(f"  Anomalies:      {summary.get('anomalies_found', 0)}")
    print(f"  Hypotheses:     {summary.get('n_hypotheses_tested', 0)}")

    top = summary.get("top_targets", [])
    if top:
        print(f"\n  Top {min(len(top), 10)} EXODUS Targets:")
        for i, t in enumerate(top[:10]):
            print(
                f"    {i+1:>2}. {t.get('target_id', 'N/A'):>30s}  "
                f"score={t.get('total_score', 0):.4f}  "
                f"channels={t.get('n_active_channels', 0)}"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()
