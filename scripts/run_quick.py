#!/usr/bin/env python3
"""
Project EXODUS -- Quick Research Run
=====================================

Minimal version of the full EXODUS pipeline, designed to complete in under
one hour on a standard laptop.  Useful for testing, demos, and quick scans.

Constraints
-----------
- Tier 1 targets only, capped at 20
- Single iteration (no evolution loop)
- 5/6 detection channels: IR excess, IR variability (NEOWISE), transit anomaly (BLS), radio (BL), Gaia photometry
- Lightcurves downsampled to 30k points for laptop-scale BLS
- Hypothesis cycle runs but with limited scope

This script delegates to :class:`ExodusRunner` from ``run_exodus.py`` with
the ``--quick`` flag, then adds a concise summary.

Usage
-----
    python scripts/run_quick.py
    python scripts/run_quick.py --targets 10        # override target count
    python scripts/run_quick.py --no-hypotheses     # skip hypothesis cycle
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Ensure project root is on sys.path ──────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Project utilities ────────────────────────────────────────────────
from src.utils import get_config, get_logger, save_result, safe_json_dump, PROJECT_ROOT as _PR

log = get_logger("quick_runner")

# ── Graceful imports ─────────────────────────────────────────────────
try:
    from src.ingestion.exoplanet_archive import get_hz_planets
except ImportError as e:
    log.warning("exoplanet_archive unavailable: %s", e)
    get_hz_planets = None

try:
    from src.ingestion.gaia_query import get_stellar_params, get_astrometry, get_epoch_photometry
except ImportError as e:
    log.warning("gaia_query unavailable: %s", e)
    get_stellar_params = get_astrometry = get_epoch_photometry = None

try:
    from src.ingestion.ir_surveys import get_ir_photometry
except ImportError as e:
    log.warning("ir_surveys unavailable: %s", e)
    get_ir_photometry = None

try:
    from src.processing.ir_excess import compute_ir_excess
except ImportError as e:
    log.warning("ir_excess processing unavailable: %s", e)
    compute_ir_excess = None

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
    from src.ingestion.gaia_query import batch_cone_search as gaia_batch_cone_search
except ImportError as e:
    log.warning("gaia cone_search unavailable: %s", e)
    gaia_cone_search = None
    gaia_batch_cone_search = None

try:
    from src.ingestion.lightcurves import get_lightcurve, stitch_lightcurves
except ImportError as e:
    log.warning("lightcurves unavailable: %s", e)
    get_lightcurve = stitch_lightcurves = None

try:
    from src.processing.transit_anomaly import detect_transit_anomaly, detect_irregular_dimming
except ImportError as e:
    log.warning("transit_anomaly unavailable: %s", e)
    detect_transit_anomaly = detect_irregular_dimming = None

try:
    from src.ingestion.breakthrough_listen import get_spectrogram as bl_get_spectrogram
except ImportError as e:
    log.warning("breakthrough_listen unavailable: %s", e)
    bl_get_spectrogram = None

try:
    from src.processing.radio_processor import process_spectrogram as radio_process_spectrogram
except ImportError as e:
    log.warning("radio_processor unavailable: %s", e)
    radio_process_spectrogram = None

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
    from src.ingestion.neowise_timeseries import query_neowise_timeseries
except ImportError:
    query_neowise_timeseries = None

try:
    from src.detection.ir_variability import compute_ir_variability
except ImportError:
    compute_ir_variability = None

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

try:
    from src.vetting.galaxy_contamination import check_galaxy_contamination, check_pm_ir_correlation
except ImportError:
    check_galaxy_contamination = None
    check_pm_ir_correlation = None

try:
    from src.ingestion.erosita_catalog import query_erosita_cone
except ImportError:
    query_erosita_cone = None

try:
    from src.ingestion.herschel_catalog import query_herschel, interpret_herschel_sed
except ImportError:
    query_herschel = None
    interpret_herschel_sed = None

try:
    from src.ingestion.sb9_catalog import query_sb9_cone
except ImportError:
    query_sb9_cone = None

try:
    from src.ingestion.simbad_catalog import query_simbad_cone
except ImportError:
    query_simbad_cone = None

try:
    from src.vetting.dust_extinction import get_extinction_context
except ImportError:
    get_extinction_context = None

try:
    from src.ingestion.galex_catalog import query_galex_cone, compute_uv_metrics
except ImportError:
    query_galex_cone = compute_uv_metrics = None

try:
    from src.ingestion.vlass_catalog import query_radio_continuum, is_radio_continuum_detected
except ImportError:
    query_radio_continuum = is_radio_continuum_detected = None

try:
    from src.detection.uv_anomaly import compute_uv_anomaly
except ImportError:
    compute_uv_anomaly = None

try:
    from src.detection.radio_emission import compute_radio_emission
except ImportError:
    compute_radio_emission = None

try:
    from src.detection.hr_anomaly import compute_hr_anomaly
except ImportError:
    compute_hr_anomaly = None

try:
    from src.detection.abundance_anomaly import compute_abundance_anomaly
except ImportError:
    compute_abundance_anomaly = None

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
    from src.core.provenance import provenance_logger
except ImportError:
    provenance_logger = None

try:
    from src.core.evidence import EvidenceBundle, save_evidence_bundle
except ImportError:
    EvidenceBundle = save_evidence_bundle = None


# =====================================================================
#  Quick Runner
# =====================================================================

class QuickRunner:
    """Lightweight single-pass runner for Project EXODUS.

    Supports tiered scanning for laptop-scale analysis:

    - **Tier 0** (~25s/target): IR excess + IR variability (NEOWISE) +
      astrometry + HZ prior.  Multi-messenger crossmatches (Fermi,
      IceCube, GW, Pulsar, FRB) run in batch.
    - **Tier 1** (~150s/target): Tier 0 + Gaia epoch photometry +
      lightcurve download + BLS transit anomaly. Standard depth.
    - **Tier 2** (full pipeline): Tier 1 + radio (BL) spectrograms.
      Deep dive for targets showing multi-channel convergence.

    Parameters
    ----------
    max_targets : int
        Maximum number of targets to process (default: 20).
    run_hypotheses : bool
        Whether to run the hypothesis generation/validation cycle
        (default: True).
    tier : int
        Scanning depth (0, 1, or 2). Default: 1.
    """

    REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

    CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"

    def __init__(
        self,
        max_targets: int = 20,
        run_hypotheses: bool = True,
        target_file: str = None,
        tier: int = 1,
        resume: bool = False,
        convergence_priority: bool = False,
    ):
        self.max_targets = max_targets
        self.run_hypotheses = run_hypotheses
        self.target_file = target_file
        self.tier = tier
        self.resume = resume
        self.convergence_priority = convergence_priority
        self.campaign_metadata = None
        self.cfg = get_config()

        self.scorer = (
            EXODUSScorer(convergence_priority=convergence_priority)
            if EXODUSScorer else None
        )
        self.generator = HypothesisGenerator() if (HypothesisGenerator and run_hypotheses) else None
        self.analyst = AnalystEngine() if (AnalystEngine and run_hypotheses) else None

        self.thresholds: Dict[str, float] = {
            "anomaly_sigma": self.cfg.get("search", {}).get("anomaly_sigma", 3.0),
        }

        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # --- Checkpoint infrastructure ---
        self._checkpoint_path = self._get_checkpoint_path()
        self._checkpoint_data: Dict[str, Any] = {}
        self._checkpoint_lock = threading.Lock()
        self._shutdown_requested = False
        if self.resume:
            self._load_checkpoint()

    # ── Checkpoint helpers ──────────────────────────────────────────

    def _get_checkpoint_path(self) -> Path:
        """Deterministic checkpoint path based on target file + tier."""
        import hashlib
        key = f"{self.target_file or 'default'}:tier{self.tier}:n{self.max_targets}"
        h = hashlib.sha256(key.encode()).hexdigest()[:12]
        path = self.CHECKPOINT_DIR / h
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_checkpoint(self) -> None:
        """Load checkpoint state from disk."""
        ckpt_file = self._checkpoint_path / "checkpoint.json"
        if ckpt_file.exists():
            with open(ckpt_file) as f:
                self._checkpoint_data = json.load(f)
            n_gathered = len(self._checkpoint_data.get("gathered_targets", {}))
            n_controls = len(self._checkpoint_data.get("control_scores", {}).get("ir_excess", []))
            phase = self._checkpoint_data.get("phase", "unknown")
            log.info(
                "Resuming from checkpoint: phase=%s, %d targets gathered, "
                "%d control scores",
                phase, n_gathered, n_controls,
            )
        else:
            log.info("No checkpoint found at %s — starting fresh", ckpt_file)
            self._checkpoint_data = {}

    def _save_checkpoint(self, phase: str, **kwargs: Any) -> None:
        """Save checkpoint state to disk.

        Parameters
        ----------
        phase : str
            Current pipeline phase (e.g. "gather", "controls", "scored").
        **kwargs
            Additional data to merge into the checkpoint.
        """
        with self._checkpoint_lock:
            self._checkpoint_data["phase"] = phase
            self._checkpoint_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._checkpoint_data.update(kwargs)
            ckpt_file = self._checkpoint_path / "checkpoint.json"
            # Write atomically (tmp + rename)
            tmp = ckpt_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                safe_json_dump(self._checkpoint_data, f, indent=2)
            tmp.rename(ckpt_file)

    def _clear_checkpoint(self) -> None:
        """Remove checkpoint after successful completion."""
        ckpt_file = self._checkpoint_path / "checkpoint.json"
        if ckpt_file.exists():
            ckpt_file.unlink()
            log.info("Checkpoint cleared (run complete)")

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        def _handler(signum, frame):
            sig_name = signal.Signals(signum).name
            log.warning(
                "Received %s — saving checkpoint and shutting down gracefully...",
                sig_name,
            )
            self._shutdown_requested = True
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def run(self) -> Dict[str, Any]:
        """Execute a single-pass quick scan.

        Returns
        -------
        dict
            Summary with top targets, timing, and anomaly counts.
        """
        log.info(
            "=== EXODUS Quick Run ===  max_targets=%d  tier=%d  hypotheses=%s  resume=%s",
            self.max_targets, self.tier, self.run_hypotheses, self.resume,
        )
        start = time.time()
        self._install_signal_handlers()

        # Step 1: Load targets
        targets = self._load_targets()
        if not targets:
            log.error("No targets loaded. Aborting.")
            return {"error": "No targets loaded."}

        log.info("Loaded %d targets.", len(targets))

        # Step 2: Gather data (IR + Gaia) — with checkpoint/resume
        self._gather_data(targets)
        if self._shutdown_requested:
            self._save_checkpoint("gather_interrupted")
            log.warning("Shutdown after gather phase. Resume with --resume.")
            return {"status": "interrupted", "phase": "gather"}

        # Step 3: Process (IR excess + Gaia variability)
        anomaly_count = self._process_data(targets)

        # Step 3a: Multi-messenger cross-matching
        mm_summary = self._run_multi_messenger(targets)

        # Step 3b: Select matched controls and calibrate scorer — with checkpoint
        control_scores = self._select_controls_and_score(targets)
        self._control_scores = control_scores or {}
        if control_scores:
            self.scorer = EXODUSScorer(
                    control_scores=control_scores,
                    convergence_priority=self.convergence_priority,
                ) if EXODUSScorer else None

        if self._shutdown_requested:
            self._save_checkpoint("controls_interrupted")
            log.warning("Shutdown after controls phase. Resume with --resume.")
            return {"status": "interrupted", "phase": "controls"}

        # Step 4: Score (now with calibrated p-values if controls available)
        scored = self._score_targets(targets)

        # Step 4a: Unexplainability evaluation (astrophysical template matching)
        self._evaluate_unexplainability(scored)

        # Step 4b: Red-Team falsification (automated skeptical vetting)
        self._run_red_team(scored)

        # Store scored targets for downstream access (e.g., calibration runner)
        self.scored_targets = scored

        # Step 5: Optional hypothesis cycle
        hyp_summary = {}
        if self.run_hypotheses:
            hyp_summary = self._run_hypothesis_cycle(targets)

        elapsed = time.time() - start
        summary = self._build_summary(targets, anomaly_count, hyp_summary, elapsed, mm_summary)

        # Save report
        self._save_report(summary)

        # Step 6: Provenance manifest + evidence bundles
        self._save_provenance(summary, targets, scored)

        # Clear checkpoint on successful completion
        self._clear_checkpoint()

        log.info(
            "=== Quick run complete in %.1f s (%.1f min) ===",
            elapsed, elapsed / 60.0,
        )
        return summary

    # ── Load targets ─────────────────────────────────────────────────

    def _load_targets(self) -> List[Dict[str, Any]]:
        """Load targets from custom file or tier-1 HZ planets, capped at max_targets."""
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
                # Respect max_targets cap
                targets = targets[:self.max_targets]
                # Set population_tag for population-conditional FDR
                # Priority: per-target source_catalog > campaign name
                for t in targets:
                    if "population_tag" not in t:
                        t["population_tag"] = (
                            t.get("source_catalog")
                            or campaign.campaign
                            or "default"
                        )
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
        if get_hz_planets is None:
            log.warning("Exoplanet archive not available.")
            return []

        try:
            df = get_hz_planets(max_distance_pc=100)
            if df is None or df.empty:
                return []

            df = df.head(self.max_targets)
            targets = []
            for _, row in df.iterrows():
                targets.append({
                    "target_id": str(row.get("planet_name", row.get("host_star", ""))),
                    "host_star": str(row.get("host_star", "")),
                    "ra": float(row["ra_deg"]),
                    "dec": float(row["dec_deg"]),
                    "distance_pc": float(row.get("distance_pc", 0)) if row.get("distance_pc") else None,
                    "hz_flag": True,
                })
            return targets

        except Exception:
            log.error("Failed to load targets:\n%s", traceback.format_exc())
            return []

    # ── Gather data (IR + Gaia only) ─────────────────────────────────

    @staticmethod
    def _search_radius_for_target(target: Dict[str, Any], catalog_epoch: float = 2010.0) -> float:
        """Compute a proper-motion-aware search radius for a target.

        Nearby stars have high proper motions and their positions drift
        significantly between the J2000 coordinates in our target files
        and the catalog observation epochs (2MASS ~2000, WISE ~2010,
        Gaia ~2016).  This method estimates the required search radius
        so that ``_pick_closest()`` can reliably find the correct source.

        Parameters
        ----------
        target : dict
            Target dict with at least ``distance_pc``.
        catalog_epoch : float
            Approximate observation epoch of the catalog being queried.
            2MASS: 2000.0, AllWISE: 2010.0, Gaia DR3: 2016.0.

        Returns
        -------
        float
            Search radius in arcseconds (minimum 5.0, maximum 120.0).
        """
        from src.utils import get_config
        try:
            cfg = get_config()
            base_radius = float(cfg["search"]["crossmatch_radius_arcsec"])
        except Exception:
            base_radius = 5.0

        dist = target.get("distance_pc")
        if dist is None or dist <= 0:
            return base_radius

        # Use known proper motion if available (from target file);
        # otherwise estimate from distance.
        known_pmra = target.get("pmra_mas")
        known_pmdec = target.get("pmdec_mas")
        if (known_pmra is not None and known_pmdec is not None):
            import numpy as np
            total_pm_mas = np.sqrt(known_pmra**2 + known_pmdec**2)
            estimated_pm = total_pm_mas / 1000.0  # arcsec/yr
        else:
            # Estimate from distance: nearby stars typically have
            # PM ~ 1-10 arcsec/yr for d < 5 pc, scaling as ~5/d.
            estimated_pm = min(5.0 / dist, 10.0)  # arcsec/yr, capped

        # Time between J2000 target coords and catalog epoch
        dt = abs(catalog_epoch - 2000.0)

        # Required radius = base + PM * dt, with safety factor 1.5
        pm_radius = estimated_pm * dt * 1.5
        radius = max(base_radius, base_radius + pm_radius)

        return min(radius, 300.0)  # cap at 5 arcmin

    @staticmethod
    def _propagate_coords(
        ra_j2000: float, dec_j2000: float,
        pmra_mas: float, pmdec_mas: float,
        target_epoch: float,
    ) -> tuple:
        """Propagate J2000 coordinates to a target epoch using proper motion.

        Parameters
        ----------
        ra_j2000, dec_j2000 : float
            Position at J2000 (degrees).
        pmra_mas, pmdec_mas : float
            Proper motion in mas/yr (pmra already includes cos(dec) factor).
        target_epoch : float
            Epoch to propagate to (e.g. 2010.0 for WISE, 2016.0 for Gaia).

        Returns
        -------
        (ra, dec) at target_epoch in degrees.
        """
        import numpy as np
        dt = target_epoch - 2000.0
        # Convert mas/yr to deg/yr
        dra = (pmra_mas / 1000.0 / 3600.0) / np.cos(np.radians(dec_j2000))
        ddec = pmdec_mas / 1000.0 / 3600.0
        return ra_j2000 + dra * dt, dec_j2000 + ddec * dt

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

    @staticmethod
    def _pick_target_from_gaia(
        gaia_df: "pd.DataFrame",
        target_ra_j2000: float,
        target_dec_j2000: float,
        gaia_epoch: float = 2016.0,
    ) -> "pd.Series":
        """Select the correct target from a wide-radius Gaia query.

        For each Gaia source, back-propagate its position from
        ``gaia_epoch`` to J2000 using its own proper motion, then
        pick the one closest to the target's J2000 coordinates.

        Parameters
        ----------
        gaia_df : DataFrame
            Gaia query results (must have ra, dec, pmra, pmdec columns).
        target_ra_j2000, target_dec_j2000 : float
            Target position at J2000 (degrees).
        gaia_epoch : float
            Reference epoch of Gaia positions (default 2016.0 for DR3).

        Returns
        -------
        pd.Series
            The best-matching row from gaia_df.
        """
        import numpy as np

        if len(gaia_df) == 1:
            return gaia_df.iloc[0]

        dt = 2000.0 - gaia_epoch  # negative: propagate backward
        best_idx = 0
        best_sep = float("inf")

        for i in range(len(gaia_df)):
            row = gaia_df.iloc[i]
            gaia_ra = float(row.get("ra", 0))
            gaia_dec = float(row.get("dec", 0))
            pmra = row.get("pmra", 0)   # mas/yr, includes cos(dec)
            pmdec = row.get("pmdec", 0)  # mas/yr

            # Handle missing/NaN proper motions
            if pmra is None or pmdec is None:
                pmra, pmdec = 0.0, 0.0
            elif not (np.isfinite(pmra) and np.isfinite(pmdec)):
                pmra, pmdec = 0.0, 0.0

            # Back-propagate to J2000
            dra_deg = (float(pmra) / 1000.0 / 3600.0) / np.cos(
                np.radians(gaia_dec)
            )
            ddec_deg = float(pmdec) / 1000.0 / 3600.0
            ra_at_j2000 = gaia_ra + dra_deg * dt
            dec_at_j2000 = gaia_dec + ddec_deg * dt

            # Angular separation (small-angle approx, arcsec)
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

    def _gather_data(self, targets: List[Dict[str, Any]]) -> None:
        """Ingest IR photometry and Gaia data for each target.

        For high proper motion stars, uses a two-pass strategy:
        1. Query Gaia with wide radius, pick correct source via J2000 matching.
        2. Extract PM from the matched source.
        3. Use PM-corrected coordinates for epoch-specific 2MASS/WISE queries.
        """
        log.info("Gathering data for %d targets (tier %d) ...", len(targets), self.tier)

        from src.utils import get_config
        try:
            cfg = get_config()
            base_radius = float(cfg["search"]["crossmatch_radius_arcsec"])
        except Exception:
            base_radius = 5.0

        # --- Resume: restore already-gathered target data from checkpoint ---
        gathered_cache = self._checkpoint_data.get("gathered_targets", {})
        n_resumed = 0
        if gathered_cache:
            for t in targets:
                tid = t["target_id"]
                if tid in gathered_cache:
                    cached = gathered_cache[tid]
                    for key in ("gaia_astrometry", "gaia_params", "ir_photometry",
                                "gaia_data", "lightcurve_metadata", "epoch_photometry",
                                "erosita", "herschel"):
                        if key in cached:
                            t[key] = cached[key]
                    n_resumed += 1
            if n_resumed:
                log.info("  Restored %d/%d targets from checkpoint", n_resumed, len(targets))

        # ── Parallel vs sequential gathering ───────────────────────────
        cfg = get_config()
        n_parallel = cfg.get("performance", {}).get("parallel_targets", 1)

        # Build list of targets that still need gathering
        pending = [
            (idx, t) for idx, t in enumerate(targets)
            if t["target_id"] not in gathered_cache
        ]
        cached_count = len(targets) - len(pending)
        if cached_count:
            for idx, t in enumerate(targets):
                if t["target_id"] in gathered_cache:
                    log.info("  [%d/%d] %s (cached — skipping)",
                             idx + 1, len(targets), t["target_id"])

        if n_parallel > 1 and len(pending) > 1:
            # ── PARALLEL MODE ─────────────────────────────────────────
            log.info("  Parallel gathering: %d workers, %d pending targets",
                     n_parallel, len(pending))
            target_pool = ThreadPoolExecutor(max_workers=n_parallel)
            futures_map: Dict[Future, Tuple[int, Dict[str, Any]]] = {}

            for idx, t in pending:
                if self._shutdown_requested:
                    break
                fut = target_pool.submit(
                    self._gather_single_target, idx, t,
                    len(targets), base_radius,
                )
                futures_map[fut] = (idx, t)

            completed = 0
            for fut in as_completed(futures_map):
                idx, t = futures_map[fut]
                tid = t["target_id"]
                try:
                    snapshot = fut.result(timeout=600)
                    # Merge gathered data back into target dict
                    for key, val in snapshot.items():
                        t[key] = val
                    gathered_cache[tid] = snapshot
                    completed += 1
                    if completed % 10 == 0 or completed == len(pending):
                        self._save_checkpoint(
                            "gathering",
                            gathered_targets=gathered_cache,
                            n_gathered=len(gathered_cache),
                            n_total=len(targets),
                        )
                        log.info("  Checkpoint saved: %d/%d targets gathered",
                                 len(gathered_cache), len(targets))
                except Exception as exc:
                    log.warning("  Target %s failed: %s", tid, exc)

            target_pool.shutdown(wait=True)
        else:
            # ── SEQUENTIAL MODE (original behavior) ───────────────────
            _concurrent_pool = ThreadPoolExecutor(max_workers=4)

            for idx, t in pending:
                if self._shutdown_requested:
                    log.warning("Shutdown requested at target %d/%d",
                                idx + 1, len(targets))
                    break

                snapshot = self._gather_single_target(
                    idx, t, len(targets), base_radius,
                    sub_pool=_concurrent_pool,
                )
                for key, val in snapshot.items():
                    t[key] = val
                gathered_cache[t["target_id"]] = snapshot

                if (len(gathered_cache)) % 10 == 0 or (idx + 1) == len(targets):
                    self._save_checkpoint(
                        "gathering",
                        gathered_targets=gathered_cache,
                        n_gathered=len(gathered_cache),
                        n_total=len(targets),
                    )
                    log.info("  Checkpoint saved: %d/%d targets gathered",
                             len(gathered_cache), len(targets))

            _concurrent_pool.shutdown(wait=False)

    def _gather_single_target(
        self,
        idx: int,
        t: Dict[str, Any],
        targets_len: int,
        base_radius: float,
        sub_pool: Optional[ThreadPoolExecutor] = None,
    ) -> Dict[str, Any]:
        """Gather all data for a single target.

        Returns a snapshot dict suitable for the checkpoint cache.
        If *sub_pool* is None (parallel mode), creates a temporary pool.
        """
        ra, dec = t["ra"], t["dec"]
        tid = t["target_id"]
        log.info("  [%d/%d] %s", idx + 1, targets_len, tid)

        # Create per-target sub-pool in parallel mode
        _own_pool = sub_pool is None
        if _own_pool:
            sub_pool = ThreadPoolExecutor(max_workers=4)

        try:
            return self._do_gather_target(idx, t, base_radius, sub_pool)
        finally:
            if _own_pool:
                sub_pool.shutdown(wait=False)

    def _do_gather_target(
        self,
        idx: int,
        t: Dict[str, Any],
        base_radius: float,
        _concurrent_pool: ThreadPoolExecutor,
    ) -> Dict[str, Any]:
        """Inner gathering logic for a single target (extracted from loop)."""
        ra, dec = t["ra"], t["dec"]
        tid = t["target_id"]

        # --- Use known PMs from target file if available ---
        gaia_radius = self._search_radius_for_target(t, catalog_epoch=2016.0)
        _known_pmra = t.get("pmra_mas")
        _known_pmdec = t.get("pmdec_mas")
        if _known_pmra is not None and _known_pmdec is not None:
            pmra_mas = float(_known_pmra)
            pmdec_mas = float(_known_pmdec)
            log.info("    Using known PM: pmra=%.1f pmdec=%.1f mas/yr",
                     pmra_mas, pmdec_mas)
        else:
            pmra_mas = 0.0
            pmdec_mas = 0.0
        matched_source_id = None

        # ── Submit independent queries concurrently (before Gaia chain) ──
        concurrent_futures = self._submit_concurrent_queries(
            ra, dec, t, tid, _concurrent_pool
        )

        # --- Step 1: Query Gaia ASTROMETRY first ---
        if get_astrometry is not None:
            try:
                astro_df = get_astrometry(ra, dec, radius_arcsec=gaia_radius)
                if astro_df is not None and not astro_df.empty:
                    best_row = self._pick_target_from_gaia(
                        astro_df, ra, dec, gaia_epoch=2016.0,
                    )
                    t["gaia_astrometry"] = best_row.to_dict()
                    matched_source_id = best_row.get("source_id")
                    if abs(pmra_mas) < 0.1 and abs(pmdec_mas) < 0.1:
                        _pmra = best_row.get("pmra")
                        _pmdec = best_row.get("pmdec")
                        if _pmra is not None and _pmdec is not None:
                            import numpy as np
                            if np.isfinite(_pmra) and np.isfinite(_pmdec):
                                pmra_mas = float(_pmra)
                                pmdec_mas = float(_pmdec)
            except Exception as exc:
                log.debug("Gaia astrometry failed for %s: %s", tid, exc)

        # --- Step 1b: Query stellar params, match by source_id ---
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

        # --- Step 2: Compute epoch-corrected coordinates ---
        has_pm = abs(pmra_mas) > 0.1 or abs(pmdec_mas) > 0.1
        gaia_astro = t.get("gaia_astrometry", {})
        gaia_ra_2016 = gaia_astro.get("ra")
        gaia_dec_2016 = gaia_astro.get("dec")
        use_gaia_ref = (
            has_pm
            and gaia_ra_2016 is not None
            and gaia_dec_2016 is not None
        )

        if use_gaia_ref:
            import numpy as np
            gaia_ra_2016 = float(gaia_ra_2016)
            gaia_dec_2016 = float(gaia_dec_2016)
            total_pm = (pmra_mas**2 + pmdec_mas**2)**0.5 / 1000.0
            log.info("    PM detected: %.1f mas/yr (%.2f\"/yr)",
                     (pmra_mas**2 + pmdec_mas**2)**0.5, total_pm)
            dt_2mass = 2000.0 - 2016.0
            dra_per_yr = (pmra_mas / 1000.0 / 3600.0) / np.cos(
                np.radians(gaia_dec_2016))
            ddec_per_yr = pmdec_mas / 1000.0 / 3600.0
            ra_2mass = gaia_ra_2016 + dra_per_yr * dt_2mass
            dec_2mass = gaia_dec_2016 + ddec_per_yr * dt_2mass
            dt_wise = 2010.0 - 2016.0
            ra_wise = gaia_ra_2016 + dra_per_yr * dt_wise
            dec_wise = gaia_dec_2016 + ddec_per_yr * dt_wise
            log.info("    Gaia 2016: (%.4f, %+.4f)  →  2MASS J2000: (%.4f, %+.4f)"
                     "  →  WISE 2010: (%.4f, %+.4f)",
                     gaia_ra_2016, gaia_dec_2016,
                     ra_2mass, dec_2mass,
                     ra_wise, dec_wise)
        elif has_pm:
            total_pm = (pmra_mas**2 + pmdec_mas**2)**0.5 / 1000.0
            log.info("    PM detected: %.1f mas/yr (%.2f\"/yr) [no Gaia ref]",
                     (pmra_mas**2 + pmdec_mas**2)**0.5, total_pm)
            ra_2mass, dec_2mass = self._propagate_coords(
                ra, dec, pmra_mas, pmdec_mas, 2000.0)
            ra_wise, dec_wise = self._propagate_coords(
                ra, dec, pmra_mas, pmdec_mas, 2010.0)
        else:
            ra_2mass, dec_2mass = ra, dec
            ra_wise, dec_wise = ra, dec

        # --- Step 3: Query IR photometry at epoch-corrected positions ---
        if get_ir_photometry is not None:
            try:
                from src.ingestion.ir_surveys import get_2mass, get_wise

                ir_merged = {"ra": ra, "dec": dec}

                twomass = get_2mass(ra_2mass, dec_2mass,
                                   radius_arcsec=base_radius)
                if twomass:
                    for band in ("J", "H", "Ks", "J_err", "H_err", "Ks_err"):
                        if band in twomass:
                            ir_merged[band] = twomass[band]
                    ir_merged["twomass_designation"] = twomass.get("designation")
                    ir_merged["twomass_sep_arcsec"] = twomass.get("match_sep_arcsec")

                wise = get_wise(ra_wise, dec_wise,
                               radius_arcsec=base_radius)
                if wise:
                    for band in ("W1", "W2", "W3", "W4",
                                 "W1_err", "W2_err", "W3_err", "W4_err"):
                        if band in wise:
                            ir_merged[band] = wise[band]
                    ir_merged["wise_designation"] = wise.get("designation")
                    ir_merged["wise_sep_arcsec"] = wise.get("match_sep_arcsec")
                    if wise.get("cc_flags") is not None:
                        ir_merged["cc_flags"] = wise["cc_flags"]
                    if wise.get("ext_flg") is not None:
                        ir_merged["ext_flg"] = wise["ext_flg"]

                if get_catwise is not None:
                    try:
                        catwise = get_catwise(ra_wise, dec_wise,
                                              radius_arcsec=base_radius)
                        if catwise:
                            for ck in ("W1_catwise", "W2_catwise",
                                       "pmra_wise", "e_pmra_wise",
                                       "pmdec_wise", "e_pmdec_wise"):
                                if ck in catwise:
                                    ir_merged[ck] = catwise[ck]
                            ir_merged["catwise_designation"] = catwise.get("designation")
                            ir_merged["catwise_sep_arcsec"] = catwise.get("match_sep_arcsec")
                    except Exception as exc:
                        log.debug("CatWISE query failed for %s: %s", tid, exc)

                gaia_p = t.get("gaia_params", {})
                if gaia_p:
                    import numpy as np
                    for band_key, col in [("G", "phot_g_mean_mag"),
                                          ("BP", "phot_bp_mean_mag"),
                                          ("RP", "phot_rp_mean_mag")]:
                        val = gaia_p.get(col)
                        if val is not None and np.isfinite(val):
                            ir_merged[band_key] = float(val)
                            ir_merged[f"{band_key}_err"] = 0.01

                t["ir_photometry"] = ir_merged
            except Exception as exc:
                log.debug("IR photometry failed for %s: %s", tid, exc)

        # ── Collect concurrent results ────────────────────────────────
        self._collect_concurrent_results(t, tid, concurrent_futures)

        # Gaia epoch photometry (tier >= 1)
        if self.tier >= 1 and get_epoch_photometry is not None:
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

        # Light curves (tier >= 1)
        if self.tier >= 1 and get_lightcurve is not None:
            try:
                lk_name = self._lightcurve_query_name(t)
                log.info("    Querying lightcurve for '%s' ...", lk_name)
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
                else:
                    log.info("    No lightcurve found for '%s'", lk_name)
            except Exception as exc:
                log.debug("Lightcurve failed for %s: %s", tid, exc)

        # Breakthrough Listen (tier >= 2)
        if self.tier >= 2 and bl_get_spectrogram is not None:
            try:
                bl_name = self._lightcurve_query_name(t)
                spec, freqs, times, radio_source = bl_get_spectrogram(bl_name)
                if spec is not None:
                    t["radio_spectrogram"] = {
                        "spectrogram": spec,
                        "frequencies_mhz": freqs,
                        "timestamps_sec": times,
                        "data_source": radio_source,
                    }
                    log.info("    BL radio: %dx%d spectrogram (%s)",
                             spec.shape[0], spec.shape[1], radio_source)
            except Exception as exc:
                log.debug("BL radio failed for %s: %s", tid, exc)

        # --- Build checkpoint snapshot ---
        target_snapshot: Dict[str, Any] = {}
        for key in ("gaia_astrometry", "gaia_params", "ir_photometry",
                    "gaia_data", "lightcurve_metadata", "epoch_photometry",
                    "erosita", "herschel"):
            if key in t:
                target_snapshot[key] = t[key]
        return target_snapshot

    # ── Concurrent I/O helpers ────────────────────────────────────────

    def _submit_concurrent_queries(
        self,
        ra: float,
        dec: float,
        t: Dict[str, Any],
        tid: str,
        pool: ThreadPoolExecutor,
    ) -> Dict[str, Future]:
        """Submit independent data queries to thread pool.

        These queries use the original (ra, dec) and don't depend on
        Gaia results, so they can run concurrently with the Gaia+IR chain.
        Returns a dict of future_name -> Future.
        """
        futures = {}

        # eROSITA X-ray
        if query_erosita_cone is not None:
            futures["erosita"] = pool.submit(
                self._query_erosita_safe, ra, dec, tid
            )

        # Herschel far-IR
        if query_herschel is not None:
            futures["herschel"] = pool.submit(
                self._query_herschel_safe, ra, dec, tid
            )

        # SB9 spectroscopic binaries
        if query_sb9_cone is not None:
            futures["sb9"] = pool.submit(
                self._query_sb9_safe, ra, dec, tid
            )

        # 3D dust extinction (fast, local — but included for completeness)
        dist_pc = t.get("distance_pc")
        if get_extinction_context is not None and dist_pc and dist_pc > 0:
            futures["dust"] = pool.submit(
                self._query_dust_safe, ra, dec, dist_pc, tid
            )

        # GALEX UV
        if query_galex_cone is not None:
            futures["galex"] = pool.submit(
                self._query_galex_safe, ra, dec, tid
            )

        # Radio continuum (FIRST/NVSS)
        if query_radio_continuum is not None:
            futures["radio"] = pool.submit(
                self._query_radio_safe, ra, dec, tid
            )

        # SIMBAD object type (audit fix D1)
        if query_simbad_cone is not None:
            futures["simbad"] = pool.submit(
                self._query_simbad_safe, ra, dec, tid
            )

        # NEOWISE time-series — THE BOTTLENECK (60-120s)
        # This is why concurrent I/O matters: NEOWISE runs in background
        # while the Gaia+IR sequential chain proceeds.
        if query_neowise_timeseries is not None:
            futures["neowise"] = pool.submit(
                self._query_neowise_safe, ra, dec, tid
            )

        return futures

    def _collect_concurrent_results(
        self,
        t: Dict[str, Any],
        tid: str,
        futures: Dict[str, Future],
    ) -> None:
        """Collect results from concurrent queries and merge into target dict."""
        for name, fut in futures.items():
            try:
                result = fut.result(timeout=180)  # 3-minute per-query timeout
                if result is None:
                    continue

                if name == "erosita":
                    t["erosita"] = result
                    flux = result.get("flux_0p2_2p3", 0) or 0
                    log.info("    eROSITA: X-ray source at %.1f\" (flux=%.1e)",
                             result.get("sep_arcsec", 0), flux)

                elif name == "herschel":
                    t["herschel"] = result
                    if result.get("has_data"):
                        log.info(
                            "    Herschel: PACS=%s/%s/%s  SPIRE=%s/%s/%s (sep=%.1f\")",
                            result.get("pacs_70"), result.get("pacs_100"),
                            result.get("pacs_160"),
                            result.get("spire_250"), result.get("spire_350"),
                            result.get("spire_500"),
                            result.get("separation_arcsec", -1),
                        )
                        # SED interpretation
                        if interpret_herschel_sed is not None:
                            t["herschel_sed"] = interpret_herschel_sed(result)

                elif name == "sb9":
                    t["sb9"] = result
                    if result.get("match"):
                        log.info("    SB9: spectroscopic binary '%s' at %.1f\"",
                                 result.get("name", ""), result.get("sep_arcsec", 0))

                elif name == "dust":
                    if result.get("available"):
                        t["dust_extinction"] = result
                        ebv = result.get("ebv", 0) or 0
                        if ebv > 0.05:
                            log.info("    Dust: E(B-V)=%.3f [%s]",
                                     ebv, result.get("concern_level", "?"))

                elif name == "galex":
                    t["galex"] = result
                    # Compute UV metrics (needs Gaia params which are now available)
                    gaia_p = t.get("gaia_params", {})
                    uv_metrics = compute_uv_metrics(result, gaia_params=gaia_p)
                    t["uv_metrics"] = uv_metrics
                    nuv = result.get("nuv_mag")
                    fuv = result.get("fuv_mag")
                    log.info("    GALEX: FUV=%.1f NUV=%.1f (UV score=%.2f)",
                             fuv if fuv else -99, nuv if nuv else -99,
                             uv_metrics.get("uv_anomaly_score", 0))

                elif name == "radio":
                    t["radio_continuum"] = result
                    detected = is_radio_continuum_detected(result)
                    if detected:
                        log.info("    Radio continuum: %s at %.1f\" — %.2f mJy",
                                 result.get("survey", "?"),
                                 result.get("sep_arcsec", 0),
                                 result.get("peak_flux_mjy", 0))

                elif name == "simbad":
                    t["simbad"] = result
                    if result.get("match"):
                        log.info("    SIMBAD: %s (%s) at %.1f\" [risk=%.1f]",
                                 result.get("main_id", "?"),
                                 result.get("otype", "?"),
                                 result.get("sep_arcsec", 0),
                                 result.get("risk_level", 0))

                elif name == "neowise":
                    if result.n_epochs > 0:
                        t["neowise_timeseries"] = result
                        log.info("    NEOWISE: %d epochs over %.1f yr",
                                 result.n_epochs, result.time_baseline_years)

            except Exception as exc:
                log.debug("Concurrent query '%s' failed for %s: %s", name, tid, exc)

    # ── Thread-safe query wrappers ────────────────────────────────────

    @staticmethod
    def _query_erosita_safe(ra, dec, tid):
        try:
            return query_erosita_cone(ra, dec, radius_arcsec=30.0)
        except Exception as exc:
            log.debug("eROSITA query failed for %s: %s", tid, exc)
            return None

    @staticmethod
    def _query_herschel_safe(ra, dec, tid):
        try:
            return query_herschel(ra, dec, radius_arcsec=15.0)
        except Exception as exc:
            log.debug("Herschel query failed for %s: %s", tid, exc)
            return None

    @staticmethod
    def _query_sb9_safe(ra, dec, tid):
        try:
            return query_sb9_cone(ra, dec, radius_arcsec=30.0)
        except Exception as exc:
            log.debug("SB9 query failed for %s: %s", tid, exc)
            return None

    @staticmethod
    def _query_dust_safe(ra, dec, dist_pc, tid):
        try:
            return get_extinction_context(ra, dec, dist_pc)
        except Exception as exc:
            log.debug("Dust extinction query failed for %s: %s", tid, exc)
            return None

    @staticmethod
    def _query_galex_safe(ra, dec, tid):
        try:
            return query_galex_cone(ra, dec, radius_arcsec=30.0)
        except Exception as exc:
            log.debug("GALEX query failed for %s: %s", tid, exc)
            return None

    @staticmethod
    def _query_radio_safe(ra, dec, tid):
        try:
            return query_radio_continuum(ra, dec, radius_arcsec=15.0)
        except Exception as exc:
            log.debug("Radio continuum query failed for %s: %s", tid, exc)
            return None

    @staticmethod
    def _query_simbad_safe(ra, dec, tid):
        try:
            return query_simbad_cone(ra, dec, radius_arcsec=5.0)
        except Exception as exc:
            log.debug("SIMBAD query failed for %s: %s", tid, exc)
            return None

    @staticmethod
    def _query_neowise_safe(ra, dec, tid):
        try:
            return query_neowise_timeseries(ra, dec)
        except Exception as exc:
            log.debug("NEOWISE timeseries failed for %s: %s", tid, exc)
            return None

    # ── Process data ─────────────────────────────────────────────────

    def _process_data(self, targets: List[Dict[str, Any]]) -> int:
        """Process IR excess and Gaia variability.  Returns anomaly count."""
        log.info("Processing data ...")
        anomaly_count = 0

        for t in targets:
            tid = t["target_id"]

            # IR excess
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
                        t["ir_excess"] = {"reason": "no data provided"}
                except Exception as exc:
                    log.debug("IR excess failed for %s: %s", tid, exc)

            # IR variability (NEOWISE time-series analysis) — all tiers
            # INDEPENDENT from ir_excess: measures temporal changes over 10+ years,
            # not current SED anomaly.  Detects secular brightening (Dyson sphere
            # construction) and excess scatter (variable dust / megastructure transit).
            if compute_ir_variability is not None:
                try:
                    # Pass pre-fetched data if available (avoids re-query);
                    # otherwise compute_ir_variability auto-queries NEOWISE
                    neowise_data = t.get("neowise_timeseries")
                    irv_result = compute_ir_variability(
                        t.get("ra"), t.get("dec"),
                        neowise_data=neowise_data,
                    )
                    t["ir_variability"] = irv_result.to_dict()
                    if irv_result.is_anomalous:
                        anomaly_count += 1
                        log.info("  IR variability anomaly for %s: score=%.3f, "
                                 "excess_scatter=%.1f, trend_sigma=%.1f",
                                 tid, irv_result.variability_score,
                                 max(irv_result.w1_excess_scatter, irv_result.w2_excess_scatter),
                                 max(abs(irv_result.w1_trend_sigma), abs(irv_result.w2_trend_sigma)))
                except Exception as exc:
                    # Audit fix N1: escalate from debug to warning so silent
                    # channel drops are visible; record the failure in target.
                    log.warning("IR variability failed for %s: %s", tid, exc)
                    t["ir_variability"] = {"error": str(exc), "channel_dropped": True}

            # UV anomaly (from GALEX data already gathered) — all tiers
            if compute_uv_anomaly is not None and t.get("uv_metrics"):
                try:
                    uv_result = compute_uv_anomaly(
                        uv_metrics=t.get("uv_metrics"),
                        galex_raw=t.get("galex"),
                        ir_excess_data=t.get("ir_excess"),
                    )
                    t["uv_anomaly"] = uv_result.to_dict()
                    if uv_result.is_anomalous:
                        anomaly_count += 1
                        log.info("  UV anomaly for %s: score=%.3f, deficit=%s, active=%s",
                                 tid, uv_result.anomaly_score,
                                 uv_result.is_uv_deficit, uv_result.is_uv_active)
                except Exception as exc:
                    log.debug("UV anomaly failed for %s: %s", tid, exc)

            # Radio continuum emission (from FIRST/NVSS data already gathered) — all tiers
            if compute_radio_emission is not None and t.get("radio_continuum"):
                try:
                    radio_result = compute_radio_emission(
                        radio_continuum=t.get("radio_continuum"),
                        distance_pc=t.get("distance_pc"),
                    )
                    t["radio_emission"] = radio_result.to_dict()
                    if radio_result.is_anomalous:
                        anomaly_count += 1
                        log.info("  Radio emission for %s: score=%.3f, %.1f mJy (%s)",
                                 tid, radio_result.anomaly_score,
                                 radio_result.peak_flux_mjy, radio_result.survey)
                except Exception as exc:
                    log.debug("Radio emission failed for %s: %s", tid, exc)

            # HR diagram outlier (from Gaia params already gathered) — all tiers
            if compute_hr_anomaly is not None and t.get("gaia_params"):
                try:
                    hr_result = compute_hr_anomaly(
                        gaia_params=t.get("gaia_params"),
                        astrometry=t.get("gaia_astrometry"),
                        distance_pc=t.get("distance_pc"),
                    )
                    t["hr_anomaly"] = hr_result.to_dict()
                    if hr_result.is_anomalous:
                        anomaly_count += 1
                        log.info("  HR anomaly for %s: score=%.3f, sigma=%.1f (%s MS)",
                                 tid, hr_result.anomaly_score, hr_result.ms_sigma,
                                 "above" if hr_result.is_above_ms else "below")
                except Exception as exc:
                    log.debug("HR anomaly failed for %s: %s", tid, exc)

            # Stellar abundance anomaly (cross-match APOGEE/GALAH) — all tiers
            if compute_abundance_anomaly is not None:
                try:
                    abund_result = compute_abundance_anomaly(
                        ra=t["ra"], dec=t["dec"],
                    )
                    t["abundance_anomaly"] = abund_result.to_dict()
                    if abund_result.is_anomalous:
                        anomaly_count += 1
                        log.info("  Abundance anomaly for %s: score=%.3f, %d ratios (%s)",
                                 tid, abund_result.anomaly_score,
                                 abund_result.n_anomalous_ratios,
                                 ", ".join(abund_result.anomalous_ratios))
                except Exception as exc:
                    log.debug("Abundance anomaly failed for %s: %s", tid, exc)

            # Transit anomaly (from lightcurve data) — tier >= 1
            if self.tier >= 1 and detect_transit_anomaly is not None and t.get("lightcurve"):
                try:
                    lc_data = t["lightcurve"]
                    time_arr = lc_data["time"]
                    flux_arr = lc_data["flux"]
                    flux_err_arr = lc_data.get("flux_err")

                    # Downsample large lightcurves to cap BLS runtime
                    # BLS is ~O(N^2); 30k pts keeps analysis under ~2 min
                    _MAX_LC_POINTS = 30_000
                    n_orig = len(time_arr)
                    if n_orig > _MAX_LC_POINTS:
                        import numpy as np
                        step = max(2, -(-n_orig // _MAX_LC_POINTS))  # ceiling division, min step=2
                        time_arr = time_arr[::step] if isinstance(time_arr, np.ndarray) else time_arr[::step]
                        flux_arr = flux_arr[::step] if isinstance(flux_arr, np.ndarray) else flux_arr[::step]
                        if flux_err_arr is not None:
                            flux_err_arr = flux_err_arr[::step] if isinstance(flux_err_arr, np.ndarray) else flux_err_arr[::step]
                        log.info("  Lightcurve downsampled: %d -> %d points (step=%d)",
                                 n_orig, len(time_arr), step)

                    if len(time_arr) >= 50:
                        # Run BLS-based periodic transit anomaly detection
                        ta_result = detect_transit_anomaly(
                            time=time_arr,
                            flux=flux_arr,
                            flux_err=flux_err_arr,
                        )

                        # Also run irregular dimming detection (for Tabby's Star-type events)
                        irreg_score = 0.0
                        irreg_events = 0
                        if detect_irregular_dimming is not None and len(time_arr) >= 100:
                            try:
                                irreg_result = detect_irregular_dimming(
                                    time=time_arr,
                                    flux=flux_arr,
                                )
                                irreg_score = irreg_result.anomaly_score
                                irreg_events = irreg_result.n_events
                            except Exception:
                                pass

                        # Combine: take the max of periodic and irregular scores
                        combined_score = max(ta_result.anomaly_score, irreg_score)

                        # Starspot rotation filter: if shape_residual dominates
                        # (signal is sinusoidal, not box-like) and transit-specific
                        # metrics are low, this is likely starspot modulation
                        starspot_flag = False
                        if (ta_result.anomaly_score > 0.2
                                and ta_result.shape_residual > 0.5
                                and ta_result.symmetry_score < 0.3
                                and ta_result.depth_variability < 0.3):
                            starspot_flag = True
                            log.info("  Starspot filter: %s BLS period=%.2fd, shape_res=%.3f "
                                     ">> suppressing periodic score (likely rotation)",
                                     tid, ta_result.period, ta_result.shape_residual)
                            # Keep irregular score but zero periodic contribution
                            combined_score = irreg_score

                        t["transit_anomaly"] = {
                            "anomaly_score": combined_score,
                            "is_anomalous": (not starspot_flag and ta_result.is_anomalous) or combined_score > 0.3,
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
                            log.info("  Transit anomaly for %s: score=%.3f (periodic=%.3f, irregular=%.3f)",
                                     tid, combined_score, ta_result.anomaly_score, irreg_score)
                    else:
                        log.info("  Lightcurve for %s too short (%d pts) for transit analysis",
                                 tid, len(time_arr))
                except Exception as exc:
                    log.debug("Transit anomaly failed for %s: %s", tid, exc)

            # Radio anomaly (Breakthrough Listen spectrogram) — tier >= 2
            if self.tier >= 2 and radio_process_spectrogram is not None and t.get("radio_spectrogram"):
                try:
                    rs = t["radio_spectrogram"]
                    radio_result = radio_process_spectrogram(
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
                        "data_source": rs.get("data_source", "simulated"),
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
                        log.info("  Radio anomaly for %s: %d candidates (max SNR=%.1f)",
                                 tid, len(non_rfi), max_snr)
                except Exception as exc:
                    log.debug("Radio processing failed for %s: %s", tid, exc)

            # Gaia photometric variability — tier >= 1
            if self.tier >= 1 and t.get("gaia_epoch_photometry"):
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

            # HZ flag
            t["habitable_zone_planet"] = {
                "has_hz_planet": t.get("hz_flag", False),
                "n_hz_planets": 1 if t.get("hz_flag") else 0,
                "data_source": "real",
            }

            # Astrometry
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
                            phot_g_mean_mag=float(t.get("gaia_params", {}).get("phot_g_mean_mag"))
                            if t.get("gaia_params", {}).get("phot_g_mean_mag") is not None else None,
                        )
                        pm_anomaly["wise_gaia_pm"] = pm_check
                        if pm_check.get("catwise_systematic_flag"):
                            log.info("  CATWISE SYSTEMATIC for %s: "
                                     "PM offset suppressed (floor=%.1f mas/yr)",
                                     tid, pm_check.get("wise_sys_floor", 0))
                        elif pm_check["is_discrepant"]:
                            log.info("  PM discrepancy for %s: %.1f sigma "
                                     "(dRA=%+.1f dDec=%+.1f mas/yr)",
                                     tid, pm_check["pm_discrepancy_sigma"],
                                     pm_check["delta_pmra"],
                                     pm_check["delta_pmdec"])
                    except Exception as exc:
                        log.debug("PM consistency check failed for %s: %s",
                                  tid, exc)

                t["proper_motion_anomaly"] = pm_anomaly

        log.info("Processing done. %d IR excess candidates.", anomaly_count)
        return anomaly_count

    # ── Multi-messenger cross-matching ───────────────────────────────

    def _run_multi_messenger(
        self, targets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run multi-messenger cross-match modules against the target list.

        Annotates each target with any cross-match hits and returns a
        summary dict.  Uses simulation fallback when live catalogs are
        unavailable.
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
                gamma_result = crossmatch_fermi_exoplanets(fermi, exoplanet_hosts)
                summary["gamma"] = {
                    "n_matches": gamma_result.n_matches,
                    "n_escalations": gamma_result.n_escalations,
                    "data_source": "simulation" if (fermi and fermi[0].get("simulated")) else "real",
                }
                gamma_ds = summary["gamma"]["data_source"]
                for m in gamma_result.matches:
                    if not m.escalation:
                        continue  # Only attach significance-qualified matches
                    host = m.host_name
                    for t in targets:
                        if t.get("target_id") == host or t.get("host_star") == host:
                            mm_dict = m.to_dict()
                            mm_dict["data_source"] = gamma_ds  # F-03: per-hit provenance
                            t.setdefault("multi_messenger", {})["gamma"] = mm_dict
                log.info("  Gamma-ray: %d matches (%d escalations)",
                         gamma_result.n_matches, gamma_result.n_escalations)
            except Exception as exc:
                log.warning("  Gamma-ray crossmatch failed: %s", exc)
                summary["gamma"] = {"error": str(exc)}

        # Neutrino (IceCube) — use fast array path when available
        if crossmatch_neutrino_exoplanets is not None:
            try:
                from src.ingestion.icecube_catalog import get_all_events, get_arrays as get_nu_arrays
                nu_arrays = get_nu_arrays()
                if nu_arrays is not None:
                    neutrinos = []  # not needed when using array path
                    nu_result = crossmatch_neutrino_exoplanets(
                        neutrinos, exoplanet_hosts, neutrino_arrays=nu_arrays,
                    )
                else:
                    nu_raw = get_all_events()
                    neutrinos = [e.to_dict() for e in nu_raw]
                    nu_result = crossmatch_neutrino_exoplanets(neutrinos, exoplanet_hosts)
                summary["neutrino"] = {
                    "n_significant": nu_result.n_significant,
                    "n_hosts_with_excess": len(nu_result.hosts_with_excess),
                    "data_source": "real" if nu_arrays is not None else (
                        "simulation" if (neutrinos and neutrinos[0].get("source") == "simulated") else "real"
                    ),
                }
                nu_ds = summary["neutrino"]["data_source"]
                for host_excess in nu_result.hosts_with_excess:
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
                log.info("  Neutrino: %d significant hosts", nu_result.n_significant)
            except Exception as exc:
                log.warning("  Neutrino crossmatch failed: %s", exc)
                summary["neutrino"] = {"error": str(exc)}

        # Gravitational waves
        if crossmatch_gw_exoplanets is not None:
            try:
                from src.ingestion.gw_events import get_all_events as get_gw_events
                gw_raw = get_gw_events()
                gw_events = [e.to_dict() for e in gw_raw]
                gw_result = crossmatch_gw_exoplanets(gw_events, exoplanet_hosts)
                summary["gw"] = {
                    "n_coincidences": gw_result.n_coincidences,
                    "data_source": "simulation" if (gw_events and gw_events[0].get("source") == "simulated") else "real",
                }
                gw_ds = summary["gw"]["data_source"]
                for c in gw_result.coincidences:
                    if not c.is_low_fa:
                        continue  # Only attach low-false-alarm coincidences
                    host = c.host_name
                    for t in targets:
                        if t.get("target_id") == host or t.get("host_star") == host:
                            mm_dict = c.to_dict()
                            mm_dict["data_source"] = gw_ds  # F-03: per-hit provenance
                            t.setdefault("multi_messenger", {})["gw"] = mm_dict
                log.info("  GW: %d coincidences (%d low-FA)",
                         gw_result.n_coincidences, len(gw_result.low_fa_coincidences))
            except Exception as exc:
                log.warning("  GW crossmatch failed: %s", exc)
                summary["gw"] = {"error": str(exc)}

        # Pulsar Shapiro delay
        if search_pulsar_los is not None:
            try:
                from src.ingestion.nanograv import get_all_pulsars
                pulsar_raw = get_all_pulsars()
                pulsars = [p.to_dict() for p in pulsar_raw]
                pulsar_result = search_pulsar_los(pulsars, exoplanet_hosts)
                summary["pulsar"] = {
                    "n_candidates": pulsar_result.n_candidates,
                    "n_los_matches": pulsar_result.n_los_matches_total,
                    "data_source": "simulation" if (pulsars and pulsars[0].get("source") == "simulated") else "real",
                }
                pulsar_ds = summary["pulsar"]["data_source"]
                for pr in pulsar_result.candidates:  # Only flagged candidates
                    for los_match in pr.los_matches:
                        host = los_match.host_star
                        for t in targets:
                            if t.get("target_id") == host or t.get("host_star") == host:
                                mm_dict = pr.to_dict()
                                mm_dict["data_source"] = pulsar_ds  # F-03: per-hit provenance
                                t.setdefault("multi_messenger", {})["pulsar"] = mm_dict
                log.info("  Pulsar: %d candidates, %d LOS matches",
                         pulsar_result.n_candidates, pulsar_result.n_los_matches_total)
            except Exception as exc:
                log.warning("  Pulsar search failed: %s", exc)
                summary["pulsar"] = {"error": str(exc)}

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
                for pm in frb_result.matches:
                    host = pm.host_name
                    for t in targets:
                        if t.get("target_id") == host or t.get("host_star") == host:
                            t.setdefault("multi_messenger", {})["frb"] = pm.__dict__ if hasattr(pm, '__dict__') else {}
                log.info("  FRB: %d spatial, %d period matches (%.1fσ)",
                         frb_result.n_spatial_matches, frb_result.n_period_matches,
                         frb_result.significance_sigma)
            except Exception as exc:
                log.warning("  FRB correlation failed: %s", exc)
                summary["frb"] = {"error": str(exc)}

        # Temporal Archaeology (NVSS vs VLASS radio survey comparison) — tier >= 1
        if TemporalArchaeology is not None and self.tier >= 1:
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
                # Attach high-priority hits (exoplanet-matched radio changes) to targets
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
                summary["temporal_archaeology"] = {"error": str(exc)}

        log.info("=== Multi-messenger complete: %d modules ran ===", len(summary))
        return summary

    # ── Matched control selection ────────────────────────────────────

    def _select_controls_and_score(
        self, targets: List[Dict[str, Any]],
    ) -> Dict[str, List[float]]:
        """Query field stars, select matched controls, run detectors on them.

        Returns a dict of channel_name -> list of control scores suitable
        for passing to ``EXODUSScorer(control_scores=...)``.

        This enables calibrated p-values, Fisher combination, and FDR
        correction -- the statistical rigor layer required for publication.
        """
        control_scores: Dict[str, List[float]] = {}

        if select_matched_controls is None or gaia_cone_search is None:
            log.warning(
                "Matched controls unavailable (missing imports). "
                "Scoring will remain UNCALIBRATED."
            )
            return control_scores

        if compute_ir_excess is None:
            log.warning("IR excess not available; cannot score controls.")
            return control_scores

        log.info("=== Selecting matched controls for calibration ===")

        # Step 1: Build a field catalog by querying Gaia in a wide cone
        # around the median target position.  For small campaigns (< 50
        # targets within ~50 pc), a single 30-arcmin cone usually captures
        # enough field stars for matching.
        import numpy as np

        ras = [t["ra"] for t in targets]
        decs = [t["dec"] for t in targets]
        median_ra = float(np.median(ras))
        median_dec = float(np.median(decs))

        # For widely separated targets, query around each target
        # and concatenate; for clustered targets, a single cone suffices.
        ra_spread = float(np.ptp(ras))
        dec_spread = float(np.ptp(decs))
        spread_deg = max(ra_spread, dec_spread)

        catalog_dicts: List[Dict[str, Any]] = []

        if spread_deg < 5.0:
            # Targets are clustered -- single wide cone search
            log.info(
                "  Targets clustered (spread=%.1f°), querying single cone "
                "at (%.2f, %.2f) r=1800\"",
                spread_deg, median_ra, median_dec,
            )
            field_df = gaia_cone_search(
                median_ra, median_dec, radius_arcsec=1800.0, top_n=500,
            )
            if field_df is not None and not field_df.empty:
                for _, row in field_df.iterrows():
                    catalog_dicts.append(row.to_dict())
        else:
            # Targets are spread out -- query per-target cones
            log.info(
                "  Targets spread (%.1f°), querying per-target cones", spread_deg,
            )
            seen_source_ids = set()
            if gaia_batch_cone_search is not None:
                # ── Batched cone search (Optimisation Step 2) ─────────
                cfg = get_config()
                _bs = cfg.get("performance", {}).get(
                    "controls_cone_batch_size", 10,
                )
                positions = [(t["ra"], t["dec"]) for t in targets]
                batch_results = gaia_batch_cone_search(
                    positions,
                    radius_arcsec=600.0,
                    top_n_per_position=100,
                    batch_size=_bs,
                )
                for i, _t in enumerate(targets):
                    field_df = batch_results.get(i)
                    if field_df is not None and not field_df.empty:
                        for _, row in field_df.iterrows():
                            sid = row.get("source_id")
                            if sid not in seen_source_ids:
                                seen_source_ids.add(sid)
                                catalog_dicts.append(row.to_dict())
            else:
                # Fallback: individual cone searches
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

        # Compute distance_pc and b_gal for matching
        for d in catalog_dicts:
            plx = d.get("parallax")
            if plx is not None and plx > 0:
                d["distance_pc"] = 1000.0 / plx
            else:
                d["distance_pc"] = None
            # Galactic latitude from RA/Dec (approximate)
            try:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                c = SkyCoord(ra=d["ra"] * u.deg, dec=d["dec"] * u.deg)
                d["b_gal"] = float(c.galactic.b.deg)
            except Exception:
                d["b_gal"] = 0.0

        # Prepare target dicts for matching (need matching features)
        target_match_dicts = []
        for t in targets:
            td = {"target_id": t["target_id"]}
            gp = t.get("gaia_params", {})
            ga = t.get("gaia_astrometry", {})
            td["phot_g_mean_mag"] = gp.get("phot_g_mean_mag")
            td["bp_rp"] = gp.get("bp_rp")
            # Compute distance_pc from Gaia parallax if not in target file
            dist = t.get("distance_pc")
            if dist is None:
                plx = ga.get("parallax") or gp.get("parallax")
                if plx is not None and plx > 0:
                    dist = 1000.0 / float(plx)
            td["distance_pc"] = dist
            # Carry source_id for control-pool exclusion (prevents
            # target from appearing in its own control cohort).
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

        # Step 2: Select matched controls
        try:
            cohort = select_matched_controls(
                target_match_dicts,
                catalog_dicts,
                n_per_target=10,
                match_on=["phot_g_mean_mag", "bp_rp", "distance_pc", "b_gal"],
                target_id_key="source_id",
            )
            log.info("  %s", cohort.summary())
            # Store matching caveats for inclusion in run summary
            if cohort.matching_caveats:
                self._matching_caveats = cohort.matching_caveats
        except Exception as exc:
            log.warning("Control selection failed: %s", exc)
            return control_scores

        if cohort.n_controls == 0:
            log.warning("No controls selected; scoring remains uncalibrated.")
            return control_scores

        # Step 3: Run IR excess detector on each control star
        # Resume from checkpoint if available
        cached_controls = self._checkpoint_data.get("control_scores", {})
        if cached_controls:
            log.info(
                "  Resuming control scores from checkpoint: "
                "IR=%d PM=%d HR=%d UV=%d radio=%d IRvar=%d",
                len(cached_controls.get("ir_excess", [])),
                len(cached_controls.get("proper_motion_anomaly", [])),
                len(cached_controls.get("hr_anomaly", [])),
                len(cached_controls.get("uv_anomaly", [])),
                len(cached_controls.get("radio_emission", [])),
                len(cached_controls.get("ir_variability", [])),
            )
            return cached_controls

        log.info("  Scoring %d control stars across 6 channels ...", cohort.n_controls)
        ir_control_scores: List[float] = []
        astro_control_scores: List[float] = []
        hr_control_scores: List[float] = []
        uv_control_scores: List[float] = []
        radio_control_scores: List[float] = []
        irv_control_scores: List[float] = []

        for ci, ctrl in enumerate(cohort.controls):
            # Check for graceful shutdown
            if self._shutdown_requested:
                log.warning("Shutdown requested at control %d/%d", ci + 1, cohort.n_controls)
                break
            ctrl_ra = ctrl.get("ra")
            ctrl_dec = ctrl.get("dec")
            if ctrl_ra is None or ctrl_dec is None:
                continue

            # Query IR photometry for control star
            # Uses a thread with 10s timeout to avoid blocking on IRSA/VizieR
            # downtime (uncached queries can take 60-120s to timeout).
            try:
                from src.ingestion.ir_surveys import get_2mass, get_wise
                from concurrent.futures import ThreadPoolExecutor as _TPE

                def _ir_control_query():
                    """Run 2MASS + WISE queries in a short-lived thread."""
                    _ir = {"ra": ctrl_ra, "dec": ctrl_dec}
                    _2m = get_2mass(ctrl_ra, ctrl_dec, radius_arcsec=5.0)
                    if _2m:
                        for b in ("J", "H", "Ks", "J_err", "H_err", "Ks_err"):
                            if b in _2m:
                                _ir[b] = _2m[b]
                    _wi = get_wise(ctrl_ra, ctrl_dec, radius_arcsec=5.0)
                    if _wi:
                        for b in ("W1", "W2", "W3", "W4",
                                  "W1_err", "W2_err", "W3_err", "W4_err"):
                            if b in _wi:
                                _ir[b] = _wi[b]
                    return _ir

                with _TPE(max_workers=1) as _pool:
                    _fut = _pool.submit(_ir_control_query)
                    ir_merged = _fut.result(timeout=10)  # 10s timeout

                # Add Gaia optical bands (always available, no network)
                g_mag = ctrl.get("phot_g_mean_mag")
                bp_mag = ctrl.get("phot_bp_mean_mag")
                rp_mag = ctrl.get("phot_rp_mean_mag")
                if g_mag is not None and np.isfinite(g_mag):
                    ir_merged["G"] = float(g_mag)
                    ir_merged["G_err"] = 0.01
                if bp_mag is not None and np.isfinite(bp_mag):
                    ir_merged["BP"] = float(bp_mag)
                    ir_merged["BP_err"] = 0.01
                if rp_mag is not None and np.isfinite(rp_mag):
                    ir_merged["RP"] = float(rp_mag)
                    ir_merged["RP_err"] = 0.01

                # Run IR excess
                ir_result = compute_ir_excess(ir_merged)
                # Convert to scorer-compatible dict and extract score
                ir_dict = {
                    "sigma_W3": ir_result.sigma_W3,
                    "sigma_W4": ir_result.sigma_W4,
                    "is_candidate": ir_result.is_candidate,
                    "excess_W3": ir_result.excess_W3,
                    "excess_W4": ir_result.excess_W4,
                }
                score = EXODUSScorer._get_ir_excess_score(ir_dict)
                ir_control_scores.append(score)
            except Exception as exc:
                log.debug("IR excess control %d failed: %s", ci, exc)

            # HZ is a prior channel (not calibrated against controls)

            # Astrometric score for controls
            ruwe = ctrl.get("ruwe", 1.0)
            if ruwe is not None and np.isfinite(ruwe):
                astro_dict = {
                    "ruwe": float(ruwe),
                    "astrometric_excess_noise_sig": 0.0,
                }
                astro_score = EXODUSScorer._get_astrometric_score(astro_dict)
                astro_control_scores.append(astro_score)

            # ── HR anomaly (zero network cost — Gaia data already in ctrl) ──
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
                    hr_control_scores.append(hr_score)
                except Exception as exc:
                    log.debug("HR anomaly control %d failed: %s", ci, exc)

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
                        uv_control_scores.append(uv_score)
                except Exception as exc:
                    log.debug("UV anomaly control %d failed: %s", ci, exc)

            # ── Radio emission (FIRST/NVSS/VLASS catalog query — 10s timeout) ──
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
                    radio_control_scores.append(re_score)
                except Exception as exc:
                    log.debug("Radio emission control %d failed: %s", ci, exc)

            # ── IR variability (NEOWISE time-series — VERY SLOW ~60-120s/star) ──
            # NEOWISE TAP queries average ~90s each; for 355 controls ≈ 9 hours.
            # Enabled only via env var EXODUS_CONTROL_NEOWISE=1 or when NEOWISE
            # data is already disk-cached for this position.
            if compute_ir_variability is not None:
                _force_neowise = os.environ.get("EXODUS_CONTROL_NEOWISE") == "1"
                try:
                    neowise_ts = None
                    if query_neowise_timeseries is not None:
                        # Check disk cache; skip 90s network query unless forced
                        _cache_key = f"neowise_ts_{ctrl_ra:.6f}_{ctrl_dec:.6f}_10.0"
                        from src.utils import load_cache as _lc
                        _cached = _lc(_cache_key)
                        if _cached is not None or _force_neowise:
                            neowise_ts = query_neowise_timeseries(
                                ctrl_ra, ctrl_dec,
                            )
                    if neowise_ts is not None:
                        irv_res = compute_ir_variability(
                            ctrl_ra, ctrl_dec,
                            neowise_data=neowise_ts,
                        )
                        irv_dict = irv_res.to_dict()
                        # Only include if real data was available (not
                        # "none"/"insufficient" — those are missing-data, not nulls)
                        ds = irv_dict.get("data_source", "none")
                        if ds not in ("none", "insufficient", "simulated"):
                            irv_score = EXODUSScorer._get_ir_variability_score(
                                irv_dict
                            )
                            irv_control_scores.append(irv_score)
                except Exception as exc:
                    log.debug("IR variability control %d failed: %s", ci, exc)

            # Checkpoint every 20 controls (more frequent for NEOWISE latency)
            if (ci + 1) % 20 == 0:
                partial = {}
                if ir_control_scores:
                    partial["ir_excess"] = ir_control_scores
                if astro_control_scores:
                    partial["proper_motion_anomaly"] = astro_control_scores
                if hr_control_scores:
                    partial["hr_anomaly"] = hr_control_scores
                if uv_control_scores:
                    partial["uv_anomaly"] = uv_control_scores
                if radio_control_scores:
                    partial["radio_emission"] = radio_control_scores
                if irv_control_scores:
                    partial["ir_variability"] = irv_control_scores
                self._save_checkpoint(
                    "controls",
                    control_scores=partial,
                    n_controls_done=ci + 1,
                    n_controls_total=cohort.n_controls,
                )
                log.info(
                    "  Control checkpoint: %d/%d scored "
                    "(IR=%d PM=%d HR=%d UV=%d radio=%d IRvar=%d)",
                    ci + 1, cohort.n_controls,
                    len(ir_control_scores), len(astro_control_scores),
                    len(hr_control_scores), len(uv_control_scores),
                    len(radio_control_scores), len(irv_control_scores),
                )

        # Build control_scores dict
        if ir_control_scores:
            control_scores["ir_excess"] = ir_control_scores
            log.info(
                "  IR excess controls: %d scores, median=%.4f",
                len(ir_control_scores), float(np.median(ir_control_scores)),
            )
        if astro_control_scores:
            control_scores["proper_motion_anomaly"] = astro_control_scores
            log.info(
                "  Astrometric controls: %d scores, median=%.4f",
                len(astro_control_scores), float(np.median(astro_control_scores)),
            )
        if hr_control_scores:
            control_scores["hr_anomaly"] = hr_control_scores
            log.info(
                "  HR anomaly controls: %d scores, median=%.4f",
                len(hr_control_scores), float(np.median(hr_control_scores)),
            )
        if uv_control_scores:
            control_scores["uv_anomaly"] = uv_control_scores
            log.info(
                "  UV anomaly controls: %d scores, median=%.4f",
                len(uv_control_scores), float(np.median(uv_control_scores)),
            )
        if radio_control_scores:
            control_scores["radio_emission"] = radio_control_scores
            log.info(
                "  Radio emission controls: %d scores, median=%.4f",
                len(radio_control_scores), float(np.median(radio_control_scores)),
            )
        if irv_control_scores:
            control_scores["ir_variability"] = irv_control_scores
            log.info(
                "  IR variability controls: %d scores, median=%.4f",
                len(irv_control_scores), float(np.median(irv_control_scores)),
            )

        log.info(
            "=== Control selection complete: %d channels calibrated ===",
            len(control_scores),
        )
        return control_scores

    # ── Score targets ────────────────────────────────────────────────

    def _score_targets(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank all targets using score_all() for FDR/ranking.

        Uses :meth:`EXODUSScorer.score_all` instead of per-target
        ``score_target()`` so that:
        - Targets are ranked globally (``rank`` field populated).
        - FDR correction (Benjamini-Hochberg) is applied across all targets.
        - ``get_top_targets()`` is available downstream.
        """
        if self.scorer is None:
            log.warning("Scorer not available; skipping.")
            return targets

        log.info("Scoring %d targets with score_all() (FDR + ranking) ...", len(targets))
        try:
            results = self.scorer.score_all(targets)
            # Attach scored results back to target dicts
            for exodus_score in results:
                # Find the matching target and attach
                for t in targets:
                    if str(t.get("target_id", "")) == exodus_score.target_id:
                        t["exodus_score"] = exodus_score.to_dict()
                        break
        except Exception as exc:
            log.warning("score_all() failed, falling back to per-target: %s", exc)
            for t in targets:
                try:
                    score = self.scorer.score_target(t)
                    t["exodus_score"] = score.to_dict()
                except Exception as exc2:
                    log.debug("Scoring failed for %s: %s", t.get("target_id"), exc2)

        return targets

    # ── Unexplainability evaluation ──────────────────────────────────

    def _evaluate_unexplainability(
        self, targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate Unexplainability Score for targets with active channels.

        For each target that has an EXODUS score with 1+ active detection
        channels, run the astrophysical template matcher and attach the
        result.  This transforms every anomaly from "this is weird" to
        "this is weird AND here's why it can't be a binary/disk/YSO."

        Only runs on Tier 1+ targets by default, or any target with
        2+ active channels at Tier 0 (to avoid wasting time on
        single-channel noise during blitz scans).
        """
        if UnexplainabilityScorer is None:
            log.debug("UnexplainabilityScorer not available; skipping.")
            return targets

        # At Tier 0, only evaluate targets with 2+ active channels
        min_channels = 2 if self.tier == 0 else 1

        # Audit fix E6: pass convergence_priority threshold (0.25) to
        # the unexplainability scorer so it matches the EXODUSScorer threshold.
        unex_threshold = 0.25 if self.convergence_priority else 0.3
        scorer = UnexplainabilityScorer(activation_threshold=unex_threshold)
        n_evaluated = 0

        for t in targets:
            exodus = t.get("exodus_score")
            if not exodus:
                continue

            n_active = exodus.get("n_active_channels", 0)
            if n_active < min_channels:
                continue

            try:
                # Extract channel scores from the EXODUS score dict
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
                log.debug(
                    "Unexplainability evaluation failed for %s: %s",
                    t.get("target_id"), exc,
                )

        if n_evaluated > 0:
            log.info(
                "Unexplainability evaluation: %d targets assessed "
                "(min %d active channels)",
                n_evaluated, min_channels,
            )

            # Log any UNEXPLAINED targets
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

    # ── Red-Team falsification ─────────────────────────────────────

    def _run_red_team(
        self, targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run Red-Team falsification engine on scored targets.

        Adds ``red_team`` dict to each target that has >= 1 active channel
        or is flagged for escalation.  The Red-Team aggressively seeks
        natural explanations — works entirely from existing pipeline data.
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
                # Only evaluate targets with at least 1 active channel
                if n_active < 1:
                    continue

                verdict = engine.evaluate(t)
                t["red_team"] = verdict.to_dict()
                n_evaluated += 1

                # Galaxy contamination check (supplements red-team)
                if check_galaxy_contamination is not None:
                    try:
                        gc = check_galaxy_contamination(t)
                        t["galaxy_contamination"] = gc.to_dict()
                        if gc.contamination_likely:
                            log.info(
                                "  Galaxy contamination LIKELY for %s: %s",
                                t.get("target_id"), gc.explanation,
                            )
                    except Exception:
                        pass

                # PM-IR correlation check
                if check_pm_ir_correlation is not None:
                    try:
                        pm_ir = check_pm_ir_correlation(t)
                        t["pm_ir_correlation"] = pm_ir
                        if pm_ir.get("pm_ir_correlated"):
                            log.info(
                                "  PM-IR correlation for %s: %s",
                                t.get("target_id"), pm_ir["explanation"],
                            )
                    except Exception:
                        pass

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

            # Log DEMOTE targets prominently
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

    # ── Hypothesis cycle (limited) ───────────────────────────────────

    def _run_hypothesis_cycle(
        self, targets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run a simplified hypothesis validation pass.

        Only tests pending hypotheses whose method is ``ir_excess_comparison``,
        since that is the only channel with data in quick mode.
        """
        summary: Dict[str, Any] = {
            "tested": 0,
            "unexplained": 0,
            "natural": 0,
            "artifact": 0,
            "inconclusive": 0,
        }

        if self.generator is None or self.analyst is None:
            return summary

        try:
            pending = self.generator.get_pending()
        except Exception:
            return summary

        if not pending:
            log.info("No pending hypotheses.")
            return summary

        # Build IR excess data for validation
        target_excess = []
        control_excess = []
        for t in targets:
            ir = t.get("ir_excess", {})
            if ir.get("sigma_W4") is not None:
                if ir.get("is_candidate"):
                    target_excess.append(float(ir["sigma_W4"]))
                else:
                    control_excess.append(float(ir["sigma_W4"]))

        ir_data = {
            "target_excess": target_excess or control_excess[:5],
            "control_excess": control_excess or [0.0] * 5,
        }

        for hyp in pending:
            method = hyp.get("method", "")

            # In quick mode, only test IR-related hypotheses
            if method != "ir_excess_comparison":
                continue

            analyst_hyp = {
                "hypothesis_id": hyp["hypothesis_id"],
                "test_method": method,
            }

            try:
                vresult = self.analyst.validate(analyst_hyp, ir_data)
                summary["tested"] += 1

                status_str = vresult.status.value if hasattr(vresult.status, "value") else str(vresult.status)

                if vresult.status == ValidationStatus.UNEXPLAINED:
                    summary["unexplained"] += 1
                    self.generator.update_status(hyp["hypothesis_id"], "confirmed", results=vresult.to_dict())
                elif vresult.status == ValidationStatus.CONFIRMED_NATURAL:
                    summary["natural"] += 1
                    self.generator.update_status(hyp["hypothesis_id"], "rejected", results=vresult.to_dict())
                elif vresult.status == ValidationStatus.CONFIRMED_ARTIFACT:
                    summary["artifact"] += 1
                    self.generator.update_status(hyp["hypothesis_id"], "rejected", results=vresult.to_dict())
                else:
                    summary["inconclusive"] += 1

                log.info(
                    "  Hypothesis %s -> %s (det=%.3f)",
                    hyp["hypothesis_id"], status_str, vresult.detection_score,
                )

            except Exception as exc:
                log.debug("Validation failed for %s: %s", hyp["hypothesis_id"], exc)

        return summary

    # ── Summary & reporting ──────────────────────────────────────────

    def _build_summary(
        self,
        targets: List[Dict[str, Any]],
        anomaly_count: int,
        hyp_summary: Dict[str, Any],
        elapsed: float,
        mm_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the final quick-run summary."""
        top_targets = []
        all_scored = []
        if self.scorer:
            try:
                # Save ALL scored targets for population analysis (R8 fix)
                all_results = self.scorer.get_top_targets(
                    n=len(self.scorer._results)
                )
                all_scored_raw = [s.to_dict() for s in all_results]
                # Enrich with distance, host_star, and vetting data from target data
                target_lookup = {t.get("target_id"): t for t in targets}
                _ENRICH_KEYS = [
                    "distance_pc", "host_star",
                    # Vetting results
                    "galaxy_contamination", "red_team", "pm_ir_correlation",
                    # Far-IR data
                    "herschel", "herschel_sed",
                    # Raw photometry for diagnostics
                    "ir_photometry", "gaia_astrometry", "gaia_params",
                    "unexplainability",
                    # IR excess raw
                    "ir_excess",
                ]
                for tt in all_scored_raw:
                    src = target_lookup.get(tt.get("target_id"), {})
                    for ek in _ENRICH_KEYS:
                        val = src.get(ek)
                        if val is not None:
                            tt[ek] = val
                all_scored = all_scored_raw
                # top_targets = first 20 (backward compatibility)
                top_targets = all_scored_raw[:20]
            except RuntimeError:
                log.warning("get_top_targets() failed: score_all() was not called.")
            except Exception as exc:
                log.debug("get_top_targets() failed: %s", exc)

        summary = {
            "project": "EXODUS",
            "mode": "quick",
            "tier": self.tier,
            "n_targets": len(targets),
            "anomaly_count": anomaly_count,
            "channels_active": [
                ch for ch in [
                    "ir_excess",
                    "proper_motion_anomaly",
                    "transit_anomaly" if any(t.get("transit_anomaly") for t in targets) else None,
                    "radio_anomaly" if any(t.get("radio_anomaly") for t in targets) else None,
                    "gaia_photometric_anomaly" if self.tier >= 1 else None,
                    "ir_variability" if any(t.get("ir_variability") for t in targets) else None,
                    "uv_anomaly" if any(t.get("uv_metrics") for t in targets) else None,
                    "radio_emission" if any(t.get("radio_continuum") for t in targets) else None,
                    "hr_anomaly" if any(t.get("hr_anomaly") for t in targets) else None,
                    "abundance_anomaly" if any(t.get("abundance_anomaly") for t in targets) else None,
                    "habitable_zone_planet",
                ]
                if ch is not None
            ],
            "channels_calibrated": list(self._control_scores.keys()) if self._control_scores else [],
            "calibration_note": (
                "FDR significance is based only on calibrated channels; "
                "uncalibrated channels are excluded from Stouffer/Fisher combination"
            ) if self._control_scores else "No controls obtained — results are UNCALIBRATED",
            "matching_caveats": getattr(self, "_matching_caveats", []),
            "multi_messenger": mm_summary or {},
            "hypothesis_results": hyp_summary,
            "top_targets": top_targets,
            "all_scored": all_scored,
            "n_scored": len(all_scored),
            "n_anomalies": sum(
                1 for s in all_scored if s.get("n_active_channels", 0) > 0
            ),
            "n_fdr_significant": sum(
                1 for s in all_scored if s.get("fdr_significant")
            ),
            "elapsed_sec": round(elapsed, 1),
            "elapsed_min": round(elapsed / 60.0, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Tier 0 escalation: flag targets with anomalies for Tier 1 follow-up
        # Audit fix N2: only count MM hits from real (non-simulated) data.
        # When network queries fail, MM catalogs fall back to simulation;
        # simulated hits must NOT drive escalation decisions.
        if self.tier == 0:
            mm_sources = summary.get("multi_messenger", {})
            _mm_real = {
                key for key in ("gamma", "neutrino", "gw", "pulsar", "frb")
                if mm_sources.get(key, {}).get("data_source") == "real"
            }
            escalation_targets = []
            for t in targets:
                has_ir = t.get("ir_excess", {}).get("is_candidate", False)
                has_astro = t.get("proper_motion_anomaly", {}).get("ruwe", 1.0) > 1.4
                # Only count MM hits where the parent catalog was real data
                mm_data = t.get("multi_messenger", {})
                has_mm = any(
                    bool(mm_data.get(key)) for key in _mm_real
                )
                if has_ir or has_astro or has_mm:
                    escalation_targets.append(t["target_id"])
            summary["tier0_escalations"] = escalation_targets
            if escalation_targets:
                log.info("Tier 0 flagged %d targets for escalation: %s",
                         len(escalation_targets), escalation_targets)
            else:
                log.info("Tier 0: no targets flagged for escalation")

        # Attach unexplainability summary for top targets
        unexplained_targets = []
        for t in targets:
            unex = t.get("unexplainability", {})
            if unex.get("classification") in ("UNEXPLAINED", "PARTIALLY_EXPLAINED"):
                unexplained_targets.append({
                    "target_id": t.get("target_id"),
                    "unexplainability_score": unex.get("unexplainability_score"),
                    "classification": unex.get("classification"),
                    "best_template": unex.get("best_template"),
                    "residual_channels": unex.get("residual_channels", []),
                })
        if unexplained_targets:
            summary["unexplained_targets"] = unexplained_targets

        # Attach Red-Team falsification summary
        red_team_results = []
        for t in targets:
            rt = t.get("red_team", {})
            if rt:
                red_team_results.append({
                    "target_id": rt.get("target_id"),
                    "overall_risk": rt.get("overall_risk"),
                    "risk_level": rt.get("risk_level"),
                    "recommendation": rt.get("recommendation"),
                    "n_risk_flags": rt.get("n_risk_flags"),
                    "top_risk": rt.get("top_risk"),
                    "natural_explanations": rt.get("natural_explanations", [])[:3],
                })
        if red_team_results:
            n_escalate = sum(1 for r in red_team_results if r["recommendation"] == "ESCALATE")
            n_demote = sum(1 for r in red_team_results if r["recommendation"] == "DEMOTE")
            n_monitor = len(red_team_results) - n_escalate - n_demote
            summary["red_team"] = {
                "n_evaluated": len(red_team_results),
                "n_escalate": n_escalate,
                "n_monitor": n_monitor,
                "n_demote": n_demote,
                "results": sorted(red_team_results, key=lambda r: r.get("overall_risk", 0)),
            }

        return summary

    def _save_provenance(
        self,
        summary: Dict[str, Any],
        targets: List[Dict[str, Any]],
        scored: List[Any],
    ) -> None:
        """Create provenance manifest and evidence bundles for escalated targets."""
        # --- Provenance manifest ---
        if provenance_logger is not None:
            try:
                config = {
                    "max_targets": self.max_targets,
                    "tier": self.tier,
                    "run_hypotheses": self.run_hypotheses,
                    "target_file": str(self.target_file) if self.target_file else None,
                }
                manifest = provenance_logger.create_manifest(
                    config=config,
                    seed=0,
                    n_targets=len(targets),
                    thresholds=self.thresholds,
                    notes=f"QuickRunner tier={self.tier}, {len(targets)} targets",
                )
                provenance_logger.save_manifest(manifest)
                provenance_logger.save_query_log(manifest.run_id)
                log.info("Provenance manifest saved: run_id=%s", manifest.run_id)

                # --- Evidence bundles for escalated targets ---
                if EvidenceBundle is not None and save_evidence_bundle is not None:
                    escalations = summary.get("tier0_escalations", [])
                    unexplained = [
                        t.get("target_id")
                        for t in targets
                        if t.get("unexplainability", {}).get("classification")
                        in ("UNEXPLAINED", "PARTIALLY_EXPLAINED")
                    ]
                    bundle_targets = set(escalations) | set(unexplained)

                    for tid in bundle_targets:
                        # Find target data (scored is List[Dict], not EXODUSScore objects)
                        tdata = next((t for t in scored if t.get("target_id") == tid), None)
                        if tdata is None:
                            continue
                        # Extract channel scores from the exodus_score dict attached to target
                        exodus_dict = tdata.get("exodus_score", {})

                        bundle = EvidenceBundle(
                            target_id=tid,
                            run_id=manifest.run_id,
                            channel_results=exodus_dict.get("channel_scores", {}),
                            multi_messenger_results=tdata.get("multi_messenger", {}),
                            raw_data_refs=[],
                            query_provenance=[q.to_dict() for q in provenance_logger.get_queries()],
                            plots=[],
                            detector_versions={},
                            breakthrough_level=1 if tid in escalations else 0,
                            notes=f"Tier {self.tier} escalation" if tid in escalations else "Unexplained target",
                        )
                        save_evidence_bundle(bundle)
                        log.info("Evidence bundle saved for %s", tid)

                    if bundle_targets:
                        log.info("Saved %d evidence bundles", len(bundle_targets))

                provenance_logger.clear()
            except Exception as exc:
                log.warning("Provenance/evidence save failed: %s", exc)

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
        # CX-7 fix: iterate both top_targets and all_scored (not "targets")
        for list_key in ("top_targets", "all_scored"):
            for target in summary.get(list_key, []):
                mm = target.get("multi_messenger", {})
                for key in list(mm.keys()):
                    if isinstance(mm[key], dict) and mm[key].get("data_source") == "simulation":
                        del mm[key]
                        stripped += 1
        return stripped

    def _save_report(self, summary: Dict[str, Any]) -> None:
        """Save a JSON report to the reports directory."""
        import json
        # CX-7: strip simulated MM data before saving publication-ready report
        n_stripped = self._strip_simulated_mm(summary)
        if n_stripped:
            log.info("Stripped %d simulated multi-messenger entries from report", n_stripped)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.REPORTS_DIR / f"quick_run_{ts}.json"
        with open(path, "w") as f:
            safe_json_dump(summary, f, indent=2)
        log.info("Quick run report saved: %s", path)


# =====================================================================
#  CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project EXODUS -- Quick Research Run (<1 hour on laptop)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_quick.py                       # default: 20 targets from archive
  python scripts/run_quick.py --targets 10          # fewer targets
  python scripts/run_quick.py --target-file t.json  # all targets in file (no limit)
  python scripts/run_quick.py --no-hypotheses       # skip hypothesis cycle
        """,
    )
    parser.add_argument(
        "--targets", "-n",
        type=int,
        default=None,
        help="Maximum number of targets to process. "
             "Default: 20 when using Exoplanet Archive, "
             "unlimited when --target-file is provided.",
    )
    parser.add_argument(
        "--no-hypotheses",
        action="store_true",
        help="Skip the hypothesis generation/validation cycle.",
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
        "--tier",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Scanning depth: 0=fast triage (~25s/target, IR+astrometry+MM batch), "
             "1=standard with lightcurves (~150s/target), "
             "2=deep dive with radio+all channels (full pipeline). Default: 1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint. Skips targets that were already gathered "
             "and restores control scores from a previous (interrupted) run.",
    )
    parser.add_argument(
        "--convergence-priority",
        action="store_true",
        help="Enable convergence-priority scoring mode. Lowers activation "
             "threshold to 0.25 and amplifies multi-channel convergence "
             "bonus (4^(n-1) vs 2^(n-1)). Single-channel results are "
             "penalised to 25%%. Designed to detect weak-but-convergent "
             "signals (Type I-1.5 civilisations, partial structures).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve target count: default 20 for archive, unlimited for target files
    max_targets = args.targets
    if max_targets is None:
        max_targets = 999_999 if args.target_file else 20

    runner = QuickRunner(
        max_targets=max_targets,
        run_hypotheses=not args.no_hypotheses,
        target_file=args.target_file,
        tier=args.tier,
        resume=args.resume,
        convergence_priority=args.convergence_priority,
    )

    summary = runner.run()

    # Print results
    print("\n" + "=" * 60)
    print("  EXODUS QUICK RUN COMPLETE")
    print("=" * 60)
    print(f"  Targets:    {summary.get('n_targets', 0)}")
    n_scored = summary.get("n_scored", 0)
    n_anom = summary.get("n_anomalies", summary.get("anomaly_count", 0))
    n_fdr = summary.get("n_fdr_significant", 0)
    print(f"  Scored:     {n_scored} ({n_anom} anomalies, {n_fdr} FDR significant)")
    print(f"  Elapsed:    {summary.get('elapsed_min', 0):.1f} min")
    print(f"  Channels:   {', '.join(summary.get('channels_active', []))}")

    hyp = summary.get("hypothesis_results", {})
    if hyp.get("tested", 0) > 0:
        print(f"\n  Hypotheses tested:     {hyp['tested']}")
        print(f"  Unexplained:           {hyp.get('unexplained', 0)}")
        print(f"  Natural/Artifact:      {hyp.get('natural', 0) + hyp.get('artifact', 0)}")

    # Multi-messenger summary
    mm = summary.get("multi_messenger", {})
    mm_hits = []
    for key in ("gamma", "neutrino", "gw", "pulsar", "frb", "temporal_archaeology"):
        val = mm.get(key, {})
        if isinstance(val, dict) and not val.get("error"):
            count = val.get("n_coincidences", val.get("n_high_priority", 0))
            if count and count > 0:
                mm_hits.append(f"{key.replace('_', ' ')}={count}")
    if mm_hits:
        print(f"  MM hits:    {', '.join(mm_hits)}")

    # Unexplainability summary
    unex_list = summary.get("unexplained_targets", [])
    if unex_list:
        n_unexplained = sum(1 for u in unex_list if u.get("classification") == "UNEXPLAINED")
        n_partial = sum(1 for u in unex_list if u.get("classification") == "PARTIALLY_EXPLAINED")
        print(f"  Unexplained: {n_unexplained} UNEXPLAINED, {n_partial} PARTIAL")

    # Red-Team falsification summary
    rt = summary.get("red_team", {})
    if rt.get("n_evaluated", 0) > 0:
        print(
            f"  Red-Team:   {rt['n_evaluated']} assessed — "
            f"{rt.get('n_escalate', 0)} ESCALATE, "
            f"{rt.get('n_monitor', 0)} MONITOR, "
            f"{rt.get('n_demote', 0)} DEMOTE"
        )

    # Top targets
    top = summary.get("top_targets", [])
    if top:
        print(f"\n  Top {min(len(top), 10)} Targets:")
        for i, t in enumerate(top[:10]):
            dist = t.get("distance_pc")
            dist_str = f"{dist:.1f}pc" if dist else "?"
            fdr_str = " FDR*" if t.get("fdr_significant") else ""
            print(
                f"    {i+1:>2}. {t.get('target_id', 'N/A'):<28s}  "
                f"score={t.get('total_score', 0):.4f}  "
                f"ch={t.get('n_active_channels', 0)}  "
                f"d={dist_str}{fdr_str}"
            )

    # Escalation flags
    esc = summary.get("tier0_escalations", [])
    if esc:
        print(f"\n  Escalation flags: {len(esc)} targets flagged for Tier 1")

    print("=" * 60)


if __name__ == "__main__":
    main()
