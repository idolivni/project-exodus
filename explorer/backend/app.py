"""
EXODUS Galaxy Explorer — FastAPI Backend (Phase 1 MVP)

Serves EXODUS pipeline data as REST endpoints for the sky explorer.
Includes checkpoint-aware endpoints for live pipeline monitoring.

Usage:
    cd explorer/backend
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import json
import glob
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="EXODUS Galaxy Explorer API",
    version="0.2.0",
    description="REST API for the EXODUS multi-channel technosignature sky explorer",
)

app.add_middleware(
    CORSMiddleware,
    # CX-10: restrict to local development origins
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ── Helpers ──────────────────────────────────────────────────


def _sanitize(val: Any) -> Any:
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize(v) for v in val]
    return val


def _load_latest_report() -> Optional[Dict[str, Any]]:
    """Load the most recent quick_run report."""
    reports = sorted(glob.glob(str(DATA_DIR / "reports" / "quick_run_*.json")))
    if not reports:
        return None
    with open(reports[-1]) as f:
        return json.load(f)


def _load_all_reports() -> List[Dict[str, Any]]:
    """Load all quick_run reports, newest first."""
    reports = sorted(glob.glob(str(DATA_DIR / "reports" / "quick_run_*.json")), reverse=True)
    results = []
    for rpath in reports:
        try:
            with open(rpath) as f:
                data = json.load(f)
            data["_filepath"] = rpath
            results.append(data)
        except Exception:
            pass
    return results


def _load_target_file(name: str) -> Optional[Dict[str, Any]]:
    """Load a target file by name."""
    path = DATA_DIR / "targets" / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_catalog_cache(name: str) -> Optional[Any]:
    """Load a cached catalog (Fermi, IceCube, etc.)."""
    for subdir in ["catalogs", "results"]:
        path = DATA_DIR / subdir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def _load_active_checkpoints() -> List[Dict[str, Any]]:
    """Load all checkpoint files, sorted by modification time (newest first)."""
    cp_dir = DATA_DIR / "checkpoints"
    if not cp_dir.exists():
        return []
    results = []
    for subdir in cp_dir.iterdir():
        if not subdir.is_dir():
            continue
        cp_file = subdir / "checkpoint.json"
        if cp_file.exists():
            try:
                with open(cp_file) as f:
                    data = json.load(f)
                data["_checkpoint_id"] = subdir.name
                data["_mtime"] = os.path.getmtime(cp_file)
                results.append(data)
            except Exception:
                pass
    return sorted(results, key=lambda x: x.get("_mtime", 0), reverse=True)


def _extract_target_from_gathered(target_id: str, g: Dict) -> Dict[str, Any]:
    """Extract a displayable target from a gathered-data dict.

    Checkpoint format: gathered_targets is a dict of {target_id: {gaia_astrometry: {...}, ...}}.
    """
    gaia = g.get("gaia_astrometry", {}) or {}
    ir = g.get("ir_photometry", {}) or {}

    # Extract coordinates from Gaia astrometry
    ra = gaia.get("ra")
    dec = gaia.get("dec")

    # Distance from parallax
    distance_pc = None
    parallax = gaia.get("parallax")
    if parallax and parallax > 0:
        distance_pc = 1000.0 / parallax

    return _sanitize({
        "id": target_id,
        "ra": ra,
        "dec": dec,
        "distance_pc": distance_pc,
        "host_star": gaia.get("designation"),
        "hz_flag": False,
        "status": "gathered",
        "has_ir_data": bool(ir),
        "has_gaia_data": bool(gaia),
        "has_mm_data": bool(g.get("mm_matches")),
    })


# IDs of prime candidates whose deep vetting (ZTF, X-ray, TESS, SED)
# supersedes the pipeline's automated template matcher. These are exempt
# from red-team unexplainability overrides — the channel-pattern heuristic
# assigns their scores instead.
# (Populated lazily from CANDIDATES list at module load time — see bottom of file.)
_CANDIDATE_SURVIVOR_IDS: set = set()


def _build_red_team_index(report: Dict) -> Dict[str, Dict]:
    """Build a lookup from target_id → red team assessment.

    Red team results live in report["red_team"]["results"] as a list,
    separate from the target entries. This builds an O(1) lookup so we
    can attach the real unexplainability/risk to each target.
    """
    rt = report.get("red_team", {})
    if not rt:
        return {}
    results = rt.get("results", [])
    if not isinstance(results, list):
        return {}
    index: Dict[str, Dict] = {}
    for entry in results:
        if isinstance(entry, dict):
            tid = entry.get("target_id")
            if tid:
                index[tid] = entry
    return index


def _enrich_target_with_red_team(t: Dict, rt_index: Dict[str, Dict]) -> Dict:
    """Attach red team assessment fields to a target entry if available.

    This bridges the gap between the separate red_team section and
    individual target entries so _format_scored_target picks them up.

    EXCEPTION: Prime candidates (CANDIDATES list with verdict='SURVIVES')
    are exempt from pipeline red-team overrides. Their template-matcher
    scores are superseded by deeper manual vetting (ZTF, X-ray, TESS, SED).
    For these, we let the channel-pattern heuristic in _estimate_unexplainability
    assign the value instead.
    """
    tid = t.get("target_id")
    if not tid or not rt_index:
        return t

    # Don't let pipeline template matcher override deeply-vetted candidates
    if tid in _CANDIDATE_SURVIVOR_IDS:
        return t

    rt = rt_index.get(tid)
    if not rt:
        return t
    # Only set if not already present on the target
    if t.get("red_team_risk_level") is None and rt.get("risk_level"):
        t = {**t, "red_team_risk_level": rt["risk_level"]}
    if t.get("unexplainability_score") is None:
        # Extract from natural_explanations text (format: "unexplainability=X.XXX")
        for expl in rt.get("natural_explanations", []):
            if "unexplainability=" in str(expl):
                try:
                    val = float(str(expl).split("unexplainability=")[1].split(")")[0].split(",")[0])
                    t = {**t, "unexplainability_score": val}
                    break
                except (ValueError, IndexError):
                    pass
    return t


def _estimate_unexplainability(t: Dict) -> Optional[float]:
    """Estimate unexplainability from channel activation patterns.

    Uses known astrophysical patterns to assign an explainability heuristic
    when the full pipeline red-team analysis hasn't been run.

    Returns 0.0 (fully explained) to 1.0 (completely unexplained), or None.
    Known patterns:
      - PM+HR (no IR) → M-dwarf binary (very common) → 0.05
      - PM+UV+HR → M-dwarf binary → 0.05
      - PM only or single channel → noise/marginal → 0.10
      - IR+PM (no radio, no UV) → possible binary or debris → 0.30
      - IR+PM+HR (no radio, no UV) → binary pattern → 0.15
      - IR+PM+radio → highly unusual → 0.85
      - IR+PM+UV (FGK star) → rare, IR-selected FGK pattern → 0.75
      - Extreme RUWE (>4) → strong binary indicator → lowers unexplainability
    Note: habitable_zone_planet is a contextual prior, not an independent
    detection, so it does NOT count toward multi-channel convergence.
    """
    # Collect pipeline unexplainability from all sources.
    # We do NOT short-circuit: the heuristic may give a LOWER (more explained)
    # value than the pipeline, so we always compute both and take min().
    # Candidate survivors are exempt — deep vetting supersedes the automated
    # template matcher, so they always use the heuristic only.
    pipeline_unex = None
    tid = t.get("target_id")
    if tid not in _CANDIDATE_SURVIVOR_IDS:
        # Source 1: flat field (set by _enrich_target_with_red_team from natural_explanations)
        if t.get("unexplainability_score") is not None:
            pipeline_unex = t["unexplainability_score"]
        # Source 2: nested dict from pipeline template matcher
        unex_dict = t.get("unexplainability")
        if isinstance(unex_dict, dict) and unex_dict.get("unexplainability_score") is not None:
            nested_val = unex_dict["unexplainability_score"]
            pipeline_unex = min(pipeline_unex, nested_val) if pipeline_unex is not None else nested_val

    # NOTE: We intentionally do NOT map red_team_risk_level → unexplainability.
    # The red team's "risk_level" refers to risk of false positive detection,
    # not to whether the signal is astrophysically explained. E.g. a single-channel
    # PM target gets risk=LOW (real detection) but unexplainability should be ~0.10
    # (trivially explained). The channel-pattern heuristic below handles this correctly.

    channels = t.get("channel_scores", {})
    if not channels:
        channels = t.get("channel_details", {})
    if not channels:
        return pipeline_unex  # no channels → use pipeline if available

    active = set()
    for ch_name, ch_data in channels.items():
        if isinstance(ch_data, dict) and ch_data.get("is_active", ch_data.get("active", False)):
            active.add(ch_name)

    # HZ prior is a contextual boost, not an independent detection.
    # Don't count it toward multi-channel convergence patterns.
    PRIOR_CHANNELS = {"habitable_zone_planet", "hz_prior"}
    detection_channels = active - PRIOR_CHANNELS
    n_detect = len(detection_channels)
    n_active = len(active)

    if n_detect == 0:
        return pipeline_unex  # no detections → use pipeline if available

    has_ir = "ir_excess" in detection_channels
    has_pm = "proper_motion_anomaly" in detection_channels or "pm_anomaly" in detection_channels
    has_hr = "hr_anomaly" in detection_channels
    has_uv = "uv_anomaly" in detection_channels
    has_radio = "radio_anomaly" in detection_channels or "radio_emission" in detection_channels
    has_gaia = "gaia_photometric_anomaly" in detection_channels or "gaia_photometric" in detection_channels

    # Check for extreme RUWE — strong binary indicator
    ruwe = None
    pm_ch = channels.get("proper_motion_anomaly") or channels.get("pm_anomaly")
    if isinstance(pm_ch, dict):
        details = pm_ch.get("details", {})
        if isinstance(details, dict):
            ruwe = details.get("ruwe")
    extreme_ruwe = ruwe is not None and ruwe > 4.0

    # ── Channel-pattern heuristic ──
    # Uses known astrophysical patterns to estimate explainability.
    heuristic = None

    # Single detection channel — likely noise or marginal
    if n_detect == 1:
        heuristic = 0.20 if has_ir else 0.10
    # Known binary patterns (very well explained)
    elif has_pm and has_hr and not has_ir and not has_radio:
        heuristic = 0.05  # Classic M-dwarf binary
    elif has_pm and has_uv and has_hr and not has_ir:
        heuristic = 0.05  # M-dwarf binary with UV
    # PM + Gaia photometric (common astrometric binary)
    elif has_pm and has_gaia and not has_ir and not has_radio:
        heuristic = 0.08
    # Two detection channels without IR — usually explained
    elif n_detect == 2 and not has_ir:
        heuristic = 0.15
    # IR + PM only (could be binary or genuine excess)
    elif has_ir and has_pm and n_detect == 2:
        heuristic = 0.15 if extreme_ruwe else 0.35
    # IR + PM + HR (common M-dwarf binary pattern — well explained)
    elif has_ir and has_pm and has_hr and not has_radio and not has_uv:
        heuristic = 0.15
    # IR + radio (unusual — synchrotron?)
    elif has_ir and has_radio:
        heuristic = 0.85 if n_detect >= 3 else 0.70
    # IR + PM + UV (rare for FGK stars, no binary template match)
    elif has_ir and has_pm and has_uv:
        heuristic = 0.75
    # 3+ detection channels with IR but not matching known patterns
    elif has_ir and n_detect >= 3:
        heuristic = 0.60
    # Default: moderate uncertainty
    elif n_detect >= 3:
        heuristic = 0.45
    else:
        heuristic = 0.25

    # Combine: use min(heuristic, pipeline) when both available.
    # The heuristic encodes astrophysical knowledge (e.g. IRvar+HR = noise),
    # while the pipeline template matcher may over-flag targets when its
    # 9-template library lacks a matching natural template.
    # min() ensures we take the MORE explained estimate — either the pipeline
    # found a partial template match, OR the heuristic recognizes the pattern.
    if pipeline_unex is not None:
        return min(heuristic, pipeline_unex)
    return heuristic


def _format_scored_target(t: Dict) -> Dict[str, Any]:
    """Format a scored target for the API response."""
    channel_details = t.get("channel_details")
    if not channel_details and "channel_scores" in t:
        channel_details = {}
        for ch_name, ch_data in t.get("channel_scores", {}).items():
            if isinstance(ch_data, dict):
                channel_details[ch_name] = {
                    "score": ch_data.get("score", 0),
                    "active": ch_data.get("is_active", False),
                    "calibrated_p": ch_data.get("calibrated_p"),
                    "details": ch_data.get("details"),
                }

    # Compute or retrieve unexplainability score
    unex = _estimate_unexplainability(t)
    # Derive red-team risk from unexplainability if not available
    risk = t.get("red_team_risk_level")
    if risk is None and unex is not None:
        if unex < 0.15:
            risk = "HIGH"  # highly explained = high confidence it's natural
        elif unex < 0.30:
            risk = "MODERATE"
        elif unex < 0.60:
            risk = "LOW"  # low risk = low chance it's natural = interesting
        else:
            risk = None  # truly unexplained

    return _sanitize({
        "id": t.get("target_id"),
        "ra": t.get("ra"),
        "dec": t.get("dec"),
        "distance_pc": t.get("distance_pc"),
        "host_star": t.get("host_star"),
        "total_score": t.get("total_score"),
        "n_active_channels": t.get("n_active_channels"),
        "stouffer_p": t.get("stouffer_p"),
        "fdr_significant": t.get("fdr_significant"),
        "q_value": t.get("q_value"),
        "unexplainability_score": unex,
        "red_team_risk": risk,
        "hz_flag": t.get("hz_flag"),
        "channel_details": channel_details,
        "status": "scored",
    })


# ── Endpoints ────────────────────────────────────────────────


@app.get("/")
def root():
    return {"project": "EXODUS Galaxy Explorer", "version": "0.2.0", "status": "ok"}


@app.get("/api/targets")
def get_targets(campaign: str = "exodus_500"):
    """Get target list with coordinates for sky overlay."""
    data = _load_target_file(f"{campaign}.json")
    if not data:
        return {"error": f"Campaign '{campaign}' not found"}

    targets = data.get("targets", [])
    return {
        "campaign": data.get("campaign"),
        "description": data.get("description"),
        "count": len(targets),
        "targets": [
            {
                "id": t.get("target_id"),
                "ra": t.get("ra"),
                "dec": t.get("dec"),
                "distance_pc": t.get("distance_pc"),
                "hz_flag": t.get("hz_flag", False),
            }
            for t in targets
        ],
    }


@app.get("/api/campaigns")
def list_campaigns():
    """List available target campaigns."""
    target_dir = DATA_DIR / "targets"
    campaigns = []
    for f in sorted(target_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            campaigns.append({
                "filename": f.name,
                "campaign": data.get("campaign", f.stem),
                "description": data.get("description", ""),
                "n_targets": len(data.get("targets", [])),
            })
        except Exception:
            pass
    return {"campaigns": campaigns}


@app.get("/api/reports")
def list_reports():
    """List all available run reports."""
    reports = []
    for rpath in sorted(glob.glob(str(DATA_DIR / "reports" / "quick_run_*.json")), reverse=True):
        try:
            with open(rpath) as f:
                data = json.load(f)
            reports.append({
                "filename": Path(rpath).name,
                "timestamp": data.get("timestamp"),
                "n_targets": data.get("n_targets"),
                "anomaly_count": data.get("anomaly_count"),
                "tier": data.get("tier"),
                "n_scored": len(data.get("all_scored", data.get("top_targets", []))),
            })
        except Exception:
            pass
    return {"reports": reports}


@app.get("/api/report/latest")
def get_latest_report():
    """Get the latest quick_run report (scored results)."""
    report = _load_latest_report()
    if not report:
        return {"error": "No reports found"}
    return {
        "n_targets": report.get("n_targets"),
        "anomaly_count": report.get("anomaly_count"),
        "channels_active": report.get("channels_active"),
        "channels_calibrated": report.get("channels_calibrated"),
        "calibration_note": report.get("calibration_note"),
        "multi_messenger": report.get("multi_messenger"),
        "top_targets": report.get("top_targets", []),
        "red_team": report.get("red_team"),
        "tier0_escalations": report.get("tier0_escalations"),
        "timestamp": report.get("timestamp"),
        "elapsed_min": report.get("elapsed_min"),
    }


@app.get("/api/report/scored")
def get_scored_targets():
    """Get ALL scored target details for sky overlay."""
    report = _load_latest_report()
    if not report:
        return {"error": "No reports found", "targets": [], "count": 0}

    # Prefer all_scored (full population); fall back to top_targets (legacy)
    scored = report.get("all_scored", report.get("top_targets", []))
    return {
        "count": len(scored),
        "report_timestamp": report.get("timestamp"),
        "tier": report.get("tier"),
        "targets": [_format_scored_target(t) for t in scored],
    }


@app.get("/api/report/{filename}")
def get_report_by_name(filename: str):
    """Load a specific report by filename."""
    # Security: reject path traversal attempts
    if "/" in filename or "\\" in filename or ".." in filename:
        return {"error": "Invalid filename"}
    if not filename.endswith(".json"):
        return {"error": "Only .json report files are supported"}
    reports_dir = (DATA_DIR / "reports").resolve()
    path = (reports_dir / filename).resolve()
    if not path.is_relative_to(reports_dir):
        return {"error": "Invalid filename"}
    if not path.exists():
        return {"error": f"Report '{filename}' not found"}
    with open(path) as f:
        data = json.load(f)
    scored = data.get("all_scored", data.get("top_targets", []))
    return {
        "count": len(scored),
        "report_timestamp": data.get("timestamp"),
        "tier": data.get("tier"),
        "targets": [_format_scored_target(t) for t in scored],
    }


@app.get("/api/target/{target_id}")
def get_target_detail(target_id: str):
    """Get full detail for a single target from any available report."""
    for report in _load_all_reports():
        rt_index = _build_red_team_index(report)
        # Search all_scored first (full population), fall back to top_targets
        scored = report.get("all_scored", report.get("top_targets", []))
        for t in scored:
            if t.get("target_id") == target_id:
                t_enriched = _enrich_target_with_red_team(t, rt_index)
                result = _format_scored_target(t_enriched)
                result["_from_report"] = Path(report.get("_filepath", "")).name
                return result
    return {"error": f"Target '{target_id}' not found in any report"}


@app.get("/api/checkpoints")
def get_checkpoints():
    """List active checkpoints (live pipeline runs)."""
    checkpoints = _load_active_checkpoints()
    return {
        "count": len(checkpoints),
        "checkpoints": [
            {
                "id": cp.get("_checkpoint_id"),
                "phase": cp.get("phase"),
                "n_gathered": cp.get("n_gathered", 0),
                "n_total": cp.get("n_total"),
                "n_controls_done": cp.get("n_controls_done"),
                "n_controls_total": cp.get("n_controls_total"),
                "timestamp": cp.get("timestamp"),
            }
            for cp in checkpoints
        ],
    }


@app.get("/api/checkpoint/{checkpoint_id}/targets")
def get_checkpoint_targets(checkpoint_id: str):
    """Get gathered targets from a checkpoint for live sky overlay."""
    cp_file = DATA_DIR / "checkpoints" / checkpoint_id / "checkpoint.json"
    if not cp_file.exists():
        return {"error": f"Checkpoint '{checkpoint_id}' not found"}

    with open(cp_file) as f:
        data = json.load(f)

    gathered = data.get("gathered_targets", {})
    # gathered_targets can be dict (target_id -> data) or list
    if isinstance(gathered, dict):
        targets = [_extract_target_from_gathered(tid, gdata) for tid, gdata in gathered.items()]
    else:
        targets = []
    return {
        "checkpoint_id": checkpoint_id,
        "phase": data.get("phase"),
        "n_gathered": data.get("n_gathered", len(targets)),
        "n_total": data.get("n_total"),
        "targets": targets,
    }


@app.get("/api/sky/all")
def get_all_sky_targets():
    """Unified sky overlay: merge scored reports + active checkpoints.

    This is the main endpoint the Galaxy Explorer uses to populate the map.
    Returns a combined view of:
    - Scored targets from completed reports (with full EXODUS scores)
    - Gathered targets from in-progress checkpoints (positions only)
    """
    seen_ids: set[str] = set()
    all_targets: list[Dict[str, Any]] = []

    # 1. Scored targets from reports (highest priority)
    # Audit fix E2: read all_scored (full list) first, fall back to top_targets.
    # Previously only top_targets were loaded, missing lower-ranked scored targets.
    for report in _load_all_reports():
        rt_index = _build_red_team_index(report)
        scored_list = report.get("all_scored", report.get("top_targets", []))
        for t in scored_list:
            tid = t.get("target_id")
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                t_enriched = _enrich_target_with_red_team(t, rt_index)
                fmt = _format_scored_target(t_enriched)
                fmt["_source"] = "report"
                all_targets.append(fmt)

    # 2. Gathered targets from active checkpoints (lower priority)
    for cp in _load_active_checkpoints():
        gathered = cp.get("gathered_targets", {})
        if isinstance(gathered, dict):
            for tid, gdata in gathered.items():
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    fmt = _extract_target_from_gathered(tid, gdata)
                    fmt["_source"] = "checkpoint"
                    all_targets.append(fmt)

    n_scored = sum(1 for t in all_targets if t.get("status") == "scored")
    n_gathered = sum(1 for t in all_targets if t.get("status") == "gathered")

    return {
        "count": len(all_targets),
        "n_scored": n_scored,
        "n_gathered": n_gathered,
        "targets": all_targets,
    }


@app.get("/api/catalogs/fermi")
def get_fermi_sources(unid_only: bool = False):
    """Get Fermi 4FGL sources for overlay."""
    # Try direct cache name first, then look in fermi_catalog subdirectory
    data = _load_catalog_cache("fermi_4fgl_cache.json")
    if not data:
        fermi_dir = DATA_DIR / "cache" / "fermi_catalog"
        if fermi_dir.exists():
            for f in sorted(fermi_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                with open(f) as fh:
                    data = json.load(fh)
                break
    if not data:
        return {"error": "Fermi catalog not cached", "count": 0, "sources": []}

    sources = data if isinstance(data, list) else data.get("sources", [])
    if unid_only:
        sources = [s for s in sources if s.get("is_unidentified") or not s.get("source_class")]
    # Normalize field names for frontend
    normalized = []
    for s in sources:
        normalized.append({
            "source_name": s.get("source_name"),
            "ra": s.get("ra"),
            "dec": s.get("dec"),
            "class1": s.get("source_class", ""),
            "signif_avg": s.get("variability_index"),
        })
    return {"count": len(normalized), "sources": normalized}


@app.get("/api/pipeline/status")
def get_pipeline_status():
    """Get current pipeline run status from checkpoints."""
    status_file = DATA_DIR / "pipeline_status.json"
    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)

    # Fall back to reading active checkpoints
    checkpoints = _load_active_checkpoints()
    for cp in checkpoints:
        phase = cp.get("phase", "")
        if phase not in ("", "completed", "gather_interrupted"):
            n_gathered = cp.get("n_gathered", 0)
            n_total = cp.get("n_total", 0)
            n_ctrl = cp.get("n_controls_done", 0)
            n_ctrl_total = cp.get("n_controls_total", 0)

            if phase == "controls":
                msg = f"Control calibration: {n_ctrl}/{n_ctrl_total}"
                progress = n_ctrl
                total = n_ctrl_total
            elif phase == "gathering":
                msg = f"Gathering data: {n_gathered}/{n_total}"
                progress = n_gathered
                total = n_total
            else:
                msg = f"Phase: {phase}"
                progress = n_gathered
                total = n_total

            return {
                "running": True,
                "message": msg,
                "phase": phase,
                "progress": progress,
                "total": total,
                "checkpoint_id": cp.get("_checkpoint_id"),
            }

    return {"running": False, "message": "No active pipeline run"}


@app.get("/api/search/{query}")
def search_targets(query: str):
    """Search for targets by ID or host star name (substring match, deduped)."""
    query_lower = query.lower()
    results = []
    seen_ids: set[str] = set()

    for report in _load_all_reports():
        # F-07 fix: search all_scored (full population), not just top_targets
        scored_list = report.get("all_scored", report.get("top_targets", []))
        for t in scored_list:
            tid = t.get("target_id") or ""
            if tid in seen_ids:
                continue
            host = (t.get("host_star") or "").lower()
            if query_lower in tid.lower() or query_lower in host:
                seen_ids.add(tid)
                results.append(_format_scored_target(t))
                if len(results) >= 20:
                    return {"count": len(results), "results": results, "truncated": True}

    for cp in _load_active_checkpoints():
        gathered = cp.get("gathered_targets", {})
        if isinstance(gathered, dict):
            for tid, gdata in gathered.items():
                if tid in seen_ids:
                    continue
                gaia = (gdata or {}).get("gaia_astrometry", {}) or {}
                host = (gaia.get("designation") or "").lower()
                if query_lower in tid.lower() or query_lower in host:
                    seen_ids.add(tid)
                    results.append(_extract_target_from_gathered(tid, gdata))
                    if len(results) >= 20:
                        return {"count": len(results), "results": results, "truncated": True}

    return {"count": len(results), "results": results, "truncated": False}


@app.get("/api/catalogs/icecube")
def get_icecube_events(top_n: int = Query(500, le=5000)):
    """Get IceCube neutrino events for overlay (highest energy first)."""
    cache_dir = DATA_DIR / "cache" / "icecube"
    if not cache_dir.exists():
        return {"error": "IceCube catalog not cached", "count": 0, "events": []}

    events: list = []
    for f in cache_dir.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                events = data
                break
        except Exception:
            pass

    if not events:
        return {"count": 0, "events": []}

    # Sort by energy (highest first) and take top N
    events_sorted = sorted(events, key=lambda e: e.get("energy_gev", 0), reverse=True)
    top_events = events_sorted[:top_n]
    return {
        "count": len(top_events),
        "total_in_catalog": len(events),
        "events": [
            _sanitize({
                "ra": e.get("ra"),
                "dec": e.get("dec"),
                "energy_gev": e.get("energy_gev"),
                "angular_err_deg": e.get("angular_err_deg"),
            })
            for e in top_events
            if e.get("ra") is not None
        ],
    }


@app.get("/api/catalogs/frb")
def get_frb_repeaters():
    """Get FRB repeaters for overlay."""
    cache_file = DATA_DIR / "cache" / "frb_catalog_v1.json"
    if not cache_file.exists():
        return {"error": "FRB catalog not cached", "count": 0, "repeaters": []}

    with open(cache_file) as f:
        data = json.load(f)

    repeaters = data.get("repeaters", [])
    return {
        "count": len(repeaters),
        "repeaters": [
            _sanitize({
                "name": r.get("name"),
                "ra": r.get("ra"),
                "dec": r.get("dec"),
                "dm": r.get("dm"),
                "n_bursts": r.get("n_bursts"),
            })
            for r in repeaters
            if r.get("ra") is not None
        ],
    }


@app.get("/api/catalogs/pulsars")
def get_nanograv_pulsars():
    """Get NANOGrav pulsars for overlay."""
    cache_dir = DATA_DIR / "cache" / "nanograv"
    if not cache_dir.exists():
        return {"error": "NANOGrav data not cached", "count": 0, "pulsars": []}

    pulsars: list = []
    for f in cache_dir.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list) and data and "ra_deg" in data[0]:
                pulsars = data
                break
        except Exception:
            pass

    if not pulsars:
        return {"count": 0, "pulsars": []}

    return {
        "count": len(pulsars),
        "pulsars": [
            _sanitize({
                "name": p.get("name"),
                "ra": p.get("ra_deg"),
                "dec": p.get("dec_deg"),
                "period_ms": p.get("period_ms"),
                "dm": p.get("dm"),
            })
            for p in pulsars
            if p.get("ra_deg") is not None
        ],
    }


@app.get("/api/annotations")
def get_annotations():
    """Get saved user annotations."""
    ann_file = DATA_DIR / "annotations" / "explorer_marks.json"
    if ann_file.exists():
        with open(ann_file) as f:
            return json.load(f)
    return {"annotations": []}


from pydantic import BaseModel


class AnnotationCreate(BaseModel):
    type: str = "interesting"
    ra_center: float
    dec_center: float
    radius_deg: float = 0.1
    notes: str = ""
    targets_in_region: list = []


@app.post("/api/annotations")
def save_annotation(body: AnnotationCreate):
    """Save a new annotation."""
    import datetime

    ann_dir = DATA_DIR / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_file = ann_dir / "explorer_marks.json"

    # Load existing
    existing = {"annotations": []}
    if ann_file.exists():
        with open(ann_file) as f:
            existing = json.load(f)

    annotation = body.dict()
    annotation["id"] = f"ann_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(existing['annotations'])}"
    annotation["created"] = datetime.datetime.now().isoformat()
    existing["annotations"].append(annotation)

    with open(ann_file, "w") as f:
        json.dump(existing, f, indent=2)

    return {"status": "ok", "annotation": annotation, "total": len(existing["annotations"])}


@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: str):
    """Delete an annotation by ID."""
    ann_file = DATA_DIR / "annotations" / "explorer_marks.json"
    if not ann_file.exists():
        return {"error": "No annotations file"}

    with open(ann_file) as f:
        data = json.load(f)

    before = len(data["annotations"])
    data["annotations"] = [a for a in data["annotations"] if a.get("id") != annotation_id]

    with open(ann_file, "w") as f:
        json.dump(data, f, indent=2)

    return {"status": "ok", "deleted": before - len(data["annotations"]), "remaining": len(data["annotations"])}


# ── Candidate Highlights ─────────────────────────────────────

# Candidate data is loaded from data/candidates.json (not tracked in git).
# To populate: create data/candidates.json with a list of candidate dicts,
# each having at minimum: id, name, ra, dec, verdict.
def _load_candidates() -> list:
    """Load candidate data from private data file."""
    path = DATA_DIR / "candidates.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


CANDIDATES = _load_candidates()

# Populate the candidate survivor set for red-team override exemption
_CANDIDATE_SURVIVOR_IDS.update(
    c["id"] for c in CANDIDATES if c.get("verdict") == "SURVIVES"
)


# Campaign comparison data loaded from data file or empty
def _load_campaign_comparison() -> list:
    path = DATA_DIR / "campaign_comparison.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


CAMPAIGN_COMPARISON = _load_campaign_comparison()


def _load_peer_review(candidate_id: str) -> Optional[Dict[str, Any]]:
    """Load peer review JSON for a candidate.

    Looks for peer_review_<candidate_id>.json in data/reports/.
    """
    # Derive filename from candidate ID (sanitize for filesystem)
    safe_id = candidate_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    filename = f"peer_review_{safe_id}.json"
    reports_dir = (DATA_DIR / "reports").resolve()
    path = (reports_dir / filename).resolve()
    if not path.is_relative_to(reports_dir):
        return None
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Fallback: try extracting RA from candidate ID (e.g., "..._RA<degrees>_...")
    import re
    m = re.search(r"RA([\d.]+)", candidate_id)
    if m:
        ra_str = m.group(1)
        fallback = f"peer_review_ra{ra_str}.json"
        fpath = (reports_dir / fallback).resolve()
        if fpath.is_relative_to(reports_dir) and fpath.exists():
            with open(fpath) as f:
                return json.load(f)
    return None


@app.get("/api/candidates")
def get_candidates():
    """Get curated prime candidate targets with research stories."""
    return {
        "count": len(CANDIDATES),
        "candidates": CANDIDATES,
    }


@app.get("/api/candidates/{candidate_id}")
def get_candidate_detail(candidate_id: str):
    """Get full detail for a candidate, including peer review results."""
    for c in CANDIDATES:
        if c["id"] == candidate_id:
            result = dict(c)
            # Attach peer review if available
            pr = _load_peer_review(candidate_id)
            if pr:
                result["peer_review"] = _sanitize(pr)
            return result
    return {"error": f"Candidate '{candidate_id}' not found"}


@app.get("/api/campaign_comparison")
def get_campaign_comparison():
    """Get binary dominance comparison across all campaigns."""
    total_n = sum(c["n"] for c in CAMPAIGN_COMPARISON)
    total_3ch = sum(c["three_ch"] for c in CAMPAIGN_COMPARISON)
    total_non_binary = sum(c["non_binary_3ch"] for c in CAMPAIGN_COMPARISON)
    return {
        "campaigns": CAMPAIGN_COMPARISON,
        "totals": {"n": total_n, "three_ch": total_3ch, "non_binary_3ch": total_non_binary},
        "summary": f"{total_n} targets across {len(CAMPAIGN_COMPARISON)} populations: {total_non_binary} non-binary 3ch",
    }


@app.get("/api/candidates/{candidate_id}/peer_review")
def get_candidate_peer_review(candidate_id: str):
    """Get peer review results for a candidate."""
    pr = _load_peer_review(candidate_id)
    if pr:
        return _sanitize(pr)
    return {"error": f"No peer review found for '{candidate_id}'"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
