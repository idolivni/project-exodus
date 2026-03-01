"""
Project EXODUS — UV Anomaly Detection Channel
===============================================

Promotes GALEX UV data from vetting-only to a real scoring channel.

UV anomaly is HIGHLY independent of IR and PM channels. A star with
both IR excess and UV anomaly is far harder to explain with binarity
alone — binaries don't typically produce UV deficits or unexpected
UV-IR colour combinations.

The key discriminant: background galaxy confusion produces IR excess
but NOT UV emission (at the same position). A star with UV + IR anomaly
is definitively NOT a background source.

Scoring logic
-------------
Uses the uv_anomaly_score already computed in galex_catalog.py
(compute_uv_metrics), but applies additional quality gates and
normalisation for scoring integration.

UV anomaly types:
  1. UV deficit (NUV fainter than expected) → possible absorption
  2. UV excess (FUV-NUV < 1.0) → chromospheric/circumstellar
  3. UV-IR anti-correlation → highly suspicious (cold dust + UV source)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class UVAnomalyResult:
    """Result of UV anomaly detection."""

    has_data: bool = False
    data_source: str = "none"  # "galex_vizier", "none"

    # Raw measurements
    fuv_mag: Optional[float] = None
    nuv_mag: Optional[float] = None
    fuv_nuv_color: Optional[float] = None

    # Computed metrics
    uv_anomaly_score: float = 0.0  # [0, 1]
    is_uv_active: bool = False     # Chromospheric activity (FUV-NUV < 1.0)
    is_uv_deficit: bool = False    # NUV fainter than expected
    nuv_residual: float = 0.0     # Deviation from Teff-based expectation

    # Quality
    artifact_clean: bool = True

    # Composite
    anomaly_score: float = 0.0     # Final score for scorer [0, 1]
    is_anomalous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_data": self.has_data,
            "data_source": self.data_source,
            "fuv_mag": self.fuv_mag,
            "nuv_mag": self.nuv_mag,
            "fuv_nuv_color": self.fuv_nuv_color,
            "uv_anomaly_score": round(self.uv_anomaly_score, 4),
            "is_uv_active": self.is_uv_active,
            "is_uv_deficit": self.is_uv_deficit,
            "nuv_residual": round(self.nuv_residual, 4),
            "artifact_clean": self.artifact_clean,
            "anomaly_score": round(self.anomaly_score, 4),
            "is_anomalous": self.is_anomalous,
        }


def compute_uv_anomaly(
    uv_metrics: Optional[Dict[str, Any]],
    galex_raw: Optional[Dict[str, Any]] = None,
    ir_excess_data: Optional[Dict[str, Any]] = None,
) -> UVAnomalyResult:
    """Compute UV anomaly score from pre-gathered GALEX data.

    Parameters
    ----------
    uv_metrics : dict or None
        Output from galex_catalog.compute_uv_metrics(). Contains
        uv_anomaly_score, is_uv_active, is_uv_deficit, etc.
    galex_raw : dict or None
        Raw GALEX match data (fuv_mag, nuv_mag, artifact flags).
    ir_excess_data : dict or None
        IR excess results (for UV-IR cross-correlation bonus).

    Returns
    -------
    UVAnomalyResult
    """
    result = UVAnomalyResult()

    if not uv_metrics or not uv_metrics.get("has_data"):
        return result

    result.has_data = True
    result.data_source = uv_metrics.get("data_source", "galex_vizier")

    # Extract metrics
    result.uv_anomaly_score = float(uv_metrics.get("uv_anomaly_score", 0.0))
    result.is_uv_active = bool(uv_metrics.get("is_uv_active", False))
    result.is_uv_deficit = bool(uv_metrics.get("is_uv_deficit", False))
    result.nuv_residual = float(uv_metrics.get("nuv_residual", 0.0))
    result.fuv_nuv_color = uv_metrics.get("fuv_nuv_color")

    if galex_raw:
        result.fuv_mag = galex_raw.get("fuv_mag")
        result.nuv_mag = galex_raw.get("nuv_mag")
        result.artifact_clean = bool(galex_raw.get("artifact_clean", True))

    # Quality gate: reject if artifact contamination
    if not result.artifact_clean:
        result.anomaly_score = 0.0
        result.is_anomalous = False
        return result

    # Base score from UV metrics
    score = result.uv_anomaly_score

    # UV-IR cross-correlation bonus REMOVED (audit fix B2).
    # Channel independence requires each channel to score on its own merit.
    # The convergence bonus in exodus_score.py already rewards multi-channel
    # signals; double-counting here violated the independence assumption
    # underpinning the Fisher/Stouffer combination.

    result.anomaly_score = score
    result.is_anomalous = score > 0.3

    return result
