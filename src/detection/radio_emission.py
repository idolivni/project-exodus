"""
Project EXODUS — Radio Continuum Emission Detection Channel
=============================================================

Promotes FIRST/NVSS radio continuum data from vetting-only to a
real scoring channel.

Radio emission at 1.4 GHz is HIGHLY independent of IR excess and
proper motion anomaly. A star with radio + IR + PM convergence is
genuinely hard to explain with binarity alone.

Most main-sequence stars are radio-quiet. Detectable 1.4 GHz emission
from a nearby (<50 pc) star typically indicates:
  - Active M-dwarf flares (common, but transient)
  - Gyrosynchrotron emission from magnetically active stars
  - Thermal bremsstrahlung from stellar winds (rare at 1.4 GHz)
  - Something else entirely (our target signal)

The key: if a star has radio emission AND IR excess AND PM anomaly,
all three channels are independently pointing to something unusual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RadioEmissionResult:
    """Result of radio continuum emission detection."""

    has_data: bool = False
    data_source: str = "none"  # "first_vizier", "nvss_vizier", "none"
    survey: str = "none"       # "FIRST", "NVSS"

    # Measurements
    peak_flux_mjy: float = 0.0
    integrated_flux_mjy: float = 0.0
    sep_arcsec: float = 0.0
    snr: float = 0.0

    # Derived
    is_detected: bool = False
    anomaly_score: float = 0.0  # [0, 1]
    is_anomalous: bool = False

    # Context
    radio_luminosity_proxy: Optional[float] = None  # flux * dist^2 proxy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_data": self.has_data,
            "data_source": self.data_source,
            "survey": self.survey,
            "peak_flux_mjy": round(self.peak_flux_mjy, 3),
            "integrated_flux_mjy": round(self.integrated_flux_mjy, 3),
            "sep_arcsec": round(self.sep_arcsec, 2),
            "snr": round(self.snr, 2),
            "is_detected": self.is_detected,
            "anomaly_score": round(self.anomaly_score, 4),
            "is_anomalous": self.is_anomalous,
            "radio_luminosity_proxy": (
                round(self.radio_luminosity_proxy, 2)
                if self.radio_luminosity_proxy is not None else None
            ),
        }


def compute_radio_emission(
    radio_continuum: Optional[Dict[str, Any]],
    distance_pc: Optional[float] = None,
) -> RadioEmissionResult:
    """Compute radio emission anomaly score from FIRST/NVSS data.

    Parameters
    ----------
    radio_continuum : dict or None
        Output from vlass_catalog.query_radio_continuum().
        Contains peak_flux_mjy, snr, survey, sep_arcsec, etc.
    distance_pc : float or None
        Distance for luminosity estimation.

    Returns
    -------
    RadioEmissionResult
    """
    result = RadioEmissionResult()

    if not radio_continuum:
        return result

    result.has_data = True
    result.data_source = radio_continuum.get("data_source", "unknown")
    result.survey = radio_continuum.get("survey", "unknown")
    result.peak_flux_mjy = float(radio_continuum.get("peak_flux_mjy", 0.0))
    result.integrated_flux_mjy = float(radio_continuum.get("integrated_flux_mjy", 0.0))
    result.sep_arcsec = float(radio_continuum.get("sep_arcsec", 0.0))
    result.snr = float(radio_continuum.get("snr", 0.0))

    # Quality gates
    # Require minimum flux and reasonable positional match
    min_flux = 1.0   # mJy
    max_sep = 10.0    # arcsec (FIRST has 5" beam, NVSS 45")

    if result.peak_flux_mjy < min_flux or result.sep_arcsec > max_sep:
        result.is_detected = False
        result.anomaly_score = 0.0
        return result

    result.is_detected = True

    # Luminosity proxy if distance available
    if distance_pc and distance_pc > 0:
        result.radio_luminosity_proxy = result.peak_flux_mjy * (distance_pc ** 2)

    # Score based on flux strength with saturating function
    # Most nearby stars are radio-quiet (< 0.1 mJy)
    # Detection above 1 mJy at < 50 pc is unusual
    # Score = 1 - exp(-flux / 5.0)
    flux_score = 1.0 - np.exp(-result.peak_flux_mjy / 5.0)

    # SNR boost: higher SNR → more confident detection
    snr_factor = min(1.0, result.snr / 10.0)

    # Positional closeness bonus
    sep_factor = 1.0 - (result.sep_arcsec / max_sep)

    result.anomaly_score = float(np.clip(
        flux_score * 0.6 + snr_factor * 0.3 + sep_factor * 0.1,
        0.0, 1.0
    ))

    result.is_anomalous = result.anomaly_score > 0.3

    return result
