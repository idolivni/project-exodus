"""
Multi-Modal Correlation Engine for Project EXODUS.

THE CORE INNOVATION: stacks ALL datasets on top of each other for every
target star and finds correlations that no single dataset can reveal.

The fundamental insight is that technosignatures are MULTI-CHANNEL phenomena.
A Dyson sphere blocks optical light (transit anomaly), re-emits in mid-IR
(infrared excess), and the civilization may also transmit radio signals.  No
single dataset can distinguish these from natural astrophysical noise with
confidence.  But the probability of a target appearing anomalous in THREE
independent datasets simultaneously by chance is vanishingly small.

Architecture
------------
For each target (RA, Dec) we build a ``TargetProfile`` that pulls from:

    Layer 1: Gaia optical  (magnitude, color, variability)
    Layer 2: 2MASS near-IR  (J, H, K)
    Layer 3: WISE mid-IR  (W1-W4, especially W3/W4 for waste heat)
    Layer 4: Gaia epoch photometry  (brightness over years)
    Layer 5: Kepler / TESS transits  (light curve shape, depth, symmetry)
    Layer 6: Breakthrough Listen radio  (signal candidates)
    Layer 7: NASA Exoplanet Archive  (planet properties, HZ status)
    Layer 8: Gaia astrometry  (proper motion, parallax anomalies)

Four correlation modes operate on the stacked profile:

    1. **Spatial correlation** -- same star, different wavelengths, same epoch.
       IR excess + transit anomaly + radio signal = highest priority.
    2. **Temporal correlation** -- same position, different decades.
       Did something CHANGE over time?
    3. **Cross-band temporal correlation** -- THE MOST NOVEL.  When a star
       dims in optical, does it simultaneously brighten in IR?  Anti-
       correlation = Dyson swarm blocking light and re-emitting as heat.
    4. **Anomaly stacking** -- across ALL targets, find outliers in MULTIPLE
       independent datasets.  IsolationForest on each channel; targets
       flagged in >= N channels are statistically extraordinary.

Key references
--------------
- Wright et al. 2014, ApJ 792, 26 (Glimpsing Heat from Alien Technologies)
- Suazo et al. 2024, MNRAS 527, 1 (Project Hephaistos)
- Boyajian et al. 2016, MNRAS 457, 3988 (Tabby's Star)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_config, get_logger, save_result

log = get_logger("correlation.multi_modal")

# ---------------------------------------------------------------------------
# Lazy imports -- ingestion modules
# ---------------------------------------------------------------------------
# Using try/except so the engine can still be instantiated even when some
# heavy dependencies (astroquery, lightkurve, blimpy, etc.) are not
# installed.  Each accessor method checks its own availability.

try:
    from src.ingestion.gaia_query import (
        get_stellar_params as _gaia_stellar_params,
        get_astrometry as _gaia_astrometry,
        get_epoch_photometry as _gaia_epoch_photometry,
    )
    _HAS_GAIA = True
except ImportError:
    _HAS_GAIA = False
    log.debug("Gaia ingestion module not available")

try:
    from src.ingestion.ir_surveys import (
        get_2mass as _get_2mass,
        get_wise as _get_wise,
    )
    _HAS_IR = True
except ImportError:
    _HAS_IR = False
    log.debug("IR survey ingestion module not available")

try:
    from src.ingestion.lightcurves import (
        get_lightcurve as _get_lightcurve,
    )
    _HAS_LC = True
except ImportError:
    _HAS_LC = False
    log.debug("Lightcurve ingestion module not available")

try:
    from src.ingestion.breakthrough_listen import (
        get_observation as _bl_get_observation,
        get_spectrogram as _bl_get_spectrogram,
    )
    _HAS_BL = True
except ImportError:
    _HAS_BL = False
    log.debug("Breakthrough Listen ingestion module not available")

try:
    from src.ingestion.exoplanet_archive import (
        query_exoplanet_archive as _query_exoplanets,
    )
    _HAS_EXOARCH = True
except ImportError:
    _HAS_EXOARCH = False
    log.debug("Exoplanet Archive ingestion module not available")

# ---------------------------------------------------------------------------
# Lazy imports -- processing modules
# ---------------------------------------------------------------------------

try:
    from src.processing.ir_excess import compute_ir_excess as _compute_ir_excess
    _HAS_IR_EXCESS = True
except ImportError:
    _HAS_IR_EXCESS = False
    log.debug("IR excess processing module not available")

try:
    from src.processing.transit_anomaly import (
        detect_transit_anomaly as _detect_transit_anomaly,
        detect_irregular_dimming as _detect_irregular_dimming,
    )
    _HAS_TRANSIT = True
except ImportError:
    _HAS_TRANSIT = False
    log.debug("Transit anomaly processing module not available")

try:
    from src.processing.radio_processor import (
        process_spectrogram as _process_spectrogram,
    )
    _HAS_RADIO = True
except ImportError:
    _HAS_RADIO = False
    log.debug("Radio processor module not available")


# ============================================================================
#  Configuration helpers
# ============================================================================

def _anomaly_sigma() -> float:
    """Return the anomaly sigma threshold from project config."""
    try:
        cfg = get_config()
        return float(cfg["search"]["anomaly_sigma"])
    except Exception:
        return 3.0


def _min_convergence_channels() -> int:
    """Return the minimum number of channels required for convergence."""
    try:
        cfg = get_config()
        return int(cfg["search"]["min_convergence_channels"])
    except Exception:
        return 2


# ============================================================================
#  Data classes
# ============================================================================

@dataclass
class TargetProfile:
    """Unified multi-wavelength profile for a single star.

    Every field beyond (ra, dec) is Optional -- populated only when
    the corresponding dataset has coverage for this target.

    Attributes
    ----------
    ra, dec : float
        ICRS coordinates in decimal degrees.
    optical : dict or None
        Gaia photometry (G, BP, RP magnitudes, Teff, etc.).
    near_ir : dict or None
        2MASS photometry (J, H, Ks magnitudes).
    mid_ir : dict or None
        AllWISE photometry (W1-W4 magnitudes).
    epoch_photometry : dict or None
        Gaia epoch photometry time-series (transit times + magnitudes).
    lightcurve : dict or None
        Kepler / TESS light curve data (time, flux, flux_err arrays).
    radio : dict or None
        Breakthrough Listen observation metadata and processing results.
    exoplanets : dict or None
        NASA Exoplanet Archive data (planet properties, HZ status).
    astrometry : dict or None
        Gaia astrometric solution (parallax, proper motion, RUWE).
    archival_radio : dict or None
        Historical radio observations from other archives (if available).
    mid_ir_timeseries : dict or None
        NEOWISE multi-epoch mid-IR photometry for temporal analysis.
    """
    ra: float
    dec: float
    optical: Optional[Dict[str, Any]] = None
    near_ir: Optional[Dict[str, Any]] = None
    mid_ir: Optional[Dict[str, Any]] = None
    epoch_photometry: Optional[Dict[str, Any]] = None
    lightcurve: Optional[Dict[str, Any]] = None
    radio: Optional[Dict[str, Any]] = None
    exoplanets: Optional[Dict[str, Any]] = None
    astrometry: Optional[Dict[str, Any]] = None
    archival_radio: Optional[Dict[str, Any]] = None
    mid_ir_timeseries: Optional[Dict[str, Any]] = None

    @property
    def available_layers(self) -> List[str]:
        """Return a list of layer names that have data."""
        layers = []
        for name in (
            "optical", "near_ir", "mid_ir", "epoch_photometry",
            "lightcurve", "radio", "exoplanets", "astrometry",
            "archival_radio", "mid_ir_timeseries",
        ):
            if getattr(self, name) is not None:
                layers.append(name)
        return layers

    @property
    def n_layers(self) -> int:
        """Number of populated data layers."""
        return len(self.available_layers)

    @property
    def label(self) -> str:
        """Short human-readable identifier."""
        return f"({self.ra:.4f}, {self.dec:.4f})"


@dataclass
class AnomalyResult:
    """Result of anomaly detection in a single dataset channel.

    Attributes
    ----------
    score : float
        Anomaly score normalized to [0, 1].  Higher = more anomalous.
    details : dict
        Channel-specific details (e.g. sigma of IR excess, transit
        anomaly sub-scores, radio SNR, etc.).
    dataset_name : str
        Name of the dataset that produced this result (e.g. "ir_excess",
        "transit_anomaly", "radio", "astrometry").
    """
    score: float
    details: Dict[str, Any]
    dataset_name: str


@dataclass
class SpatialCorrelation:
    """Result of spatial (multi-wavelength, same-epoch) correlation.

    Attributes
    ----------
    anomalies : dict
        Mapping of channel name -> AnomalyResult for each flagged channel.
    n_channels_flagged : int
        Number of independent channels showing anomalous signal.
    channel_names : list of str
        Names of the flagged channels.
    """
    anomalies: Dict[str, AnomalyResult]
    n_channels_flagged: int
    channel_names: List[str]


@dataclass
class TemporalCorrelation:
    """Result of temporal (same position, different decades) correlation.

    Attributes
    ----------
    changes : dict
        Mapping of change description -> details dict.  For example:
        ``{"optical_dimming": {"delta_mag": -0.3, "epoch_span_yr": 5.2}, ...}``
    """
    changes: Dict[str, Any]


@dataclass
class CrossBandCorrelation:
    """Result of cross-band temporal correlation.

    The key diagnostic: when optical dims, does IR brighten?  Anti-
    correlation is the smoking gun for a Dyson swarm.

    Attributes
    ----------
    correlation_coefficient : float
        Pearson r between optical and IR light curves on overlapping
        epochs.  Ranges from -1 (perfect anti-correlation) to +1.
    is_anticorrelated : bool
        True if the correlation is significantly negative, consistent
        with blocked starlight re-emitted as waste heat.
    aligned_data : dict
        The time-aligned optical and IR data used for the correlation.
    """
    correlation_coefficient: float
    is_anticorrelated: bool
    aligned_data: Dict[str, Any]


@dataclass
class AnomalyStackResult:
    """Result of anomaly stacking across ALL targets.

    Attributes
    ----------
    ranked_targets : list of dict
        Targets ranked by the number (and strength) of independent
        anomaly detections.  Each entry contains the target coordinates,
        number of channels flagged, individual channel scores, and a
        combined convergence score.
    """
    ranked_targets: List[Dict[str, Any]]


# ============================================================================
#  Multi-Modal Correlator
# ============================================================================

class MultiModalCorrelator:
    """The core multi-modal correlation engine.

    Builds unified target profiles by pulling from ALL available datasets,
    then searches for correlations that only appear when you look across
    datasets simultaneously.
    """

    def __init__(self):
        self._sigma = _anomaly_sigma()
        self._min_channels = _min_convergence_channels()
        log.info(
            "MultiModalCorrelator initialized (sigma=%.1f, min_channels=%d)",
            self._sigma, self._min_channels,
        )

    # ------------------------------------------------------------------ #
    #  1. BUILD TARGET PROFILE                                            #
    # ------------------------------------------------------------------ #

    def build_target_profile(self, ra: float, dec: float) -> TargetProfile:
        """Pull everything we have for one star and build a unified profile.

        Layers populated:
            1. Gaia optical (G, BP, RP magnitudes, color, Teff, variability)
            2. 2MASS near-IR (J, H, Ks)
            3. WISE mid-IR (W1-W4, especially W3/W4 for waste heat)
            4. Gaia epoch photometry (brightness over years)
            5. Kepler/TESS transits (light curve shape, depth, symmetry)
            6. Breakthrough Listen radio (signal candidates)
            7. NASA Exoplanet Archive (planet properties, HZ status)
            8. Gaia astrometry (proper motion, parallax, RUWE)

        Each layer is populated independently; failures in one do not
        prevent others from loading.  Returns a TargetProfile with all
        available layers, and None for layers that are unavailable.

        Parameters
        ----------
        ra : float
            Right ascension in decimal degrees (ICRS).
        dec : float
            Declination in decimal degrees (ICRS).

        Returns
        -------
        TargetProfile
        """
        log.info("Building target profile for (%.4f, %.4f)", ra, dec)
        profile = TargetProfile(ra=ra, dec=dec)

        # -- Layer 1: Gaia optical photometry --------------------------------
        if _HAS_GAIA:
            try:
                stellar = _gaia_stellar_params(ra, dec)
                if stellar is not None and not stellar.empty:
                    row = stellar.iloc[0]
                    profile.optical = {
                        "source_id": int(row.get("source_id", 0)),
                        "G": row.get("phot_g_mean_mag"),
                        "BP": row.get("phot_bp_mean_mag"),
                        "RP": row.get("phot_rp_mean_mag"),
                        "bp_rp": row.get("bp_rp"),
                        "teff": row.get("teff_gspphot"),
                        "logg": row.get("logg_gspphot"),
                    }
                    log.info("  Layer 1 (optical): G=%.3f, BP-RP=%.3f",
                             row.get("phot_g_mean_mag", 0),
                             row.get("bp_rp", 0))
            except Exception as exc:
                log.warning("  Layer 1 (optical) failed: %s", exc)

        # -- Layer 2: 2MASS near-IR -----------------------------------------
        if _HAS_IR:
            try:
                twomass = _get_2mass(ra, dec)
                if twomass:
                    profile.near_ir = twomass
                    log.info("  Layer 2 (near-IR): J=%.3f, H=%.3f, Ks=%.3f",
                             twomass.get("J", 0), twomass.get("H", 0),
                             twomass.get("Ks", 0))
            except Exception as exc:
                log.warning("  Layer 2 (near-IR) failed: %s", exc)

        # -- Layer 3: WISE mid-IR -------------------------------------------
        if _HAS_IR:
            try:
                wise = _get_wise(ra, dec)
                if wise:
                    profile.mid_ir = wise
                    log.info("  Layer 3 (mid-IR): W1=%.3f, W3=%.3f, W4=%.3f",
                             wise.get("W1", 0), wise.get("W3", 0),
                             wise.get("W4", 0))
            except Exception as exc:
                log.warning("  Layer 3 (mid-IR) failed: %s", exc)

        # -- Layer 4: Gaia epoch photometry ----------------------------------
        if _HAS_GAIA and profile.optical and profile.optical.get("source_id"):
            try:
                source_id = profile.optical["source_id"]
                epoch = _gaia_epoch_photometry(source_id)
                if epoch is not None and not epoch.empty:
                    profile.epoch_photometry = {
                        "source_id": source_id,
                        "n_transits": len(epoch),
                        "time": epoch["g_obs_time"].values.tolist()
                            if "g_obs_time" in epoch.columns else [],
                        "mag": epoch["g_transit_mag"].values.tolist()
                            if "g_transit_mag" in epoch.columns else [],
                    }
                    log.info("  Layer 4 (epoch phot): %d transits", len(epoch))
            except Exception as exc:
                log.warning("  Layer 4 (epoch photometry) failed: %s", exc)

        # -- Layer 5: Kepler/TESS light curves --------------------------------
        if _HAS_LC:
            try:
                lc = _get_lightcurve((ra, dec))
                if lc is not None:
                    profile.lightcurve = {
                        "time": np.asarray(lc.time.value, dtype=np.float64).tolist(),
                        "flux": np.asarray(lc.flux.value, dtype=np.float64).tolist(),
                        "flux_err": np.asarray(
                            lc.flux_err.value, dtype=np.float64
                        ).tolist(),
                        "n_points": len(lc.flux),
                        "mission": getattr(lc, "mission", "unknown"),
                    }
                    log.info("  Layer 5 (lightcurve): %d points", len(lc.flux))
            except Exception as exc:
                log.warning("  Layer 5 (lightcurve) failed: %s", exc)

        # -- Layer 6: Breakthrough Listen radio --------------------------------
        if _HAS_BL:
            try:
                # BL observations are indexed by target name, not (ra, dec).
                # We query by coordinate string as a best-effort lookup.
                target_str = f"ra{ra:.4f}_dec{dec:.4f}"
                obs = _bl_get_observation(target_str)
                if obs and obs.meta:
                    profile.radio = {
                        "target": obs.meta.target,
                        "telescope": obs.meta.telescope,
                        "freq_start_mhz": obs.meta.freq_start_mhz,
                        "freq_end_mhz": obs.meta.freq_end_mhz,
                        "has_spectrogram": obs.spectrogram is not None,
                        "source": obs.meta.source,
                    }
                    log.info("  Layer 6 (radio): %s via %s",
                             obs.meta.target, obs.meta.telescope)
            except Exception as exc:
                log.warning("  Layer 6 (radio) failed: %s", exc)

        # -- Layer 7: NASA Exoplanet Archive -----------------------------------
        if _HAS_EXOARCH:
            try:
                all_planets = _query_exoplanets()
                if all_planets is not None and not all_planets.empty:
                    # Find planets whose host star is within crossmatch
                    # radius of the target.  Use simple angular distance.
                    if "ra_deg" in all_planets.columns and "dec_deg" in all_planets.columns:
                        dra = all_planets["ra_deg"] - ra
                        ddec = all_planets["dec_deg"] - dec
                        cos_dec = np.cos(np.radians(dec))
                        sep_arcsec = 3600.0 * np.sqrt(
                            (dra * cos_dec) ** 2 + ddec ** 2
                        )
                        nearby = all_planets[sep_arcsec < 5.0]
                        if not nearby.empty:
                            planets = []
                            for _, row in nearby.iterrows():
                                planets.append({
                                    "name": row.get("planet_name"),
                                    "host": row.get("host_star"),
                                    "period_days": row.get("orbital_period_days"),
                                    "radius_earth": row.get("radius_earth"),
                                    "eq_temp_k": row.get("eq_temp_k"),
                                    "hz_flag": bool(row.get("hz_flag", False)),
                                })
                            profile.exoplanets = {
                                "n_planets": len(planets),
                                "planets": planets,
                                "has_hz_planet": any(
                                    p["hz_flag"] for p in planets
                                ),
                            }
                            log.info("  Layer 7 (exoplanets): %d planet(s)",
                                     len(planets))
            except Exception as exc:
                log.warning("  Layer 7 (exoplanets) failed: %s", exc)

        # -- Layer 8: Gaia astrometry ----------------------------------------
        if _HAS_GAIA:
            try:
                astrom = _gaia_astrometry(ra, dec)
                if astrom is not None and not astrom.empty:
                    row = astrom.iloc[0]
                    profile.astrometry = {
                        "source_id": int(row.get("source_id", 0)),
                        "parallax": row.get("parallax"),
                        "parallax_error": row.get("parallax_error"),
                        "pmra": row.get("pmra"),
                        "pmdec": row.get("pmdec"),
                        "ruwe": row.get("ruwe"),
                        "astrometric_excess_noise":
                            row.get("astrometric_excess_noise"),
                    }
                    log.info("  Layer 8 (astrometry): parallax=%.3f, RUWE=%.3f",
                             row.get("parallax", 0), row.get("ruwe", 0))
            except Exception as exc:
                log.warning("  Layer 8 (astrometry) failed: %s", exc)

        log.info(
            "Profile built for %s: %d/%d layers populated (%s)",
            profile.label, profile.n_layers, 10,
            ", ".join(profile.available_layers),
        )
        return profile

    # ------------------------------------------------------------------ #
    #  2. SPATIAL CORRELATION                                             #
    # ------------------------------------------------------------------ #

    def correlate_spatial(self, profile: TargetProfile) -> SpatialCorrelation:
        """Same star, different wavelengths, same epoch.

        Runs independent anomaly detectors on each available dataset and
        identifies extraordinary multi-channel combinations:

        - IR excess + transit anomaly = partial Dyson swarm
        - IR excess + radio signal = waste heat AND communication
        - Transit anomaly + radio signal = occultation + techno-emission
        - ALL THREE = highest priority target

        Parameters
        ----------
        profile : TargetProfile
            Fully populated target profile.

        Returns
        -------
        SpatialCorrelation
        """
        log.info("Running spatial correlation for %s", profile.label)
        anomalies: Dict[str, AnomalyResult] = {}

        # -- Channel 1: IR excess -------------------------------------------
        if (profile.optical or profile.near_ir) and profile.mid_ir:
            ir_score, ir_details = self._check_ir_excess(profile)
            if ir_score > 0:
                anomalies["ir_excess"] = AnomalyResult(
                    score=ir_score, details=ir_details,
                    dataset_name="ir_excess",
                )

        # -- Channel 2: Transit anomaly -------------------------------------
        if profile.lightcurve:
            transit_score, transit_details = self._check_transit_anomaly(profile)
            if transit_score > 0:
                anomalies["transit_anomaly"] = AnomalyResult(
                    score=transit_score, details=transit_details,
                    dataset_name="transit_anomaly",
                )

        # -- Channel 3: Radio signal ----------------------------------------
        if profile.radio:
            radio_score, radio_details = self._check_radio_signal(profile)
            if radio_score > 0:
                anomalies["radio"] = AnomalyResult(
                    score=radio_score, details=radio_details,
                    dataset_name="radio",
                )

        # -- Channel 4: Astrometric anomaly ---------------------------------
        if profile.astrometry:
            astro_score, astro_details = self._check_astrometry_anomaly(profile)
            if astro_score > 0:
                anomalies["astrometry"] = AnomalyResult(
                    score=astro_score, details=astro_details,
                    dataset_name="astrometry",
                )

        flagged = [name for name, a in anomalies.items() if a.score >= 0.5]
        n_flagged = len(flagged)

        # Log extraordinary combinations
        if n_flagged >= 3:
            log.warning(
                "*** TRIPLE CONVERGENCE at %s: %s ***",
                profile.label, ", ".join(flagged),
            )
        elif n_flagged >= 2:
            combo = " + ".join(flagged)
            log.info(
                "** DOUBLE CONVERGENCE at %s: %s **",
                profile.label, combo,
            )
            # Log specific combinations of interest
            has_ir = "ir_excess" in flagged
            has_transit = "transit_anomaly" in flagged
            has_radio = "radio" in flagged
            if has_ir and has_transit:
                log.info("  -> IR excess + transit anomaly = partial Dyson swarm candidate")
            if has_ir and has_radio:
                log.info("  -> IR excess + radio = waste heat AND communication")
            if has_transit and has_radio:
                log.info("  -> Transit anomaly + radio signal = occultation + techno-emission")

        result = SpatialCorrelation(
            anomalies=anomalies,
            n_channels_flagged=n_flagged,
            channel_names=flagged,
        )
        log.info(
            "Spatial correlation for %s: %d channels flagged: %s",
            profile.label, n_flagged, flagged,
        )
        return result

    # ------------------------------------------------------------------ #
    #  3. TEMPORAL CORRELATION                                            #
    # ------------------------------------------------------------------ #

    def correlate_temporal(self, profile: TargetProfile) -> TemporalCorrelation:
        """Same position, different decades.

        Examines whether any properties CHANGED over time:
        - Optical brightness changed (star dimmed / brightened over years)
        - IR emission appeared or increased
        - Radio signal appeared or disappeared
        - Astrometric solution changed (unexpected proper motion shift)

        These changes could indicate an evolving megastructure -- a swarm
        being built, or a civilization ramping up energy collection.

        Parameters
        ----------
        profile : TargetProfile

        Returns
        -------
        TemporalCorrelation
        """
        log.info("Running temporal correlation for %s", profile.label)
        changes: Dict[str, Any] = {}

        # -- Optical brightness trend from epoch photometry -----------------
        if profile.epoch_photometry:
            times = profile.epoch_photometry.get("time", [])
            mags = profile.epoch_photometry.get("mag", [])
            if len(times) >= 10 and len(mags) >= 10:
                times_arr = np.array(times, dtype=np.float64)
                mags_arr = np.array(mags, dtype=np.float64)

                # Remove NaN/Inf
                valid = np.isfinite(times_arr) & np.isfinite(mags_arr)
                times_arr = times_arr[valid]
                mags_arr = mags_arr[valid]

                if len(times_arr) >= 10:
                    # Fit a linear trend: mag = a * time + b
                    # Positive slope = dimming (fainter = higher magnitude)
                    coeffs = np.polyfit(times_arr, mags_arr, 1)
                    slope = coeffs[0]  # mag per unit time
                    time_span = times_arr[-1] - times_arr[0]

                    # Estimate variability amplitude
                    mag_std = np.std(mags_arr)
                    mag_range = np.ptp(mags_arr)

                    changes["optical_trend"] = {
                        "slope_mag_per_day": float(slope),
                        "time_span_days": float(time_span),
                        "total_change_mag": float(slope * time_span),
                        "variability_std_mag": float(mag_std),
                        "variability_range_mag": float(mag_range),
                        "n_epochs": len(times_arr),
                        "is_significant": bool(
                            abs(slope * time_span) > self._sigma * mag_std
                        ),
                    }
                    if abs(slope * time_span) > self._sigma * mag_std:
                        direction = "dimming" if slope > 0 else "brightening"
                        log.info(
                            "  Significant optical %s detected: %.4f mag over %.0f days",
                            direction, slope * time_span, time_span,
                        )

        # -- Light curve variability from Kepler/TESS -----------------------
        if profile.lightcurve:
            flux_arr = np.array(profile.lightcurve.get("flux", []),
                                dtype=np.float64)
            time_arr = np.array(profile.lightcurve.get("time", []),
                                dtype=np.float64)
            valid = np.isfinite(flux_arr) & np.isfinite(time_arr)
            flux_arr = flux_arr[valid]
            time_arr = time_arr[valid]

            if len(flux_arr) >= 20:
                # Split into halves and compare median flux
                mid = len(flux_arr) // 2
                early_median = float(np.median(flux_arr[:mid]))
                late_median = float(np.median(flux_arr[mid:]))
                delta = late_median - early_median
                flux_std = float(np.std(flux_arr))

                changes["lightcurve_evolution"] = {
                    "early_median_flux": early_median,
                    "late_median_flux": late_median,
                    "delta_flux": float(delta),
                    "flux_std": flux_std,
                    "time_span_days": float(time_arr[-1] - time_arr[0]),
                    "is_significant": bool(
                        abs(delta) > self._sigma * flux_std / np.sqrt(mid)
                    ),
                }

        # -- Radio appearance/disappearance ---------------------------------
        if profile.radio:
            changes["radio_status"] = {
                "detected": True,
                "telescope": profile.radio.get("telescope"),
                "freq_range_mhz": (
                    profile.radio.get("freq_start_mhz"),
                    profile.radio.get("freq_end_mhz"),
                ),
                "note": "Radio observation exists; compare with future epochs "
                        "to detect appearance/disappearance",
            }

        # -- Astrometric anomalies ------------------------------------------
        if profile.astrometry:
            ruwe = profile.astrometry.get("ruwe")
            excess_noise = profile.astrometry.get("astrometric_excess_noise")
            if ruwe is not None:
                # RUWE > 1.4 indicates a poor single-star astrometric fit,
                # which could indicate an unresolved companion, a nearby
                # megastructure perturbing the photocenter, or other anomaly.
                is_anomalous = float(ruwe) > 1.4
                changes["astrometric_quality"] = {
                    "ruwe": float(ruwe),
                    "excess_noise": float(excess_noise) if excess_noise else None,
                    "is_anomalous": is_anomalous,
                    "note": ("RUWE > 1.4 suggests non-single-star behavior or "
                             "photocenter wobble" if is_anomalous else "Normal"),
                }

        result = TemporalCorrelation(changes=changes)
        log.info(
            "Temporal correlation for %s: %d change indicators found",
            profile.label, len(changes),
        )
        return result

    # ------------------------------------------------------------------ #
    #  4. CROSS-BAND TEMPORAL CORRELATION (THE MOST NOVEL)                #
    # ------------------------------------------------------------------ #

    def correlate_cross_band_temporal(
        self, profile: TargetProfile,
    ) -> CrossBandCorrelation:
        """When a star dims in optical, does it simultaneously brighten in IR?

        Anti-correlation between optical and infrared brightness is the
        smoking gun for a Dyson swarm: the structure blocks starlight
        (optical dimming) and re-emits the absorbed energy as waste heat
        (infrared brightening).

        This correlation is nearly impossible to produce naturally.  Dust
        causes correlated dimming across ALL wavelengths.  A Dyson swarm
        causes ANTI-correlated behavior: optical down, IR up.

        The method requires overlapping time coverage between optical
        and IR surveys.  We align the time axes, interpolate to a common
        grid, and compute the Pearson correlation coefficient.

        Parameters
        ----------
        profile : TargetProfile

        Returns
        -------
        CrossBandCorrelation
        """
        log.info("Running cross-band temporal correlation for %s", profile.label)

        # Default: no meaningful correlation possible
        default = CrossBandCorrelation(
            correlation_coefficient=0.0,
            is_anticorrelated=False,
            aligned_data={
                "note": "Insufficient overlapping time coverage",
            },
        )

        # Need both optical time-series AND mid-IR time-series
        optical_times = None
        optical_vals = None
        ir_times = None
        ir_vals = None

        # Extract optical time-series from epoch photometry
        if profile.epoch_photometry:
            opt_t = profile.epoch_photometry.get("time", [])
            opt_m = profile.epoch_photometry.get("mag", [])
            if len(opt_t) >= 10 and len(opt_m) >= 10:
                optical_times = np.array(opt_t, dtype=np.float64)
                optical_vals = np.array(opt_m, dtype=np.float64)
                # Convert magnitudes to relative flux (inverted sense:
                # brighter = more flux = lower magnitude)
                valid = np.isfinite(optical_times) & np.isfinite(optical_vals)
                optical_times = optical_times[valid]
                optical_vals = optical_vals[valid]
                # Convert mag to linear flux relative to median
                median_mag = np.median(optical_vals)
                optical_flux = 10.0 ** (-0.4 * (optical_vals - median_mag))

        # Extract mid-IR time-series from mid_ir_timeseries layer
        if profile.mid_ir_timeseries:
            ir_t = profile.mid_ir_timeseries.get("time", [])
            ir_m = profile.mid_ir_timeseries.get("flux", [])
            if not ir_m:
                ir_m = profile.mid_ir_timeseries.get("mag", [])
            if len(ir_t) >= 5 and len(ir_m) >= 5:
                ir_times = np.array(ir_t, dtype=np.float64)
                ir_vals = np.array(ir_m, dtype=np.float64)
                valid = np.isfinite(ir_times) & np.isfinite(ir_vals)
                ir_times = ir_times[valid]
                ir_vals = ir_vals[valid]

        # Also try to use the lightcurve as optical if epoch_phot is missing
        if optical_times is None and profile.lightcurve:
            lc_t = profile.lightcurve.get("time", [])
            lc_f = profile.lightcurve.get("flux", [])
            if len(lc_t) >= 10 and len(lc_f) >= 10:
                optical_times = np.array(lc_t, dtype=np.float64)
                optical_flux = np.array(lc_f, dtype=np.float64)
                valid = np.isfinite(optical_times) & np.isfinite(optical_flux)
                optical_times = optical_times[valid]
                optical_flux = optical_flux[valid]

        if optical_times is None or ir_times is None:
            log.info(
                "  Insufficient data for cross-band correlation "
                "(optical=%s, IR=%s)",
                optical_times is not None, ir_times is not None,
            )
            return default

        if len(optical_times) < 10 or len(ir_times) < 5:
            return default

        # -- Find overlapping time range ------------------------------------
        t_start = max(optical_times[0], ir_times[0])
        t_end = min(optical_times[-1], ir_times[-1])

        if t_end <= t_start:
            log.info("  No overlapping time coverage between optical and IR")
            return default

        # -- Interpolate to a common time grid ------------------------------
        n_grid = min(100, len(optical_times), len(ir_times) * 5)
        common_times = np.linspace(t_start, t_end, n_grid)

        try:
            from scipy.interpolate import interp1d

            opt_interp = interp1d(
                optical_times, optical_flux,
                kind="linear", bounds_error=False,
                fill_value=np.nan,
            )
            ir_interp = interp1d(
                ir_times, ir_vals,
                kind="linear", bounds_error=False,
                fill_value=np.nan,
            )

            opt_aligned = opt_interp(common_times)
            ir_aligned = ir_interp(common_times)
        except Exception as exc:
            log.warning("  Interpolation failed: %s", exc)
            return default

        # Remove points where either series is NaN
        valid = np.isfinite(opt_aligned) & np.isfinite(ir_aligned)
        opt_aligned = opt_aligned[valid]
        ir_aligned = ir_aligned[valid]
        common_times_valid = common_times[valid]

        if len(opt_aligned) < 5:
            log.info("  Too few overlapping points (%d) after alignment",
                     len(opt_aligned))
            return default

        # -- Compute Pearson correlation ------------------------------------
        opt_centered = opt_aligned - np.mean(opt_aligned)
        ir_centered = ir_aligned - np.mean(ir_aligned)

        numerator = np.sum(opt_centered * ir_centered)
        denominator = np.sqrt(
            np.sum(opt_centered ** 2) * np.sum(ir_centered ** 2)
        )

        if denominator == 0:
            corr = 0.0
        else:
            corr = float(numerator / denominator)

        # -- Assess significance --------------------------------------------
        # For n data points, under the null hypothesis of no correlation,
        # r * sqrt(n-2) / sqrt(1-r^2) follows a t-distribution with n-2 dof.
        n = len(opt_aligned)
        if abs(corr) < 1.0 and n > 2:
            t_stat = abs(corr) * np.sqrt(n - 2) / np.sqrt(1.0 - corr ** 2)
            # Approximate: t > 3 is significant at p < 0.01 for n > 10
            is_significant = t_stat > self._sigma
        else:
            is_significant = False

        is_anticorrelated = corr < -0.3 and is_significant

        if is_anticorrelated:
            log.warning(
                "*** ANTI-CORRELATION DETECTED at %s: r=%.4f (t=%.2f) ***",
                profile.label, corr,
                abs(corr) * np.sqrt(n - 2) / np.sqrt(1.0 - corr ** 2)
                if abs(corr) < 1.0 else float("inf"),
            )
            log.warning(
                "  -> Optical dims while IR brightens: consistent with "
                "Dyson swarm blocking light and re-emitting as heat"
            )

        result = CrossBandCorrelation(
            correlation_coefficient=corr,
            is_anticorrelated=is_anticorrelated,
            aligned_data={
                "n_overlap_points": n,
                "time_range": [float(common_times_valid[0]),
                               float(common_times_valid[-1])],
                "optical_mean": float(np.mean(opt_aligned)),
                "optical_std": float(np.std(opt_aligned)),
                "ir_mean": float(np.mean(ir_aligned)),
                "ir_std": float(np.std(ir_aligned)),
                "t_statistic": float(
                    abs(corr) * np.sqrt(n - 2) / np.sqrt(1.0 - corr ** 2)
                ) if abs(corr) < 1.0 and n > 2 else 0.0,
            },
        )

        log.info(
            "Cross-band correlation for %s: r=%.4f, anti-correlated=%s, "
            "n_points=%d",
            profile.label, corr, is_anticorrelated, n,
        )
        return result

    # ------------------------------------------------------------------ #
    #  5. ANOMALY STACKING (IsolationForest across all targets)           #
    # ------------------------------------------------------------------ #

    def correlate_anomaly_stacking(
        self, all_profiles: List[TargetProfile],
    ) -> AnomalyStackResult:
        """Across ALL targets, find outliers in MULTIPLE independent datasets.

        The key insight: run independent anomaly detection on each dataset
        channel using IsolationForest.  Then count how many independent
        channels flag each target.

        Probability of false positive:
            - Flagged in 1 dataset:  ~1%   (common, probably noise)
            - Flagged in 2 datasets: ~0.01% (interesting)
            - Flagged in 3 datasets: ~0.0001% (very interesting)

        This finds things we DON'T KNOW TO LOOK FOR -- any kind of multi-
        channel outlier, not just the specific signatures we defined above.

        Parameters
        ----------
        all_profiles : list of TargetProfile
            All target profiles to analyze.

        Returns
        -------
        AnomalyStackResult
        """
        log.info(
            "Running anomaly stacking across %d targets", len(all_profiles),
        )

        if len(all_profiles) < 5:
            log.warning(
                "Too few targets (%d) for meaningful anomaly stacking; "
                "need at least 5",
                len(all_profiles),
            )
            return AnomalyStackResult(ranked_targets=[])

        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            log.error("scikit-learn not available; cannot run anomaly stacking")
            return AnomalyStackResult(ranked_targets=[])

        # -- Build feature matrices for each independent channel ------------
        channels = self._build_channel_features(all_profiles)

        if not channels:
            log.warning("No channels with sufficient data for stacking")
            return AnomalyStackResult(ranked_targets=[])

        # -- Run IsolationForest independently on each channel ---------------
        n_targets = len(all_profiles)
        channel_flags: Dict[str, np.ndarray] = {}
        channel_scores: Dict[str, np.ndarray] = {}

        for chan_name, (features, valid_mask) in channels.items():
            # features: shape (n_targets, n_features) but only rows where
            # valid_mask is True have real data.
            valid_features = features[valid_mask]

            if len(valid_features) < 5:
                log.debug(
                    "  Channel '%s': too few valid targets (%d), skipping",
                    chan_name, len(valid_features),
                )
                continue

            # Set contamination based on expected anomaly fraction
            contamination = min(0.05, max(0.01, 1.0 / len(valid_features)))

            iso = IsolationForest(
                contamination=contamination,
                n_estimators=200,
                random_state=42,
            )
            predictions = iso.fit_predict(valid_features)
            scores = iso.decision_function(valid_features)

            # Map back to full target array
            flags = np.zeros(n_targets, dtype=bool)
            score_arr = np.zeros(n_targets, dtype=np.float64)
            valid_indices = np.where(valid_mask)[0]
            for i, idx in enumerate(valid_indices):
                flags[idx] = predictions[i] == -1
                # Normalize score: lower decision_function = more anomalous
                # Map to [0, 1] where 1 = most anomalous
                score_arr[idx] = float(
                    np.clip(-scores[i], 0, 1)
                )

            channel_flags[chan_name] = flags
            channel_scores[chan_name] = score_arr

            n_flagged = int(np.sum(flags))
            log.info(
                "  Channel '%s': %d/%d targets flagged as anomalous",
                chan_name, n_flagged, int(np.sum(valid_mask)),
            )

        # -- Count multi-channel outliers ------------------------------------
        ranked: List[Dict[str, Any]] = []

        for i, profile in enumerate(all_profiles):
            flagged_channels = []
            channel_detail = {}

            for chan_name in channel_flags:
                if channel_flags[chan_name][i]:
                    flagged_channels.append(chan_name)
                    channel_detail[chan_name] = {
                        "flagged": True,
                        "anomaly_score": float(channel_scores[chan_name][i]),
                    }
                elif chan_name in channel_scores and channel_scores[chan_name][i] > 0:
                    channel_detail[chan_name] = {
                        "flagged": False,
                        "anomaly_score": float(channel_scores[chan_name][i]),
                    }

            n_flagged = len(flagged_channels)

            # Compute combined convergence score
            # The convergence score weights both the number of channels and
            # the strength of the anomaly in each channel.
            if n_flagged > 0:
                mean_score = np.mean([
                    channel_scores[c][i] for c in flagged_channels
                ])
                # Exponential boost for multi-channel convergence
                convergence = float(
                    mean_score * (10.0 ** (n_flagged - 1))
                )
            else:
                convergence = 0.0

            ranked.append({
                "ra": profile.ra,
                "dec": profile.dec,
                "label": profile.label,
                "n_channels_flagged": n_flagged,
                "flagged_channels": flagged_channels,
                "channel_details": channel_detail,
                "convergence_score": convergence,
                "n_layers_available": profile.n_layers,
            })

        # Sort by convergence score (highest first), then by n_channels
        ranked.sort(key=lambda x: (
            -x["n_channels_flagged"], -x["convergence_score"]
        ))

        # Log summary statistics
        for n in range(4, 0, -1):
            count = sum(1 for t in ranked if t["n_channels_flagged"] >= n)
            if count > 0:
                log.info(
                    "  Targets flagged in >= %d channels: %d (%.4f%%)",
                    n, count, 100.0 * count / n_targets,
                )

        result = AnomalyStackResult(ranked_targets=ranked)
        return result

    # ------------------------------------------------------------------ #
    #  6. CORRELATE ALL                                                   #
    # ------------------------------------------------------------------ #

    def correlate_all(
        self, profile: TargetProfile,
    ) -> Dict[str, Any]:
        """Run all correlation types on a single profile.

        Parameters
        ----------
        profile : TargetProfile

        Returns
        -------
        dict
            Keys: "spatial", "temporal", "cross_band_temporal",
            plus metadata.
        """
        log.info("Running all correlations for %s", profile.label)

        spatial = self.correlate_spatial(profile)
        temporal = self.correlate_temporal(profile)
        cross_band = self.correlate_cross_band_temporal(profile)

        # Overall assessment
        is_high_priority = (
            spatial.n_channels_flagged >= self._min_channels
            or cross_band.is_anticorrelated
        )

        result = {
            "target": {"ra": profile.ra, "dec": profile.dec},
            "n_layers": profile.n_layers,
            "available_layers": profile.available_layers,
            "spatial": {
                "n_channels_flagged": spatial.n_channels_flagged,
                "channel_names": spatial.channel_names,
                "anomalies": {
                    name: {"score": a.score, "details": a.details}
                    for name, a in spatial.anomalies.items()
                },
            },
            "temporal": {
                "changes": temporal.changes,
            },
            "cross_band_temporal": {
                "correlation_coefficient": cross_band.correlation_coefficient,
                "is_anticorrelated": cross_band.is_anticorrelated,
                "aligned_data": cross_band.aligned_data,
            },
            "is_high_priority": is_high_priority,
        }

        if is_high_priority:
            log.warning(
                "*** HIGH PRIORITY TARGET: %s (spatial=%d channels, "
                "anti-corr=%s) ***",
                profile.label, spatial.n_channels_flagged,
                cross_band.is_anticorrelated,
            )

        return result

    # ------------------------------------------------------------------ #
    #  7. PROCESS TARGET LIST                                             #
    # ------------------------------------------------------------------ #

    def process_target_list(
        self, targets: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """Process a list of (ra, dec) targets end-to-end.

        For each target: builds a profile, runs all correlations.
        Then runs anomaly stacking across all profiles.

        Parameters
        ----------
        targets : list of (float, float)
            List of (RA, Dec) coordinate pairs.

        Returns
        -------
        dict
            Complete results including per-target correlations and
            the anomaly stacking ranking.
        """
        log.info(
            "Processing %d targets through multi-modal correlation engine",
            len(targets),
        )

        profiles: List[TargetProfile] = []
        per_target_results: List[Dict[str, Any]] = []

        for i, (ra, dec) in enumerate(targets):
            log.info("--- Target %d/%d: (%.4f, %.4f) ---", i + 1, len(targets), ra, dec)

            # Build profile
            profile = self.build_target_profile(ra, dec)
            profiles.append(profile)

            # Run all correlations
            result = self.correlate_all(profile)
            per_target_results.append(result)

        # Run anomaly stacking across all profiles
        stacking = self.correlate_anomaly_stacking(profiles)

        # Build summary
        n_high_priority = sum(
            1 for r in per_target_results if r.get("is_high_priority")
        )
        n_multi_channel = sum(
            1 for t in stacking.ranked_targets
            if t["n_channels_flagged"] >= self._min_channels
        )

        summary = {
            "n_targets": len(targets),
            "n_high_priority_from_correlations": n_high_priority,
            "n_multi_channel_from_stacking": n_multi_channel,
            "per_target": per_target_results,
            "anomaly_stacking": {
                "ranked_targets": stacking.ranked_targets,
            },
        }

        log.info(
            "Processing complete: %d targets, %d high-priority, "
            "%d multi-channel outliers",
            len(targets), n_high_priority, n_multi_channel,
        )

        return summary

    # ================================================================== #
    #  PRIVATE: anomaly detection helpers                                 #
    # ================================================================== #

    def _check_ir_excess(
        self, profile: TargetProfile,
    ) -> Tuple[float, Dict[str, Any]]:
        """Check for infrared excess using the IR excess processor.

        Returns (score, details) where score is in [0, 1].
        """
        # Build a photometry dict from the profile layers
        phot: Dict[str, Any] = {}

        if profile.optical:
            for key in ("G", "BP", "RP"):
                val = profile.optical.get(key)
                if val is not None:
                    phot[key] = val

        if profile.near_ir:
            for key in ("J", "H", "Ks", "J_err", "H_err", "Ks_err"):
                val = profile.near_ir.get(key)
                if val is not None:
                    phot[key] = val

        if profile.mid_ir:
            for key in ("W1", "W2", "W3", "W4",
                        "W1_err", "W2_err", "W3_err", "W4_err"):
                val = profile.mid_ir.get(key)
                if val is not None:
                    phot[key] = val

        if not phot.get("W3") and not phot.get("W4"):
            return 0.0, {"note": "No WISE W3/W4 data"}

        if _HAS_IR_EXCESS:
            try:
                result = _compute_ir_excess(phot)
                # Convert to 0-1 score based on sigma significance
                max_sigma = max(
                    result.sigma_W3 or 0.0,
                    result.sigma_W4 or 0.0,
                )
                # Map sigma to [0, 1]: 0 at sigma=0, 0.5 at sigma=3, ~1 at sigma=10
                score = float(np.clip(max_sigma / (2.0 * self._sigma), 0.0, 1.0))

                details = {
                    "fitted_teff": result.fitted_teff,
                    "excess_W3": result.excess_W3,
                    "excess_W4": result.excess_W4,
                    "sigma_W3": result.sigma_W3,
                    "sigma_W4": result.sigma_W4,
                    "is_candidate": result.is_candidate,
                }
                return score, details
            except Exception as exc:
                log.warning("IR excess computation failed: %s", exc)

        # Fallback: crude Ks - W3 / Ks - W4 excess check
        ks = phot.get("Ks")
        w3 = phot.get("W3")
        w4 = phot.get("W4")
        details: Dict[str, Any] = {"method": "crude_color_excess"}

        score = 0.0
        if ks is not None and w3 is not None:
            excess_w3 = ks - w3  # positive = W3 brighter than expected
            details["ks_minus_w3"] = excess_w3
            if excess_w3 > 0.5:
                score = max(score, min(excess_w3 / 2.0, 1.0))
        if ks is not None and w4 is not None:
            excess_w4 = ks - w4
            details["ks_minus_w4"] = excess_w4
            if excess_w4 > 1.0:
                score = max(score, min(excess_w4 / 3.0, 1.0))

        return score, details

    def _check_transit_anomaly(
        self, profile: TargetProfile,
    ) -> Tuple[float, Dict[str, Any]]:
        """Check for transit anomalies using the transit anomaly detector."""
        if not profile.lightcurve:
            return 0.0, {"note": "No light curve data"}

        time_arr = np.array(profile.lightcurve["time"], dtype=np.float64)
        flux_arr = np.array(profile.lightcurve["flux"], dtype=np.float64)
        flux_err = profile.lightcurve.get("flux_err")
        if flux_err:
            flux_err = np.array(flux_err, dtype=np.float64)
        else:
            flux_err = None

        if len(time_arr) < 20:
            return 0.0, {"note": "Too few data points"}

        if _HAS_TRANSIT:
            try:
                result = _detect_transit_anomaly(time_arr, flux_arr, flux_err)
                details = {
                    "period": result.period,
                    "depth": result.depth,
                    "symmetry_score": result.symmetry_score,
                    "depth_variability": result.depth_variability,
                    "shape_residual": result.shape_residual,
                    "is_anomalous": result.is_anomalous,
                    "n_dips": len(result.detected_dips),
                }

                # Also run irregular dimming detection
                irreg = _detect_irregular_dimming(time_arr, flux_arr)
                details["irregular_events"] = irreg.n_events
                details["irregular_max_depth"] = irreg.max_depth
                details["irregular_score"] = irreg.anomaly_score

                # Combined score: max of periodic and irregular
                score = float(np.clip(
                    max(result.anomaly_score, irreg.anomaly_score), 0.0, 1.0
                ))
                return score, details

            except Exception as exc:
                log.warning("Transit anomaly detection failed: %s", exc)

        # Fallback: simple flux statistics
        flux_std = np.std(flux_arr)
        flux_range = np.ptp(flux_arr)
        max_dip = 1.0 - np.min(flux_arr)

        score = float(np.clip(max_dip / 0.05, 0.0, 1.0))  # 5% dip = score 1.0
        details = {
            "method": "simple_statistics",
            "flux_std": float(flux_std),
            "flux_range": float(flux_range),
            "max_dip_depth": float(max_dip),
        }
        return score, details

    def _check_radio_signal(
        self, profile: TargetProfile,
    ) -> Tuple[float, Dict[str, Any]]:
        """Check for radio signal candidates."""
        if not profile.radio:
            return 0.0, {"note": "No radio data"}

        details: Dict[str, Any] = {
            "telescope": profile.radio.get("telescope"),
            "freq_range_mhz": (
                profile.radio.get("freq_start_mhz"),
                profile.radio.get("freq_end_mhz"),
            ),
        }

        has_spectrogram = profile.radio.get("has_spectrogram", False)
        candidates = profile.radio.get("candidates", [])

        if candidates:
            # Real candidates from radio processing
            non_rfi = [c for c in candidates if not c.get("is_rfi", False)]
            best_snr = max((c.get("snr", 0) for c in non_rfi), default=0)
            details["n_candidates"] = len(candidates)
            details["n_non_rfi"] = len(non_rfi)
            details["best_snr"] = best_snr

            # Score based on best non-RFI candidate SNR
            score = float(np.clip(best_snr / 30.0, 0.0, 1.0))
            return score, details

        # If we have observation metadata but no processed candidates,
        # assign a low baseline score for having radio coverage.
        if has_spectrogram:
            details["note"] = "Spectrogram available but not yet processed"
            return 0.1, details

        details["note"] = "Observation metadata only"
        return 0.05, details

    def _check_astrometry_anomaly(
        self, profile: TargetProfile,
    ) -> Tuple[float, Dict[str, Any]]:
        """Check for astrometric anomalies (RUWE, excess noise, etc.)."""
        if not profile.astrometry:
            return 0.0, {"note": "No astrometry data"}

        ruwe = profile.astrometry.get("ruwe")
        excess_noise = profile.astrometry.get("astrometric_excess_noise")
        parallax = profile.astrometry.get("parallax")
        parallax_error = profile.astrometry.get("parallax_error")

        details: Dict[str, Any] = {
            "ruwe": ruwe,
            "excess_noise": excess_noise,
            "parallax": parallax,
            "parallax_error": parallax_error,
        }

        score = 0.0

        # RUWE > 1.4 is anomalous (poor single-star fit)
        if ruwe is not None and np.isfinite(ruwe):
            if ruwe > 1.4:
                # Scale: RUWE 1.4 -> 0.5, RUWE 3.0 -> 1.0
                score = max(score, float(np.clip(
                    (ruwe - 1.4) / 1.6 * 0.5 + 0.5, 0.0, 1.0
                )))
                details["ruwe_anomalous"] = True

        # Parallax quality: large relative error suggests something odd
        if (parallax is not None and parallax_error is not None
                and parallax > 0 and parallax_error > 0):
            frac_err = parallax_error / parallax
            if frac_err > 0.2:
                score = max(score, float(np.clip(frac_err, 0.0, 0.5)))
                details["parallax_quality"] = "poor"

        # Excess astrometric noise
        if excess_noise is not None and np.isfinite(excess_noise):
            if excess_noise > 1.0:
                score = max(score, float(np.clip(
                    excess_noise / 5.0, 0.0, 1.0
                )))
                details["excess_noise_anomalous"] = True

        return score, details

    # ================================================================== #
    #  PRIVATE: feature extraction for anomaly stacking                  #
    # ================================================================== #

    def _build_channel_features(
        self, profiles: List[TargetProfile],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Build feature matrices for each independent dataset channel.

        Returns a dict mapping channel_name -> (features, valid_mask)
        where features is (n_targets, n_features) and valid_mask is
        (n_targets,) boolean.
        """
        n = len(profiles)
        channels: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # -- Channel: optical photometry ------------------------------------
        opt_features = np.zeros((n, 4), dtype=np.float64)
        opt_valid = np.zeros(n, dtype=bool)
        for i, p in enumerate(profiles):
            if p.optical:
                g = p.optical.get("G")
                bp_rp = p.optical.get("bp_rp")
                teff = p.optical.get("teff")
                if g is not None and np.isfinite(g):
                    opt_features[i, 0] = g
                    opt_features[i, 1] = bp_rp if (
                        bp_rp is not None and np.isfinite(bp_rp)
                    ) else 0.0
                    opt_features[i, 2] = teff if (
                        teff is not None and np.isfinite(teff)
                    ) else 5000.0
                    opt_features[i, 3] = p.optical.get("logg", 4.0) or 4.0
                    opt_valid[i] = True
        if np.sum(opt_valid) >= 5:
            channels["optical"] = (opt_features, opt_valid)

        # -- Channel: infrared colors ----------------------------------------
        ir_features = np.zeros((n, 6), dtype=np.float64)
        ir_valid = np.zeros(n, dtype=bool)
        for i, p in enumerate(profiles):
            if p.near_ir and p.mid_ir:
                j = p.near_ir.get("J")
                ks = p.near_ir.get("Ks")
                w1 = p.mid_ir.get("W1")
                w2 = p.mid_ir.get("W2")
                w3 = p.mid_ir.get("W3")
                w4 = p.mid_ir.get("W4")
                if all(v is not None and np.isfinite(v)
                       for v in [j, ks, w1, w3]):
                    ir_features[i, 0] = j - ks if ks else 0.0
                    ir_features[i, 1] = ks - w1 if (ks and w1) else 0.0
                    ir_features[i, 2] = w1 - w2 if (w1 and w2) else 0.0
                    ir_features[i, 3] = w1 - w3 if (w1 and w3) else 0.0
                    ir_features[i, 4] = w3 - w4 if (w3 and w4) else 0.0
                    ir_features[i, 5] = ks - w3 if (ks and w3) else 0.0
                    ir_valid[i] = True
        if np.sum(ir_valid) >= 5:
            channels["infrared"] = (ir_features, ir_valid)

        # -- Channel: lightcurve statistics ----------------------------------
        lc_features = np.zeros((n, 5), dtype=np.float64)
        lc_valid = np.zeros(n, dtype=bool)
        for i, p in enumerate(profiles):
            if p.lightcurve and len(p.lightcurve.get("flux", [])) >= 20:
                flux = np.array(p.lightcurve["flux"], dtype=np.float64)
                valid = np.isfinite(flux)
                flux = flux[valid]
                if len(flux) >= 20:
                    lc_features[i, 0] = np.std(flux)
                    lc_features[i, 1] = np.ptp(flux)
                    lc_features[i, 2] = 1.0 - np.min(flux)  # max dip depth
                    lc_features[i, 3] = float(
                        np.mean(((flux - np.mean(flux)) / max(np.std(flux), 1e-10)) ** 3)
                    )  # skewness
                    lc_features[i, 4] = float(
                        np.mean(((flux - np.mean(flux)) / max(np.std(flux), 1e-10)) ** 4)
                    )  # kurtosis
                    lc_valid[i] = True
        if np.sum(lc_valid) >= 5:
            channels["lightcurve"] = (lc_features, lc_valid)

        # -- Channel: astrometry --------------------------------------------
        ast_features = np.zeros((n, 4), dtype=np.float64)
        ast_valid = np.zeros(n, dtype=bool)
        for i, p in enumerate(profiles):
            if p.astrometry:
                ruwe = p.astrometry.get("ruwe")
                plx = p.astrometry.get("parallax")
                plx_err = p.astrometry.get("parallax_error")
                exc_noise = p.astrometry.get("astrometric_excess_noise")
                if ruwe is not None and np.isfinite(ruwe):
                    ast_features[i, 0] = ruwe
                    ast_features[i, 1] = float(plx or 0.0)
                    ast_features[i, 2] = float(plx_err or 0.0)
                    ast_features[i, 3] = float(exc_noise or 0.0)
                    ast_valid[i] = True
        if np.sum(ast_valid) >= 5:
            channels["astrometry"] = (ast_features, ast_valid)

        # -- Channel: epoch photometry variability ---------------------------
        ep_features = np.zeros((n, 3), dtype=np.float64)
        ep_valid = np.zeros(n, dtype=bool)
        for i, p in enumerate(profiles):
            if p.epoch_photometry:
                mags = p.epoch_photometry.get("mag", [])
                if len(mags) >= 10:
                    mag_arr = np.array(mags, dtype=np.float64)
                    valid = np.isfinite(mag_arr)
                    mag_arr = mag_arr[valid]
                    if len(mag_arr) >= 10:
                        ep_features[i, 0] = np.std(mag_arr)
                        ep_features[i, 1] = np.ptp(mag_arr)
                        ep_features[i, 2] = float(np.mean(
                            ((mag_arr - np.mean(mag_arr)) /
                             max(np.std(mag_arr), 1e-10)) ** 3
                        ))  # skewness
                        ep_valid[i] = True
        if np.sum(ep_valid) >= 5:
            channels["epoch_photometry"] = (ep_features, ep_valid)

        log.info(
            "Built %d feature channels for anomaly stacking: %s",
            len(channels), list(channels.keys()),
        )
        return channels


# ============================================================================
#  CLI demonstration with synthetic data
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- Multi-Modal Correlation Engine Demo")
    print("  THE CORE INNOVATION: stacking all datasets to find")
    print("  what no single dataset can reveal.")
    print("=" * 72)
    print()

    # We create synthetic TargetProfiles to demonstrate all four
    # correlation modes without requiring network access to real catalogs.

    rng = np.random.default_rng(seed=42)

    # ---- Helper: generate a synthetic light curve -------------------------
    def _make_lc(n=2000, period=3.5, depth=0.01, noise=0.001, anomalous=False):
        t = np.linspace(0, 30, n)
        f = np.ones(n) + rng.normal(0, noise, n)
        if anomalous:
            # Asymmetric, variable-depth transits
            for k in range(int(30 / period) + 1):
                tc = 0.5 * period + k * period
                d = depth * (0.5 + rng.random())
                mask = np.abs(t - tc) < 0.15
                f[mask] -= d * rng.random(int(np.sum(mask)))
        else:
            phase = ((t % period) / period)
            in_transit = np.abs(phase - 0.5) < (0.1 / period) / 2
            f[in_transit] -= depth
        return t.tolist(), f.tolist(), (np.ones(n) * noise).tolist()

    # ---- Helper: generate synthetic epoch photometry -----------------------
    def _make_epoch_phot(n=50, trend=0.0, scatter=0.02):
        times = np.sort(rng.uniform(0, 1800, n))  # ~5 years in days
        mags = 10.0 + trend * times / 365.25 + rng.normal(0, scatter, n)
        return times.tolist(), mags.tolist()

    # ---- Create synthetic target profiles ---------------------------------
    profiles = []

    # Target 1: NORMAL STAR (no anomalies)
    t, f, fe = _make_lc(depth=0.005, noise=0.001, anomalous=False)
    ep_t, ep_m = _make_epoch_phot(trend=0.0, scatter=0.01)
    p1 = TargetProfile(
        ra=180.0, dec=45.0,
        optical={"G": 10.5, "BP": 11.0, "RP": 10.1, "bp_rp": 0.9,
                 "teff": 5500.0, "logg": 4.3, "source_id": 100001},
        near_ir={"J": 9.5, "H": 9.2, "Ks": 9.1},
        mid_ir={"W1": 9.0, "W2": 9.0, "W3": 9.0, "W4": 8.9},
        epoch_photometry={"source_id": 100001, "n_transits": 50,
                          "time": ep_t, "mag": ep_m},
        lightcurve={"time": t, "flux": f, "flux_err": fe, "n_points": len(t)},
        astrometry={"source_id": 100001, "parallax": 5.0,
                    "parallax_error": 0.02, "pmra": 10.0, "pmdec": -5.0,
                    "ruwe": 1.05, "astrometric_excess_noise": 0.1},
    )
    profiles.append(p1)
    print("[Target 1] Normal star at (180.0, 45.0)")
    print(f"  Layers: {p1.available_layers}")

    # Target 2: IR EXCESS CANDIDATE (possible Dyson sphere)
    t, f, fe = _make_lc(depth=0.008, noise=0.001, anomalous=False)
    ep_t, ep_m = _make_epoch_phot(trend=0.0, scatter=0.015)
    p2 = TargetProfile(
        ra=200.0, dec=30.0,
        optical={"G": 11.0, "BP": 11.5, "RP": 10.5, "bp_rp": 1.0,
                 "teff": 5200.0, "logg": 4.4, "source_id": 200001},
        near_ir={"J": 9.8, "H": 9.5, "Ks": 9.4},
        mid_ir={"W1": 9.3, "W2": 9.2, "W3": 6.5, "W4": 4.0},  # HUGE W3/W4 excess!
        epoch_photometry={"source_id": 200001, "n_transits": 45,
                          "time": ep_t, "mag": ep_m},
        lightcurve={"time": t, "flux": f, "flux_err": fe, "n_points": len(t)},
        astrometry={"source_id": 200001, "parallax": 3.0,
                    "parallax_error": 0.05, "pmra": 8.0, "pmdec": -3.0,
                    "ruwe": 1.1, "astrometric_excess_noise": 0.2},
    )
    profiles.append(p2)
    print("[Target 2] IR excess candidate at (200.0, 30.0)")
    print(f"  W3={p2.mid_ir['W3']}, W4={p2.mid_ir['W4']} (anomalously bright!)")

    # Target 3: TRANSIT ANOMALY + IR EXCESS (double convergence!)
    t, f, fe = _make_lc(depth=0.02, noise=0.001, anomalous=True)
    ep_t, ep_m = _make_epoch_phot(trend=0.05, scatter=0.02)  # dimming trend
    p3 = TargetProfile(
        ra=150.0, dec=60.0,
        optical={"G": 12.0, "BP": 12.6, "RP": 11.4, "bp_rp": 1.2,
                 "teff": 4800.0, "logg": 4.5, "source_id": 300001},
        near_ir={"J": 10.2, "H": 9.8, "Ks": 9.7},
        mid_ir={"W1": 9.5, "W2": 9.4, "W3": 7.0, "W4": 5.5},  # IR excess
        epoch_photometry={"source_id": 300001, "n_transits": 60,
                          "time": ep_t, "mag": ep_m},
        lightcurve={"time": t, "flux": f, "flux_err": fe, "n_points": len(t)},
        radio={"target": "EXODUS-003", "telescope": "GBT",
               "freq_start_mhz": 1000, "freq_end_mhz": 2000,
               "has_spectrogram": True, "source": "simulated",
               "candidates": [
                   {"snr": 12.5, "is_rfi": False, "drift_rate": 1.2},
               ]},
        exoplanets={"n_planets": 1, "has_hz_planet": True,
                    "planets": [{"name": "EXODUS-003 b", "hz_flag": True,
                                 "radius_earth": 1.2}]},
        astrometry={"source_id": 300001, "parallax": 2.0,
                    "parallax_error": 0.08, "pmra": 5.0, "pmdec": -2.0,
                    "ruwe": 1.8, "astrometric_excess_noise": 2.5},
    )
    profiles.append(p3)
    print("[Target 3] MULTI-ANOMALY at (150.0, 60.0)")
    print("  Transit anomaly + IR excess + radio signal + astrometric anomaly")

    # Target 4: OPTICAL-IR ANTI-CORRELATION (the Dyson swarm smoking gun)
    # Create time-series where optical dims while IR brightens.
    # Epoch photometry stores magnitudes: higher mag = fainter (less flux).
    # The correlator converts mag -> flux: flux = 10^(-0.4*(mag - median)).
    # So when mag goes UP (star fainter), optical flux goes DOWN.
    # For anti-correlation: when optical flux is LOW, IR flux should be HIGH.
    # Therefore: IR flux should track the SAME sine pattern as the magnitude
    # (both increase together), because rising mag = falling optical flux.
    ep_t_sync = np.sort(rng.uniform(0, 1800, 80)).tolist()
    sin_pattern = np.sin(2 * np.pi * np.array(ep_t_sync) / 400)
    ep_m_sync = (10.0 + 0.1 * sin_pattern
                 + rng.normal(0, 0.005, 80)).tolist()  # optical mag oscillation
    ir_t_sync = ep_t_sync  # same time stamps
    # IR BRIGHTENS when optical DIMS (same phase as mag, opposite phase to flux)
    ir_flux_sync = (5.0 + 0.08 * sin_pattern
                    + rng.normal(0, 0.003, 80)).tolist()

    t4, f4, fe4 = _make_lc(depth=0.015, noise=0.001, anomalous=True)
    p4 = TargetProfile(
        ra=270.0, dec=-10.0,
        optical={"G": 9.0, "BP": 9.4, "RP": 8.6, "bp_rp": 0.8,
                 "teff": 5800.0, "logg": 4.2, "source_id": 400001},
        near_ir={"J": 8.2, "H": 7.9, "Ks": 7.8},
        mid_ir={"W1": 7.6, "W2": 7.5, "W3": 5.0, "W4": 3.5},  # IR excess
        epoch_photometry={"source_id": 400001, "n_transits": 80,
                          "time": ep_t_sync, "mag": ep_m_sync},
        lightcurve={"time": t4, "flux": f4, "flux_err": fe4, "n_points": len(t4)},
        mid_ir_timeseries={"time": ir_t_sync, "flux": ir_flux_sync,
                           "n_epochs": 80},
        astrometry={"source_id": 400001, "parallax": 8.0,
                    "parallax_error": 0.01, "pmra": 15.0, "pmdec": -8.0,
                    "ruwe": 2.1, "astrometric_excess_noise": 3.0},
    )
    profiles.append(p4)
    print("[Target 4] ANTI-CORRELATION CANDIDATE at (270.0, -10.0)")
    print("  Optical dims while IR brightens (Dyson swarm signature)")

    # Target 5: Another normal star for stacking baseline
    t5, f5, fe5 = _make_lc(depth=0.003, noise=0.001, anomalous=False)
    ep_t5, ep_m5 = _make_epoch_phot(trend=0.0, scatter=0.01)
    p5 = TargetProfile(
        ra=90.0, dec=20.0,
        optical={"G": 9.5, "BP": 9.9, "RP": 9.1, "bp_rp": 0.8,
                 "teff": 5700.0, "logg": 4.3, "source_id": 500001},
        near_ir={"J": 8.8, "H": 8.5, "Ks": 8.4},
        mid_ir={"W1": 8.3, "W2": 8.3, "W3": 8.2, "W4": 8.1},
        epoch_photometry={"source_id": 500001, "n_transits": 40,
                          "time": ep_t5, "mag": ep_m5},
        lightcurve={"time": t5, "flux": f5, "flux_err": fe5, "n_points": len(t5)},
        astrometry={"source_id": 500001, "parallax": 10.0,
                    "parallax_error": 0.01, "pmra": 20.0, "pmdec": -10.0,
                    "ruwe": 1.0, "astrometric_excess_noise": 0.05},
    )
    profiles.append(p5)
    print("[Target 5] Normal star at (90.0, 20.0)")

    # Target 6: Third normal star
    t6, f6, fe6 = _make_lc(depth=0.004, noise=0.001, anomalous=False)
    ep_t6, ep_m6 = _make_epoch_phot(trend=0.0, scatter=0.012)
    p6 = TargetProfile(
        ra=310.0, dec=50.0,
        optical={"G": 11.5, "BP": 12.0, "RP": 11.0, "bp_rp": 1.0,
                 "teff": 5100.0, "logg": 4.4, "source_id": 600001},
        near_ir={"J": 10.0, "H": 9.7, "Ks": 9.6},
        mid_ir={"W1": 9.5, "W2": 9.5, "W3": 9.4, "W4": 9.3},
        epoch_photometry={"source_id": 600001, "n_transits": 35,
                          "time": ep_t6, "mag": ep_m6},
        lightcurve={"time": t6, "flux": f6, "flux_err": fe6, "n_points": len(t6)},
        astrometry={"source_id": 600001, "parallax": 4.0,
                    "parallax_error": 0.03, "pmra": 6.0, "pmdec": -4.0,
                    "ruwe": 1.1, "astrometric_excess_noise": 0.15},
    )
    profiles.append(p6)
    print("[Target 6] Normal star at (310.0, 50.0)")

    print()

    # ---- Initialize the correlator ----------------------------------------
    correlator = MultiModalCorrelator()

    # ---- 1. Spatial Correlation -------------------------------------------
    print("=" * 72)
    print("  [1] SPATIAL CORRELATION (same star, different wavelengths)")
    print("=" * 72)
    for p in profiles:
        spatial = correlator.correlate_spatial(p)
        flag_str = (
            f"FLAGGED: {', '.join(spatial.channel_names)}"
            if spatial.n_channels_flagged > 0
            else "clean"
        )
        print(f"  {p.label}: {spatial.n_channels_flagged} channels -- {flag_str}")
        for name, anom in spatial.anomalies.items():
            if anom.score >= 0.5:
                print(f"    -> {name}: score={anom.score:.3f}")
    print()

    # ---- 2. Temporal Correlation ------------------------------------------
    print("=" * 72)
    print("  [2] TEMPORAL CORRELATION (same position, different decades)")
    print("=" * 72)
    for p in profiles:
        temporal = correlator.correlate_temporal(p)
        print(f"  {p.label}: {len(temporal.changes)} change indicators")
        for change_name, change_data in temporal.changes.items():
            if isinstance(change_data, dict) and change_data.get("is_significant"):
                print(f"    -> {change_name}: SIGNIFICANT")
            elif isinstance(change_data, dict) and change_data.get("is_anomalous"):
                print(f"    -> {change_name}: ANOMALOUS")
    print()

    # ---- 3. Cross-Band Temporal Correlation --------------------------------
    print("=" * 72)
    print("  [3] CROSS-BAND TEMPORAL CORRELATION (the Dyson swarm detector)")
    print("=" * 72)
    for p in profiles:
        cross = correlator.correlate_cross_band_temporal(p)
        status = "*** ANTI-CORRELATED ***" if cross.is_anticorrelated else "normal"
        r_str = f"r={cross.correlation_coefficient:.4f}" if (
            cross.aligned_data.get("n_overlap_points", 0) > 0
        ) else "no overlap"
        print(f"  {p.label}: {r_str} -- {status}")
    print()

    # ---- 4. Anomaly Stacking ----------------------------------------------
    print("=" * 72)
    print("  [4] ANOMALY STACKING (IsolationForest across all targets)")
    print("=" * 72)
    stacking = correlator.correlate_anomaly_stacking(profiles)
    print()
    for target in stacking.ranked_targets:
        n = target["n_channels_flagged"]
        chans = ", ".join(target["flagged_channels"]) if target["flagged_channels"] else "none"
        conv = target["convergence_score"]
        marker = ""
        if n >= 3:
            marker = " <<< EXTRAORDINARY"
        elif n >= 2:
            marker = " << INTERESTING"
        elif n >= 1:
            marker = " < notable"
        print(
            f"  {target['label']}: {n} channels flagged [{chans}] "
            f"convergence={conv:.4f}{marker}"
        )

    # ---- Summary -----------------------------------------------------------
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    multi_flagged = [
        t for t in stacking.ranked_targets
        if t["n_channels_flagged"] >= 2
    ]
    print(f"  Targets flagged by >= 2 independent channels: {len(multi_flagged)}")
    for t in multi_flagged:
        print(
            f"    {t['label']}: {t['n_channels_flagged']} channels "
            f"({', '.join(t['flagged_channels'])})"
        )

    # Find anti-correlated targets
    print()
    for p in profiles:
        cross = correlator.correlate_cross_band_temporal(p)
        if cross.is_anticorrelated:
            print(
                f"  ANTI-CORRELATION at {p.label}: r={cross.correlation_coefficient:.4f}"
            )
            print(
                "    -> Optical dims while IR brightens = Dyson swarm signature"
            )

    print()
    print("=" * 72)
    print("  Demo complete.")
    print("=" * 72)
