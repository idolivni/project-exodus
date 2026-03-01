"""
Infrared excess computation for Dyson sphere candidate detection.

Implements the photospheric-extrapolation method used by the Project Hephaistos
pipeline (Suazo et al. 2024): fit a Planck blackbody to the optical / near-IR
spectral energy distribution (Gaia G, BP, RP + 2MASS J, H, Ks), extrapolate
to WISE mid-IR wavelengths (W3 at 12 um, W4 at 22 um), and flag sources whose
observed WISE magnitudes are significantly brighter than the model prediction.

A star surrounded by a partial Dyson sphere (or any large warm circumstellar
structure) re-radiates absorbed starlight in the mid-infrared, producing an
excess above the bare photosphere that this module quantifies.

Key references
--------------
- Suazo et al. 2024 (Project Hephaistos), MNRAS 527, 1
- Wright et al. 2014, ApJ 792, 26 (Glimpsing Heat from Alien Technologies)
- Griffith et al. 2015, ApJS 217, 25
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))

from src.utils import get_config, get_logger

log = get_logger("processing.ir_excess")

# =====================================================================
#  Physical constants (SI)
# =====================================================================
_H_PLANCK = 6.62607015e-34   # Planck constant  [J s]
_K_BOLTZ = 1.380649e-23      # Boltzmann const   [J / K]
_C_LIGHT = 2.99792458e8      # Speed of light    [m / s]

# =====================================================================
#  Band definitions
# =====================================================================
#  Effective wavelengths in microns, converted to metres internally.

BAND_WAVELENGTHS_UM: Dict[str, float] = {
    # Gaia DR3
    "G":   0.673,
    "BP":  0.532,
    "RP":  0.797,
    # 2MASS
    "J":   1.235,
    "H":   1.662,
    "Ks":  2.159,
    # WISE
    "W1":  3.353,
    "W2":  4.603,
    "W3":  11.561,
    "W4":  22.088,
}

# Zero-point fluxes in Jy for the Vega magnitude system (used to convert
# magnitudes to physical flux densities and back).
# Gaia uses its own Vega-based system; we adopt representative values that
# are sufficient for the broadband blackbody fit.
ZERO_POINT_FLUX_JY: Dict[str, float] = {
    "G":   3228.75,
    "BP":  3552.01,
    "RP":  2554.95,
    "J":   1594.0,
    "H":   1024.0,
    "Ks":  666.7,
    "W1":  309.540,
    "W2":  171.787,
    "W3":  31.674,
    "W4":  8.363,
}

# Bands used to *fit* the stellar photosphere (optical + near-IR only).
FIT_BANDS: Sequence[str] = ("G", "BP", "RP", "J", "H", "Ks")

# Bands where we look for excess (mid-IR).
EXCESS_BANDS: Sequence[str] = ("W3", "W4")

# Default photometric uncertainty (mag) assigned when none is provided.
# Audit fix F1: band-dependent defaults (W3/W4 scatter is 3-6x larger)
_DEFAULT_MAG_ERR_BY_BAND = {
    "G": 0.003, "BP": 0.01, "RP": 0.01,
    "J": 0.03, "H": 0.03, "Ks": 0.03,
    "W1": 0.03, "W2": 0.03,
    "W3": 0.15, "W4": 0.30,
}
_DEFAULT_MAG_ERR = 0.05  # fallback for unknown bands


def _get_default_mag_err(band: str) -> float:
    """Return band-appropriate default magnitude error."""
    return _DEFAULT_MAG_ERR_BY_BAND.get(band, _DEFAULT_MAG_ERR)

# =====================================================================
#  Anomaly threshold (from project config)
# =====================================================================

def _anomaly_sigma() -> float:
    """Return the detection threshold in sigma from config."""
    try:
        cfg = get_config()
        return float(cfg["search"]["anomaly_sigma"])
    except Exception:
        return 3.0


# =====================================================================
#  Result container
# =====================================================================

@dataclass
class IRExcessResult:
    """Results of the infrared-excess computation for a single source.

    Attributes
    ----------
    source_id : str or None
        Optional identifier carried through from input photometry.
    fitted_teff : float
        Best-fit effective temperature from the blackbody model [K].
    fitted_scale : float
        Multiplicative scaling factor of the blackbody (encodes solid
        angle / distance information).
    excess_W3 : float or None
        Observed minus predicted W3 magnitude (negative = brighter than
        model, i.e. excess emission).
    excess_W4 : float or None
        Same for W4.
    sigma_W3 : float or None
        Significance of the W3 excess in sigma (|excess| / uncertainty).
    sigma_W4 : float or None
        Significance of the W4 excess.
    is_candidate : bool
        True if any WISE band shows excess above the configured threshold.
    fit_bands_used : int
        Number of photometric bands actually used in the fit.
    fit_chi2_reduced : float
        Reduced chi-squared of the blackbody fit to the optical/NIR bands.
    warnings : list of str
        Diagnostic messages (e.g. missing bands, poor fit quality).
    """

    source_id: Optional[str] = None
    fitted_teff: float = np.nan
    fitted_scale: float = np.nan
    excess_W3: Optional[float] = None
    excess_W4: Optional[float] = None
    sigma_W3: Optional[float] = None
    sigma_W4: Optional[float] = None
    is_candidate: bool = False
    fit_bands_used: int = 0
    fit_chi2_reduced: float = np.nan
    # --- Hephaistos contamination flags (Jan 2025 lesson) ---
    # If the WISE centroid shifts between short-wave (W1/W2) and
    # long-wave (W3/W4) bands, the excess likely comes from a
    # background source (AGN / Hot DOG), not the star itself.
    centroid_shift_arcsec: Optional[float] = None
    contamination_flag: bool = False
    contamination_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable summary."""
        flag = "** CANDIDATE **" if self.is_candidate else "normal"
        if self.contamination_flag:
            flag = "CONTAMINATED"
        parts = [
            f"Teff={self.fitted_teff:.0f}K" if np.isfinite(self.fitted_teff) else "Teff=N/A",
            f"bands={self.fit_bands_used}",
        ]
        if self.excess_W3 is not None:
            parts.append(f"dW3={self.excess_W3:+.3f}mag({self.sigma_W3:.1f}sig)")
        if self.excess_W4 is not None:
            parts.append(f"dW4={self.excess_W4:+.3f}mag({self.sigma_W4:.1f}sig)")
        if self.centroid_shift_arcsec is not None:
            parts.append(f"shift={self.centroid_shift_arcsec:.1f}\"")
        parts.append(flag)
        return "  ".join(parts)


# =====================================================================
#  Blackbody model
# =====================================================================

def _planck_flux_density(wavelength_m: np.ndarray, teff: float) -> np.ndarray:
    """Planck spectral radiance B_nu(T) as a function of wavelength.

    Computes B_nu (per unit frequency) since our data is in Jy (flux
    density per unit frequency).  The conversion from B_lambda to B_nu
    is: B_nu = B_lambda * lambda^2 / c, or equivalently:

        B_nu = (2 h nu^3 / c^2) / (exp(h nu / k T) - 1)

    where nu = c / lambda.  The absolute scaling is handled by the fit
    parameter *scale* in the calling function.

    Parameters
    ----------
    wavelength_m : ndarray
        Wavelengths in metres.
    teff : float
        Effective temperature in Kelvin.

    Returns
    -------
    ndarray
        B_nu in W / m^2 / Hz / sr (proportional to Jy).
    """
    # B_nu = (2 h / c^2) * (c / lam)^3 / (exp(hc / lam k T) - 1)
    #      = 2 h c / lam^3 / (exp(hc / lam k T) - 1)
    with np.errstate(over="ignore", under="ignore"):
        exponent = (_H_PLANCK * _C_LIGHT) / (wavelength_m * _K_BOLTZ * teff)
        # Clamp to avoid overflow in exp
        exponent = np.clip(exponent, 0.0, 500.0)
        B_nu = (2.0 * _H_PLANCK * _C_LIGHT / wavelength_m ** 3) / (
            np.exp(exponent) - 1.0
        )
    return B_nu


def _bb_model(wavelength_m: np.ndarray, teff: float, log_scale: float) -> np.ndarray:
    """Scaled blackbody model for curve_fit.

    Parameters
    ----------
    wavelength_m : ndarray
        Wavelengths in metres.
    teff : float
        Effective temperature [K].
    log_scale : float
        log10 of the multiplicative scaling factor.

    Returns
    -------
    ndarray
        Model flux density values (same units as the data fed to curve_fit).
    """
    scale = 10.0 ** log_scale
    return scale * _planck_flux_density(wavelength_m, teff)


# =====================================================================
#  Magnitude <-> flux conversion helpers
# =====================================================================

def _mag_to_flux(mag: float, band: str) -> float:
    """Convert a Vega magnitude to flux density in Jy."""
    zp = ZERO_POINT_FLUX_JY[band]
    return zp * 10.0 ** (-0.4 * mag)


def _flux_to_mag(flux_jy: float, band: str) -> float:
    """Convert flux density in Jy to Vega magnitude."""
    zp = ZERO_POINT_FLUX_JY[band]
    if flux_jy <= 0:
        return np.nan
    return -2.5 * np.log10(flux_jy / zp)


def _mag_err_to_flux_err(mag: float, mag_err: float, band: str) -> float:
    """Propagate magnitude uncertainty to flux uncertainty (Jy)."""
    flux = _mag_to_flux(mag, band)
    # d(flux)/d(mag) = flux * ln(10) * 0.4
    return flux * np.log(10.0) * 0.4 * mag_err


# =====================================================================
#  Core computation
# =====================================================================

def _extract_fit_data(
    phot: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract wavelengths, fluxes, and uncertainties for the fit bands.

    Parameters
    ----------
    phot : dict
        Photometry dict.  Expected keys: band names (e.g. "G", "J") for
        magnitudes; band + "_err" for uncertainties.

    Returns
    -------
    wavelengths_m : ndarray
        Wavelengths in metres for available fit bands.
    fluxes : ndarray
        Flux density in Jy.
    flux_errs : ndarray
        1-sigma flux uncertainty in Jy.
    band_names : list[str]
        Names of the bands actually used.
    """
    wavelengths = []
    fluxes = []
    flux_errs = []
    band_names = []

    for band in FIT_BANDS:
        mag = phot.get(band)
        if mag is None or not np.isfinite(mag):
            continue
        mag_err = phot.get(f"{band}_err")
        if mag_err is None or not np.isfinite(mag_err) or mag_err <= 0:
            mag_err = _get_default_mag_err(band)

        lam_m = BAND_WAVELENGTHS_UM[band] * 1e-6  # um -> m
        flux = _mag_to_flux(mag, band)
        ferr = _mag_err_to_flux_err(mag, mag_err, band)

        wavelengths.append(lam_m)
        fluxes.append(flux)
        flux_errs.append(ferr)
        band_names.append(band)

    return (
        np.array(wavelengths),
        np.array(fluxes),
        np.array(flux_errs),
        band_names,
    )


def _fit_blackbody(
    wavelengths_m: np.ndarray,
    fluxes: np.ndarray,
    flux_errs: np.ndarray,
) -> tuple[float, float, float]:
    """Fit a scaled Planck blackbody to photometric data.

    Parameters
    ----------
    wavelengths_m : ndarray  (N,)
    fluxes : ndarray  (N,)
        Observed flux densities in Jy.
    flux_errs : ndarray  (N,)
        1-sigma uncertainties in Jy.

    Returns
    -------
    teff : float
        Best-fit effective temperature [K].
    log_scale : float
        Best-fit log10(scale).
    chi2_red : float
        Reduced chi-squared of the fit.

    Raises
    ------
    RuntimeError
        If the fit fails to converge.
    """
    n = len(wavelengths_m)
    if n < 2:
        raise RuntimeError("Need at least 2 bands to fit a blackbody")

    # Add a systematic model-uncertainty floor: a Planck blackbody cannot
    # reproduce a real stellar photosphere better than ~2% (molecular
    # absorption features, non-LTE effects, filter bandpass mismatch).
    # Without this floor, very small photometric errors would produce
    # artificially high chi2 values.
    model_floor = 0.02 * fluxes  # 2% of flux
    flux_errs = np.sqrt(flux_errs ** 2 + model_floor ** 2)

    # Initial guesses: estimate Teff from the peak of the SED using
    # Wien's displacement law: lambda_max * T = 2898 um K
    # Use the band with the highest flux as an approximation.
    peak_idx = np.argmax(fluxes)
    lam_peak_um = wavelengths_m[peak_idx] * 1e6  # m -> um
    teff_guess = 2898.0 / lam_peak_um
    teff_guess = np.clip(teff_guess, 2500.0, 50000.0)

    # Estimate scale from the peak flux
    model_peak = _planck_flux_density(wavelengths_m[peak_idx:peak_idx + 1], teff_guess)[0]
    if model_peak > 0:
        scale_guess = fluxes[peak_idx] / model_peak
        log_scale_guess = np.log10(max(scale_guess, 1e-100))
    else:
        log_scale_guess = -30.0

    p0 = [teff_guess, log_scale_guess]

    # Bounds: Teff in [2000, 60000] K; log_scale quite free.
    # B_nu values are very small (SI units), so the scale factor needed
    # to match Jy-level fluxes can be very large (log_scale >> 0).
    bounds = ([2000.0, -100.0], [60000.0, 100.0])

    try:
        popt, pcov = curve_fit(
            _bb_model,
            wavelengths_m,
            fluxes,
            p0=p0,
            sigma=flux_errs,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=10000,
        )
    except Exception as exc:
        raise RuntimeError(f"Blackbody fit failed: {exc}") from exc

    teff_fit, log_scale_fit = popt

    # Reduced chi-squared
    residuals = fluxes - _bb_model(wavelengths_m, teff_fit, log_scale_fit)
    chi2 = np.sum((residuals / flux_errs) ** 2)
    dof = max(n - 2, 1)
    chi2_red = chi2 / dof

    return teff_fit, log_scale_fit, chi2_red


def _compute_excess_for_band(
    band: str,
    phot: Dict[str, Any],
    teff: float,
    log_scale: float,
) -> tuple[Optional[float], Optional[float]]:
    """Compute excess and significance for a single WISE band.

    Parameters
    ----------
    band : str
        "W3" or "W4".
    phot : dict
        Input photometry (magnitudes and errors).
    teff : float
        Fitted effective temperature.
    log_scale : float
        Fitted log10(scale).

    Returns
    -------
    excess_mag : float or None
        Observed - predicted magnitude.  Negative means the source is
        brighter than the model (i.e. genuine excess emission).
    sigma : float or None
        Significance: |excess| / uncertainty.
    """
    obs_mag = phot.get(band)
    if obs_mag is None or not np.isfinite(obs_mag):
        return None, None

    obs_err = phot.get(f"{band}_err")
    if obs_err is None or not np.isfinite(obs_err) or obs_err <= 0:
        obs_err = _get_default_mag_err(band)

    # Model-predicted flux density at this wavelength
    lam_m = np.array([BAND_WAVELENGTHS_UM[band] * 1e-6])
    model_flux = _bb_model(lam_m, teff, log_scale)[0]
    model_mag = _flux_to_mag(model_flux, band)

    if not np.isfinite(model_mag):
        return None, None

    # Excess: observed minus predicted.  A Dyson sphere makes the star
    # brighter in mid-IR, so obs_mag < model_mag => excess_mag < 0.
    excess_mag = obs_mag - model_mag

    # Significance: combine observational error with a model uncertainty
    # term.  We adopt a conservative 5% model uncertainty (in flux),
    # converted to magnitudes: d_mag ~ 2.5 / ln(10) * (d_flux / flux) ~ 0.054 mag.
    model_err_mag = 0.054
    total_err = np.sqrt(obs_err ** 2 + model_err_mag ** 2)
    sigma = abs(excess_mag) / total_err if total_err > 0 else 0.0

    return excess_mag, sigma


# =====================================================================
#  WISE centroid contamination check (Hephaistos lesson)
# =====================================================================
#
#  The January 2025 follow-up of the Project Hephaistos Dyson sphere
#  candidates revealed that ALL 7 M-dwarf candidates had mid-IR excess
#  from background AGN / Hot DOGs (dust-obscured galaxies), not from the
#  stars themselves.  The smoking gun: the WISE W3/W4 centroid positions
#  were offset from the W1/W2 centroids by several arcseconds, indicating
#  two spatially distinct sources blended in the WISE beam.
#
#  WISE angular resolution:
#    W1 (3.4 um): 6.1"   W2 (4.6 um): 6.4"
#    W3 (12 um):  6.5"   W4 (22 um):  12.0"
#
#  If the W3/W4 centroid is offset from W1/W2 by more than ~2", the
#  excess emission almost certainly comes from a different physical
#  source.  This check catches the exact failure mode that Hephaistos
#  missed.

# Default contamination threshold in arcseconds.
# WISE astrometric precision is ~0.5" for bright sources (W1 SNR>20).
# A shift > 2" is highly suspicious; > 3" is definitive contamination.
_CENTROID_SHIFT_WARNING_ARCSEC = 2.0
_CENTROID_SHIFT_CONTAMINATED_ARCSEC = 3.0


def _check_wise_centroid_contamination(
    phot: Dict[str, Any],
) -> Dict[str, Any]:
    """Check for background source contamination via WISE centroid shift.

    Compares the centroid position of the short-wave WISE bands (W1/W2)
    against the long-wave bands (W3/W4).  A significant shift indicates
    that the mid-IR excess is coming from a spatially distinct background
    source (AGN, Hot DOG, or other confusing object) rather than from
    the target star itself.

    Parameters
    ----------
    phot : dict
        Photometry dictionary.  Looks for these centroid keys:

        - ``"w1_ra"``, ``"w1_dec"`` -- W1 centroid position (deg)
        - ``"w2_ra"``, ``"w2_dec"`` -- W2 centroid position (deg)
        - ``"w3_ra"``, ``"w3_dec"`` -- W3 centroid position (deg)
        - ``"w4_ra"``, ``"w4_dec"`` -- W4 centroid position (deg)

        If short-wave AND long-wave centroid positions are both present,
        the angular separation is computed.  If centroid data is missing,
        no contamination flag is raised (we can't rule it out without
        the data, but we note the absence).

    Returns
    -------
    dict
        Keys:

        - ``"shift_arcsec"`` : float or None -- angular offset between
          short-wave and long-wave centroids.
        - ``"is_contaminated"`` : bool -- True if shift exceeds the
          contamination threshold.
        - ``"reason"`` : str or None -- human-readable explanation.
    """
    result: Dict[str, Any] = {
        "shift_arcsec": None,
        "is_contaminated": False,
        "reason": None,
    }

    # ---- Gather short-wave centroid (W1 preferred, W2 fallback) ----
    sw_ra, sw_dec = None, None
    for band in ("w1", "w2"):
        ra_key = f"{band}_ra"
        dec_key = f"{band}_dec"
        ra_val = phot.get(ra_key)
        dec_val = phot.get(dec_key)
        if ra_val is not None and dec_val is not None:
            try:
                sw_ra, sw_dec = float(ra_val), float(dec_val)
                break
            except (TypeError, ValueError):
                continue

    # ---- Gather long-wave centroid (W3 preferred, W4 fallback) ----
    lw_ra, lw_dec = None, None
    for band in ("w3", "w4"):
        ra_key = f"{band}_ra"
        dec_key = f"{band}_dec"
        ra_val = phot.get(ra_key)
        dec_val = phot.get(dec_key)
        if ra_val is not None and dec_val is not None:
            try:
                lw_ra, lw_dec = float(ra_val), float(dec_val)
                break
            except (TypeError, ValueError):
                continue

    # ---- If we don't have both, we can't check ----
    if sw_ra is None or lw_ra is None:
        log.debug(
            "No WISE centroid data available for contamination check "
            "[source=%s]",
            phot.get("source_id"),
        )
        return result

    # ---- Compute angular separation ----
    # Use the small-angle approximation with cos(dec) correction.
    # At WISE resolution (~6-12"), this is perfectly accurate.
    dec_mean_rad = np.radians((sw_dec + lw_dec) / 2.0)
    cos_dec = np.cos(dec_mean_rad)

    d_ra_arcsec = (lw_ra - sw_ra) * 3600.0 * cos_dec
    d_dec_arcsec = (lw_dec - sw_dec) * 3600.0

    shift_arcsec = np.sqrt(d_ra_arcsec**2 + d_dec_arcsec**2)
    result["shift_arcsec"] = float(shift_arcsec)

    log.debug(
        "WISE centroid shift: %.2f arcsec  (dRA=%.2f\", dDec=%.2f\")  "
        "[source=%s]",
        shift_arcsec, d_ra_arcsec, d_dec_arcsec,
        phot.get("source_id"),
    )

    # ---- Apply thresholds ----
    if shift_arcsec >= _CENTROID_SHIFT_CONTAMINATED_ARCSEC:
        result["is_contaminated"] = True
        result["reason"] = (
            f"WISE W3/W4 centroid offset by {shift_arcsec:.1f}\" from "
            f"W1/W2 position (threshold: {_CENTROID_SHIFT_CONTAMINATED_ARCSEC}\")"
            f" -- probable background AGN/Hot DOG contamination"
        )
        log.warning(
            "CONTAMINATION DETECTED: centroid shift %.1f\" > %.1f\" "
            "[source=%s]",
            shift_arcsec, _CENTROID_SHIFT_CONTAMINATED_ARCSEC,
            phot.get("source_id"),
        )
    elif shift_arcsec >= _CENTROID_SHIFT_WARNING_ARCSEC:
        # Suspicious but not definitive -- flag as warning only
        result["is_contaminated"] = False
        result["reason"] = (
            f"WISE centroid shift {shift_arcsec:.1f}\" is suspicious "
            f"(warning threshold: {_CENTROID_SHIFT_WARNING_ARCSEC}\", "
            f"contamination threshold: {_CENTROID_SHIFT_CONTAMINATED_ARCSEC}\")"
            f" -- further investigation recommended"
        )
        log.info(
            "CENTROID WARNING: shift %.1f\" is suspicious but below "
            "contamination threshold [source=%s]",
            shift_arcsec, phot.get("source_id"),
        )

    return result


# =====================================================================
#  Public API
# =====================================================================

def compute_ir_excess(phot: Dict[str, Any]) -> IRExcessResult:
    """Compute infrared excess for a single source.

    Fits a Planck blackbody to the optical / near-IR photometry (Gaia
    G, BP, RP + 2MASS J, H, Ks) and compares the extrapolated mid-IR
    flux to observed WISE W3 and W4 magnitudes.

    Parameters
    ----------
    phot : dict
        Photometry dictionary.  Expected magnitude keys: ``"G"``,
        ``"BP"``, ``"RP"``, ``"J"``, ``"H"``, ``"Ks"``, ``"W3"``,
        ``"W4"``; error keys: ``"G_err"``, ``"J_err"``, etc.
        Missing bands are gracefully skipped.  An optional ``"source_id"``
        key is carried through to the result.

    Returns
    -------
    IRExcessResult
        Dataclass with fitted temperature, excess magnitudes, sigma
        significance, and Dyson-sphere candidate flag.
    """
    result = IRExcessResult(source_id=phot.get("source_id"))
    threshold = _anomaly_sigma()

    # ---- Gather fit data ----
    wavelengths_m, fluxes, flux_errs, band_names = _extract_fit_data(phot)
    result.fit_bands_used = len(band_names)

    if len(band_names) < 2:
        msg = (
            f"Insufficient photometry for blackbody fit: "
            f"only {len(band_names)} bands available ({band_names}); need >= 2."
        )
        result.warnings.append(msg)
        log.warning("%s [source=%s]", msg, result.source_id)
        return result

    if len(band_names) < 4:
        msg = (
            f"Fit uses only {len(band_names)} bands ({band_names}); "
            f"results may be unreliable."
        )
        result.warnings.append(msg)
        log.info("%s [source=%s]", msg, result.source_id)

    # ---- Fit the blackbody ----
    try:
        teff, log_scale, chi2_red = _fit_blackbody(wavelengths_m, fluxes, flux_errs)
    except RuntimeError as exc:
        msg = f"Blackbody fit failed: {exc}"
        result.warnings.append(msg)
        log.warning("%s [source=%s]", msg, result.source_id)
        return result

    result.fitted_teff = teff
    result.fitted_scale = 10.0 ** log_scale
    result.fit_chi2_reduced = chi2_red

    if chi2_red > 50.0:
        msg = (
            f"Poor blackbody fit quality (chi2_red={chi2_red:.1f}); "
            f"excess measurements should be interpreted with caution."
        )
        result.warnings.append(msg)
        log.info("%s [source=%s]", msg, result.source_id)

    log.debug(
        "Blackbody fit: Teff=%.0f K, log_scale=%.2f, chi2_red=%.2f, bands=%s [source=%s]",
        teff, log_scale, chi2_red, band_names, result.source_id,
    )

    # ---- Compute excess in WISE bands ----
    exc_w3, sig_w3 = _compute_excess_for_band("W3", phot, teff, log_scale)
    exc_w4, sig_w4 = _compute_excess_for_band("W4", phot, teff, log_scale)

    result.excess_W3 = exc_w3
    result.excess_W4 = exc_w4
    result.sigma_W3 = sig_w3
    result.sigma_W4 = sig_w4

    # ---- Candidate flagging ----
    # A candidate must show *negative* excess_mag (observed brighter than
    # model) with significance above threshold.  Following Hephaistos:
    # flag if excess is significantly negative (excess emission).
    #
    # QUALITY GATE 1 — chi2_red: If the blackbody fit is very poor
    # (chi2_red > 50), the Planck model cannot be trusted for
    # extrapolation to mid-IR wavelengths.  This is common for late
    # M dwarfs (M4+) where molecular absorption (TiO, VO, H2O) makes
    # the SED deviate dramatically from a Planck curve.  Without a
    # reliable photospheric model, any "excess" is model error, not
    # astrophysical signal.
    #
    # QUALITY GATE 2 — NIR anchor: The fit MUST include at least one
    # near-IR band (J, H, or Ks) to reliably extrapolate to mid-IR.
    # A fit using only optical bands (G, BP, RP at 0.5-0.8 um)
    # extrapolates 15-30x in wavelength to W3/W4, which is physically
    # unreliable for cool stars.
    # Audit fix F2: lowered from 50 to 25.  chi2_red=25 still means
    # the model is poor, but 50 was far too lenient — a model that bad
    # should not be trusted for excess significance calculations.
    _CHI2_RED_SUPPRESS_THRESHOLD = 25.0

    nir_bands = {"J", "H", "Ks"}
    has_nir = bool(nir_bands & set(band_names))
    fit_reliable = chi2_red <= _CHI2_RED_SUPPRESS_THRESHOLD

    candidate = False

    # Determine suppression reason (if any) before checking bands
    suppress_reason = None
    if not fit_reliable:
        suppress_reason = (
            f"chi2_red={chi2_red:.1f} >> {_CHI2_RED_SUPPRESS_THRESHOLD} "
            f"-- Planck model unreliable for mid-IR extrapolation"
        )
    elif not has_nir:
        suppress_reason = (
            f"fit uses only optical bands {band_names} -- no NIR anchor"
        )

    if exc_w3 is not None and exc_w3 < 0 and sig_w3 is not None and sig_w3 >= threshold:
        if suppress_reason is None:
            candidate = True
            log.info(
                "W3 excess detected: %.3f mag (%.1f sigma) [source=%s]",
                exc_w3, sig_w3, result.source_id,
            )
        else:
            msg = (
                f"W3 excess ({exc_w3:+.3f} mag, {sig_w3:.1f}sig) SUPPRESSED: "
                f"{suppress_reason}"
            )
            result.warnings.append(msg)
            log.info("%s [source=%s]", msg, result.source_id)

    if exc_w4 is not None and exc_w4 < 0 and sig_w4 is not None and sig_w4 >= threshold:
        if suppress_reason is None:
            candidate = True
            log.info(
                "W4 excess detected: %.3f mag (%.1f sigma) [source=%s]",
                exc_w4, sig_w4, result.source_id,
            )
        else:
            msg = (
                f"W4 excess ({exc_w4:+.3f} mag, {sig_w4:.1f}sig) SUPPRESSED: "
                f"{suppress_reason}"
            )
            result.warnings.append(msg)
            log.info("%s [source=%s]", msg, result.source_id)

    result.is_candidate = candidate

    # ---- AllWISE catalog quality flags ----
    # cc_flags: 4-char string (one per band W1-W4). Non-zero = contaminated
    #   by diffraction spike, persistence, halo, or optical ghost.
    # ext_flg: integer; 0 = point source, >0 = extended/resolved source
    #   (likely a background galaxy blended into the photometry).
    # These are the cheapest false-positive filters — already in the catalog.
    if candidate:
        cc = phot.get("cc_flags") or phot.get("cc_flg") or ""
        ext = phot.get("ext_flg")

        # cc_flags: any non-zero character in W3/W4 positions (index 2,3)
        # means the long-wave band is contaminated
        if len(cc) >= 4 and (cc[2] != "0" or cc[3] != "0"):
            candidate = False
            result.is_candidate = False
            reason = f"WISE cc_flags={cc} (W3/W4 contaminated)"
            result.contamination_flag = True
            result.contamination_reason = reason
            result.warnings.append(reason)
            log.warning(
                "IR excess DEMOTED (cc_flags): source=%s  cc_flags=%s",
                result.source_id, cc,
            )

        # ext_flg > 0 means the source is resolved/extended (background galaxy)
        if candidate and ext is not None:
            try:
                ext_val = int(ext)
            except (TypeError, ValueError):
                ext_val = 0
            if ext_val > 0:
                candidate = False
                result.is_candidate = False
                reason = f"WISE ext_flg={ext_val} (extended source — likely galaxy)"
                result.contamination_flag = True
                result.contamination_reason = reason
                result.warnings.append(reason)
                log.warning(
                    "IR excess DEMOTED (ext_flg): source=%s  ext_flg=%d",
                    result.source_id, ext_val,
                )

    # ---- Hephaistos contamination check (Jan 2025 lesson) ----
    # If WISE centroids shift between short-wave (W1/W2) and long-wave
    # (W3/W4) bands, the IR excess likely comes from a background source
    # (AGN / Hot DOG galaxy), not the star.  This check would have caught
    # all 7 Hephaistos false positives.
    if candidate:
        contamination = _check_wise_centroid_contamination(phot)
        result.centroid_shift_arcsec = contamination.get("shift_arcsec")
        result.contamination_flag = contamination.get("is_contaminated", False)
        result.contamination_reason = contamination.get("reason")

        if result.contamination_flag:
            result.is_candidate = False  # demote to non-candidate
            result.warnings.append(
                f"CONTAMINATION: {result.contamination_reason} "
                f"(centroid shift {result.centroid_shift_arcsec:.1f}\")"
            )
            log.warning(
                "IR excess DEMOTED (contamination): source=%s  reason=%s  "
                "shift=%.1f arcsec",
                result.source_id, result.contamination_reason,
                result.centroid_shift_arcsec or 0,
            )
        else:
            log.info(
                "DYSON SPHERE CANDIDATE: source=%s  Teff=%.0f K  dW3=%s  dW4=%s  "
                "contamination_check=CLEAN",
                result.source_id,
                teff,
                f"{exc_w3:+.3f} ({sig_w3:.1f}sig)" if exc_w3 is not None else "N/A",
                f"{exc_w4:+.3f} ({sig_w4:.1f}sig)" if exc_w4 is not None else "N/A",
            )

    return result


def compute_ir_excess_batch(
    photometry_list: List[Dict[str, Any]],
) -> List[IRExcessResult]:
    """Compute infrared excess for a batch of sources.

    Parameters
    ----------
    photometry_list : list of dict
        Each dict has the same format accepted by :func:`compute_ir_excess`.

    Returns
    -------
    list of IRExcessResult
        One result per input source, in the same order.
    """
    total = len(photometry_list)
    log.info("Computing IR excess for %d sources", total)

    results: List[IRExcessResult] = []
    n_candidates = 0

    for idx, phot in enumerate(photometry_list, 1):
        sid = phot.get("source_id", f"target_{idx}")
        if "source_id" not in phot:
            phot = {**phot, "source_id": sid}

        try:
            result = compute_ir_excess(phot)
        except Exception as exc:
            log.error("Unhandled error for source %s: %s", sid, exc)
            result = IRExcessResult(
                source_id=str(sid),
                warnings=[f"Unhandled error: {exc}"],
            )

        results.append(result)

        if result.is_candidate:
            n_candidates += 1

        if idx % 100 == 0 or idx == total:
            log.info(
                "  Progress: %d / %d  (%d candidates so far)",
                idx, total, n_candidates,
            )

    log.info(
        "Batch complete: %d sources processed, %d candidates flagged (%.1f%%)",
        total,
        n_candidates,
        100.0 * n_candidates / max(total, 1),
    )

    return results


# =====================================================================
#  CLI demo
# =====================================================================

if __name__ == "__main__":
    print()
    print("=" * 68)
    print("  Project EXODUS -- Infrared Excess / Dyson Sphere Detector Demo")
    print("=" * 68)

    # ------------------------------------------------------------------
    #  Example 1: Normal solar-type star (no excess expected)
    # ------------------------------------------------------------------
    #
    #  Approximate photometry for a G2V star at ~100 pc (similar to a
    #  Sun twin).  The WISE magnitudes are consistent with a ~5780 K
    #  photosphere -- no excess.
    normal_star = {
        "source_id": "Sun_twin_100pc",
        # Gaia (consistent with T~5780 K blackbody at G=8.5)
        "G":    8.50,   "G_err":   0.003,
        "BP":   8.92,   "BP_err":  0.003,
        "RP":   8.15,   "RP_err":  0.004,
        # 2MASS
        "J":    7.76,   "J_err":   0.02,
        "H":    7.56,   "H_err":   0.03,
        "Ks":   7.44,   "Ks_err":  0.02,
        # WISE -- consistent with photosphere (no excess)
        "W3":   7.20,   "W3_err":  0.02,
        "W4":   7.11,   "W4_err":  0.10,
    }

    print("\n--- Example 1: Normal G2V star (no excess) ---")
    print(f"  Input photometry: G={normal_star['G']}, J={normal_star['J']}, "
          f"Ks={normal_star['Ks']}, W3={normal_star['W3']}, W4={normal_star['W4']}")

    result1 = compute_ir_excess(normal_star)

    print(f"  Fitted Teff     : {result1.fitted_teff:.0f} K")
    print(f"  Bands used      : {result1.fit_bands_used}")
    print(f"  chi2_red        : {result1.fit_chi2_reduced:.2f}")
    if result1.excess_W3 is not None:
        print(f"  W3 excess       : {result1.excess_W3:+.3f} mag  ({result1.sigma_W3:.1f} sigma)")
    if result1.excess_W4 is not None:
        print(f"  W4 excess       : {result1.excess_W4:+.3f} mag  ({result1.sigma_W4:.1f} sigma)")
    print(f"  Candidate?      : {result1.is_candidate}")
    if result1.warnings:
        for w in result1.warnings:
            print(f"  WARNING: {w}")
    print(f"  Summary: {result1.summary()}")

    # ------------------------------------------------------------------
    #  Example 2: Anomalous star with strong mid-IR excess
    # ------------------------------------------------------------------
    #
    #  Same stellar photosphere as above, but WISE W3/W4 are anomalously
    #  bright (2+ magnitudes brighter than expected), simulating a partial
    #  Dyson sphere re-radiating intercepted starlight in the mid-IR.
    anomalous_star = {
        "source_id": "EXODUS_candidate_001",
        # Gaia -- same photosphere as the normal star
        "G":    8.50,   "G_err":   0.003,
        "BP":   8.92,   "BP_err":  0.003,
        "RP":   8.15,   "RP_err":  0.004,
        # 2MASS -- same photosphere
        "J":    7.76,   "J_err":   0.02,
        "H":    7.56,   "H_err":   0.03,
        "Ks":   7.44,   "Ks_err":  0.02,
        # WISE -- anomalously bright (excess emission from warm structure!)
        "W3":   4.50,   "W3_err":  0.03,   # ~2.7 mag brighter than expected
        "W4":   3.00,   "W4_err":  0.08,   # ~4.1 mag brighter than expected
    }

    print("\n--- Example 2: Anomalous star (simulated Dyson sphere) ---")
    print(f"  Input photometry: G={anomalous_star['G']}, J={anomalous_star['J']}, "
          f"Ks={anomalous_star['Ks']}, W3={anomalous_star['W3']}, W4={anomalous_star['W4']}")

    result2 = compute_ir_excess(anomalous_star)

    print(f"  Fitted Teff     : {result2.fitted_teff:.0f} K")
    print(f"  Bands used      : {result2.fit_bands_used}")
    print(f"  chi2_red        : {result2.fit_chi2_reduced:.2f}")
    if result2.excess_W3 is not None:
        print(f"  W3 excess       : {result2.excess_W3:+.3f} mag  ({result2.sigma_W3:.1f} sigma)")
    if result2.excess_W4 is not None:
        print(f"  W4 excess       : {result2.excess_W4:+.3f} mag  ({result2.sigma_W4:.1f} sigma)")
    print(f"  Candidate?      : {result2.is_candidate}")
    if result2.warnings:
        for w in result2.warnings:
            print(f"  WARNING: {w}")
    print(f"  Summary: {result2.summary()}")

    # ------------------------------------------------------------------
    #  Example 3: Batch run with both
    # ------------------------------------------------------------------
    print("\n--- Example 3: Batch processing ---")
    batch_results = compute_ir_excess_batch([normal_star, anomalous_star])
    for r in batch_results:
        print(f"  [{r.source_id}]  {r.summary()}")

    n_cand = sum(1 for r in batch_results if r.is_candidate)
    print(f"\n  Total candidates: {n_cand} / {len(batch_results)}")

    print()
    print("=" * 68)
    print("  Done.")
    print("=" * 68)
    print()
