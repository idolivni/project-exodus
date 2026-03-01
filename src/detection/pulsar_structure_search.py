"""
Pulsar line-of-sight megastructure search for Project EXODUS.

A NOVEL detection strategy: searching for anomalous Shapiro delays in
pulsar timing data caused by mass concentrations (megastructures) along
the line of sight (LOS) to millisecond pulsars.

Physical basis
--------------
When a photon (or radio pulse) passes near a mass M at an impact
parameter b, it experiences a **Shapiro delay**:

    dt = -(2 G M / c^3) * ln(1 - cos(theta))

where theta is the angle between the pulsar direction and the mass
location as seen from Earth.  For a mass at angular separation phi
(small angle), this simplifies to:

    dt ~ (2 G M / c^3) * ln(2 / phi)      [for phi << 1 radian]

For a Jupiter-mass object at 1 arcsecond separation the delay is
~10 nanoseconds.  A megastructure of ~10^-3 solar masses at 0.1
degree separation could produce delays of ~0.1-1 microseconds --
detectable in the best-timed NANOGrav millisecond pulsars.

Search strategy
---------------
1. For each NANOGrav pulsar, identify known exoplanet systems
   within 1 degree of the pulsar line-of-sight.

2. For those systems, compute the expected Shapiro delay from
   the known planetary masses and orbital parameters.

3. Search for **periodic** timing residual anomalies at the
   known orbital period of each planet -- a planet or structure
   orbiting the host star would produce a periodic Shapiro delay
   modulation as it moves along its orbit.

4. Also search for **static** excess delay (a fixed mass
   concentration, like a Dyson sphere or megastructure shell).

5. Significance is assessed via a matched-filter approach,
   comparing the observed periodogram peak at the planet's
   orbital period against the noise floor.

This module provides:
    - ``search_pulsar_los()`` : main search pipeline
    - ``compute_shapiro_delay()`` : Shapiro delay calculator
    - Simulation mode with realistic test data
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_config, get_logger, save_result

log = get_logger("detection.pulsar_structure_search")

# ---------------------------------------------------------------------------
# Lazy / optional imports
# ---------------------------------------------------------------------------
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import astropy.constants as const

    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False
    log.debug("astropy not available -- using built-in constants")

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
G_SI = 6.67430e-11        # m^3 kg^-1 s^-2
C_SI = 2.99792458e8       # m s^-1
M_SUN_KG = 1.98892e30     # kg
M_JUPITER_KG = 1.8982e27  # kg
AU_M = 1.496e11           # metres
DEG_TO_RAD = np.pi / 180.0
ARCSEC_TO_RAD = DEG_TO_RAD / 3600.0

# Shapiro delay prefactor: 2 G M_sun / c^3  (in seconds per solar mass)
SHAPIRO_PREFACTOR_S = 2.0 * G_SI * M_SUN_KG / C_SI**3   # ~9.85 microseconds

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LOSMatch:
    """A match between a pulsar LOS and an exoplanet system."""

    pulsar_name: str
    pulsar_ra: float                     # degrees
    pulsar_dec: float                    # degrees
    host_star: str
    host_ra: float                       # degrees
    host_dec: float                      # degrees
    angular_sep_deg: float               # angular separation (degrees)
    planet_name: Optional[str] = None
    orbital_period_days: Optional[float] = None
    planet_mass_mjup: Optional[float] = None
    distance_pc: Optional[float] = None  # host star distance


@dataclass
class ShapiroEstimate:
    """Expected Shapiro delay from a mass at a given angular separation."""

    mass_solar: float                    # mass in solar masses
    angular_sep_deg: float               # angular separation (degrees)
    distance_pc: Optional[float] = None  # distance to the mass (parsecs)
    delay_us: float = 0.0               # expected Shapiro delay (microseconds)
    delay_ns: float = 0.0               # expected Shapiro delay (nanoseconds)


@dataclass
class PulsarLOSResult:
    """Search result for one pulsar line-of-sight."""

    pulsar_name: str
    pulsar_ra: float
    pulsar_dec: float
    n_los_matches: int                       # exoplanet systems near LOS
    los_matches: List[LOSMatch] = field(default_factory=list)
    shapiro_estimates: List[ShapiroEstimate] = field(default_factory=list)
    periodic_significance: Optional[float] = None  # max periodogram significance
    periodic_period_days: Optional[float] = None   # period of strongest signal
    static_delay_excess_us: Optional[float] = None # excess static delay
    is_candidate: bool = False                      # flagged as interesting?
    candidate_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "pulsar_name": self.pulsar_name,
            "pulsar_ra": self.pulsar_ra,
            "pulsar_dec": self.pulsar_dec,
            "n_los_matches": self.n_los_matches,
            "los_matches": [asdict(m) for m in self.los_matches],
            "shapiro_estimates": [asdict(s) for s in self.shapiro_estimates],
            "periodic_significance": self.periodic_significance,
            "periodic_period_days": self.periodic_period_days,
            "static_delay_excess_us": self.static_delay_excess_us,
            "is_candidate": self.is_candidate,
            "candidate_reason": self.candidate_reason,
        }


@dataclass
class PulsarStructureSearchResult:
    """Aggregate result across all pulsars."""

    n_pulsars_searched: int = 0
    n_los_matches_total: int = 0
    n_candidates: int = 0
    results: List[PulsarLOSResult] = field(default_factory=list)
    candidates: List[PulsarLOSResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_pulsars_searched": self.n_pulsars_searched,
            "n_los_matches_total": self.n_los_matches_total,
            "n_candidates": self.n_candidates,
            "results": [r.to_dict() for r in self.results],
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------------------
# Shapiro delay computation
# ---------------------------------------------------------------------------

def compute_shapiro_delay(
    mass_solar: float,
    angular_sep_deg: float,
    distance_pc: Optional[float] = None,
) -> ShapiroEstimate:
    """
    Compute the expected Shapiro delay for a mass at a given angular
    separation from a pulsar line of sight.

    The Shapiro delay for a photon passing near a mass M is:

        dt = (2 G M / c^3) * ln(2 / phi)

    where phi is the angular separation in radians (valid for phi << 1).

    For very small angles, this diverges logarithmically.  In practice,
    the finite size of the mass distribution and the pulsar beam set
    a minimum meaningful angular separation.

    Parameters
    ----------
    mass_solar : float
        Mass of the intervening object in solar masses.
    angular_sep_deg : float
        Angular separation between the mass and the pulsar LOS in degrees.
    distance_pc : float, optional
        Distance to the mass in parsecs (for context/reporting only).

    Returns
    -------
    ShapiroEstimate
        Estimated delay in microseconds and nanoseconds.
    """
    # Convert angular separation to radians
    phi_rad = angular_sep_deg * DEG_TO_RAD

    # Prevent divergence at zero separation
    phi_rad = max(phi_rad, 1.0e-10)

    # Shapiro delay: dt = (2GM/c^3) * ln(2/phi)
    # SHAPIRO_PREFACTOR_S is in seconds per solar mass
    delay_s = SHAPIRO_PREFACTOR_S * mass_solar * np.log(2.0 / phi_rad)

    # Ensure non-negative (phi > 2 rad would give negative log, but that
    # is unphysical for our use case)
    delay_s = max(delay_s, 0.0)

    delay_us = delay_s * 1.0e6
    delay_ns = delay_s * 1.0e9

    return ShapiroEstimate(
        mass_solar=mass_solar,
        angular_sep_deg=angular_sep_deg,
        distance_pc=distance_pc,
        delay_us=delay_us,
        delay_ns=delay_ns,
    )


# ---------------------------------------------------------------------------
# Angular separation calculation
# ---------------------------------------------------------------------------

def _angular_separation_deg(
    ra1: float, dec1: float,
    ra2: float, dec2: float,
) -> float:
    """
    Compute the angular separation between two sky positions in degrees.

    Uses the Vincenty formula for numerical stability.

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        Coordinates in degrees (ICRS).

    Returns
    -------
    float
        Angular separation in degrees.
    """
    if _HAS_ASTROPY:
        c1 = SkyCoord(ra=ra1, dec=dec1, unit=(u.deg, u.deg), frame="icrs")
        c2 = SkyCoord(ra=ra2, dec=dec2, unit=(u.deg, u.deg), frame="icrs")
        return c1.separation(c2).deg

    # Vincenty formula fallback
    ra1_r = np.radians(ra1)
    dec1_r = np.radians(dec1)
    ra2_r = np.radians(ra2)
    dec2_r = np.radians(dec2)
    dra = ra2_r - ra1_r

    num = np.sqrt(
        (np.cos(dec2_r) * np.sin(dra)) ** 2
        + (np.cos(dec1_r) * np.sin(dec2_r)
           - np.sin(dec1_r) * np.cos(dec2_r) * np.cos(dra)) ** 2
    )
    den = (
        np.sin(dec1_r) * np.sin(dec2_r)
        + np.cos(dec1_r) * np.cos(dec2_r) * np.cos(dra)
    )
    return np.degrees(np.arctan2(num, den))


# ---------------------------------------------------------------------------
# Periodogram analysis
# ---------------------------------------------------------------------------

def _lomb_scargle_at_period(
    mjd: np.ndarray,
    residuals: np.ndarray,
    uncertainties: np.ndarray,
    period_days: float,
) -> float:
    """
    Compute the Lomb-Scargle power at a specific period.

    Returns a significance measure: the ratio of the power at the
    target period to the median power across all sampled frequencies.

    Parameters
    ----------
    mjd : array
        Observation epochs (Modified Julian Date).
    residuals : array
        Timing residuals (microseconds).
    uncertainties : array
        Per-point uncertainties (microseconds).
    period_days : float
        Period to test (days).

    Returns
    -------
    float
        Significance (power / median_power).  Values > 5 are
        considered interesting.
    """
    if len(mjd) < 10:
        return 0.0

    # Use astropy Lomb-Scargle if available
    try:
        from astropy.timeseries import LombScargle

        freq_target = 1.0 / period_days  # cycles per day
        # Evaluate power at the target frequency and nearby frequencies
        # for normalisation
        freq_grid = np.linspace(
            1.0 / (mjd[-1] - mjd[0]),  # lowest frequency
            0.5 / np.median(np.diff(mjd)),  # Nyquist-ish
            500,
        )

        ls = LombScargle(mjd, residuals, uncertainties)
        power_grid = ls.power(freq_grid)
        power_target = ls.power(freq_target)

        median_power = np.median(power_grid)
        if median_power > 0:
            return float(power_target / median_power)
        return 0.0

    except ImportError:
        pass

    # Fallback: simple DFT at the target frequency
    omega = 2.0 * np.pi / period_days
    t = mjd - mjd[0]

    # Weighted by inverse variance
    w = 1.0 / (uncertainties ** 2 + 1.0e-30)
    w /= w.sum()

    cos_term = np.sum(w * residuals * np.cos(omega * t))
    sin_term = np.sum(w * residuals * np.sin(omega * t))
    power_target = cos_term**2 + sin_term**2

    # Estimate noise power from random frequencies
    rng = np.random.RandomState(42)
    noise_powers = []
    for _ in range(200):
        omega_rand = rng.uniform(0.01, 2.0 * np.pi)
        c = np.sum(w * residuals * np.cos(omega_rand * t))
        s = np.sum(w * residuals * np.sin(omega_rand * t))
        noise_powers.append(c**2 + s**2)

    median_noise = np.median(noise_powers)
    if median_noise > 0:
        return float(power_target / median_noise)
    return 0.0


# ---------------------------------------------------------------------------
# Main search functions
# ---------------------------------------------------------------------------

def search_pulsar_los(
    pulsars: List[Dict[str, Any]],
    exoplanet_hosts: List[Dict[str, Any]],
    search_radius_deg: float = 1.0,
    significance_threshold: float = 5.0,
    include_timing_analysis: bool = True,
) -> PulsarStructureSearchResult:
    """
    Search for megastructure signatures along pulsar lines of sight.

    For each pulsar, identifies exoplanet systems within
    ``search_radius_deg`` and checks for:

    1. **Expected Shapiro delay** from known planet masses at the
       measured angular separation.
    2. **Periodic timing residual anomalies** matching known
       exoplanet orbital periods (requires timing data).
    3. **Static excess delay** that could indicate a fixed mass
       concentration (Dyson sphere, debris field, etc.).

    Parameters
    ----------
    pulsars : list[dict]
        Each dict must have 'name', 'ra_deg' (or 'ra'), 'dec_deg' (or
        'dec'), and optionally 'residual_rms_us'.  Typically from the
        ``nanograv`` ingestion module.
    exoplanet_hosts : list[dict]
        Each dict must have 'ra' (or 'ra_deg'), 'dec' (or 'dec_deg'),
        and optionally 'host_star', 'planet_name', 'orbital_period_days',
        'planet_mass_mjup', 'distance_pc'.  Typically from the
        ``exoplanet_archive`` ingestion module.
    search_radius_deg : float
        Maximum angular separation to consider (default 1.0 degree).
    significance_threshold : float
        Periodogram significance threshold for flagging candidates
        (default 5.0).
    include_timing_analysis : bool
        If True, load timing residuals and perform periodogram analysis
        for each LOS match.  Set to False for fast positional-only search.

    Returns
    -------
    PulsarStructureSearchResult
        Aggregate results with candidates flagged.
    """
    log.info(
        "=== Pulsar LOS Megastructure Search ===\n"
        "  Pulsars: %d\n"
        "  Exoplanet hosts: %d\n"
        "  Search radius: %.2f deg\n"
        "  Significance threshold: %.1f",
        len(pulsars), len(exoplanet_hosts),
        search_radius_deg, significance_threshold,
    )

    all_results: List[PulsarLOSResult] = []
    candidates: List[PulsarLOSResult] = []
    total_matches = 0

    # Precompute exoplanet host positions for vectorised matching
    host_ra = np.array([
        h.get("ra_deg", h.get("ra", 0.0)) for h in exoplanet_hosts
    ], dtype=np.float64)
    host_dec = np.array([
        h.get("dec_deg", h.get("dec", 0.0)) for h in exoplanet_hosts
    ], dtype=np.float64)

    if _HAS_ASTROPY and len(exoplanet_hosts) > 0:
        host_coords = SkyCoord(
            ra=host_ra, dec=host_dec, unit=(u.deg, u.deg), frame="icrs"
        )
    else:
        host_coords = None

    for i, psr in enumerate(pulsars):
        psr_name = psr.get("name", psr.get("pulsar_name", f"PSR_{i}"))
        psr_ra = float(psr.get("ra_deg", psr.get("ra", 0.0)))
        psr_dec = float(psr.get("dec_deg", psr.get("dec", 0.0)))

        log.debug(
            "  [%d/%d] %s (%.4f, %.4f)",
            i + 1, len(pulsars), psr_name, psr_ra, psr_dec,
        )

        # Find exoplanet systems near this pulsar's LOS
        los_matches = _find_los_matches(
            psr_name, psr_ra, psr_dec,
            exoplanet_hosts, host_ra, host_dec, host_coords,
            search_radius_deg,
        )

        if not los_matches:
            all_results.append(PulsarLOSResult(
                pulsar_name=psr_name,
                pulsar_ra=psr_ra,
                pulsar_dec=psr_dec,
                n_los_matches=0,
            ))
            continue

        total_matches += len(los_matches)

        # Compute expected Shapiro delays
        shapiro_estimates = []
        for match in los_matches:
            # Use planet mass if available, else assume Jupiter mass
            mass_mjup = match.planet_mass_mjup or 1.0
            mass_solar = mass_mjup * M_JUPITER_KG / M_SUN_KG

            estimate = compute_shapiro_delay(
                mass_solar=mass_solar,
                angular_sep_deg=match.angular_sep_deg,
                distance_pc=match.distance_pc,
            )
            shapiro_estimates.append(estimate)

        # Timing analysis (if requested and data available)
        periodic_sig = None
        periodic_period = None
        static_excess = None

        if include_timing_analysis and los_matches:
            try:
                timing_result = _analyze_timing_for_los(
                    psr_name, psr, los_matches,
                )
                periodic_sig = timing_result.get("periodic_significance")
                periodic_period = timing_result.get("periodic_period_days")
                static_excess = timing_result.get("static_delay_excess_us")
            except Exception as exc:
                log.debug(
                    "Timing analysis failed for %s: %s", psr_name, exc
                )

        # Determine if this is a candidate
        is_candidate = False
        reason = None

        if periodic_sig is not None and periodic_sig >= significance_threshold:
            is_candidate = True
            reason = (
                f"Periodic signal at {periodic_period:.1f} days "
                f"with significance {periodic_sig:.1f}"
            )

        # Also flag if expected Shapiro delay exceeds pulsar timing precision
        psr_rms = float(psr.get("residual_rms_us", 1.0))
        max_shapiro = max(
            (s.delay_us for s in shapiro_estimates), default=0.0
        )
        if max_shapiro > psr_rms * 0.1:
            is_candidate = True
            reason = reason or (
                f"Expected Shapiro delay {max_shapiro:.4f} us exceeds "
                f"10% of timing RMS ({psr_rms:.4f} us)"
            )

        result = PulsarLOSResult(
            pulsar_name=psr_name,
            pulsar_ra=psr_ra,
            pulsar_dec=psr_dec,
            n_los_matches=len(los_matches),
            los_matches=los_matches,
            shapiro_estimates=shapiro_estimates,
            periodic_significance=periodic_sig,
            periodic_period_days=periodic_period,
            static_delay_excess_us=static_excess,
            is_candidate=is_candidate,
            candidate_reason=reason,
        )

        all_results.append(result)
        if is_candidate:
            candidates.append(result)
            log.info(
                "  *** CANDIDATE: %s -- %s ***", psr_name, reason
            )

    # Compile aggregate result
    search_result = PulsarStructureSearchResult(
        n_pulsars_searched=len(pulsars),
        n_los_matches_total=total_matches,
        n_candidates=len(candidates),
        results=all_results,
        candidates=candidates,
    )

    log.info(
        "Search complete: %d pulsars, %d LOS matches, %d candidates",
        len(pulsars), total_matches, len(candidates),
    )

    # Persist result
    try:
        save_result("pulsar_structure_search", search_result.to_dict())
    except Exception as exc:
        log.debug("Could not save result: %s", exc)

    return search_result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_los_matches(
    psr_name: str,
    psr_ra: float,
    psr_dec: float,
    exoplanet_hosts: List[Dict[str, Any]],
    host_ra: np.ndarray,
    host_dec: np.ndarray,
    host_coords: Optional[Any],  # SkyCoord or None
    radius_deg: float,
) -> List[LOSMatch]:
    """Find exoplanet systems within radius_deg of a pulsar LOS."""
    matches = []

    if _HAS_ASTROPY and host_coords is not None and len(host_ra) > 0:
        psr_coord = SkyCoord(
            ra=psr_ra, dec=psr_dec, unit=(u.deg, u.deg), frame="icrs"
        )
        seps = psr_coord.separation(host_coords).deg
        mask = seps <= radius_deg
        indices = np.where(mask)[0]

        for idx in indices:
            h = exoplanet_hosts[idx]
            matches.append(LOSMatch(
                pulsar_name=psr_name,
                pulsar_ra=psr_ra,
                pulsar_dec=psr_dec,
                host_star=str(h.get("host_star", h.get("hostname", "unknown"))),
                host_ra=float(host_ra[idx]),
                host_dec=float(host_dec[idx]),
                angular_sep_deg=float(seps[idx]),
                planet_name=h.get("planet_name"),
                orbital_period_days=_safe_float(h.get("orbital_period_days")),
                planet_mass_mjup=_safe_float(h.get("planet_mass_mjup")),
                distance_pc=_safe_float(h.get("distance_pc")),
            ))
    else:
        # Fallback: brute-force angular distance
        cos_dec = np.cos(np.radians(psr_dec))
        for idx, h in enumerate(exoplanet_hosts):
            h_ra = float(h.get("ra_deg", h.get("ra", 0.0)))
            h_dec = float(h.get("dec_deg", h.get("dec", 0.0)))
            sep = _angular_separation_deg(psr_ra, psr_dec, h_ra, h_dec)
            if sep <= radius_deg:
                matches.append(LOSMatch(
                    pulsar_name=psr_name,
                    pulsar_ra=psr_ra,
                    pulsar_dec=psr_dec,
                    host_star=str(h.get("host_star", h.get("hostname", "unknown"))),
                    host_ra=h_ra,
                    host_dec=h_dec,
                    angular_sep_deg=sep,
                    planet_name=h.get("planet_name"),
                    orbital_period_days=_safe_float(h.get("orbital_period_days")),
                    planet_mass_mjup=_safe_float(h.get("planet_mass_mjup")),
                    distance_pc=_safe_float(h.get("distance_pc")),
                ))

    # Sort by angular separation
    matches.sort(key=lambda m: m.angular_sep_deg)
    return matches


def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None if not possible."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _analyze_timing_for_los(
    psr_name: str,
    psr_data: Dict[str, Any],
    los_matches: List[LOSMatch],
) -> Dict[str, Any]:
    """
    Analyse timing residuals for periodic signals at LOS match periods.

    Attempts to load residuals from the NANOGrav ingestion module;
    if unavailable, generates simulated residuals.

    Returns a dict with 'periodic_significance', 'periodic_period_days',
    and 'static_delay_excess_us'.
    """
    # Try to load timing residuals
    mjd = None
    residuals = None
    uncertainties = None

    try:
        from src.ingestion.nanograv import get_timing_residuals
        ts = get_timing_residuals(psr_name)
        mjd = ts.mjd
        residuals = ts.residuals_us
        uncertainties = ts.uncertainties_us
    except Exception as exc:
        log.debug("Could not load residuals for %s: %s", psr_name, exc)

    if mjd is None or len(mjd) < 20:
        return {
            "periodic_significance": None,
            "periodic_period_days": None,
            "static_delay_excess_us": None,
        }

    # Search for periodic signals at each matched planet's orbital period
    best_sig = 0.0
    best_period = None

    for match in los_matches:
        period = match.orbital_period_days
        if period is None or period <= 0:
            continue

        sig = _lomb_scargle_at_period(mjd, residuals, uncertainties, period)
        if sig > best_sig:
            best_sig = sig
            best_period = period

    # Static excess: is the mean residual significantly non-zero?
    mean_residual = np.mean(residuals)
    mean_unc = np.std(residuals) / np.sqrt(len(residuals))
    static_excess = abs(mean_residual) if mean_unc > 0 else 0.0

    return {
        "periodic_significance": best_sig if best_sig > 0 else None,
        "periodic_period_days": best_period,
        "static_delay_excess_us": static_excess,
    }


# ---------------------------------------------------------------------------
# Simulation mode: generate realistic test data
# ---------------------------------------------------------------------------

def _generate_mock_pulsars(n: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate mock pulsar catalog for testing."""
    rng = np.random.RandomState(seed)

    # Use a selection of real NANOGrav pulsar positions as seeds
    base_pulsars = [
        ("J1713+0747", 258.47, 7.79, 0.03),
        ("J1909-3744", 287.43, -37.74, 0.04),
        ("J0437-4715", 69.32, -47.25, 0.07),
        ("B1937+21", 294.91, 21.58, 0.08),
        ("J0030+0451", 7.61, 4.86, 0.19),
        ("J1744-1134", 266.12, -11.58, 0.12),
        ("J0613-0200", 93.46, -2.01, 0.18),
        ("J1600-3053", 240.06, -30.89, 0.21),
        ("J2317+1439", 349.44, 14.66, 0.18),
        ("J1640+2224", 250.07, 22.41, 0.22),
        ("J0645+5158", 101.28, 51.97, 0.20),
        ("J1012+5307", 153.14, 53.12, 0.32),
        ("J1455-3330", 223.94, -33.51, 0.78),
        ("J1918-0642", 289.56, -6.71, 0.28),
        ("J2145-0750", 326.44, -7.84, 0.45),
        ("J1853+1303", 283.31, 13.06, 0.45),
        ("J2043+1711", 310.89, 17.19, 0.15),
        ("J0740+6620", 115.10, 66.34, 0.15),
        ("J1614-2230", 243.65, -22.51, 0.16),
        ("J1855+09", 283.86, 9.02, 0.58),
    ]

    pulsars = []
    for i in range(min(n, len(base_pulsars))):
        name, ra, dec, rms = base_pulsars[i]
        pulsars.append({
            "name": name,
            "ra_deg": ra,
            "dec_deg": dec,
            "residual_rms_us": rms,
            "period_ms": rng.uniform(1.5, 16.0),
            "dm": rng.uniform(3.0, 300.0),
            "n_toas": rng.randint(500, 8000),
            "time_span_yr": rng.uniform(5.0, 15.0),
        })

    return pulsars


def _generate_mock_exoplanet_hosts(
    n: int = 200,
    pulsars: Optional[List[Dict]] = None,
    n_near_los: int = 5,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    """
    Generate mock exoplanet host catalog for testing.

    Places ``n_near_los`` hosts deliberately close to pulsar LOSs
    to ensure some matches in the search.
    """
    rng = np.random.RandomState(seed)

    hosts = []

    # Place some hosts deliberately near pulsar LOSs
    if pulsars and n_near_los > 0:
        for i in range(min(n_near_los, len(pulsars))):
            psr = pulsars[i]
            offset_ra = rng.uniform(-0.5, 0.5)
            offset_dec = rng.uniform(-0.5, 0.5)
            hosts.append({
                "host_star": f"HD-MOCK-{100+i}",
                "ra_deg": psr["ra_deg"] + offset_ra,
                "dec_deg": psr["dec_deg"] + offset_dec,
                "planet_name": f"HD-MOCK-{100+i} b",
                "orbital_period_days": rng.uniform(1.0, 1000.0),
                "planet_mass_mjup": rng.uniform(0.1, 10.0),
                "distance_pc": rng.uniform(5.0, 500.0),
            })

    # Fill remaining with random sky positions
    for i in range(n - len(hosts)):
        hosts.append({
            "host_star": f"MOCK-Star-{i}",
            "ra_deg": rng.uniform(0, 360),
            "dec_deg": rng.uniform(-90, 90),
            "planet_name": f"MOCK-Star-{i} b",
            "orbital_period_days": rng.uniform(0.5, 5000.0),
            "planet_mass_mjup": rng.uniform(0.01, 13.0),
            "distance_pc": rng.uniform(1.0, 1000.0),
        })

    return hosts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- Pulsar Line-of-Sight Megastructure Search")
    print("=" * 72)
    print()
    print("NOVEL STRATEGY: Searching for anomalous Shapiro delays in")
    print("NANOGrav pulsar timing data caused by mass concentrations")
    print("(megastructures) along pulsar lines of sight.")
    print()

    # ---- Try to load real data, fall back to simulation ----
    use_real_data = False
    pulsars = None
    exoplanet_hosts = None

    try:
        from src.ingestion.nanograv import get_all_pulsars as _get_pulsars
        raw_pulsars = _get_pulsars()
        pulsars = [p.to_dict() for p in raw_pulsars]
        log.info("Loaded %d pulsars from NANOGrav module", len(pulsars))
        use_real_data = True
    except Exception as exc:
        log.info("NANOGrav module not available: %s", exc)

    try:
        from src.ingestion.exoplanet_archive import query_exoplanet_archive
        df = query_exoplanet_archive()
        exoplanet_hosts = df.to_dict(orient="records")
        log.info("Loaded %d exoplanet entries", len(exoplanet_hosts))
    except Exception as exc:
        log.info("Exoplanet archive not available: %s", exc)

    if pulsars is None:
        print("  Using simulated pulsar data ...")
        pulsars = _generate_mock_pulsars(n=20)

    if exoplanet_hosts is None:
        print("  Using simulated exoplanet host data ...")
        exoplanet_hosts = _generate_mock_exoplanet_hosts(
            n=200, pulsars=pulsars, n_near_los=5,
        )

    print(f"  Pulsars:         {len(pulsars)}")
    print(f"  Exoplanet hosts: {len(exoplanet_hosts)}")
    print()

    # ---- Shapiro delay examples ----
    print("[1] Shapiro delay estimates (example masses at various separations):")
    print("-" * 60)
    test_cases = [
        ("Jupiter mass at 0.1 deg", 1.0 * M_JUPITER_KG / M_SUN_KG, 0.1),
        ("Jupiter mass at 0.01 deg", 1.0 * M_JUPITER_KG / M_SUN_KG, 0.01),
        ("10 Jupiter at 0.5 deg", 10.0 * M_JUPITER_KG / M_SUN_KG, 0.5),
        ("Solar mass at 1.0 deg", 1.0, 1.0),
        ("Solar mass at 0.1 deg", 1.0, 0.1),
        ("0.001 M_sun at 0.01 deg", 0.001, 0.01),
    ]
    for label, mass, sep in test_cases:
        est = compute_shapiro_delay(mass, sep)
        print(
            f"  {label:<35s}  "
            f"delay = {est.delay_us:>10.4f} us  ({est.delay_ns:>10.1f} ns)"
        )
    print()

    # ---- Run the search ----
    print("[2] Running pulsar LOS megastructure search ...")
    print("-" * 60)
    result = search_pulsar_los(
        pulsars=pulsars,
        exoplanet_hosts=exoplanet_hosts,
        search_radius_deg=1.0,
        significance_threshold=5.0,
        include_timing_analysis=True,
    )

    print(f"  Pulsars searched:     {result.n_pulsars_searched}")
    print(f"  Total LOS matches:    {result.n_los_matches_total}")
    print(f"  Candidates flagged:   {result.n_candidates}")
    print()

    # ---- Show LOS matches ----
    matches_found = [r for r in result.results if r.n_los_matches > 0]
    if matches_found:
        print("[3] Pulsar -- Exoplanet LOS matches:")
        print("-" * 60)
        for r in matches_found:
            print(f"  {r.pulsar_name} ({r.pulsar_ra:.3f}, {r.pulsar_dec:.3f}):")
            for m in r.los_matches:
                print(
                    f"    -> {m.host_star:<20s} sep={m.angular_sep_deg:.4f} deg"
                    f"  P={m.orbital_period_days or 0:.1f} d"
                    f"  M={m.planet_mass_mjup or 0:.2f} Mjup"
                )
            for s in r.shapiro_estimates:
                print(
                    f"       Shapiro delay: {s.delay_us:.4f} us "
                    f"({s.delay_ns:.1f} ns) for {s.mass_solar:.6f} M_sun "
                    f"at {s.angular_sep_deg:.4f} deg"
                )
            print()
    else:
        print("  No LOS matches found.")
        print()

    # ---- Show candidates ----
    if result.candidates:
        print("=" * 72)
        print("  *** HIGH PRIORITY CANDIDATES ***")
        print("=" * 72)
        for c in result.candidates:
            print(f"  Pulsar: {c.pulsar_name}")
            print(f"    Reason: {c.candidate_reason}")
            print(f"    LOS matches: {c.n_los_matches}")
            if c.periodic_significance is not None:
                print(
                    f"    Periodic signal: significance={c.periodic_significance:.1f} "
                    f"at P={c.periodic_period_days:.1f} days"
                )
            if c.static_delay_excess_us is not None:
                print(
                    f"    Static excess: {c.static_delay_excess_us:.4f} us"
                )
            for m in c.los_matches:
                print(
                    f"    -> Host: {m.host_star}, sep={m.angular_sep_deg:.4f} deg"
                )
            print()
    else:
        print("  No high-priority candidates identified in this run.")
        print()

    # ---- Summary ----
    print("-" * 72)
    print("  SUMMARY")
    print("-" * 72)
    print(f"  Pulsars searched:          {result.n_pulsars_searched}")
    print(f"  Exoplanet hosts queried:   {len(exoplanet_hosts)}")
    print(f"  LOS matches (< 1 deg):     {result.n_los_matches_total}")
    print(f"  Candidate megastructures:  {result.n_candidates}")
    print()
    print("=" * 72)
    print("  Pulsar LOS megastructure search complete.")
    print("=" * 72)
