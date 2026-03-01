"""
FRB-Orbital Period Correlation for Project EXODUS.

Tests whether repeating FRB intervals match planetary orbital periods of
nearby exoplanet systems.  Nobody has checked this correlation — even a
null result is a first-of-its-kind analysis.

For each repeating FRB:
  1. Extract inter-burst time intervals
  2. Cross-match position against exoplanet hosts within localization error
  3. If spatial match: test whether FRB repeat interval is a harmonic of
     any known planetary orbital period in that system
  4. Apply Kepler's third law: given host star mass and FRB repeat period,
     what orbital radius does that imply?  Is it physically reasonable?
  5. Statistical test: how many FRB-orbital period matches would we expect
     by chance?

Public API
----------
correlate_frb_orbits(repeaters, exoplanet_hosts, max_sep_arcmin=10)
    Run the full FRB-orbital period correlation analysis.

test_period_match(frb_intervals, orbital_period, tolerance=0.05)
    Test whether FRB intervals are harmonics of an orbital period.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger

log = get_logger("detection.frb_orbital_correlation")


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class PeriodMatch:
    """A single FRB-orbital period match."""
    frb_name: str
    host_name: str
    planet_name: str
    frb_mean_interval: float    # days
    orbital_period: float       # days
    harmonic_ratio: float       # frb_interval / orbital_period
    nearest_harmonic: int       # nearest integer harmonic
    residual_fraction: float    # |ratio - nearest_integer| / nearest_integer
    angular_separation_arcmin: float
    is_significant: bool
    implied_orbital_radius_au: float = 0.0
    host_mass_msun: float = 1.0


@dataclass
class FRBOrbitalResult:
    """Full FRB-orbital correlation analysis result."""
    n_repeaters_tested: int
    n_spatial_matches: int
    n_period_matches: int
    n_expected_by_chance: float
    significance_sigma: float
    matches: List[PeriodMatch] = field(default_factory=list)
    spatial_matches: List[Dict[str, Any]] = field(default_factory=list)
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_repeaters_tested": self.n_repeaters_tested,
            "n_spatial_matches": self.n_spatial_matches,
            "n_period_matches": self.n_period_matches,
            "n_expected_by_chance": self.n_expected_by_chance,
            "significance_sigma": self.significance_sigma,
            "interpretation": self.interpretation,
            "matches": [
                {
                    "frb": m.frb_name,
                    "host": m.host_name,
                    "planet": m.planet_name,
                    "frb_interval": m.frb_mean_interval,
                    "orbital_period": m.orbital_period,
                    "harmonic_ratio": m.harmonic_ratio,
                    "nearest_harmonic": m.nearest_harmonic,
                    "residual": m.residual_fraction,
                    "sep_arcmin": m.angular_separation_arcmin,
                }
                for m in self.matches
            ],
        }


# =====================================================================
#  Period matching
# =====================================================================

def test_period_match(
    frb_intervals: List[float],
    orbital_period: float,
    tolerance: float = 0.05,
    max_harmonic: int = 10,
) -> Tuple[bool, float, int, float]:
    """Test whether FRB intervals are harmonics of an orbital period.

    Checks if the mean FRB interval divided by the orbital period is
    close to an integer (within tolerance fraction).

    Parameters
    ----------
    frb_intervals : list of float
        Inter-burst intervals in days.
    orbital_period : float
        Known planetary orbital period in days.
    tolerance : float
        Fractional tolerance for harmonic matching (default 5%).
    max_harmonic : int
        Maximum harmonic number to test.

    Returns
    -------
    is_match : bool
    harmonic_ratio : float
    nearest_harmonic : int
    residual_fraction : float
    """
    if not frb_intervals or orbital_period <= 0:
        return False, 0.0, 0, 1.0

    mean_interval = np.mean(frb_intervals)
    ratio = mean_interval / orbital_period

    # Find nearest integer harmonic (both N and 1/N)
    best_residual = 1.0
    best_harmonic = 0
    best_ratio = ratio

    # Check if mean_interval = N * orbital_period
    for n in range(1, max_harmonic + 1):
        residual = abs(ratio - n) / n
        if residual < best_residual:
            best_residual = residual
            best_harmonic = n
            best_ratio = ratio

    # Check if orbital_period = N * mean_interval (sub-harmonic)
    inv_ratio = orbital_period / mean_interval if mean_interval > 0 else 0
    for n in range(1, max_harmonic + 1):
        residual = abs(inv_ratio - n) / n
        if residual < best_residual:
            best_residual = residual
            best_harmonic = -n  # negative means sub-harmonic
            best_ratio = ratio

    is_match = best_residual < tolerance

    return is_match, float(best_ratio), int(best_harmonic), float(best_residual)


def _kepler_orbital_radius(period_days: float, mass_msun: float) -> float:
    """Compute orbital radius from Kepler's third law.

    a^3 = (M_star / M_sun) * (P / yr)^2  [in AU]

    Parameters
    ----------
    period_days : float
        Orbital period in days.
    mass_msun : float
        Host star mass in solar masses.

    Returns
    -------
    float
        Orbital semi-major axis in AU.
    """
    period_yr = period_days / 365.25
    a_cubed = mass_msun * period_yr ** 2
    return float(a_cubed ** (1.0 / 3.0))


# =====================================================================
#  Public API
# =====================================================================

def correlate_frb_orbits(
    repeaters: List[Any],
    exoplanet_hosts: List[Dict[str, Any]],
    max_sep_arcmin: float = 10.0,
    period_tolerance: float = 0.05,
) -> FRBOrbitalResult:
    """Run the full FRB-orbital period correlation analysis.

    Parameters
    ----------
    repeaters : list
        Repeating FRB objects with .name, .ra, .dec, .burst_mjds, .n_bursts.
    exoplanet_hosts : list of dict
        Each must have: hostname, ra, dec, pl_name, pl_orbper (days),
        st_mass (solar masses).
    max_sep_arcmin : float
        Maximum angular separation for spatial matching.
    period_tolerance : float
        Fractional tolerance for harmonic period matching.

    Returns
    -------
    FRBOrbitalResult
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    log.info(
        "FRB-orbital correlation: %d repeaters, %d exoplanet hosts, "
        "max_sep=%.1f arcmin",
        len(repeaters), len(exoplanet_hosts), max_sep_arcmin,
    )

    if not repeaters or not exoplanet_hosts:
        return FRBOrbitalResult(
            n_repeaters_tested=len(repeaters),
            n_spatial_matches=0,
            n_period_matches=0,
            n_expected_by_chance=0,
            significance_sigma=0,
            interpretation="No data available for correlation",
        )

    # Build exoplanet host SkyCoord
    host_coords = SkyCoord(
        ra=[h["ra"] for h in exoplanet_hosts] * u.degree,
        dec=[h["dec"] for h in exoplanet_hosts] * u.degree,
    )

    spatial_matches = []
    period_matches = []

    for rep in repeaters:
        frb_coord = SkyCoord(ra=rep.ra * u.degree, dec=rep.dec * u.degree)
        seps = frb_coord.separation(host_coords).arcmin
        nearby = np.where(seps < max_sep_arcmin)[0]

        if len(nearby) == 0:
            continue

        # Compute FRB inter-burst intervals
        burst_mjds = sorted(rep.burst_mjds)
        intervals = np.diff(burst_mjds).tolist() if len(burst_mjds) > 1 else []

        for idx in nearby:
            host = exoplanet_hosts[idx]
            sep_arcmin = float(seps[idx])

            spatial_matches.append({
                "frb": rep.name,
                "host": host.get("hostname", "unknown"),
                "sep_arcmin": sep_arcmin,
            })

            # Get orbital periods for all planets in this system
            orbital_period = host.get("pl_orbper")
            if orbital_period is None or orbital_period <= 0:
                continue

            if not intervals:
                continue

            is_match, ratio, harmonic, residual = test_period_match(
                intervals, orbital_period, tolerance=period_tolerance,
            )

            host_mass = host.get("st_mass", 1.0)
            implied_a = _kepler_orbital_radius(
                abs(rep.mean_interval_days) if hasattr(rep, 'mean_interval_days') else np.mean(intervals),
                host_mass if host_mass and host_mass > 0 else 1.0,
            )

            if is_match:
                period_matches.append(PeriodMatch(
                    frb_name=rep.name,
                    host_name=host.get("hostname", "unknown"),
                    planet_name=host.get("pl_name", "unknown"),
                    frb_mean_interval=float(np.mean(intervals)),
                    orbital_period=float(orbital_period),
                    harmonic_ratio=ratio,
                    nearest_harmonic=harmonic,
                    residual_fraction=residual,
                    angular_separation_arcmin=sep_arcmin,
                    is_significant=True,
                    implied_orbital_radius_au=implied_a,
                    host_mass_msun=float(host_mass) if host_mass else 1.0,
                ))

    # Estimate chance coincidences
    # Sky area per FRB search cone
    search_area_sq_deg = np.pi * (max_sep_arcmin / 60.0) ** 2
    total_sky_sq_deg = 41253.0  # full sphere
    host_density = len(exoplanet_hosts) / total_sky_sq_deg

    expected_spatial = len(repeaters) * search_area_sq_deg * host_density
    # Period match probability ~10% by chance (given tolerance)
    expected_period = expected_spatial * 0.10

    # Poisson significance
    n_obs = len(period_matches)
    if expected_period > 0:
        from scipy.stats import poisson
        p_value = 1.0 - poisson.cdf(max(n_obs - 1, 0), expected_period)
        significance = float(sp_stats_norm_isf(p_value)) if p_value < 0.5 else 0.0
    else:
        significance = 0.0

    # Interpretation
    if n_obs == 0:
        interpretation = (
            f"No FRB-orbital period matches found. "
            f"Tested {len(repeaters)} repeaters against {len(exoplanet_hosts)} hosts. "
            f"Expected {expected_spatial:.1f} spatial coincidences, {expected_period:.2f} period matches by chance. "
            f"Null result is expected and still a first-of-its-kind analysis."
        )
    else:
        interpretation = (
            f"Found {n_obs} FRB-orbital period match(es)! "
            f"Expected {expected_period:.2f} by chance. "
            f"{'Likely chance coincidence.' if n_obs <= expected_period * 3 else 'POTENTIALLY SIGNIFICANT — warrants follow-up!'}"
        )

    result = FRBOrbitalResult(
        n_repeaters_tested=len(repeaters),
        n_spatial_matches=len(spatial_matches),
        n_period_matches=n_obs,
        n_expected_by_chance=float(expected_period),
        significance_sigma=significance,
        matches=period_matches,
        spatial_matches=spatial_matches,
        interpretation=interpretation,
    )

    log.info(
        "FRB-orbital result: %d spatial matches, %d period matches "
        "(expected %.2f by chance)",
        len(spatial_matches), n_obs, expected_period,
    )

    return result


def sp_stats_norm_isf(p: float) -> float:
    """Inverse survival function (sigma from p-value)."""
    from scipy.stats import norm
    return float(norm.isf(p))


# =====================================================================
#  CLI demo
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- FRB-Orbital Period Correlation Demo")
    print("=" * 70)

    from src.ingestion.frb_catalog import get_repeaters

    rng = np.random.default_rng(seed=42)

    # Get repeating FRBs
    repeaters = get_repeaters()
    print(f"\n  Repeating FRBs: {len(repeaters)}")

    # Generate simulated exoplanet hosts
    n_hosts = 5000
    hosts = []
    for i in range(n_hosts):
        ra = rng.uniform(0, 360)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
        hosts.append({
            "hostname": f"HD{100000 + i}",
            "ra": float(ra),
            "dec": float(dec),
            "pl_name": f"HD{100000 + i} b",
            "pl_orbper": float(10 ** rng.uniform(0, 3)),  # 1-1000 days
            "st_mass": float(rng.uniform(0.5, 2.0)),
        })

    print(f"  Exoplanet hosts: {n_hosts}")

    result = correlate_frb_orbits(repeaters, hosts, max_sep_arcmin=10.0)
    print(f"\n  Spatial matches:    {result.n_spatial_matches}")
    print(f"  Period matches:     {result.n_period_matches}")
    print(f"  Expected by chance: {result.n_expected_by_chance:.2f}")
    print(f"  Significance:       {result.significance_sigma:.1f} sigma")
    print(f"\n  {result.interpretation}")

    if result.matches:
        print(f"\n  Period matches:")
        for m in result.matches[:5]:
            print(f"    {m.frb_name} <-> {m.host_name} ({m.planet_name})")
            print(f"      FRB interval={m.frb_mean_interval:.1f}d, "
                  f"orbital={m.orbital_period:.1f}d, "
                  f"harmonic={m.nearest_harmonic}, "
                  f"residual={m.residual_fraction:.3f}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
