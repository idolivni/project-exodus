"""
Neutrino--Exoplanet Cross-Match detection module for Project EXODUS.

A NOVEL search strategy: cross-matching IceCube neutrino events against
known exoplanet host stars to search for anomalous neutrino emission
from the direction of planetary systems.

Astrophysical motivation
------------------------
Neutrino emission from the direction of an exoplanet host is not expected
in standard astrophysics.  Excess neutrino events above the isotropic
atmospheric background, clustered around a known planetary system, could
indicate:
    * A previously unknown high-energy astrophysical source (AGN jet, micro-
      quasar, etc.) co-located by chance.
    * An extraordinarily energetic phenomenon associated with the planetary
      system itself -- a speculative but high-impact technosignature channel.

Method
------
1. For each exoplanet host star, search IceCube events within a configurable
   angular radius (default 2 degrees -- matching IceCube track resolution).
2. Estimate the expected number of chance coincidences from the local
   neutrino event density, the search region area, and the number of hosts.
3. Compute Poisson significance of any observed excess above expectation.
4. Check for SPATIAL CLUSTERING: multiple neutrino events from the same
   direction as an exoplanet host.
5. Check for TEMPORAL CLUSTERING: do the matched neutrino events correlate
   in time (bursts, periodicity)?
6. Report a ranked list of exoplanet hosts with the most significant excess.

Simulation mode
---------------
When real IceCube or exoplanet data is unavailable, the module generates
synthetic catalogs with a few injected signals (excess events near
specific hosts) to demonstrate the analysis pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import (
    get_logger,
    get_config,
    cache_key,
    load_cache,
    save_cache,
    save_result,
    PROJECT_ROOT,
)

log = get_logger("detection.neutrino_exoplanet_crossmatch")

# ---------------------------------------------------------------------------
# Lazy / optional imports
# ---------------------------------------------------------------------------
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False
    log.debug("astropy not available -- positional matching will be approximate")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NeutrinoMatch:
    """A single neutrino event matched to an exoplanet host."""

    host_name: str               # Exoplanet host star name
    host_ra: float               # Host RA (deg)
    host_dec: float              # Host Dec (deg)
    neutrino_ra: float           # Neutrino event RA (deg)
    neutrino_dec: float          # Neutrino event Dec (deg)
    neutrino_mjd: float          # Neutrino event time (MJD)
    neutrino_energy_gev: float   # Neutrino event energy (GeV)
    separation_deg: float        # Angular separation (deg)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalCluster:
    """Description of temporal clustering among neutrino matches."""

    host_name: str
    n_events: int
    mjd_min: float
    mjd_max: float
    mjd_span_days: float
    mean_interval_days: Optional[float]   # Mean inter-event interval
    min_interval_days: Optional[float]    # Shortest interval (burst proxy)
    ks_pvalue: Optional[float]            # KS test p-value vs uniform

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HostExcess:
    """Statistical excess of neutrino events near an exoplanet host."""

    host_name: str
    host_ra: float
    host_dec: float
    n_observed: int              # Number of neutrino events in search cone
    n_expected: float            # Expected from background density
    excess: float                # n_observed - n_expected
    poisson_pvalue: float        # Poisson probability of >= n_observed given n_expected
    poisson_sigma: float         # Equivalent Gaussian sigma
    p_corrected: Optional[float] = None   # Bonferroni-corrected p-value
    n_trials: Optional[int] = None        # Number of hosts tested (multiplicity)
    matches: List[NeutrinoMatch] = field(default_factory=list)
    temporal_cluster: Optional[TemporalCluster] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matches"] = [m.to_dict() if isinstance(m, NeutrinoMatch) else m
                        for m in self.matches]
        if self.temporal_cluster is not None:
            d["temporal_cluster"] = self.temporal_cluster.to_dict() if isinstance(
                self.temporal_cluster, TemporalCluster
            ) else self.temporal_cluster
        return d


@dataclass
class CrossMatchResult:
    """Full cross-match result: all hosts with their neutrino statistics."""

    total_neutrino_events: int
    total_exoplanet_hosts: int
    search_radius_deg: float
    global_event_density_per_sqdeg: float  # Average neutrino events per sq deg
    hosts_with_excess: List[HostExcess] = field(default_factory=list)
    n_significant: int = 0                 # Hosts with Bonferroni-corrected p < 0.0027

    def to_dict(self) -> dict:
        return {
            "total_neutrino_events": self.total_neutrino_events,
            "total_exoplanet_hosts": self.total_exoplanet_hosts,
            "search_radius_deg": self.search_radius_deg,
            "global_event_density_per_sqdeg": self.global_event_density_per_sqdeg,
            "n_significant": self.n_significant,
            "hosts_with_excess": [
                h.to_dict() for h in self.hosts_with_excess
            ],
        }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _poisson_pvalue(n_observed: int, n_expected: float) -> float:
    """
    Compute the Poisson probability of observing >= n_observed events
    given an expectation of n_expected.

    P(X >= n | mu) = 1 - CDF(n-1 | mu)

    Uses scipy if available; otherwise a direct summation fallback.
    """
    if n_expected <= 0:
        # Audit fix N10: zero expected + zero observed = fully consistent (p=1.0),
        # not p=0.0 (which implies extreme significance).
        return 1.0 if n_observed == 0 else 1e-30
    if n_observed == 0:
        return 1.0

    try:
        from scipy.stats import poisson
        pval = 1.0 - poisson.cdf(n_observed - 1, n_expected)
        return float(max(pval, 1e-300))
    except ImportError:
        pass

    # Manual Poisson CDF via numpy
    k_range = np.arange(0, n_observed)
    log_pmf = k_range * np.log(n_expected) - n_expected - np.array(
        [_log_factorial(k) for k in k_range]
    )
    cdf = np.sum(np.exp(log_pmf))
    pval = 1.0 - cdf
    return float(max(pval, 1e-300))


def _log_factorial(n: int) -> float:
    """Log of n! using Stirling for large n."""
    if n <= 1:
        return 0.0
    try:
        from math import lgamma
        return lgamma(n + 1)
    except ImportError:
        return float(np.sum(np.log(np.arange(1, n + 1))))


def _pvalue_to_sigma(pvalue: float) -> float:
    """Convert a one-sided p-value to an equivalent Gaussian sigma."""
    if pvalue <= 0:
        return 10.0  # cap at 10 sigma
    if pvalue >= 1.0:
        return 0.0
    try:
        from scipy.stats import norm
        sigma = norm.isf(pvalue)
        return float(min(max(sigma, 0.0), 10.0))
    except ImportError:
        # Rough approximation using the complementary error function
        # erfc(x/sqrt(2)) / 2 = pvalue => x = sqrt(2) * erfcinv(2*pvalue)
        # Use numpy's approximation
        from math import log, sqrt
        if pvalue > 0.5:
            return 0.0
        # Abramowitz & Stegun approximation for small p
        t = sqrt(-2.0 * log(pvalue))
        sigma = t - (2.515517 + 0.802853 * t + 0.010328 * t ** 2) / (
            1.0 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3
        )
        return float(min(max(sigma, 0.0), 10.0))


def _ks_test_uniform(mjd_values: np.ndarray) -> Optional[float]:
    """
    Kolmogorov-Smirnov test against a uniform distribution over the
    observed time range.  Returns p-value, or None if insufficient data.
    """
    if len(mjd_values) < 3:
        return None
    try:
        from scipy.stats import kstest
        mjd_min = mjd_values.min()
        mjd_max = mjd_values.max()
        if mjd_max <= mjd_min:
            return None
        normalised = (mjd_values - mjd_min) / (mjd_max - mjd_min)
        stat, pval = kstest(normalised, "uniform")
        return float(pval)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Angular separation helpers
# ---------------------------------------------------------------------------

def _angular_separation_deg(
    ra1: np.ndarray, dec1: np.ndarray,
    ra2: float, dec2: float,
) -> np.ndarray:
    """
    Compute angular separation in degrees between arrays of positions
    and a single reference point.  Uses the Vincenty formula.
    """
    ra1_r = np.radians(ra1)
    dec1_r = np.radians(dec1)
    ra2_r = np.radians(ra2)
    dec2_r = np.radians(dec2)

    dra = ra1_r - ra2_r
    cos_dec2 = np.cos(dec2_r)
    sin_dec2 = np.sin(dec2_r)
    cos_dec1 = np.cos(dec1_r)
    sin_dec1 = np.sin(dec1_r)

    num = np.sqrt(
        (cos_dec2 * np.sin(dra)) ** 2
        + (cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * np.cos(dra)) ** 2
    )
    den = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * np.cos(dra)

    return np.degrees(np.arctan2(num, den))


# ---------------------------------------------------------------------------
# Core cross-match engine
# ---------------------------------------------------------------------------

def _build_kdtree(ra_deg: np.ndarray, dec_deg: np.ndarray):
    """Build a cKDTree from RA/Dec arrays using unit-sphere Cartesian coords.

    Returns (tree, xyz) where xyz is (N, 3) Cartesian coordinates.
    """
    from scipy.spatial import cKDTree

    ra_r = np.radians(ra_deg)
    dec_r = np.radians(dec_deg)
    cos_dec = np.cos(dec_r)
    x = cos_dec * np.cos(ra_r)
    y = cos_dec * np.sin(ra_r)
    z = np.sin(dec_r)
    xyz = np.column_stack([x, y, z])
    return cKDTree(xyz), xyz


def _deg_to_chord(angle_deg: float) -> float:
    """Convert angular separation in degrees to chord distance on unit sphere."""
    return 2.0 * np.sin(np.radians(angle_deg) / 2.0)


def crossmatch_neutrino_exoplanets(
    neutrino_events: List[Dict[str, Any]],
    exoplanet_hosts: List[Dict[str, Any]],
    search_radius_deg: float = 2.0,
    neutrino_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> CrossMatchResult:
    """
    Cross-match IceCube neutrino events against exoplanet host stars.

    For each exoplanet host, this function:
        1. Counts neutrino events within ``search_radius_deg``.
        2. Estimates background from the global neutrino event density.
        3. Computes Poisson significance of any observed excess.
        4. Checks for temporal clustering among matched events.

    Uses a KDTree for O(N log N + M log N) crossmatching instead of
    O(M × N) brute-force when scipy is available.

    Parameters
    ----------
    neutrino_events : list[dict]
        Each dict must have keys: 'ra', 'dec', 'mjd', 'energy_gev'.
        Optionally 'angular_err_deg'.  Ignored if neutrino_arrays is set.
    exoplanet_hosts : list[dict]
        Each dict must have keys: 'ra', 'dec'.
        Optionally 'host_star' or 'hostname' or 'name'.
    search_radius_deg : float
        Angular search radius in degrees (default 2.0).
    neutrino_arrays : dict of np.ndarray, optional
        Pre-built arrays with keys 'ra', 'dec', 'mjd', 'energy_gev'.
        If provided, neutrino_events list is ignored (fast path).

    Returns
    -------
    CrossMatchResult
        Full statistical summary of the cross-match.
    """
    # -- Build arrays for fast computation ---------------------------------
    if neutrino_arrays is not None:
        nu_ra = neutrino_arrays["ra"]
        nu_dec = neutrino_arrays["dec"]
        nu_mjd = neutrino_arrays.get("mjd", np.zeros(len(nu_ra)))
        nu_energy = neutrino_arrays.get("energy_gev", np.zeros(len(nu_ra)))
        n_neutrinos = len(nu_ra)
    else:
        n_neutrinos = len(neutrino_events)
        if n_neutrinos > 0:
            nu_ra = np.array([e["ra"] for e in neutrino_events], dtype=np.float64)
            nu_dec = np.array([e["dec"] for e in neutrino_events], dtype=np.float64)
            nu_mjd = np.array([e.get("mjd", 0.0) for e in neutrino_events], dtype=np.float64)
            nu_energy = np.array([e.get("energy_gev", 0.0) for e in neutrino_events], dtype=np.float64)
        else:
            nu_ra = nu_dec = nu_mjd = nu_energy = np.array([], dtype=np.float64)

    n_hosts = len(exoplanet_hosts)

    log.info(
        "Cross-matching %d neutrino events against %d exoplanet hosts "
        "(radius=%.2f deg)",
        n_neutrinos, n_hosts, search_radius_deg,
    )

    if n_neutrinos == 0 or n_hosts == 0:
        log.warning("Empty input -- returning empty result")
        return CrossMatchResult(
            total_neutrino_events=n_neutrinos,
            total_exoplanet_hosts=n_hosts,
            search_radius_deg=search_radius_deg,
            global_event_density_per_sqdeg=0.0,
        )

    # -- Global neutrino event density (events per square degree) ----------
    full_sky_sqdeg = 41253.0
    global_density = n_neutrinos / full_sky_sqdeg
    search_area_sqdeg = np.pi * search_radius_deg ** 2
    n_expected_per_host = global_density * search_area_sqdeg

    log.info(
        "Global density: %.2f events/sq.deg, expected per host: %.3f",
        global_density, n_expected_per_host,
    )

    # -- Build KDTree for fast cone queries --------------------------------
    use_kdtree = False
    try:
        nu_tree, nu_xyz = _build_kdtree(nu_ra, nu_dec)
        search_chord = _deg_to_chord(search_radius_deg)
        annulus_inner_chord = _deg_to_chord(search_radius_deg * 2.0)
        annulus_outer_chord = _deg_to_chord(search_radius_deg * 5.0)
        use_kdtree = True
        log.info("Using KDTree for fast crossmatch")
    except ImportError:
        log.info("scipy not available, using brute-force crossmatch")

    # -- Per-host cross-match ----------------------------------------------
    hosts_with_excess: List[HostExcess] = []
    annulus_inner_deg = search_radius_deg * 2.0
    annulus_outer_deg = search_radius_deg * 5.0
    annulus_area = np.pi * (annulus_outer_deg ** 2 - annulus_inner_deg ** 2)

    for host in exoplanet_hosts:
        host_ra = float(host["ra"])
        host_dec = float(host["dec"])
        host_name = (
            host.get("host_star")
            or host.get("hostname")
            or host.get("name")
            or f"host@({host_ra:.4f},{host_dec:.4f})"
        )

        if use_kdtree:
            # KDTree path: O(log N) per host
            ra_r = np.radians(host_ra)
            dec_r = np.radians(host_dec)
            host_xyz = np.array([
                np.cos(dec_r) * np.cos(ra_r),
                np.cos(dec_r) * np.sin(ra_r),
                np.sin(dec_r),
            ])

            # Query inner cone (search radius)
            inner_indices = nu_tree.query_ball_point(host_xyz, search_chord)
            n_observed = len(inner_indices)

            # Build match list with exact angular separations
            matches = []
            if n_observed > 0:
                for idx in inner_indices:
                    sep = _angular_separation_deg(
                        np.array([nu_ra[idx]]), np.array([nu_dec[idx]]),
                        host_ra, host_dec,
                    )[0]
                    matches.append(NeutrinoMatch(
                        host_name=host_name,
                        host_ra=host_ra,
                        host_dec=host_dec,
                        neutrino_ra=float(nu_ra[idx]),
                        neutrino_dec=float(nu_dec[idx]),
                        neutrino_mjd=float(nu_mjd[idx]),
                        neutrino_energy_gev=float(nu_energy[idx]),
                        separation_deg=float(sep),
                    ))

            # Local background: annular region
            outer_indices = set(nu_tree.query_ball_point(host_xyz, annulus_outer_chord))
            inner_excl = set(nu_tree.query_ball_point(host_xyz, annulus_inner_chord))
            annulus_indices = outer_indices - inner_excl
            n_annulus = len(annulus_indices)

        else:
            # Brute-force path: O(N) per host
            sep_deg = _angular_separation_deg(nu_ra, nu_dec, host_ra, host_dec)
            mask = sep_deg <= search_radius_deg
            n_observed = int(np.sum(mask))

            matches = []
            if n_observed > 0:
                indices = np.where(mask)[0]
                for idx in indices:
                    matches.append(NeutrinoMatch(
                        host_name=host_name,
                        host_ra=host_ra,
                        host_dec=host_dec,
                        neutrino_ra=float(nu_ra[idx]),
                        neutrino_dec=float(nu_dec[idx]),
                        neutrino_mjd=float(nu_mjd[idx]),
                        neutrino_energy_gev=float(nu_energy[idx]),
                        separation_deg=float(sep_deg[idx]),
                    ))

            annulus_mask = (sep_deg > annulus_inner_deg) & (sep_deg <= annulus_outer_deg)
            n_annulus = int(np.sum(annulus_mask))

        # Local background estimate
        if n_annulus > 5 and annulus_area > 0:
            local_density = n_annulus / annulus_area
            n_expected = local_density * search_area_sqdeg
        else:
            n_expected = n_expected_per_host

        # Poisson significance
        excess = n_observed - n_expected
        pval = _poisson_pvalue(n_observed, n_expected)
        sigma = _pvalue_to_sigma(pval)

        # Temporal clustering analysis (if multiple matches)
        temporal_cluster = None
        if n_observed >= 2:
            matched_mjd = np.array([m.neutrino_mjd for m in matches])
            temporal_cluster = _analyse_temporal_clustering(
                host_name, matched_mjd
            )

        host_excess = HostExcess(
            host_name=host_name,
            host_ra=host_ra,
            host_dec=host_dec,
            n_observed=n_observed,
            n_expected=float(n_expected),
            excess=float(excess),
            poisson_pvalue=float(pval),
            poisson_sigma=float(sigma),
            matches=matches,
            temporal_cluster=temporal_cluster,
        )

        if n_observed > 0:
            hosts_with_excess.append(host_excess)

    # Sort by significance (highest sigma first)
    hosts_with_excess.sort(key=lambda h: h.poisson_sigma, reverse=True)

    # Bonferroni multiplicity correction
    n_trials_val = max(1, n_hosts)
    for h in hosts_with_excess:
        h.n_trials = n_trials_val
        h.p_corrected = min(1.0, h.poisson_pvalue * n_trials_val)

    n_significant = sum(1 for h in hosts_with_excess if h.p_corrected < 0.0027)

    result = CrossMatchResult(
        total_neutrino_events=n_neutrinos,
        total_exoplanet_hosts=n_hosts,
        search_radius_deg=search_radius_deg,
        global_event_density_per_sqdeg=float(global_density),
        hosts_with_excess=hosts_with_excess,
        n_significant=n_significant,
    )

    log.info(
        "Cross-match complete: %d hosts with events, %d significant "
        "(Bonferroni-corrected p < 0.0027, n_trials=%d)",
        len(hosts_with_excess), n_significant, n_trials_val,
    )

    return result


# ---------------------------------------------------------------------------
# Temporal clustering analysis
# ---------------------------------------------------------------------------

def _analyse_temporal_clustering(
    host_name: str,
    mjd_values: np.ndarray,
) -> TemporalCluster:
    """
    Analyse temporal clustering of neutrino events matched to a host.

    Checks whether the arrival times are consistent with uniform
    distribution (null hypothesis: random atmospheric background).
    Clusters in time could indicate a burst or periodic emission.

    Parameters
    ----------
    host_name : str
        Name of the host star (for labelling).
    mjd_values : np.ndarray
        MJD values of the matched neutrino events.

    Returns
    -------
    TemporalCluster
    """
    sorted_mjd = np.sort(mjd_values)
    n = len(sorted_mjd)

    mjd_min = float(sorted_mjd[0])
    mjd_max = float(sorted_mjd[-1])
    span = mjd_max - mjd_min

    if n >= 2:
        intervals = np.diff(sorted_mjd)
        mean_interval = float(np.mean(intervals))
        min_interval = float(np.min(intervals))
    else:
        mean_interval = None
        min_interval = None

    # KS test against uniform distribution
    ks_pval = _ks_test_uniform(sorted_mjd)

    return TemporalCluster(
        host_name=host_name,
        n_events=n,
        mjd_min=mjd_min,
        mjd_max=mjd_max,
        mjd_span_days=float(span),
        mean_interval_days=mean_interval,
        min_interval_days=min_interval,
        ks_pvalue=ks_pval,
    )


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

def _generate_simulated_data(
    n_neutrinos: int = 100_000,
    n_hosts: int = 200,
    n_injected_signals: int = 3,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate simulated neutrino events and exoplanet hosts with
    a few injected excess signals for pipeline demonstration.

    Parameters
    ----------
    n_neutrinos : int
        Number of background neutrino events.
    n_hosts : int
        Number of simulated exoplanet hosts.
    n_injected_signals : int
        Number of hosts that will have excess neutrino events injected.
    seed : int
        Random seed.

    Returns
    -------
    neutrino_events : list[dict]
    exoplanet_hosts : list[dict]
    """
    rng = np.random.RandomState(seed)
    log.info(
        "Generating simulated data: %d neutrinos, %d hosts, %d injected signals",
        n_neutrinos, n_hosts, n_injected_signals,
    )

    # --- Simulated exoplanet hosts (uniform on sky) ---
    host_ra = rng.uniform(0, 360, n_hosts)
    sin_dec = rng.uniform(-1, 1, n_hosts)
    host_dec = np.degrees(np.arcsin(sin_dec))

    _star_names = [
        "Kepler-442", "TRAPPIST-1", "Proxima Centauri", "Ross 128",
        "GJ 273", "HD 164922", "Kepler-160", "Kepler-62",
        "TOI-700", "LHS 1140", "K2-18", "55 Cancri",
        "Tau Ceti", "GJ 667C", "HD 40307", "Kepler-22",
        "Kepler-452", "Kepler-186", "Kepler-296", "GJ 3293",
    ]

    hosts = []
    for i in range(n_hosts):
        name = (
            _star_names[i]
            if i < len(_star_names)
            else f"SIM-Host-{i}"
        )
        hosts.append({
            "ra": float(host_ra[i]),
            "dec": float(host_dec[i]),
            "host_star": name,
        })

    # --- Simulated background neutrino events (isotropic) ---
    nu_ra = rng.uniform(0, 360, n_neutrinos)
    sin_dec_nu = rng.uniform(-1, 1, n_neutrinos)
    nu_dec = np.degrees(np.arcsin(sin_dec_nu))
    nu_mjd = rng.uniform(54682, 58309, n_neutrinos)

    # Power-law energy spectrum
    gamma = 2.5
    u_e = rng.random(n_neutrinos)
    e_min = 100.0
    e_max = 1e7
    nu_energy = (
        e_min ** (1 - gamma) + u_e * (e_max ** (1 - gamma) - e_min ** (1 - gamma))
    ) ** (1.0 / (1 - gamma))

    neutrinos = []
    for i in range(n_neutrinos):
        neutrinos.append({
            "ra": float(nu_ra[i]),
            "dec": float(nu_dec[i]),
            "mjd": float(nu_mjd[i]),
            "energy_gev": float(nu_energy[i]),
            "angular_err_deg": float(
                np.clip(1.0 * (nu_energy[i] / 1000.0) ** (-0.4), 0.2, 5.0)
            ),
        })

    # --- Inject excess signals near selected hosts ---
    signal_hosts = rng.choice(n_hosts, size=min(n_injected_signals, n_hosts), replace=False)
    for idx in signal_hosts:
        h = hosts[idx]
        # Inject 5-15 extra events within 1.5 degrees
        n_inject = rng.randint(5, 16)
        base_mjd = rng.uniform(55000, 57000)
        for _ in range(n_inject):
            offset_ra = rng.normal(0, 0.5) / np.cos(np.radians(h["dec"]))
            offset_dec = rng.normal(0, 0.5)
            neutrinos.append({
                "ra": float(h["ra"] + offset_ra),
                "dec": float(h["dec"] + offset_dec),
                "mjd": float(base_mjd + rng.exponential(50.0)),
                "energy_gev": float(rng.uniform(500, 50000)),
                "angular_err_deg": float(rng.uniform(0.3, 1.5)),
            })

        log.info(
            "  Injected %d signal events near %s (%.2f, %.2f)",
            n_inject, h["host_star"], h["ra"], h["dec"],
        )

    rng.shuffle(neutrinos)
    return neutrinos, hosts


# ---------------------------------------------------------------------------
# Integration with project IceCube + exoplanet modules
# ---------------------------------------------------------------------------

def _load_icecube_events() -> Optional[List[Dict[str, Any]]]:
    """Attempt to load IceCube events from the project ingestion module."""
    try:
        from src.ingestion.icecube_catalog import get_all_events

        events = get_all_events()
        return [e.to_dict() for e in events]
    except Exception as exc:
        log.warning("Could not load IceCube events: %s", exc)
        return None


def _load_exoplanet_hosts() -> Optional[List[Dict[str, Any]]]:
    """Attempt to load exoplanet hosts from the project ingestion module."""
    try:
        from src.ingestion.exoplanet_archive import get_all_hosts

        hosts_df = get_all_hosts()
        hosts = []
        for _, row in hosts_df.iterrows():
            hosts.append({
                "ra": float(row.get("ra_deg", row.get("ra", 0))),
                "dec": float(row.get("dec_deg", row.get("dec", 0))),
                "host_star": str(
                    row.get("host_star", row.get("hostname", "unknown"))
                ),
            })
        return hosts
    except Exception as exc:
        log.warning("Could not load exoplanet hosts: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- Neutrino / Exoplanet Cross-Match")
    print("=" * 72)
    print()
    print("Searching for anomalous neutrino emission from the directions")
    print("of known exoplanet host stars -- a novel multi-messenger SETI")
    print("approach combining IceCube data with the exoplanet catalog.")
    print()

    # --- Attempt real data, fall back to simulation -----------------------
    neutrino_events = _load_icecube_events()
    exoplanet_hosts = _load_exoplanet_hosts()

    use_simulation = (neutrino_events is None) or (exoplanet_hosts is None)

    if use_simulation:
        print("  Real data not fully available. Running in SIMULATION mode.")
        print("  (Injecting 3 artificial signals for demonstration)")
        print()
        neutrino_events, exoplanet_hosts = _generate_simulated_data()

    print(f"  Neutrino events:   {len(neutrino_events):,}")
    print(f"  Exoplanet hosts:   {len(exoplanet_hosts):,}")
    print()

    # --- Run cross-match --------------------------------------------------
    print("[1] Cross-matching neutrino events against exoplanet hosts ...")
    print("-" * 40)
    result = crossmatch_neutrino_exoplanets(
        neutrino_events=neutrino_events,
        exoplanet_hosts=exoplanet_hosts,
        search_radius_deg=2.0,
    )

    print(f"  Global neutrino density:  {result.global_event_density_per_sqdeg:.2f} events/sq.deg")
    print(f"  Hosts with >= 1 event:    {len(result.hosts_with_excess)}")
    print(f"  Significant (corrected):  {result.n_significant}")
    print()

    # --- Report top excess hosts ------------------------------------------
    print("[2] Top 15 hosts by Poisson significance:")
    print("-" * 72)
    print(f"  {'Host':<25s} {'N_obs':>5s} {'N_exp':>7s} {'Excess':>7s} "
          f"{'p(raw)':>10s} {'p(corr)':>10s} {'Sigma':>6s}")
    print("-" * 82)

    for he in result.hosts_with_excess[:15]:
        print(
            f"  {he.host_name:<25s} {he.n_observed:>5d} {he.n_expected:>7.2f} "
            f"{he.excess:>+7.2f} {he.poisson_pvalue:>10.2e} "
            f"{(he.p_corrected or 0.0):>10.2e} {he.poisson_sigma:>6.2f}"
        )
    print()

    # --- Significant detections -------------------------------------------
    significant = [h for h in result.hosts_with_excess
                   if h.p_corrected is not None and h.p_corrected < 0.0027]
    if significant:
        print("=" * 72)
        print("  *** SIGNIFICANT DETECTIONS (Bonferroni-corrected p < 0.0027) ***")
        print("=" * 72)
        for he in significant:
            print(f"\n  Host:           {he.host_name}")
            print(f"  Position:       RA={he.host_ra:.4f}, Dec={he.host_dec:.4f}")
            print(f"  N_observed:     {he.n_observed}")
            print(f"  N_expected:     {he.n_expected:.2f}")
            print(f"  Excess:         {he.excess:+.2f}")
            print(f"  Poisson sigma:  {he.poisson_sigma:.2f} (raw)")
            print(f"  p-value (raw):  {he.poisson_pvalue:.2e}")
            print(f"  p-value (corr): {he.p_corrected:.2e}  (×{he.n_trials} trials)")

            # Show matched event details
            if he.matches:
                print(f"  Matched events ({len(he.matches)}):")
                for i, m in enumerate(he.matches[:5], 1):
                    print(
                        f"    {i}. E={m.neutrino_energy_gev:>10,.0f} GeV  "
                        f"sep={m.separation_deg:.3f} deg  "
                        f"MJD={m.neutrino_mjd:.2f}"
                    )
                if len(he.matches) > 5:
                    print(f"    ... and {len(he.matches) - 5} more")

            # Temporal clustering
            if he.temporal_cluster is not None:
                tc = he.temporal_cluster
                print(f"  Temporal clustering:")
                print(f"    Time span:        {tc.mjd_span_days:.1f} days")
                if tc.mean_interval_days is not None:
                    print(f"    Mean interval:    {tc.mean_interval_days:.1f} days")
                if tc.min_interval_days is not None:
                    print(f"    Min interval:     {tc.min_interval_days:.1f} days")
                if tc.ks_pvalue is not None:
                    uniform_tag = (
                        "CLUSTERED (non-uniform)"
                        if tc.ks_pvalue < 0.05
                        else "consistent with uniform"
                    )
                    print(
                        f"    KS p-value:       {tc.ks_pvalue:.4f}  ({uniform_tag})"
                    )
        print()
    else:
        print("  No hosts with significant excess found (after Bonferroni correction).")
        print("  (This is expected for isotropic background.)")
        print()

    # --- Summary ----------------------------------------------------------
    print("-" * 72)
    print("  SUMMARY")
    print("-" * 72)
    print(f"  Neutrino events:       {result.total_neutrino_events:,}")
    print(f"  Exoplanet hosts:       {result.total_exoplanet_hosts:,}")
    print(f"  Search radius:         {result.search_radius_deg:.1f} deg")
    print(f"  Global density:        {result.global_event_density_per_sqdeg:.2f} evt/sq.deg")
    print(f"  Hosts with events:     {len(result.hosts_with_excess)}")
    print(f"  Significant (corr.):   {result.n_significant}")
    print()

    # Persist result
    try:
        path = save_result("neutrino_exoplanet_crossmatch", result.to_dict())
        print(f"  Result saved to: {path}")
    except Exception as exc:
        print(f"  (Could not save result: {exc})")

    print()
    print("=" * 72)
    print("  Cross-match analysis complete.")
    print("=" * 72)
