"""
Gamma-ray / exoplanet host cross-match module for Project EXODUS.

Cross-matches Fermi 4FGL-DR4 UNIDENTIFIED gamma-ray sources against the
catalog of known exoplanet host stars.  Any positional coincidence between
an unidentified gamma-ray source and an exoplanet host star is an
IMMEDIATE ESCALATION event -- because:

    1. Unidentified Fermi sources have no known astrophysical counterpart.
    2. Normal stars do NOT produce detectable gamma-ray emission.
    3. A gamma-ray source coincident with a planet-hosting star would
       imply an extraordinary energy source at that stellar system.
    4. This is exactly the kind of signature an energy-intensive
       technological civilization might produce (propulsion beams,
       antimatter power, relativistic engineering).

The cross-match uses the Fermi 95% positional error ellipse (typically
3-10 arcminutes for unidentified sources) rather than simple point
matching, and computes the probability of chance coincidence for each
match given the local source density.

Statistical framework
---------------------
For each Fermi-exoplanet positional match, we compute:

    P_chance = 1 - exp(-pi * rho * r^2)

where *rho* is the local surface density of exoplanet hosts (per sq deg)
and *r* is the angular separation.  A low P_chance means the match is
unlikely to be due to random chance -- and is therefore more significant.
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
from src.utils import get_config, get_logger, save_result

logger = get_logger("detection.gamma_exoplanet_crossmatch")

# ---------------------------------------------------------------------------
# Lazy / optional imports
# ---------------------------------------------------------------------------
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False
    logger.debug("astropy not available -- sky matching will use fallback")

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
_DEFAULT_MAX_SEP_ARCMIN = 10.0   # Maximum separation for cross-match
_ESCALATION_PCHANCE = 0.01      # P(chance) threshold for immediate escalation


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GammaExoplanetMatch:
    """A single positional match between a Fermi source and an exoplanet host."""

    # Fermi source information
    fermi_name: str
    fermi_ra: float                            # Right ascension (deg)
    fermi_dec: float                           # Declination (deg)
    fermi_flux_1gev: Optional[float] = None    # Flux > 1 GeV (ph/cm2/s)
    fermi_spectral_index: Optional[float] = None
    fermi_variability_index: Optional[float] = None
    fermi_pos_err_arcmin: Optional[float] = None  # 95% error (arcmin)

    # Exoplanet host information
    host_name: str = ""
    host_ra: float = 0.0
    host_dec: float = 0.0
    host_distance_pc: Optional[float] = None

    # Match statistics
    separation_arcmin: float = 0.0             # Angular separation (arcmin)
    separation_sigma: Optional[float] = None   # Separation / pos_error
    p_chance: Optional[float] = None           # Probability of chance coincidence (single-test)
    p_corrected: Optional[float] = None        # P(chance) × n_trials (Bonferroni)
    n_trials: Optional[int] = None             # n_fermi_unid × n_hosts searched
    local_host_density_per_sqdeg: Optional[float] = None

    # Escalation flag
    escalation: bool = False                   # True = IMMEDIATE ESCALATION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GammaExoplanetMatch":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CrossMatchResult:
    """Aggregate result from the Fermi-exoplanet cross-match."""

    n_fermi_unidentified: int = 0        # Number of input Fermi unID sources
    n_exoplanet_hosts: int = 0           # Number of input exoplanet hosts
    n_matches: int = 0                   # Number of positional matches found
    n_escalations: int = 0               # Number of IMMEDIATE ESCALATION flags
    max_sep_arcmin: float = 0.0          # Maximum separation used
    matches: List[GammaExoplanetMatch] = field(default_factory=list)
    escalations: List[GammaExoplanetMatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_fermi_unidentified": self.n_fermi_unidentified,
            "n_exoplanet_hosts": self.n_exoplanet_hosts,
            "n_matches": self.n_matches,
            "n_escalations": self.n_escalations,
            "max_sep_arcmin": self.max_sep_arcmin,
            "matches": [m.to_dict() for m in self.matches],
            "escalations": [m.to_dict() for m in self.escalations],
        }


# ---------------------------------------------------------------------------
# Core cross-match engine
# ---------------------------------------------------------------------------

def crossmatch_fermi_exoplanets(
    fermi_sources: List[Dict[str, Any]],
    exoplanet_hosts: List[Dict[str, Any]],
    max_sep_arcmin: float = _DEFAULT_MAX_SEP_ARCMIN,
) -> CrossMatchResult:
    """Cross-match Fermi unidentified gamma-ray sources against exoplanet hosts.

    For each Fermi source, finds all exoplanet hosts within the search
    radius (or the source's own 95% positional uncertainty, whichever is
    larger).  For each match, computes the probability of chance
    coincidence.  Any match with P(chance) below the escalation threshold
    is flagged for IMMEDIATE ESCALATION.

    Parameters
    ----------
    fermi_sources : list[dict]
        Fermi sources.  Each dict must have at minimum:
            - ``source_name`` (str)
            - ``ra`` (float, degrees)
            - ``dec`` (float, degrees)
        Optional but recommended:
            - ``flux_1gev`` (float)
            - ``spectral_index`` (float)
            - ``variability_index`` (float)
            - ``pos_err_semimajor_deg`` (float)
            - ``pos_err_semiminor_deg`` (float)
            - ``pos_err_arcmin`` (float) -- average 95% error in arcmin
    exoplanet_hosts : list[dict]
        Exoplanet host stars.  Each dict must have at minimum:
            - ``ra`` or ``ra_deg`` (float, degrees)
            - ``dec`` or ``dec_deg`` (float, degrees)
        Optional:
            - ``host_star`` or ``hostname`` (str)
            - ``distance_pc`` (float)
    max_sep_arcmin : float
        Maximum angular separation for a match (arcminutes).  For each
        Fermi source, the effective search radius is
        ``max(max_sep_arcmin, source_pos_err_arcmin)``.

    Returns
    -------
    CrossMatchResult
        Contains all matches and escalation flags.
    """
    logger.info(
        "Cross-matching %d Fermi sources against %d exoplanet hosts "
        "(max_sep=%.1f arcmin)",
        len(fermi_sources), len(exoplanet_hosts), max_sep_arcmin,
    )

    if not fermi_sources or not exoplanet_hosts:
        logger.warning("Empty input catalog(s) -- no cross-match to perform")
        return CrossMatchResult(
            n_fermi_unidentified=len(fermi_sources),
            n_exoplanet_hosts=len(exoplanet_hosts),
            max_sep_arcmin=max_sep_arcmin,
        )

    # ---- Normalise exoplanet host coordinates ----------------------------
    host_ra = np.array([
        h.get("ra", h.get("ra_deg", 0.0)) for h in exoplanet_hosts
    ], dtype=np.float64)
    host_dec = np.array([
        h.get("dec", h.get("dec_deg", 0.0)) for h in exoplanet_hosts
    ], dtype=np.float64)

    # ---- Normalise Fermi source coordinates ------------------------------
    fermi_ra = np.array([s["ra"] for s in fermi_sources], dtype=np.float64)
    fermi_dec = np.array([s["dec"] for s in fermi_sources], dtype=np.float64)

    # ---- Compute local exoplanet host density ----------------------------
    # We estimate density in a 5-degree radius around each Fermi source.
    # This is used for the P(chance) calculation.
    density_radius_deg = 5.0
    density_area_sqdeg = np.pi * density_radius_deg ** 2

    # ---- Global search multiplicity (Bonferroni trials factor) ----------
    # Each Fermi unidentified source is tested against each exoplanet
    # host, so the total number of independent positional tests is
    # n_fermi × n_hosts.  This correction prevents false escalations
    # from chance alignments across the full search catalog.
    n_trials = max(1, len(fermi_sources) * len(exoplanet_hosts))
    logger.info("  Trials factor: %d Fermi × %d hosts = %d tests",
                len(fermi_sources), len(exoplanet_hosts), n_trials)

    # ---- Perform cross-match using astropy or fallback -------------------
    if _HAS_ASTROPY:
        matches = _crossmatch_astropy(
            fermi_sources, fermi_ra, fermi_dec,
            exoplanet_hosts, host_ra, host_dec,
            max_sep_arcmin,
            density_radius_deg, density_area_sqdeg,
            n_trials=n_trials,
        )
    else:
        matches = _crossmatch_fallback(
            fermi_sources, fermi_ra, fermi_dec,
            exoplanet_hosts, host_ra, host_dec,
            max_sep_arcmin,
            density_radius_deg, density_area_sqdeg,
            n_trials=n_trials,
        )

    # ---- Sort matches by significance (lowest P_chance first) -----------
    matches.sort(key=lambda m: (m.p_chance if m.p_chance is not None else 1.0))

    # ---- Identify escalations -------------------------------------------
    escalations = [m for m in matches if m.escalation]

    result = CrossMatchResult(
        n_fermi_unidentified=len(fermi_sources),
        n_exoplanet_hosts=len(exoplanet_hosts),
        n_matches=len(matches),
        n_escalations=len(escalations),
        max_sep_arcmin=max_sep_arcmin,
        matches=matches,
        escalations=escalations,
    )

    if escalations:
        logger.info(
            "*** IMMEDIATE ESCALATION: %d Fermi unidentified source(s) "
            "coincident with exoplanet host(s)! (corrected for %d trials) ***",
            len(escalations), n_trials,
        )
        for esc in escalations:
            logger.info(
                "  -> %s <-> %s  sep=%.2f'  sep_sigma=%.1f  "
                "P(chance)=%.4e  P(corrected)=%.4e  n_trials=%d",
                esc.fermi_name, esc.host_name,
                esc.separation_arcmin,
                esc.separation_sigma if esc.separation_sigma is not None else float("nan"),
                esc.p_chance or 0.0,
                esc.p_corrected or 0.0,
                esc.n_trials or 0,
            )
    else:
        logger.info(
            "No escalation-level matches found in %d total matches "
            "(after Bonferroni correction for %d trials)",
            len(matches), n_trials,
        )

    # Persist result
    try:
        save_result("gamma_exoplanet_crossmatch", result.to_dict())
    except Exception as exc:
        logger.debug("Could not save result: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Astropy-based cross-match
# ---------------------------------------------------------------------------

def _crossmatch_astropy(
    fermi_sources: List[Dict],
    fermi_ra: np.ndarray,
    fermi_dec: np.ndarray,
    exoplanet_hosts: List[Dict],
    host_ra: np.ndarray,
    host_dec: np.ndarray,
    max_sep_arcmin: float,
    density_radius_deg: float,
    density_area_sqdeg: float,
    n_trials: int = 1,
) -> List[GammaExoplanetMatch]:
    """Cross-match using astropy SkyCoord for accurate spherical geometry."""
    fermi_coords = SkyCoord(
        ra=fermi_ra, dec=fermi_dec, unit=(u.deg, u.deg), frame="icrs"
    )
    host_coords = SkyCoord(
        ra=host_ra, dec=host_dec, unit=(u.deg, u.deg), frame="icrs"
    )

    matches: List[GammaExoplanetMatch] = []

    for i, fsrc in enumerate(fermi_sources):
        # Determine effective search radius for this source
        src_err_arcmin = _get_pos_err_arcmin(fsrc)
        effective_radius_arcmin = max(max_sep_arcmin, src_err_arcmin or 0.0)
        effective_radius = effective_radius_arcmin * u.arcmin

        # Compute separations to all hosts
        seps = fermi_coords[i].separation(host_coords)

        # Find all hosts within the effective radius
        within = seps < effective_radius
        host_indices = np.where(within)[0]

        if len(host_indices) == 0:
            continue

        # Compute local host density for P(chance) calculation
        density_mask = seps.deg < density_radius_deg
        local_density = np.sum(density_mask) / density_area_sqdeg

        for j in host_indices:
            sep_arcmin = seps[j].arcmin

            match = _build_match(
                fsrc, exoplanet_hosts[j],
                sep_arcmin, src_err_arcmin,
                local_density,
                n_trials=n_trials,
            )
            matches.append(match)

    logger.info("Astropy cross-match found %d matches", len(matches))
    return matches


# ---------------------------------------------------------------------------
# Fallback cross-match (no astropy)
# ---------------------------------------------------------------------------

def _crossmatch_fallback(
    fermi_sources: List[Dict],
    fermi_ra: np.ndarray,
    fermi_dec: np.ndarray,
    exoplanet_hosts: List[Dict],
    host_ra: np.ndarray,
    host_dec: np.ndarray,
    max_sep_arcmin: float,
    density_radius_deg: float,
    density_area_sqdeg: float,
    n_trials: int = 1,
) -> List[GammaExoplanetMatch]:
    """Approximate cross-match using Cartesian distance (fallback)."""
    matches: List[GammaExoplanetMatch] = []
    max_sep_deg = max_sep_arcmin / 60.0

    for i, fsrc in enumerate(fermi_sources):
        src_err_arcmin = _get_pos_err_arcmin(fsrc)
        effective_radius_deg = max(max_sep_deg, (src_err_arcmin or 0.0) / 60.0)

        cos_dec = np.cos(np.radians(fermi_dec[i]))
        dra = (host_ra - fermi_ra[i]) * cos_dec
        ddec = host_dec - fermi_dec[i]
        sep_deg = np.sqrt(dra**2 + ddec**2)

        within = sep_deg < effective_radius_deg
        host_indices = np.where(within)[0]

        if len(host_indices) == 0:
            continue

        # Local density
        density_within = np.sum(sep_deg < density_radius_deg)
        local_density = density_within / density_area_sqdeg

        for j in host_indices:
            sep_arcmin = sep_deg[j] * 60.0

            match = _build_match(
                fsrc, exoplanet_hosts[j],
                sep_arcmin, src_err_arcmin,
                local_density,
                n_trials=n_trials,
            )
            matches.append(match)

    logger.info("Fallback cross-match found %d matches", len(matches))
    return matches


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_pos_err_arcmin(fsrc: Dict) -> Optional[float]:
    """Extract positional error in arcminutes from a Fermi source dict."""
    # Direct pos_err_arcmin key
    if fsrc.get("pos_err_arcmin") is not None:
        return float(fsrc["pos_err_arcmin"])

    # Compute from semi-major/semi-minor
    smaj = fsrc.get("pos_err_semimajor_deg")
    smin = fsrc.get("pos_err_semiminor_deg")
    if smaj is not None and smin is not None:
        return ((smaj + smin) / 2.0) * 60.0

    return None


def _build_match(
    fsrc: Dict,
    host: Dict,
    sep_arcmin: float,
    src_err_arcmin: Optional[float],
    local_density: float,
    n_trials: int = 1,
) -> GammaExoplanetMatch:
    """Build a GammaExoplanetMatch from a Fermi source and exoplanet host."""
    # Separation in units of positional error
    if src_err_arcmin and src_err_arcmin > 0:
        sep_sigma = sep_arcmin / src_err_arcmin
    else:
        sep_sigma = None

    # Probability of chance coincidence (single-test Poisson)
    # P = 1 - exp(-pi * rho * r^2)
    # where rho is density per sq arcmin and r is separation in arcmin
    local_density_per_sqarcmin = local_density / 3600.0  # convert from per sq deg
    p_chance = 1.0 - np.exp(-np.pi * local_density_per_sqarcmin * sep_arcmin**2)

    # Bonferroni-corrected p-value for global search multiplicity.
    # n_trials = n_fermi_unid × n_hosts, representing the number of
    # independent positional tests performed across the full catalog.
    p_corrected = min(1.0, p_chance * n_trials)

    # Escalation requires BOTH:
    #   1. Multiplicity-corrected P(chance) below threshold
    #   2. Separation within 3σ of positional error (when error is known)
    within_error = (sep_sigma is not None and sep_sigma < 3.0) or sep_sigma is None
    escalation = bool(p_corrected < _ESCALATION_PCHANCE and within_error)

    # Host name extraction
    host_name = (
        host.get("host_star")
        or host.get("hostname")
        or host.get("name")
        or f"host@({host.get('ra', host.get('ra_deg', 0)):.4f},"
           f"{host.get('dec', host.get('dec_deg', 0)):.4f})"
    )

    return GammaExoplanetMatch(
        fermi_name=fsrc.get("source_name", "unknown"),
        fermi_ra=float(fsrc["ra"]),
        fermi_dec=float(fsrc["dec"]),
        fermi_flux_1gev=fsrc.get("flux_1gev"),
        fermi_spectral_index=fsrc.get("spectral_index"),
        fermi_variability_index=fsrc.get("variability_index"),
        fermi_pos_err_arcmin=src_err_arcmin,
        host_name=host_name,
        host_ra=float(host.get("ra", host.get("ra_deg", 0))),
        host_dec=float(host.get("dec", host.get("dec_deg", 0))),
        host_distance_pc=host.get("distance_pc"),
        separation_arcmin=float(sep_arcmin),
        separation_sigma=float(sep_sigma) if sep_sigma is not None else None,
        p_chance=float(p_chance),
        p_corrected=float(p_corrected),
        n_trials=int(n_trials),
        local_host_density_per_sqdeg=float(local_density),
        escalation=escalation,
    )


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

def _generate_simulated_data(
    n_fermi: int = 2000,
    n_hosts: int = 500,
    n_planted_matches: int = 3,
    seed: int = 9876,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate simulated Fermi unidentified sources and exoplanet hosts.

    Plants a small number of deliberate positional coincidences to test
    the detection pipeline.

    Parameters
    ----------
    n_fermi : int
        Number of simulated unidentified Fermi sources.
    n_hosts : int
        Number of simulated exoplanet host stars.
    n_planted_matches : int
        Number of deliberate coincidences to plant.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    fermi_sources : list[dict]
    exoplanet_hosts : list[dict]
    """
    rng = np.random.RandomState(seed)

    # Generate Fermi unidentified sources
    fermi_sources = []
    for i in range(n_fermi):
        ra = rng.uniform(0, 360)
        # Cluster ~60% near galactic plane
        if rng.random() < 0.6:
            dec = rng.normal(0, 15)
            dec = np.clip(dec, -90, 90)
        else:
            dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))

        log_flux = rng.normal(-10.5, 1.0)
        spectral_idx = rng.normal(2.2, 0.4)
        spectral_idx = np.clip(spectral_idx, 1.0, 4.0)
        var_index = rng.exponential(20.0) + 10.0

        # Positional error: 0.5-10 arcmin (in degrees: 0.008 - 0.17)
        err_deg = 10.0 ** rng.normal(-1.3, 0.4)
        err_deg = np.clip(err_deg, 0.005, 0.3)
        smaj = err_deg * rng.uniform(1.5, 3.0)
        smin = smaj * rng.uniform(0.5, 1.0)

        fermi_sources.append({
            "source_name": f"4FGL J{int(ra/15):02d}{int((ra/15 % 1)*60):02d}"
                           f"{'+' if dec >= 0 else '-'}{int(abs(dec)):02d}"
                           f"{int((abs(dec) % 1)*60):02d}s",
            "ra": float(ra),
            "dec": float(dec),
            "flux_1gev": float(10.0 ** log_flux),
            "spectral_index": float(spectral_idx),
            "variability_index": float(var_index),
            "pos_err_semimajor_deg": float(smaj),
            "pos_err_semiminor_deg": float(smin),
            "is_unidentified": True,
            "simulated": True,
        })

    # Generate exoplanet host stars
    exoplanet_hosts = []
    for j in range(n_hosts):
        ra = rng.uniform(0, 360)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
        dist = rng.lognormal(np.log(50), 1.0)  # typical: 10-500 pc

        exoplanet_hosts.append({
            "host_star": f"SIM-Host-{j:04d}",
            "ra": float(ra),
            "dec": float(dec),
            "distance_pc": float(dist),
        })

    # Plant deliberate positional coincidences
    planted_host_names = [
        "PLANTED-Kepler-442",
        "PLANTED-TRAPPIST-1",
        "PLANTED-Proxima-Centauri",
        "PLANTED-TOI-700",
        "PLANTED-HD-40307",
    ]
    for k in range(min(n_planted_matches, n_fermi)):
        # Place a host star right next to the k-th Fermi source
        fsrc = fermi_sources[k]
        offset_arcmin = rng.uniform(0.5, 3.0)
        angle = rng.uniform(0, 2 * np.pi)
        dra = (offset_arcmin / 60.0) * np.cos(angle) / np.cos(np.radians(fsrc["dec"]))
        ddec = (offset_arcmin / 60.0) * np.sin(angle)

        host_name = planted_host_names[k] if k < len(planted_host_names) else f"PLANTED-Host-{k}"
        exoplanet_hosts.append({
            "host_star": host_name,
            "ra": float(fsrc["ra"] + dra),
            "dec": float(fsrc["dec"] + ddec),
            "distance_pc": float(rng.uniform(5, 100)),
        })

    logger.info(
        "Generated simulated data: %d Fermi sources, %d hosts "
        "(%d planted coincidences)",
        len(fermi_sources), len(exoplanet_hosts), n_planted_matches,
    )
    return fermi_sources, exoplanet_hosts


# ---------------------------------------------------------------------------
# High-level pipeline function
# ---------------------------------------------------------------------------

def run_crossmatch(
    force_simulated: bool = False,
    max_sep_arcmin: float = _DEFAULT_MAX_SEP_ARCMIN,
) -> CrossMatchResult:
    """Run the full Fermi-exoplanet cross-match pipeline.

    Loads the Fermi catalog and exoplanet host catalog from the
    project's ingestion modules, then performs the cross-match.

    Parameters
    ----------
    force_simulated : bool
        If True, use simulated data instead of real catalogs.
    max_sep_arcmin : float
        Maximum separation for cross-matching (arcminutes).

    Returns
    -------
    CrossMatchResult
    """
    if force_simulated:
        logger.info("Using simulated data for cross-match")
        fermi_sources, exoplanet_hosts = _generate_simulated_data()
    else:
        fermi_sources, exoplanet_hosts = _load_real_data()

    return crossmatch_fermi_exoplanets(
        fermi_sources, exoplanet_hosts, max_sep_arcmin=max_sep_arcmin,
    )


def _load_real_data() -> Tuple[List[Dict], List[Dict]]:
    """Attempt to load real Fermi and exoplanet catalogs."""
    # Load Fermi unidentified sources
    try:
        from src.ingestion.fermi_catalog import get_unidentified
        fermi_raw = get_unidentified()
        fermi_sources = [s.to_dict() if hasattr(s, "to_dict") else s for s in fermi_raw]
        logger.info("Loaded %d Fermi unidentified sources", len(fermi_sources))
    except Exception as exc:
        logger.warning("Could not load Fermi catalog: %s -- using simulation", exc)
        sim_fermi, sim_hosts = _generate_simulated_data()
        return sim_fermi, sim_hosts

    # Load exoplanet hosts
    try:
        from src.ingestion.exoplanet_archive import get_all_hosts
        hosts_df = get_all_hosts()
        exoplanet_hosts = []
        for _, row in hosts_df.iterrows():
            exoplanet_hosts.append({
                "host_star": str(row.get("host_star", row.get("hostname", "unknown"))),
                "ra": float(row.get("ra_deg", row.get("ra", 0))),
                "dec": float(row.get("dec_deg", row.get("dec", 0))),
                "distance_pc": float(row["distance_pc"]) if "distance_pc" in row and row["distance_pc"] == row["distance_pc"] else None,
            })
        logger.info("Loaded %d exoplanet hosts", len(exoplanet_hosts))
    except Exception as exc:
        logger.warning("Could not load exoplanet hosts: %s -- using simulation", exc)
        sim_fermi, sim_hosts = _generate_simulated_data()
        return sim_fermi, sim_hosts

    return fermi_sources, exoplanet_hosts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- Gamma-Ray / Exoplanet Host Cross-Match")
    print("=" * 72)
    print()
    print("Cross-matching Fermi 4FGL-DR4 unidentified gamma-ray sources")
    print("against known exoplanet host stars.")
    print()
    print("Any positional coincidence = IMMEDIATE ESCALATION.")
    print("Normal stars do NOT produce gamma-ray emission.")
    print()

    # ---- Attempt real data first, fall back to simulation ----------------
    print("[1] Loading catalogs ...")
    use_simulation = False
    try:
        from src.ingestion.fermi_catalog import get_unidentified, get_all_sources
        all_fermi = get_all_sources()
        fermi_unid_raw = get_unidentified()
        fermi_sources = [s.to_dict() if hasattr(s, "to_dict") else s for s in fermi_unid_raw]
        is_simulated = any(s.get("simulated", False) for s in fermi_sources[:5])
        print(f"    Fermi 4FGL-DR4 total:       {len(all_fermi)}")
        print(f"    Fermi unidentified:          {len(fermi_sources)}")
        if is_simulated:
            print("    (using simulated Fermi data)")
    except Exception as exc:
        print(f"    Could not load Fermi catalog: {exc}")
        use_simulation = True

    if not use_simulation:
        try:
            from src.ingestion.exoplanet_archive import get_all_hosts
            hosts_df = get_all_hosts()
            exoplanet_hosts = []
            for _, row in hosts_df.iterrows():
                exoplanet_hosts.append({
                    "host_star": str(row.get("host_star", row.get("hostname", "unknown"))),
                    "ra": float(row.get("ra_deg", row.get("ra", 0))),
                    "dec": float(row.get("dec_deg", row.get("dec", 0))),
                    "distance_pc": float(row["distance_pc"]) if "distance_pc" in row and row["distance_pc"] == row["distance_pc"] else None,
                })
            print(f"    Exoplanet hosts:             {len(exoplanet_hosts)}")
        except Exception as exc:
            print(f"    Could not load exoplanet hosts: {exc}")
            use_simulation = True

    if use_simulation:
        print()
        print("    Falling back to simulated data for demonstration ...")
        fermi_sources, exoplanet_hosts = _generate_simulated_data()
        print(f"    Simulated Fermi unidentified: {len(fermi_sources)}")
        print(f"    Simulated exoplanet hosts:    {len(exoplanet_hosts)}")
    print()

    # ---- Run cross-match -------------------------------------------------
    print("[2] Running cross-match ...")
    print(f"    Maximum separation: {_DEFAULT_MAX_SEP_ARCMIN} arcmin")
    print()

    result = crossmatch_fermi_exoplanets(
        fermi_sources, exoplanet_hosts,
        max_sep_arcmin=_DEFAULT_MAX_SEP_ARCMIN,
    )

    # ---- Report results --------------------------------------------------
    print(f"[3] Results:")
    print("-" * 60)
    print(f"    Fermi unidentified sources:   {result.n_fermi_unidentified}")
    print(f"    Exoplanet hosts:              {result.n_exoplanet_hosts}")
    print(f"    Positional matches:           {result.n_matches}")
    print(f"    ESCALATIONS:                  {result.n_escalations}")
    print()

    if result.matches:
        print("[4] All matches (sorted by P(chance)):")
        print("-" * 60)
        for i, m in enumerate(result.matches[:20]):
            p_str = f"{m.p_chance:.4e}" if m.p_chance is not None else "N/A"
            sigma_str = f"{m.separation_sigma:.2f}" if m.separation_sigma is not None else "N/A"
            dist_str = f"{m.host_distance_pc:.1f} pc" if m.host_distance_pc else "N/A"
            print(
                f"  {i+1:>3}. {m.fermi_name:<25s} <-> {m.host_name:<25s}"
            )
            print(
                f"       sep={m.separation_arcmin:.2f}'  "
                f"sigma={sigma_str}  "
                f"P(chance)={p_str}  "
                f"dist={dist_str}"
            )
            if m.escalation:
                print(f"       *** IMMEDIATE ESCALATION ***")
            print()

    if result.escalations:
        print("=" * 72)
        print("  *** IMMEDIATE ESCALATION SUMMARY ***")
        print("=" * 72)
        print()
        print("  The following unidentified gamma-ray sources are positionally")
        print("  coincident with known exoplanet host stars.  Normal stars do")
        print("  NOT produce gamma-ray emission -- these require IMMEDIATE")
        print("  follow-up investigation.")
        print()
        for esc in result.escalations:
            print(f"  Fermi source:    {esc.fermi_name}")
            print(f"  Exoplanet host:  {esc.host_name}")
            print(f"  Separation:      {esc.separation_arcmin:.2f} arcmin")
            if esc.separation_sigma is not None:
                print(f"  Sep / error:     {esc.separation_sigma:.2f} sigma")
            if esc.p_chance is not None:
                print(f"  P(chance):       {esc.p_chance:.4e}")
            if esc.fermi_flux_1gev is not None:
                print(f"  Fermi flux:      {esc.fermi_flux_1gev:.2e} ph/cm2/s")
            if esc.host_distance_pc is not None:
                print(f"  Host distance:   {esc.host_distance_pc:.1f} pc")
            print()
    else:
        print("[4] No escalation-level matches found.")
        print("    (This is expected for random sky coverage -- real matches")
        print("     would be extraordinarily significant.)")
        print()

    # ---- Match statistics ------------------------------------------------
    if result.matches:
        print("[5] Match statistics:")
        print("-" * 60)
        seps = np.array([m.separation_arcmin for m in result.matches])
        pchances = np.array([
            m.p_chance for m in result.matches if m.p_chance is not None
        ])
        print(f"    Separation min:    {seps.min():.2f} arcmin")
        print(f"    Separation max:    {seps.max():.2f} arcmin")
        print(f"    Separation median: {np.median(seps):.2f} arcmin")
        if len(pchances) > 0:
            print(f"    P(chance) min:     {pchances.min():.4e}")
            print(f"    P(chance) max:     {pchances.max():.4e}")
            print(f"    P(chance) median:  {np.median(pchances):.4e}")
        print()

    print("=" * 72)
    print("  Gamma-ray / exoplanet cross-match complete.")
    print("=" * 72)
