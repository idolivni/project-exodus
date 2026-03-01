"""
Temporal Archaeology module for Project EXODUS.

A NOVEL search strategy that has NEVER been performed for SETI: systematic
comparison of radio sky surveys from different decades to find sources that
appeared, disappeared, or changed significantly in flux density.

The key insight is that an artificial radio transmitter could switch on or off
on timescales of years to decades.  Natural radio sources do vary (AGN jets,
pulsars, transients), but a new radio source co-located with a known exoplanet
host star is extraordinarily unlikely to be astrophysical -- and would be a
HIGH PRIORITY technosignature candidate.

Survey pairs
------------
* **NVSS** (NRAO VLA Sky Survey, 1.4 GHz, observed 1993--1996): the definitive
  1.4-GHz catalog of the northern sky (Dec > -40 deg), ~1.8 million sources,
  accessed via VizieR catalog ``VIII/65/nvss``.

* **VLASS** (VLA Sky Survey, 2--4 GHz, 2017--present): the modern successor,
  accessed via the CIRADA VLASS component catalog on VizieR, or directly
  from the CIRADA catalog service if available.

By comparing these two epochs separated by ~25 years we can identify:
    a) Sources in VLASS but NOT in NVSS  --> "appeared"
    b) Sources in NVSS but NOT in VLASS  --> "disappeared"
    c) Sources matched but with >3-sigma flux change --> "brightened" / "dimmed"

Any change co-located with a known exoplanet host star is flagged as
HIGH PRIORITY for follow-up observation.

Flux comparison notes
---------------------
NVSS is at 1.4 GHz and VLASS at 3 GHz, so a direct mJy-to-mJy comparison
must account for spectral index.  For a typical synchrotron source with
spectral index alpha ~ -0.7, S_3GHz ~ 0.54 * S_1.4GHz.  We apply this
correction when computing flux ratios so that a "steady" source yields
ratio ~ 1.0.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_config, get_logger, save_result

log = get_logger("detection.temporal_archaeology")

# ---------------------------------------------------------------------------
# Lazy / optional imports
# ---------------------------------------------------------------------------
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False
    log.debug("astropy not available -- positional cross-matching disabled")

try:
    from astroquery.vizier import Vizier

    _HAS_VIZIER = True
except ImportError:
    _HAS_VIZIER = False
    log.debug("astroquery.vizier not available -- catalog queries disabled")

try:
    from src.correlation.sky_matcher import crossmatch, MatchResult

    _HAS_SKYMATCHER = True
except ImportError:
    _HAS_SKYMATCHER = False
    log.debug("sky_matcher not available -- using internal crossmatch fallback")

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
_DEFAULT_CROSSMATCH_ARCSEC: float = 3.0
_DEFAULT_ANOMALY_SIGMA: float = 3.0

# Typical synchrotron spectral index for the NVSS->VLASS frequency correction.
# S_nu ~ nu^alpha  =>  S_3GHz / S_1.4GHz = (3.0/1.4)^alpha
_DEFAULT_SPECTRAL_INDEX: float = -0.7
_FREQ_RATIO = 3.0 / 1.4  # VLASS / NVSS centre frequencies
_DEFAULT_FLUX_CORRECTION = _FREQ_RATIO ** _DEFAULT_SPECTRAL_INDEX  # ~ 0.54

# VizieR catalog IDs
_NVSS_CATALOG = "VIII/65/nvss"
_VLASS_CATALOG = "J/ApJS/255/30/table1"  # CIRADA VLASS Epoch 1 Quick Look


def _read_config_value(key_path: str, default: float) -> float:
    """Read a dot-separated key from the project config, with fallback."""
    try:
        cfg = get_config()
        val = cfg
        for k in key_path.split("."):
            val = val[k]
        return float(val)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TemporalChange:
    """A single source showing temporal variability between survey epochs."""

    ra: float                            # Right ascension (deg, ICRS)
    dec: float                           # Declination (deg, ICRS)
    change_type: str                     # appeared / disappeared / brightened / dimmed
    nvss_flux: Optional[float]           # NVSS 1.4 GHz flux density (mJy), None if absent
    vlass_flux: Optional[float]          # VLASS 3 GHz flux density (mJy), None if absent
    flux_ratio: Optional[float]          # VLASS / (NVSS * correction), None if one-sided
    sigma_change: Optional[float]        # Significance of the flux change in sigma
    exoplanet_match: bool = False        # Co-located with a known exoplanet host?
    exoplanet_host_name: Optional[str] = None  # Name of the matched host, if any

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalArchaeologyResult:
    """Aggregate result from a temporal archaeology scan."""

    n_appeared: int = 0                  # Sources in VLASS but NOT NVSS
    n_disappeared: int = 0               # Sources in NVSS but NOT VLASS
    n_changed: int = 0                   # Sources in both with significant flux change
    changes: List[TemporalChange] = field(default_factory=list)
    high_priority: List[TemporalChange] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_appeared": self.n_appeared,
            "n_disappeared": self.n_disappeared,
            "n_changed": self.n_changed,
            "n_high_priority": len(self.high_priority),
            "changes": [c.to_dict() for c in self.changes],
            "high_priority": [c.to_dict() for c in self.high_priority],
        }


# ---------------------------------------------------------------------------
# Temporal Archaeology engine
# ---------------------------------------------------------------------------

class TemporalArchaeology:
    """
    Compare radio sky surveys from different decades to discover sources
    that appeared, disappeared, or changed -- the temporal archaeology of
    the radio sky.

    Parameters
    ----------
    crossmatch_radius_arcsec : float, optional
        Angular separation threshold for positional cross-matching between
        the two survey catalogs.  Defaults to the project config value
        ``search.crossmatch_radius_arcsec`` (3.0").
    spectral_index : float, optional
        Assumed spectral index for the NVSS-to-VLASS frequency correction.
        Default is -0.7 (typical optically-thin synchrotron).
    """

    def __init__(
        self,
        crossmatch_radius_arcsec: float = None,
        spectral_index: float = _DEFAULT_SPECTRAL_INDEX,
    ):
        if crossmatch_radius_arcsec is None:
            crossmatch_radius_arcsec = _read_config_value(
                "search.crossmatch_radius_arcsec", _DEFAULT_CROSSMATCH_ARCSEC
            )
        self.crossmatch_radius_arcsec = crossmatch_radius_arcsec
        self.spectral_index = spectral_index
        self.flux_correction = _FREQ_RATIO ** self.spectral_index
        self.anomaly_sigma = _read_config_value(
            "search.anomaly_sigma", _DEFAULT_ANOMALY_SIGMA
        )

        log.info(
            "TemporalArchaeology initialised: crossmatch_radius=%.2f\", "
            "spectral_index=%.2f, flux_correction=%.4f, anomaly_sigma=%.1f",
            self.crossmatch_radius_arcsec,
            self.spectral_index,
            self.flux_correction,
            self.anomaly_sigma,
        )

    # -----------------------------------------------------------------
    # Catalog queries
    # -----------------------------------------------------------------

    def query_nvss(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Query NVSS catalog via VizieR for sources in a circular region.

        Parameters
        ----------
        ra_center, dec_center : float
            Centre of the search cone (deg, ICRS).
        radius_deg : float
            Search radius in degrees.

        Returns
        -------
        list[dict]
            Each dict has at least 'ra', 'dec', 'flux_mJy', 'e_flux_mJy'.
        """
        if not _HAS_VIZIER or not _HAS_ASTROPY:
            log.warning(
                "VizieR/astropy not available; returning empty NVSS result"
            )
            return []

        log.info(
            "Querying NVSS (VizieR %s) at (%.4f, %.4f) radius=%.2f deg",
            _NVSS_CATALOG, ra_center, dec_center, radius_deg,
        )

        try:
            viz = Vizier(
                columns=["RAJ2000", "DEJ2000", "S1.4", "e_S1.4"],
                row_limit=-1,
            )
            center = SkyCoord(
                ra=ra_center, dec=dec_center, unit=(u.deg, u.deg), frame="icrs"
            )
            tables = viz.query_region(
                center, radius=radius_deg * u.deg, catalog=_NVSS_CATALOG
            )

            if tables is None or len(tables) == 0:
                log.info("NVSS query returned no results")
                return []

            table = tables[0]
            sources = []
            for row in table:
                try:
                    sources.append({
                        "ra": float(row["RAJ2000"]),
                        "dec": float(row["DEJ2000"]),
                        "flux_mJy": float(row["S1.4"]),
                        "e_flux_mJy": float(row["e_S1.4"]) if row["e_S1.4"] else 0.5,
                        "survey": "NVSS",
                    })
                except (ValueError, KeyError) as exc:
                    log.debug("Skipping NVSS row due to parse error: %s", exc)
                    continue

            log.info("NVSS: %d sources retrieved", len(sources))
            return sources

        except Exception as exc:
            log.error("NVSS query failed: %s", exc)
            return []

    def query_vlass(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Query VLASS catalog via VizieR or CIRADA for sources in a region.

        Attempts the CIRADA VLASS component catalog on VizieR first.  Falls
        back to a simpler VizieR query if the primary catalog is not found.

        Parameters
        ----------
        ra_center, dec_center : float
            Centre of the search cone (deg, ICRS).
        radius_deg : float
            Search radius in degrees.

        Returns
        -------
        list[dict]
            Each dict has at least 'ra', 'dec', 'flux_mJy', 'e_flux_mJy'.
        """
        if not _HAS_VIZIER or not _HAS_ASTROPY:
            log.warning(
                "VizieR/astropy not available; returning empty VLASS result"
            )
            return []

        log.info(
            "Querying VLASS (VizieR %s) at (%.4f, %.4f) radius=%.2f deg",
            _VLASS_CATALOG, ra_center, dec_center, radius_deg,
        )

        try:
            viz = Vizier(
                columns=["RAJ2000", "DEJ2000", "Ftot", "e_Ftot"],
                row_limit=-1,
            )
            center = SkyCoord(
                ra=ra_center, dec=dec_center, unit=(u.deg, u.deg), frame="icrs"
            )
            tables = viz.query_region(
                center, radius=radius_deg * u.deg, catalog=_VLASS_CATALOG
            )

            if tables is None or len(tables) == 0:
                log.info(
                    "Primary VLASS catalog returned no results; "
                    "attempting fallback catalogs"
                )
                return self._query_vlass_fallback(ra_center, dec_center, radius_deg)

            table = tables[0]
            sources = []
            for row in table:
                try:
                    sources.append({
                        "ra": float(row["RAJ2000"]),
                        "dec": float(row["DEJ2000"]),
                        "flux_mJy": float(row["Ftot"]),
                        "e_flux_mJy": float(row["e_Ftot"]) if row["e_Ftot"] else 0.3,
                        "survey": "VLASS",
                    })
                except (ValueError, KeyError) as exc:
                    log.debug("Skipping VLASS row due to parse error: %s", exc)
                    continue

            log.info("VLASS: %d sources retrieved", len(sources))
            return sources

        except Exception as exc:
            log.error("VLASS query failed: %s -- trying fallback", exc)
            return self._query_vlass_fallback(ra_center, dec_center, radius_deg)

    def _query_vlass_fallback(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float,
    ) -> List[Dict[str, Any]]:
        """Fallback VLASS queries: try alternative VizieR catalog IDs."""
        fallback_catalogs = [
            ("J/ApJS/255/30", ["RAJ2000", "DEJ2000", "Ftot", "e_Ftot"]),
            # Audit fix N11: removed placeholder catalog VIII/XXX (non-existent)
        ]
        center = SkyCoord(
            ra=ra_center, dec=dec_center, unit=(u.deg, u.deg), frame="icrs"
        )

        for cat_id, col_names in fallback_catalogs:
            try:
                viz = Vizier(columns=col_names, row_limit=-1)
                tables = viz.query_region(
                    center, radius=radius_deg * u.deg, catalog=cat_id
                )
                if tables is not None and len(tables) > 0:
                    table = tables[0]
                    sources = []
                    # Determine the flux column name dynamically
                    flux_col = col_names[2]
                    eflux_col = col_names[3]
                    for row in table:
                        try:
                            sources.append({
                                "ra": float(row[col_names[0]]),
                                "dec": float(row[col_names[1]]),
                                "flux_mJy": float(row[flux_col]),
                                "e_flux_mJy": float(row[eflux_col]) if row[eflux_col] else 0.3,
                                "survey": "VLASS",
                            })
                        except (ValueError, KeyError):
                            continue
                    if sources:
                        log.info(
                            "VLASS fallback (%s): %d sources", cat_id, len(sources)
                        )
                        return sources
            except Exception:
                continue

        log.warning("All VLASS catalog queries returned no results")
        return []

    # -----------------------------------------------------------------
    # Cross-matching and change identification
    # -----------------------------------------------------------------

    def _crossmatch_catalogs(
        self,
        catalog_a: List[Dict[str, Any]],
        catalog_b: List[Dict[str, Any]],
    ) -> Tuple[
        List[Tuple[Dict, Dict, float]],  # matched pairs (a, b, sep_arcsec)
        List[Dict[str, Any]],            # unmatched in a
        List[Dict[str, Any]],            # unmatched in b
    ]:
        """Positional cross-match between two source lists.

        Uses the project's sky_matcher if available; otherwise falls back
        to a direct astropy SkyCoord match.

        Returns
        -------
        matched : list of (source_a, source_b, separation_arcsec)
        unmatched_a : sources in catalog_a with no counterpart in catalog_b
        unmatched_b : sources in catalog_b with no counterpart in catalog_a
        """
        if not catalog_a or not catalog_b:
            return [], list(catalog_a), list(catalog_b)

        if _HAS_SKYMATCHER:
            return self._crossmatch_via_skymatcher(catalog_a, catalog_b)
        elif _HAS_ASTROPY:
            return self._crossmatch_via_astropy(catalog_a, catalog_b)
        else:
            log.error(
                "Neither sky_matcher nor astropy available for crossmatch"
            )
            return [], list(catalog_a), list(catalog_b)

    def _crossmatch_via_skymatcher(
        self,
        catalog_a: List[Dict[str, Any]],
        catalog_b: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[Dict, Dict, float]], List[Dict], List[Dict]]:
        """Cross-match using the project sky_matcher module."""
        matches = crossmatch(
            catalog_a, catalog_b,
            max_sep_arcsec=self.crossmatch_radius_arcsec,
        )

        matched_pairs = []
        matched_a_indices = set()
        matched_b_indices = set()

        for m in matches:
            matched_pairs.append((m.source_a, m.source_b, m.separation_arcsec))
            # Track which sources were matched by position
            matched_a_indices.add((m.source_a["ra"], m.source_a["dec"]))
            matched_b_indices.add((m.source_b["ra"], m.source_b["dec"]))

        unmatched_a = [
            s for s in catalog_a
            if (s["ra"], s["dec"]) not in matched_a_indices
        ]
        unmatched_b = [
            s for s in catalog_b
            if (s["ra"], s["dec"]) not in matched_b_indices
        ]

        return matched_pairs, unmatched_a, unmatched_b

    def _crossmatch_via_astropy(
        self,
        catalog_a: List[Dict[str, Any]],
        catalog_b: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[Dict, Dict, float]], List[Dict], List[Dict]]:
        """Cross-match using raw astropy SkyCoord (fallback)."""
        ra_a = np.array([s["ra"] for s in catalog_a])
        dec_a = np.array([s["dec"] for s in catalog_a])
        ra_b = np.array([s["ra"] for s in catalog_b])
        dec_b = np.array([s["dec"] for s in catalog_b])

        coord_a = SkyCoord(ra=ra_a, dec=dec_a, unit=(u.deg, u.deg), frame="icrs")
        coord_b = SkyCoord(ra=ra_b, dec=dec_b, unit=(u.deg, u.deg), frame="icrs")

        max_sep = self.crossmatch_radius_arcsec * u.arcsec

        # Match A -> B
        idx_b, sep_ab, _ = coord_a.match_to_catalog_sky(coord_b)
        # Match B -> A (to find unmatched in B)
        idx_a, sep_ba, _ = coord_b.match_to_catalog_sky(coord_a)

        matched_pairs = []
        matched_a_set = set()
        matched_b_set = set()

        for i_a in range(len(catalog_a)):
            i_b = int(idx_b[i_a])
            sep = sep_ab[i_a]
            if sep <= max_sep:
                matched_pairs.append(
                    (catalog_a[i_a], catalog_b[i_b], sep.arcsec)
                )
                matched_a_set.add(i_a)
                matched_b_set.add(i_b)

        unmatched_a = [
            catalog_a[i] for i in range(len(catalog_a))
            if i not in matched_a_set
        ]
        unmatched_b = [
            catalog_b[i] for i in range(len(catalog_b))
            if i not in matched_b_set
        ]

        return matched_pairs, unmatched_a, unmatched_b

    def find_changes(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float = 1.0,
    ) -> List[TemporalChange]:
        """Query both surveys and identify all temporal changes.

        This is the core discovery method.  It:
            1. Queries NVSS and VLASS for the same sky region.
            2. Cross-matches the two catalogs by position.
            3. Classifies each source as appeared, disappeared,
               brightened, or dimmed.

        Parameters
        ----------
        ra_center, dec_center : float
            Centre of the search region (deg, ICRS).
        radius_deg : float
            Radius of the search cone in degrees.

        Returns
        -------
        list[TemporalChange]
            All detected temporal changes, sorted by |sigma_change|
            descending (most significant first).
        """
        log.info(
            "Finding temporal changes at (%.4f, %.4f) r=%.2f deg",
            ra_center, dec_center, radius_deg,
        )

        nvss_sources = self.query_nvss(ra_center, dec_center, radius_deg)
        vlass_sources = self.query_vlass(ra_center, dec_center, radius_deg)

        log.info(
            "Retrieved %d NVSS sources and %d VLASS sources",
            len(nvss_sources), len(vlass_sources),
        )

        if not nvss_sources and not vlass_sources:
            log.warning("Both catalogs empty -- no changes to detect")
            return []

        matched, unmatched_nvss, unmatched_vlass = self._crossmatch_catalogs(
            nvss_sources, vlass_sources
        )

        changes: List[TemporalChange] = []

        # (a) Sources in VLASS but NOT in NVSS -> "appeared"
        for src in unmatched_vlass:
            changes.append(TemporalChange(
                ra=src["ra"],
                dec=src["dec"],
                change_type="appeared",
                nvss_flux=None,
                vlass_flux=src.get("flux_mJy"),
                flux_ratio=None,
                sigma_change=None,
            ))

        # (b) Sources in NVSS but NOT in VLASS -> "disappeared"
        for src in unmatched_nvss:
            changes.append(TemporalChange(
                ra=src["ra"],
                dec=src["dec"],
                change_type="disappeared",
                nvss_flux=src.get("flux_mJy"),
                vlass_flux=None,
                flux_ratio=None,
                sigma_change=None,
            ))

        # (c) Matched sources with significant flux change
        for nvss_src, vlass_src, sep in matched:
            nvss_flux = nvss_src.get("flux_mJy")
            vlass_flux = vlass_src.get("flux_mJy")

            if nvss_flux is None or vlass_flux is None:
                continue
            if nvss_flux <= 0:
                continue

            # Correct NVSS flux to the VLASS frequency using spectral index
            nvss_flux_corrected = nvss_flux * self.flux_correction

            # Compute flux ratio (VLASS / corrected-NVSS)
            flux_ratio = vlass_flux / nvss_flux_corrected

            # Propagate uncertainties for significance calculation
            e_nvss = nvss_src.get("e_flux_mJy", 0.5)
            e_vlass = vlass_src.get("e_flux_mJy", 0.3)
            # Uncertainty on the corrected NVSS flux
            e_nvss_corrected = e_nvss * abs(self.flux_correction)

            flux_diff = vlass_flux - nvss_flux_corrected
            e_diff = np.sqrt(e_nvss_corrected**2 + e_vlass**2)

            if e_diff > 0:
                sigma_change = flux_diff / e_diff
            else:
                sigma_change = 0.0

            # Only flag significant changes
            if abs(sigma_change) >= self.anomaly_sigma:
                if sigma_change > 0:
                    change_type = "brightened"
                else:
                    change_type = "dimmed"

                changes.append(TemporalChange(
                    ra=vlass_src["ra"],
                    dec=vlass_src["dec"],
                    change_type=change_type,
                    nvss_flux=nvss_flux,
                    vlass_flux=vlass_flux,
                    flux_ratio=float(flux_ratio),
                    sigma_change=float(sigma_change),
                ))

        # Sort by significance: appeared/disappeared first (sigma=None treated
        # as infinite), then by |sigma| descending
        changes.sort(
            key=lambda c: (
                0 if c.sigma_change is None else 1,
                -(abs(c.sigma_change) if c.sigma_change is not None else 0),
            )
        )

        log.info(
            "Temporal changes found: %d appeared, %d disappeared, "
            "%d significantly changed flux",
            sum(1 for c in changes if c.change_type == "appeared"),
            sum(1 for c in changes if c.change_type == "disappeared"),
            sum(1 for c in changes if c.change_type in ("brightened", "dimmed")),
        )
        return changes

    # -----------------------------------------------------------------
    # Exoplanet cross-referencing
    # -----------------------------------------------------------------

    def cross_reference_exoplanets(
        self,
        changes: List[TemporalChange],
        exoplanet_hosts: List[Dict[str, Any]],
    ) -> List[TemporalChange]:
        """Check if any temporal changes co-locate with exoplanet host stars.

        Parameters
        ----------
        changes : list[TemporalChange]
            Temporal changes from ``find_changes()``.
        exoplanet_hosts : list[dict]
            Each dict must have 'ra', 'dec', and optionally 'host_star'
            or 'hostname'.  Typically from the exoplanet_archive module.

        Returns
        -------
        list[TemporalChange]
            The same list, mutated in-place with ``exoplanet_match`` and
            ``exoplanet_host_name`` set where applicable.  Also returned
            for convenience.
        """
        if not changes or not exoplanet_hosts:
            return changes

        if not _HAS_ASTROPY:
            log.warning("astropy unavailable; skipping exoplanet cross-reference")
            return changes

        log.info(
            "Cross-referencing %d temporal changes against %d exoplanet hosts",
            len(changes), len(exoplanet_hosts),
        )

        # Build SkyCoord arrays
        change_ra = np.array([c.ra for c in changes])
        change_dec = np.array([c.dec for c in changes])
        change_coords = SkyCoord(
            ra=change_ra, dec=change_dec, unit=(u.deg, u.deg), frame="icrs"
        )

        host_ra = np.array([h["ra"] for h in exoplanet_hosts])
        host_dec = np.array([h["dec"] for h in exoplanet_hosts])
        host_coords = SkyCoord(
            ra=host_ra, dec=host_dec, unit=(u.deg, u.deg), frame="icrs"
        )

        max_sep = self.crossmatch_radius_arcsec * u.arcsec
        idx_host, sep, _ = change_coords.match_to_catalog_sky(host_coords)

        n_matches = 0
        for i, (i_host, s) in enumerate(zip(idx_host, sep)):
            if s <= max_sep:
                changes[i].exoplanet_match = True
                host = exoplanet_hosts[int(i_host)]
                name = (
                    host.get("host_star")
                    or host.get("hostname")
                    or host.get("name")
                    or f"host@({host['ra']:.4f},{host['dec']:.4f})"
                )
                changes[i].exoplanet_host_name = name
                n_matches += 1

        log.info(
            "Exoplanet cross-reference: %d / %d changes co-locate with a host star",
            n_matches, len(changes),
        )
        return changes

    # -----------------------------------------------------------------
    # Full pipeline methods
    # -----------------------------------------------------------------

    def scan_region(
        self,
        ra_center: float,
        dec_center: float,
        radius_deg: float = 1.0,
        exoplanet_hosts: Optional[List[Dict[str, Any]]] = None,
    ) -> TemporalArchaeologyResult:
        """Full temporal archaeology pipeline for a sky region.

        1. Query NVSS and VLASS.
        2. Cross-match and identify changes.
        3. Cross-reference with exoplanet hosts.
        4. Compile HIGH PRIORITY list (exoplanet-matched changes).

        Parameters
        ----------
        ra_center, dec_center : float
            Centre of the search region (deg, ICRS).
        radius_deg : float
            Search cone radius in degrees.
        exoplanet_hosts : list[dict], optional
            Exoplanet host catalog.  If None, attempts to load from the
            project's exoplanet archive module.

        Returns
        -------
        TemporalArchaeologyResult
        """
        log.info(
            "=== Temporal Archaeology scan: (%.4f, %.4f) r=%.2f deg ===",
            ra_center, dec_center, radius_deg,
        )

        # Step 1-2: Find changes
        changes = self.find_changes(ra_center, dec_center, radius_deg)

        # Step 3: Load exoplanet hosts if not provided
        if exoplanet_hosts is None:
            exoplanet_hosts = self._load_exoplanet_hosts()

        # Step 4: Cross-reference with exoplanet hosts
        if exoplanet_hosts:
            self.cross_reference_exoplanets(changes, exoplanet_hosts)

        # Compile result
        n_appeared = sum(1 for c in changes if c.change_type == "appeared")
        n_disappeared = sum(1 for c in changes if c.change_type == "disappeared")
        n_changed = sum(
            1 for c in changes if c.change_type in ("brightened", "dimmed")
        )
        high_priority = [c for c in changes if c.exoplanet_match]

        result = TemporalArchaeologyResult(
            n_appeared=n_appeared,
            n_disappeared=n_disappeared,
            n_changed=n_changed,
            changes=changes,
            high_priority=high_priority,
        )

        if high_priority:
            log.info(
                "*** HIGH PRIORITY: %d temporal changes co-located with "
                "exoplanet hosts! ***",
                len(high_priority),
            )
            for hp in high_priority:
                log.info(
                    "  -> %s near %s (%.4f, %.4f) type=%s",
                    hp.exoplanet_host_name,
                    hp.change_type,
                    hp.ra, hp.dec,
                    hp.change_type,
                )

        log.info(
            "Scan complete: %d appeared, %d disappeared, %d flux-changed, "
            "%d high-priority",
            n_appeared, n_disappeared, n_changed, len(high_priority),
        )

        # Persist result
        try:
            save_result("temporal_archaeology", result.to_dict())
        except Exception as exc:
            log.debug("Could not save result: %s", exc)

        return result

    def scan_target_list(
        self,
        targets: List[Dict[str, Any]],
        search_radius_deg: float = 0.05,
        exoplanet_hosts: Optional[List[Dict[str, Any]]] = None,
    ) -> TemporalArchaeologyResult:
        """Check temporal changes near a list of specific targets.

        Useful for scanning known interesting targets (e.g., exoplanet hosts,
        SETI candidates) for temporal radio variability.

        Parameters
        ----------
        targets : list[dict]
            Each dict must have 'ra' and 'dec' (degrees).  Optionally
            'name' for logging.
        search_radius_deg : float
            Search cone radius around each target (default 0.05 deg ~ 3').
        exoplanet_hosts : list[dict], optional
            Exoplanet host catalog for cross-referencing.

        Returns
        -------
        TemporalArchaeologyResult
            Aggregated result across all targets.
        """
        log.info(
            "Scanning %d targets for temporal radio changes (r=%.4f deg)",
            len(targets), search_radius_deg,
        )

        if exoplanet_hosts is None:
            exoplanet_hosts = self._load_exoplanet_hosts()

        all_changes: List[TemporalChange] = []

        for i, target in enumerate(targets):
            ra = target["ra"]
            dec = target["dec"]
            name = target.get("name", f"target_{i}")
            log.info(
                "  [%d/%d] %s (%.4f, %.4f)", i + 1, len(targets), name, ra, dec
            )

            changes = self.find_changes(ra, dec, search_radius_deg)

            if exoplanet_hosts:
                self.cross_reference_exoplanets(changes, exoplanet_hosts)

            all_changes.extend(changes)

        # Compile aggregate result
        n_appeared = sum(1 for c in all_changes if c.change_type == "appeared")
        n_disappeared = sum(
            1 for c in all_changes if c.change_type == "disappeared"
        )
        n_changed = sum(
            1 for c in all_changes if c.change_type in ("brightened", "dimmed")
        )
        high_priority = [c for c in all_changes if c.exoplanet_match]

        result = TemporalArchaeologyResult(
            n_appeared=n_appeared,
            n_disappeared=n_disappeared,
            n_changed=n_changed,
            changes=all_changes,
            high_priority=high_priority,
        )

        log.info(
            "Target list scan complete: %d total changes across %d targets "
            "(%d high-priority)",
            len(all_changes), len(targets), len(high_priority),
        )

        try:
            save_result("temporal_archaeology_targets", result.to_dict())
        except Exception as exc:
            log.debug("Could not save result: %s", exc)

        return result

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _load_exoplanet_hosts(self) -> List[Dict[str, Any]]:
        """Attempt to load exoplanet hosts from the project ingestion module."""
        try:
            from src.ingestion.exoplanet_archive import get_all_hosts

            hosts_df = get_all_hosts()
            hosts = []
            for _, row in hosts_df.iterrows():
                hosts.append({
                    "ra": float(row.get("ra_deg", row.get("ra", 0))),
                    "dec": float(row.get("dec_deg", row.get("dec", 0))),
                    "host_star": str(row.get("host_star", row.get("hostname", "unknown"))),
                })
            log.info("Loaded %d exoplanet hosts from archive", len(hosts))
            return hosts

        except Exception as exc:
            log.warning(
                "Could not load exoplanet hosts: %s -- "
                "exoplanet cross-referencing will be skipped",
                exc,
            )
            return []


# ===========================================================================
# Mock data generator (for demo / offline testing)
# ===========================================================================

def _generate_mock_data(
    ra_center: float = 180.0,
    dec_center: float = 45.0,
    radius_deg: float = 0.5,
    n_nvss: int = 50,
    n_vlass: int = 55,
    n_common: int = 40,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Generate synthetic NVSS, VLASS, and exoplanet host catalogs.

    Returns
    -------
    nvss_sources, vlass_sources, exoplanet_hosts
    """
    rng = np.random.RandomState(seed)

    # Common sources (present in both surveys)
    common_ra = ra_center + (rng.random(n_common) - 0.5) * 2 * radius_deg
    common_dec = dec_center + (rng.random(n_common) - 0.5) * 2 * radius_deg
    common_flux_nvss = rng.uniform(2.0, 100.0, n_common)  # mJy at 1.4 GHz

    nvss_sources = []
    vlass_sources = []

    for i in range(n_common):
        nvss_flux = common_flux_nvss[i]
        e_nvss = max(0.3, nvss_flux * 0.05)

        # Apply spectral index to predict VLASS flux, with some scatter
        expected_vlass = nvss_flux * _DEFAULT_FLUX_CORRECTION
        # Most sources steady, but a few show significant variability
        if i < 3:
            # Brightened significantly
            vlass_flux = expected_vlass * rng.uniform(3.0, 8.0)
        elif i < 6:
            # Dimmed significantly
            vlass_flux = expected_vlass * rng.uniform(0.05, 0.2)
        else:
            # Normal scatter
            vlass_flux = expected_vlass * rng.normal(1.0, 0.1)
            vlass_flux = max(0.1, vlass_flux)

        e_vlass = max(0.2, vlass_flux * 0.05)

        # Small positional offset to simulate real matching
        pos_offset_arcsec = rng.uniform(0, 1.5)
        angle = rng.uniform(0, 2 * np.pi)
        dra = pos_offset_arcsec / 3600.0 * np.cos(angle) / np.cos(np.radians(common_dec[i]))
        ddec = pos_offset_arcsec / 3600.0 * np.sin(angle)

        nvss_sources.append({
            "ra": common_ra[i],
            "dec": common_dec[i],
            "flux_mJy": float(nvss_flux),
            "e_flux_mJy": float(e_nvss),
            "survey": "NVSS",
        })
        vlass_sources.append({
            "ra": common_ra[i] + dra,
            "dec": common_dec[i] + ddec,
            "flux_mJy": float(vlass_flux),
            "e_flux_mJy": float(e_vlass),
            "survey": "VLASS",
        })

    # NVSS-only sources (disappeared by the VLASS epoch)
    n_nvss_only = n_nvss - n_common
    for _ in range(n_nvss_only):
        ra = ra_center + (rng.random() - 0.5) * 2 * radius_deg
        dec = dec_center + (rng.random() - 0.5) * 2 * radius_deg
        flux = rng.uniform(2.0, 30.0)
        nvss_sources.append({
            "ra": float(ra),
            "dec": float(dec),
            "flux_mJy": float(flux),
            "e_flux_mJy": float(max(0.3, flux * 0.05)),
            "survey": "NVSS",
        })

    # VLASS-only sources (appeared since NVSS epoch)
    n_vlass_only = n_vlass - n_common
    for _ in range(n_vlass_only):
        ra = ra_center + (rng.random() - 0.5) * 2 * radius_deg
        dec = dec_center + (rng.random() - 0.5) * 2 * radius_deg
        flux = rng.uniform(0.5, 20.0)
        vlass_sources.append({
            "ra": float(ra),
            "dec": float(dec),
            "flux_mJy": float(flux),
            "e_flux_mJy": float(max(0.2, flux * 0.05)),
            "survey": "VLASS",
        })

    # Mock exoplanet hosts: place a few at positions overlapping with
    # the "appeared" and "brightened" VLASS sources for demo purposes
    exoplanet_hosts = []
    if n_vlass_only > 0:
        # Place a host near the first "appeared" VLASS source
        appeared_src = vlass_sources[n_common]
        exoplanet_hosts.append({
            "ra": appeared_src["ra"] + rng.uniform(-0.5, 0.5) / 3600.0,
            "dec": appeared_src["dec"] + rng.uniform(-0.5, 0.5) / 3600.0,
            "host_star": "MOCK-Kepler-442",
        })
    if n_common >= 3:
        # Place a host near one of the "brightened" sources
        bright_src = vlass_sources[0]
        exoplanet_hosts.append({
            "ra": bright_src["ra"] + rng.uniform(-0.5, 0.5) / 3600.0,
            "dec": bright_src["dec"] + rng.uniform(-0.5, 0.5) / 3600.0,
            "host_star": "MOCK-TRAPPIST-1",
        })
    # Add a few more random hosts that should NOT match
    for j in range(5):
        exoplanet_hosts.append({
            "ra": ra_center + rng.uniform(-5.0, 5.0),
            "dec": dec_center + rng.uniform(-5.0, 5.0),
            "host_star": f"MOCK-Host-{j}",
        })

    return nvss_sources, vlass_sources, exoplanet_hosts


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  Project EXODUS -- Temporal Archaeology of the Radio Sky")
    print("=" * 72)
    print()
    print("This is a NOVEL search strategy: comparing radio surveys from")
    print("different decades to find sources that appeared, disappeared,")
    print("or changed -- then cross-referencing against exoplanet hosts.")
    print()

    # ---- Configuration ------------------------------------------------
    ra_center = 180.0    # deg (12h 00m)
    dec_center = 45.0    # deg (+45 deg)
    radius_deg = 0.5     # ~1 sq degree patch

    print(f"Search region: RA={ra_center:.1f}, Dec={dec_center:.1f}, "
          f"radius={radius_deg:.2f} deg")
    print()

    # ---- Initialize engine --------------------------------------------
    engine = TemporalArchaeology()

    # ---- Try real data first ------------------------------------------
    print("[1] Attempting to query real survey data ...")
    nvss_sources = engine.query_nvss(ra_center, dec_center, radius_deg)
    vlass_sources = engine.query_vlass(ra_center, dec_center, radius_deg)

    use_mock = (not nvss_sources) or (not vlass_sources)

    if use_mock:
        print()
        print("    Real catalog data not available (VizieR/astroquery may not")
        print("    be installed or accessible). Generating mock demo data ...")
        print()

        mock_nvss, mock_vlass, mock_hosts = _generate_mock_data(
            ra_center=ra_center,
            dec_center=dec_center,
            radius_deg=radius_deg,
        )

        print(f"    Mock NVSS sources:  {len(mock_nvss)}")
        print(f"    Mock VLASS sources: {len(mock_vlass)}")
        print(f"    Mock exoplanet hosts: {len(mock_hosts)}")
        print()

        # Run the cross-match and change detection manually on mock data
        print("[2] Cross-matching NVSS and VLASS catalogs ...")
        matched, unmatched_nvss, unmatched_vlass = engine._crossmatch_catalogs(
            mock_nvss, mock_vlass
        )
        print(f"    Matched pairs:       {len(matched)}")
        print(f"    NVSS-only (vanished): {len(unmatched_nvss)}")
        print(f"    VLASS-only (new):     {len(unmatched_vlass)}")
        print()

        # Build change list manually from mock data
        changes: List[TemporalChange] = []

        for src in unmatched_vlass:
            changes.append(TemporalChange(
                ra=src["ra"], dec=src["dec"],
                change_type="appeared",
                nvss_flux=None, vlass_flux=src["flux_mJy"],
                flux_ratio=None, sigma_change=None,
            ))

        for src in unmatched_nvss:
            changes.append(TemporalChange(
                ra=src["ra"], dec=src["dec"],
                change_type="disappeared",
                nvss_flux=src["flux_mJy"], vlass_flux=None,
                flux_ratio=None, sigma_change=None,
            ))

        for nvss_src, vlass_src, sep in matched:
            nvss_flux = nvss_src["flux_mJy"]
            vlass_flux = vlass_src["flux_mJy"]
            if nvss_flux <= 0:
                continue
            nvss_corrected = nvss_flux * engine.flux_correction
            ratio = vlass_flux / nvss_corrected
            e_nvss_c = nvss_src.get("e_flux_mJy", 0.5) * abs(engine.flux_correction)
            e_vlass = vlass_src.get("e_flux_mJy", 0.3)
            diff = vlass_flux - nvss_corrected
            e_diff = np.sqrt(e_nvss_c**2 + e_vlass**2)
            sigma = diff / e_diff if e_diff > 0 else 0.0

            if abs(sigma) >= engine.anomaly_sigma:
                changes.append(TemporalChange(
                    ra=vlass_src["ra"], dec=vlass_src["dec"],
                    change_type="brightened" if sigma > 0 else "dimmed",
                    nvss_flux=nvss_flux, vlass_flux=vlass_flux,
                    flux_ratio=float(ratio), sigma_change=float(sigma),
                ))

        print(f"[3] Temporal changes detected: {len(changes)}")
        n_appeared = sum(1 for c in changes if c.change_type == "appeared")
        n_disappeared = sum(1 for c in changes if c.change_type == "disappeared")
        n_brightened = sum(1 for c in changes if c.change_type == "brightened")
        n_dimmed = sum(1 for c in changes if c.change_type == "dimmed")
        print(f"    Appeared (new):       {n_appeared}")
        print(f"    Disappeared:          {n_disappeared}")
        print(f"    Brightened (>3sigma): {n_brightened}")
        print(f"    Dimmed (>3sigma):     {n_dimmed}")
        print()

        # Cross-reference with exoplanet hosts
        print("[4] Cross-referencing with exoplanet host stars ...")
        engine.cross_reference_exoplanets(changes, mock_hosts)
        high_priority = [c for c in changes if c.exoplanet_match]
        print(f"    Exoplanet-matched changes: {len(high_priority)}")
        print()

        if high_priority:
            print("=" * 72)
            print("  *** HIGH PRIORITY DETECTIONS ***")
            print("=" * 72)
            for hp in high_priority:
                print(f"    Host: {hp.exoplanet_host_name}")
                print(f"      Position:    ({hp.ra:.6f}, {hp.dec:.6f})")
                print(f"      Change type: {hp.change_type}")
                if hp.nvss_flux is not None:
                    print(f"      NVSS flux:   {hp.nvss_flux:.2f} mJy")
                if hp.vlass_flux is not None:
                    print(f"      VLASS flux:  {hp.vlass_flux:.2f} mJy")
                if hp.flux_ratio is not None:
                    print(f"      Flux ratio:  {hp.flux_ratio:.2f}")
                if hp.sigma_change is not None:
                    print(f"      Sigma:       {hp.sigma_change:.1f}")
                print()
        else:
            print("    (No exoplanet-matched changes found in this patch)")
            print()

        # Summary
        print("-" * 72)
        print("  SUMMARY")
        print("-" * 72)
        print(f"  Region:     RA={ra_center:.1f}, Dec={dec_center:.1f}, "
              f"r={radius_deg:.2f} deg")
        print(f"  NVSS:       {len(mock_nvss)} sources (1993-1996 epoch)")
        print(f"  VLASS:      {len(mock_vlass)} sources (2017+ epoch)")
        print(f"  Changes:    {len(changes)} total")
        print(f"    New:      {n_appeared}")
        print(f"    Vanished: {n_disappeared}")
        print(f"    Brighter: {n_brightened}")
        print(f"    Dimmer:   {n_dimmed}")
        print(f"  High-priority (exoplanet-matched): {len(high_priority)}")

    else:
        # Real data path: use the full pipeline
        print(f"    NVSS: {len(nvss_sources)} sources")
        print(f"    VLASS: {len(vlass_sources)} sources")
        print()

        print("[2] Running full temporal archaeology pipeline ...")
        result = engine.scan_region(
            ra_center, dec_center, radius_deg
        )

        print()
        print("-" * 72)
        print("  RESULTS")
        print("-" * 72)
        print(f"  Appeared (new):       {result.n_appeared}")
        print(f"  Disappeared:          {result.n_disappeared}")
        print(f"  Flux-changed:         {result.n_changed}")
        print(f"  HIGH PRIORITY:        {len(result.high_priority)}")
        print()

        if result.high_priority:
            print("  *** HIGH PRIORITY DETECTIONS ***")
            for hp in result.high_priority:
                print(f"    {hp.exoplanet_host_name}: {hp.change_type} "
                      f"at ({hp.ra:.6f}, {hp.dec:.6f})")
                if hp.sigma_change is not None:
                    print(f"      sigma={hp.sigma_change:.1f}, "
                          f"ratio={hp.flux_ratio:.2f}")
            print()

        # Print top 10 most significant changes
        significant = [c for c in result.changes if c.sigma_change is not None]
        significant.sort(key=lambda c: abs(c.sigma_change), reverse=True)
        if significant:
            print("  Top 10 most significant flux changes:")
            for c in significant[:10]:
                exo_tag = f" [EXOPLANET: {c.exoplanet_host_name}]" if c.exoplanet_match else ""
                print(
                    f"    ({c.ra:.5f}, {c.dec:.5f}) {c.change_type:>10s}  "
                    f"sigma={c.sigma_change:+.1f}  ratio={c.flux_ratio:.2f}"
                    f"{exo_tag}"
                )

    print()
    print("=" * 72)
    print("  Temporal archaeology complete.")
    print("=" * 72)
