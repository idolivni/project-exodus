"""
Sky cross-matching infrastructure for Project EXODUS.

The backbone of multi-modal convergence: takes (RA, Dec) positions from any
dataset and cross-matches them against any other catalog within a configurable
angular separation threshold.  Uses astropy's optimised KD-tree matching
under the hood (`SkyCoord.match_to_catalog_sky()`).

Typical usage:
    from src.correlation.sky_matcher import crossmatch, find_in_all_catalogs

    matches = crossmatch(catalog_a, catalog_b, max_sep_arcsec=3.0)
    overview = find_in_all_catalogs(ra, dec, catalogs_dict)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
except ImportError as exc:
    raise ImportError(
        "astropy is required for sky cross-matching. "
        "Install it with:  pip install astropy"
    ) from exc

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ── Project utilities ────────────────────────────────────────────────
from src.utils import get_config, get_logger

logger = get_logger("correlation.sky_matcher")

# ── Configuration ────────────────────────────────────────────────────
_DEFAULT_MAX_SEP_ARCSEC: float = 3.0

def _configured_radius() -> float:
    """Read the cross-match radius from the project config, falling back
    to the hard-coded default if the config key is absent."""
    try:
        cfg = get_config()
        return float(cfg["search"]["crossmatch_radius_arcsec"])
    except Exception:
        logger.debug(
            "Could not read crossmatch_radius_arcsec from config; "
            "using default %.1f arcsec",
            _DEFAULT_MAX_SEP_ARCSEC,
        )
        return _DEFAULT_MAX_SEP_ARCSEC


# ── Result dataclasses ───────────────────────────────────────────────
@dataclass(frozen=True)
class MatchResult:
    """A single matched pair across two catalogs."""

    source_a: Dict[str, Any]
    source_b: Dict[str, Any]
    separation_arcsec: float

    def __repr__(self) -> str:
        return (
            f"MatchResult(sep={self.separation_arcsec:.4f}\", "
            f"a=({self.source_a.get('ra')}, {self.source_a.get('dec')}), "
            f"b=({self.source_b.get('ra')}, {self.source_b.get('dec')}))"
        )


@dataclass
class CatalogMatchSummary:
    """Summary of a single-target lookup in one catalog."""

    catalog_name: str
    matched: bool
    separation_arcsec: Optional[float] = None
    closest_source: Optional[Dict[str, Any]] = None


@dataclass
class BatchMatchResult:
    """Result of matching a primary catalog against one secondary catalog."""

    secondary_name: str
    matches: List[MatchResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.matches)


# ── Proper motion propagation ────────────────────────────────────────

# Gaia DR3 reference epoch (J2016.0 in Julian years)
GAIA_REF_EPOCH = 2016.0


def propagate_proper_motion(
    ra: float,
    dec: float,
    pmra: float,
    pmdec: float,
    epoch_from: float,
    epoch_to: float,
) -> Tuple[float, float]:
    """Predict a source's position at a different epoch using proper motion.

    Parameters
    ----------
    ra, dec : float
        Position at ``epoch_from`` in degrees (ICRS).
    pmra : float
        Proper motion in RA (mas/yr), already includes cos(dec) factor.
    pmdec : float
        Proper motion in Dec (mas/yr).
    epoch_from : float
        Reference epoch of the input position (Julian year, e.g. 2016.0).
    epoch_to : float
        Target epoch to propagate to (Julian year).

    Returns
    -------
    ra_new, dec_new : float
        Predicted position in degrees.
    """
    dt = epoch_to - epoch_from  # years

    if dt == 0 or (pmra == 0 and pmdec == 0):
        return ra, dec

    # Convert mas/yr to deg/yr
    pmra_deg = pmra / (3600.0 * 1000.0)
    pmdec_deg = pmdec / (3600.0 * 1000.0)

    # Account for cos(dec) factor (pmra already includes it from Gaia)
    dec_rad = np.radians(dec)
    cos_dec = np.cos(dec_rad)
    if cos_dec > 0:
        ra_new = ra + (pmra_deg * dt) / cos_dec
    else:
        ra_new = ra

    dec_new = dec + pmdec_deg * dt

    # Wrap RA to [0, 360)
    ra_new = ra_new % 360.0

    return float(ra_new), float(dec_new)


def uncertainty_aware_radius(
    source_a: Dict[str, Any],
    source_b: Dict[str, Any],
    base_radius_arcsec: float = 3.0,
    sigma_factor: float = 3.0,
) -> float:
    """Compute an uncertainty-aware match radius.

    Combines the positional uncertainties of both sources with the
    base radius.

    Parameters
    ----------
    source_a, source_b : dict
        Source dictionaries.  If they contain 'ra_error' or
        'parallax_error' keys (in mas), these are used.
    base_radius_arcsec : float
        Minimum match radius in arcseconds.
    sigma_factor : float
        Number of sigma to add in quadrature.

    Returns
    -------
    radius_arcsec : float
        Effective match radius.
    """
    # Extract positional uncertainties in arcseconds
    sigma_a = 0.0
    sigma_b = 0.0

    for src, var_name in [(source_a, "sigma_a"), (source_b, "sigma_b")]:
        pos_err = src.get("ra_error", src.get("position_error", 0))
        if pos_err:
            # Convert mas to arcsec
            err_arcsec = float(pos_err) / 1000.0
        else:
            err_arcsec = 0.0

        if var_name == "sigma_a":
            sigma_a = err_arcsec
        else:
            sigma_b = err_arcsec

    # Combined uncertainty radius
    combined_sigma = np.sqrt(sigma_a**2 + sigma_b**2)
    radius = max(base_radius_arcsec, sigma_factor * combined_sigma)

    return float(radius)


# ── Internal helpers ─────────────────────────────────────────────────
CatalogInput = Union[
    List[Dict[str, float]],
    List[Tuple[float, float]],
    "pd.DataFrame",
]


def _normalize_catalog(catalog: CatalogInput) -> List[Dict[str, float]]:
    """Convert any supported catalog format into a list of dicts with 'ra'
    and 'dec' keys (both in degrees).

    Accepted formats:
        - list of dicts:   [{'ra': 180.0, 'dec': -23.1}, ...]
        - list of tuples:  [(180.0, -23.1), ...]
        - pandas DataFrame with 'ra' and 'dec' columns
    """
    if _HAS_PANDAS and isinstance(catalog, pd.DataFrame):
        if "ra" not in catalog.columns or "dec" not in catalog.columns:
            raise ValueError(
                "DataFrame must contain 'ra' and 'dec' columns. "
                f"Found columns: {list(catalog.columns)}"
            )
        return catalog[["ra", "dec"]].to_dict(orient="records")

    if not catalog:
        return []

    first = catalog[0]

    if isinstance(first, dict):
        for i, entry in enumerate(catalog):
            if "ra" not in entry or "dec" not in entry:
                raise ValueError(
                    f"Entry at index {i} missing 'ra' and/or 'dec' keys: {entry}"
                )
        return [{"ra": float(e["ra"]), "dec": float(e["dec"]), **e} for e in catalog]

    if isinstance(first, (list, tuple)):
        results = []
        for i, entry in enumerate(catalog):
            if len(entry) < 2:
                raise ValueError(
                    f"Tuple at index {i} must have at least 2 elements (ra, dec): {entry}"
                )
            results.append({"ra": float(entry[0]), "dec": float(entry[1])})
        return results

    raise TypeError(
        f"Unsupported catalog entry type: {type(first)}. "
        "Expected dict, tuple, or pandas DataFrame rows."
    )


def _build_skycoord(sources: List[Dict[str, float]]) -> SkyCoord:
    """Build an astropy SkyCoord array from a normalised source list."""
    ra_arr = np.array([s["ra"] for s in sources], dtype=np.float64)
    dec_arr = np.array([s["dec"] for s in sources], dtype=np.float64)
    return SkyCoord(ra=ra_arr, dec=dec_arr, unit=(u.deg, u.deg), frame="icrs")


# ── Public API ───────────────────────────────────────────────────────
def crossmatch(
    catalog_a: CatalogInput,
    catalog_b: CatalogInput,
    max_sep_arcsec: Optional[float] = None,
    target_epoch: Optional[float] = None,
) -> List[MatchResult]:
    """Cross-match two catalogs within *max_sep_arcsec* angular separation.

    For every source in *catalog_a*, the closest counterpart in *catalog_b*
    is found.  Only pairs whose on-sky separation is <= *max_sep_arcsec* are
    returned.

    Parameters
    ----------
    catalog_a, catalog_b : list[dict] | list[tuple] | DataFrame
        Input catalogs.  Each entry must supply RA and Dec in degrees.
        If entries contain 'pmra' and 'pmdec' (proper motion in mas/yr)
        and 'ref_epoch' (Julian year), positions are propagated to
        ``target_epoch`` before matching.
    max_sep_arcsec : float, optional
        Maximum angular separation in arcseconds.  Falls back to the value
        in ``config/settings.yaml`` (``search.crossmatch_radius_arcsec``)
        or 3.0" if the config is unavailable.
    target_epoch : float, optional
        Epoch to propagate positions to (Julian year, e.g. 2020.0).
        If None, no proper motion correction is applied.

    Returns
    -------
    list[MatchResult]
        Matched pairs sorted by ascending separation.
    """
    if max_sep_arcsec is None:
        max_sep_arcsec = _configured_radius()

    logger.info(
        "Cross-matching catalogs (A=%d sources, B=%d sources, radius=%.2f\"%s)",
        len(catalog_a) if hasattr(catalog_a, "__len__") else -1,
        len(catalog_b) if hasattr(catalog_b, "__len__") else -1,
        max_sep_arcsec,
        f", epoch={target_epoch:.1f}" if target_epoch else "",
    )

    sources_a = _normalize_catalog(catalog_a)
    sources_b = _normalize_catalog(catalog_b)

    if not sources_a or not sources_b:
        logger.warning("One or both catalogs are empty; returning no matches.")
        return []

    # --- Proper motion propagation ---
    if target_epoch is not None:
        n_propagated = 0
        for src_list in [sources_a, sources_b]:
            for src in src_list:
                pmra = src.get("pmra", 0)
                pmdec = src.get("pmdec", 0)
                ref_epoch = src.get("ref_epoch", GAIA_REF_EPOCH)
                if pmra != 0 or pmdec != 0:
                    new_ra, new_dec = propagate_proper_motion(
                        src["ra"], src["dec"], pmra, pmdec,
                        ref_epoch, target_epoch,
                    )
                    src["ra"] = new_ra
                    src["dec"] = new_dec
                    n_propagated += 1
        if n_propagated > 0:
            logger.info(
                "Propagated proper motion for %d sources to epoch %.1f",
                n_propagated, target_epoch,
            )

    coord_a = _build_skycoord(sources_a)
    coord_b = _build_skycoord(sources_b)

    # match_to_catalog_sky returns (idx, sep2d, sep3d)
    idx_b, sep2d, _ = coord_a.match_to_catalog_sky(coord_b)

    max_sep = max_sep_arcsec * u.arcsec
    matches: List[MatchResult] = []

    for i_a, (i_b, sep) in enumerate(zip(idx_b, sep2d)):
        if sep <= max_sep:
            matches.append(
                MatchResult(
                    source_a=sources_a[i_a],
                    source_b=sources_b[int(i_b)],
                    separation_arcsec=sep.arcsec,
                )
            )

    matches.sort(key=lambda m: m.separation_arcsec)
    logger.info(
        "Cross-match complete: %d matches found within %.2f\"",
        len(matches),
        max_sep_arcsec,
    )
    return matches


def find_in_all_catalogs(
    ra: float,
    dec: float,
    catalogs_dict: Dict[str, CatalogInput],
    max_sep_arcsec: Optional[float] = None,
) -> Dict[str, CatalogMatchSummary]:
    """Look up a single sky position in every catalog provided.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in degrees (ICRS).
    catalogs_dict : dict[str, CatalogInput]
        Mapping of catalog name -> catalog data in any supported format.
    max_sep_arcsec : float, optional
        Maximum allowed separation for a match.

    Returns
    -------
    dict[str, CatalogMatchSummary]
        One entry per catalog, indicating whether a match was found and
        the closest source with its separation.
    """
    if max_sep_arcsec is None:
        max_sep_arcsec = _configured_radius()

    target = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
    results: Dict[str, CatalogMatchSummary] = {}

    logger.info(
        "Looking up target (RA=%.6f, Dec=%.6f) in %d catalogs",
        ra,
        dec,
        len(catalogs_dict),
    )

    for cat_name, cat_data in catalogs_dict.items():
        try:
            sources = _normalize_catalog(cat_data)
            if not sources:
                logger.debug("Catalog '%s' is empty, skipping.", cat_name)
                results[cat_name] = CatalogMatchSummary(
                    catalog_name=cat_name, matched=False
                )
                continue

            coord_cat = _build_skycoord(sources)
            idx, sep2d, _ = target.match_to_catalog_sky(coord_cat)

            # match_to_catalog_sky may return 0-d numpy arrays (idx)
            # and 1-element Angle arrays (sep2d) even for a scalar
            # target.  Use .item() / indexing to extract scalars safely.
            sep_val = sep2d.arcsec
            sep_arcsec = float(
                sep_val.item() if hasattr(sep_val, "item") and sep_val.ndim == 0
                else sep_val[0] if hasattr(sep_val, "__getitem__") and np.ndim(sep_val) > 0
                else sep_val
            )
            idx = int(
                idx.item() if hasattr(idx, "item") else idx
            )

            if sep_arcsec <= max_sep_arcsec:
                results[cat_name] = CatalogMatchSummary(
                    catalog_name=cat_name,
                    matched=True,
                    separation_arcsec=sep_arcsec,
                    closest_source=sources[idx],
                )
                logger.debug(
                    "  [%s] MATCH at %.4f\" — %s",
                    cat_name,
                    sep_arcsec,
                    sources[idx],
                )
            else:
                results[cat_name] = CatalogMatchSummary(
                    catalog_name=cat_name,
                    matched=False,
                    separation_arcsec=sep_arcsec,
                    closest_source=sources[idx],
                )
                logger.debug(
                    "  [%s] no match (closest=%.4f\", threshold=%.2f\")",
                    cat_name,
                    sep_arcsec,
                    max_sep_arcsec,
                )
        except Exception as exc:
            logger.error("Error searching catalog '%s': %s", cat_name, exc)
            results[cat_name] = CatalogMatchSummary(
                catalog_name=cat_name, matched=False
            )

    matched_count = sum(1 for v in results.values() if v.matched)
    logger.info(
        "Target lookup complete: matched in %d / %d catalogs",
        matched_count,
        len(catalogs_dict),
    )
    return results


def batch_crossmatch(
    primary_catalog: CatalogInput,
    secondary_catalogs: Dict[str, CatalogInput],
    max_sep_arcsec: Optional[float] = None,
) -> Dict[str, BatchMatchResult]:
    """Cross-match a primary catalog against multiple secondary catalogs.

    Parameters
    ----------
    primary_catalog : CatalogInput
        The reference catalog.
    secondary_catalogs : dict[str, CatalogInput]
        Named secondary catalogs to match against.
    max_sep_arcsec : float, optional
        Maximum separation for a valid match.

    Returns
    -------
    dict[str, BatchMatchResult]
        Results keyed by secondary catalog name.
    """
    if max_sep_arcsec is None:
        max_sep_arcsec = _configured_radius()

    logger.info(
        "Batch cross-match: primary (%d sources) vs %d secondary catalogs",
        len(primary_catalog) if hasattr(primary_catalog, "__len__") else -1,
        len(secondary_catalogs),
    )

    results: Dict[str, BatchMatchResult] = {}

    for name, sec_catalog in secondary_catalogs.items():
        try:
            matches = crossmatch(
                primary_catalog, sec_catalog, max_sep_arcsec=max_sep_arcsec
            )
            results[name] = BatchMatchResult(secondary_name=name, matches=matches)
            logger.info(
                "  [%s] %d matches found",
                name,
                len(matches),
            )
        except Exception as exc:
            logger.error("Error in batch cross-match for '%s': %s", name, exc)
            results[name] = BatchMatchResult(secondary_name=name, matches=[])

    total = sum(r.count for r in results.values())
    logger.info("Batch cross-match complete: %d total matches across all catalogs", total)
    return results


# ── CLI demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Project EXODUS — Sky Cross-Match Demo")
    print("=" * 60)

    # Two small synthetic catalogs with some overlapping sources.
    # Sources near RA~180, Dec~+45 (arbitrary patch of sky).
    catalog_alpha = [
        {"ra": 180.0000, "dec": 45.0000, "name": "alpha-1"},
        {"ra": 180.0010, "dec": 45.0010, "name": "alpha-2"},
        {"ra": 181.0000, "dec": 44.0000, "name": "alpha-3"},
        {"ra": 182.5000, "dec": 43.5000, "name": "alpha-4"},
        {"ra": 179.9990, "dec": 45.0020, "name": "alpha-5"},
    ]

    # catalog_beta has a couple of sources very close to alpha entries
    # and some unrelated ones further away.
    catalog_beta = [
        {"ra": 180.0001, "dec": 45.0001, "name": "beta-1"},   # ~0.5" from alpha-1
        {"ra": 180.0011, "dec": 45.0009, "name": "beta-2"},   # ~0.5" from alpha-2
        {"ra": 185.0000, "dec": 40.0000, "name": "beta-3"},   # no match
        {"ra": 179.9991, "dec": 45.0019, "name": "beta-4"},   # ~0.5" from alpha-5
        {"ra": 190.0000, "dec": 30.0000, "name": "beta-5"},   # no match
    ]

    # ---- Cross-match ------------------------------------------------
    print("\n--- Cross-matching catalog_alpha vs catalog_beta ---")
    matches = crossmatch(catalog_alpha, catalog_beta, max_sep_arcsec=3.0)
    if matches:
        for m in matches:
            a_name = m.source_a.get("name", "?")
            b_name = m.source_b.get("name", "?")
            print(f"  {a_name} <-> {b_name}  sep={m.separation_arcsec:.4f}\"")
    else:
        print("  No matches found.")

    print(f"\nTotal matches: {len(matches)}")

    # ---- Find in all catalogs ---------------------------------------
    print("\n--- Looking up (RA=180.0001, Dec=45.0001) in all catalogs ---")
    all_cats = {
        "alpha": catalog_alpha,
        "beta": catalog_beta,
    }
    overview = find_in_all_catalogs(180.0001, 45.0001, all_cats, max_sep_arcsec=3.0)
    for cat_name, summary in overview.items():
        status = "MATCH" if summary.matched else "no match"
        sep_str = f"{summary.separation_arcsec:.4f}\"" if summary.separation_arcsec is not None else "N/A"
        src = summary.closest_source.get("name", "?") if summary.closest_source else "N/A"
        print(f"  [{cat_name}] {status}  sep={sep_str}  closest={src}")

    # ---- Batch cross-match ------------------------------------------
    print("\n--- Batch cross-match: alpha vs {beta} ---")
    batch_results = batch_crossmatch(
        catalog_alpha,
        {"beta": catalog_beta},
        max_sep_arcsec=3.0,
    )
    for name, br in batch_results.items():
        print(f"  [{name}] {br.count} matches")
        for m in br.matches:
            a_name = m.source_a.get("name", "?")
            b_name = m.source_b.get("name", "?")
            print(f"    {a_name} <-> {b_name}  sep={m.separation_arcsec:.4f}\"")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)
