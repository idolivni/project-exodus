"""
Smart Target Selection for Project EXODUS.

Generates prioritized target lists from multiple discovery channels,
not just the NASA Exoplanet Archive.  This dramatically expands the
discovery space to include anomalous stars with no known planets.

Target Sources (4 tiers)
------------------------
* **Tier A** -- Multi-signal coincidences (highest value):
    Stars anomalous in 2+ independent catalogs.  E.g., Gaia RUWE > 2.0
    AND AllWISE IR excess within 50 pc, or stars inside Fermi unID error
    ellipses that also show astrometric anomalies.

* **Tier B** -- Single-catalog anomalies within 50 pc:
    Gaia astrometric outliers (RUWE > 2.0), AllWISE color excess,
    TESS TOI habitable-zone candidates not yet in the confirmed list.

* **Tier C** -- Positional coincidences with multi-messenger sources:
    Stars near unidentified Fermi sources, IceCube spatial hotspots,
    FRB repeater neighborhoods.

* **Tier D** -- Evolver-driven adaptive:
    Bias selection toward stellar types that yielded "unexplained"
    results in previous EXODUS runs.

Public API
----------
    st = SmartTargeter(max_targets=500, max_distance_pc=50)
    targets = st.generate()
    st.save("data/targets/smart_targets.json")
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, get_config, PROJECT_ROOT

log = get_logger("targeting.smart_targeter")


def _gaia_launch_with_retry(adql: str, max_retries: int = 3):
    """Launch a Gaia TAP job with exponential backoff on transient errors."""
    from astroquery.gaia import Gaia

    for attempt in range(max_retries):
        try:
            job = Gaia.launch_job(adql)
            return job.get_results()
        except Exception as exc:
            exc_str = str(exc).lower()
            is_transient = any(tok in exc_str for tok in (
                "408", "500", "timeout", "timed out", "connection",
                "reset", "aborted", "unknown table",
            ))
            if is_transient and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(
                    "Gaia attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                raise


# =====================================================================
#  SmartTargeter
# =====================================================================

class SmartTargeter:
    """Generate prioritized target lists from multiple discovery channels.

    Outputs standard target JSON consumable by target_loader.py and the
    existing runner infrastructure.

    Parameters
    ----------
    max_targets : int
        Maximum targets to return (default 500).
    max_distance_pc : float
        Maximum distance in parsecs (default 50).  Nearby targets have
        better data quality and more discovery channels.
    min_ruwe : float
        Minimum RUWE for Gaia astrometric anomaly flagging (default 2.0).
    max_ruwe : float or None
        Maximum RUWE (default None = no upper limit). Set to e.g. 3.0
        to select moderately anomalous stars (1.4-3.0) that are less
        likely to be binaries but still show astrometric anomalies.
    """

    def __init__(
        self,
        max_targets: int = 500,
        max_distance_pc: float = 50.0,
        min_ruwe: float = 2.0,
        max_ruwe: Optional[float] = None,
    ):
        self.max_targets = max_targets
        self.max_distance_pc = max_distance_pc
        self.min_ruwe = min_ruwe
        self.max_ruwe = max_ruwe
        self._cfg = get_config()

        # Collect all sources keyed by a canonical position key
        self._sources: Dict[str, Dict[str, Any]] = {}  # target_id -> target dict
        self._source_channels: Dict[str, Set[str]] = defaultdict(set)

    # =================================================================
    #  Main entry point
    # =================================================================

    def generate(self) -> List[Dict[str, Any]]:
        """Generate a prioritized target list.

        Returns list of target dicts compatible with target_loader.py.
        """
        ruwe_range = f"RUWE>{self.min_ruwe:.1f}"
        if self.max_ruwe is not None:
            ruwe_range = f"RUWE {self.min_ruwe:.1f}-{self.max_ruwe:.1f}"
        log.info(
            "SmartTargeter: generating targets (max=%d, d<%d pc, %s)",
            self.max_targets, self.max_distance_pc, ruwe_range,
        )

        # Phase 1: Scan individual channels
        gaia_targets = self._scan_gaia_anomalies()
        fermi_targets = self._scan_fermi_neighborhoods()
        icecube_targets = self._scan_icecube_hotspots()

        # Phase 2: Merge all sources and identify multi-channel coincidences
        all_sources = gaia_targets + fermi_targets + icecube_targets
        self._merge_sources(all_sources)

        # Phase 3: Assign tiers
        targets = self._assign_tiers()

        # Phase 4: Apply evolver weights (if available)
        targets = self._apply_evolver_weights(targets)

        # Phase 5: Sort and truncate
        targets = self._rank_and_truncate(targets)

        log.info(
            "SmartTargeter: generated %d targets "
            "(Tier A: %d, Tier B: %d, Tier C: %d, Tier D: %d)",
            len(targets),
            sum(1 for t in targets if t.get("source_tier") == "A"),
            sum(1 for t in targets if t.get("source_tier") == "B"),
            sum(1 for t in targets if t.get("source_tier") == "C"),
            sum(1 for t in targets if t.get("source_tier") == "D"),
        )

        return targets

    def save(self, path: str) -> str:
        """Save generated targets as a campaign JSON file.

        Parameters
        ----------
        path : str
            Output file path (relative to project root or absolute).

        Returns
        -------
        str
            Absolute path of the written file.
        """
        targets = self.generate()

        campaign = {
            "campaign": "smart_targets",
            "phase": "smart_discovery",
            "description": (
                f"SmartTargeter: {len(targets)} targets from multi-channel "
                f"anomaly scan (d<{self.max_distance_pc}pc, "
                f"RUWE>{self.min_ruwe}). "
                f"Sources: Gaia astrometric anomalies, Fermi unID neighborhoods, "
                f"IceCube hotspots, Evolver-driven adaptive."
            ),
            "targets": targets,
        }

        out_path = Path(path)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(campaign, f, indent=2, default=str)

        log.info("SmartTargeter: saved %d targets to %s", len(targets), out_path)
        return str(out_path)

    # =================================================================
    #  Channel scanners
    # =================================================================

    def _scan_gaia_anomalies(self) -> List[Dict[str, Any]]:
        """Query Gaia DR3 for nearby stars with anomalous astrometry.

        Selects stars within max_distance_pc that have:
        - RUWE > min_ruwe (astrometric solution doesn't fit single star)

        Uses chunked declination queries to avoid Gaia TAP timeout on
        full-table scans.  Each chunk queries a 30-degree declination
        strip.  Results are cached per chunk so failed chunks can be
        retried without re-querying successful ones.
        """
        from src.utils import load_cache, save_cache

        targets = []
        min_parallax = 1000.0 / self.max_distance_pc  # mas

        # Declination chunks — 8 strips
        dec_chunks = [
            (-90, -60), (-60, -30), (-30, -15), (-15, 0),
            (0, 15), (15, 30), (30, 60), (60, 90),
        ]

        # Load cached chunk results
        max_ruwe_tag = f"_max{self.max_ruwe}" if self.max_ruwe else ""
        cache_key = f"smart_gaia_ruwe{self.min_ruwe}{max_ruwe_tag}_d{int(self.max_distance_pc)}"
        cached_chunks = load_cache(cache_key, subfolder="targeting") or {}

        try:
            log.info(
                "Querying Gaia DR3 for anomalous nearby stars "
                "(%d dec chunks, %d cached) ...",
                len(dec_chunks),
                sum(1 for k in cached_chunks if cached_chunks[k]),
            )

            seen_ids: set = set()

            for chunk_i, (dec_lo, dec_hi) in enumerate(dec_chunks):
                chunk_key = f"{dec_lo}_{dec_hi}"

                # Restore from cache if available
                if chunk_key in cached_chunks and cached_chunks[chunk_key]:
                    chunk_targets = cached_chunks[chunk_key]
                    for ct in chunk_targets:
                        sid = ct["target_id"].replace("GAIA_", "")
                        if sid not in seen_ids:
                            seen_ids.add(sid)
                            targets.append(ct)
                    log.info(
                        "Gaia chunk %d/%d (dec %d..%d): %d stars (cached)",
                        chunk_i + 1, len(dec_chunks), dec_lo, dec_hi,
                        len(chunk_targets),
                    )
                    continue

                ruwe_clause = f"AND ruwe > {self.min_ruwe}"
                if self.max_ruwe is not None:
                    ruwe_clause += f" AND ruwe < {self.max_ruwe}"

                adql = f"""
                SELECT TOP 1000
                    source_id, ra, dec, parallax, pmra, pmdec,
                    phot_g_mean_mag, bp_rp, ruwe,
                    astrometric_excess_noise_sig
                FROM gaiadr3.gaia_source
                WHERE parallax > {min_parallax}
                  AND parallax_over_error > 10
                  {ruwe_clause}
                  AND phot_g_mean_mag < 15
                  AND dec >= {dec_lo} AND dec < {dec_hi}
                ORDER BY ruwe DESC
                """

                try:
                    table = _gaia_launch_with_retry(adql)

                    if table is None or len(table) == 0:
                        cached_chunks[chunk_key] = []
                        log.debug(
                            "Gaia chunk %d/%d (dec %d..%d): 0 results",
                            chunk_i + 1, len(dec_chunks), dec_lo, dec_hi,
                        )
                        continue

                    chunk_targets = []
                    for row in table:
                        source_id = str(row["source_id"])
                        if source_id in seen_ids:
                            continue
                        seen_ids.add(source_id)

                        parallax = float(row["parallax"])
                        distance_pc = 1000.0 / parallax if parallax > 0 else 0

                        target = {
                            "target_id": f"GAIA_{source_id}",
                            "host_star": f"Gaia DR3 {source_id}",
                            "ra": float(row["ra"]),
                            "dec": float(row["dec"]),
                            "distance_pc": distance_pc,
                            "hz_flag": False,
                            "phot_g_mean_mag": float(row["phot_g_mean_mag"]),
                            "bp_rp": (
                                float(row["bp_rp"])
                                if row["bp_rp"] is not None
                                   and not np.ma.is_masked(row["bp_rp"])
                                else None
                            ),
                            "ruwe": float(row["ruwe"]),
                            "astrometric_excess_noise_sig": (
                                float(row["astrometric_excess_noise_sig"])
                                if row["astrometric_excess_noise_sig"] is not None
                                   and not np.ma.is_masked(
                                       row["astrometric_excess_noise_sig"]
                                   )
                                else 0
                            ),
                            "discovery_channel": "gaia_astrometry",
                            "discovery_reason": (
                                f"RUWE={float(row['ruwe']):.2f}, "
                                f"d={distance_pc:.1f}pc"
                            ),
                        }
                        chunk_targets.append(target)
                        targets.append(target)

                    cached_chunks[chunk_key] = chunk_targets
                    save_cache(cache_key, cached_chunks, subfolder="targeting")

                    log.info(
                        "Gaia chunk %d/%d (dec %d..%d): %d stars",
                        chunk_i + 1, len(dec_chunks), dec_lo, dec_hi,
                        len(chunk_targets),
                    )

                except Exception as exc:
                    log.warning(
                        "Gaia chunk %d/%d (dec %d..%d) failed: %s",
                        chunk_i + 1, len(dec_chunks), dec_lo, dec_hi, exc,
                    )
                    continue

            # Final cache save
            save_cache(cache_key, cached_chunks, subfolder="targeting")

            log.info(
                "Gaia anomaly scan: %d stars with RUWE > %.1f within %d pc",
                len(targets), self.min_ruwe, self.max_distance_pc,
            )

        except Exception as exc:
            log.warning("Gaia anomaly scan failed: %s", exc)

        return targets

    def _scan_fermi_neighborhoods(self) -> List[Dict[str, Any]]:
        """Find nearby Gaia stars within Fermi unidentified source error ellipses.

        Reverse crossmatch: for each unidentified 4FGL source, find the
        nearest Gaia star within 50 pc inside the 95% error ellipse.

        Uses batched ADQL with server-side CONTAINS() to avoid 408 timeouts
        from individual cone_search calls.  Each batch sends ~20 Fermi
        source positions in a single query with OR-ed CIRCLE conditions,
        reducing 2,563 queries to ~130 batched queries.

        Results are cached per Fermi source name so re-runs fast-forward
        past already-searched sources.
        """
        targets = []

        try:
            from src.ingestion.fermi_catalog import get_unidentified
            from src.utils import load_cache, save_cache

            unid_sources = get_unidentified()
            if not unid_sources:
                log.warning("No unidentified Fermi sources available")
                return []

            # Load cached results (per-source cache)
            cache_key = f"smart_fermi_d{int(self.max_distance_pc)}"
            cached_results = load_cache(cache_key, subfolder="targeting") or {}

            # Filter sources: skip large error ellipses, skip cached
            to_search = []
            for src in unid_sources:
                search_radius = src.pos_err_semimajor_deg or 0.1
                search_radius = max(search_radius, 0.05)
                if search_radius > 0.5:
                    continue
                if src.source_name in cached_results:
                    # Restore cached target if it had a match
                    hit = cached_results[src.source_name]
                    if hit and hit.get("target_id"):
                        targets.append(hit)
                    continue
                to_search.append((src, search_radius))

            log.info(
                "Fermi scan: %d cached, %d to search (of %d unID sources)",
                len(unid_sources) - len(to_search), len(to_search),
                len(unid_sources),
            )

            if not to_search:
                return targets

            min_parallax = 1000.0 / self.max_distance_pc
            n_searched = 0
            n_with_star = len(targets)  # count cached hits

            # --- Batched ADQL: group Fermi sources into batches of 20 ---
            # Each batch sends one ADQL query with multiple CIRCLE conditions
            # joined by OR, with server-side parallax filter.
            batch_size = 20
            for batch_start in range(0, len(to_search), batch_size):
                batch = to_search[batch_start:batch_start + batch_size]

                # Build server-side ADQL with OR-ed CONTAINS conditions
                circles = []
                for src, radius_deg in batch:
                    circles.append(
                        f"CONTAINS(POINT('ICRS', ra, dec), "
                        f"CIRCLE('ICRS', {src.ra:.6f}, {src.dec:.6f}, "
                        f"{radius_deg:.4f})) = 1"
                    )

                where_clause = " OR ".join(circles)
                adql = f"""
                SELECT source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp
                FROM gaiadr3.gaia_source
                WHERE parallax > {min_parallax}
                  AND parallax_over_error > 5
                  AND phot_g_mean_mag < 15
                  AND ({where_clause})
                """

                try:
                    result_table = _gaia_launch_with_retry(adql)
                except Exception as exc:
                    log.warning(
                        "Fermi batch %d-%d ADQL failed: %s",
                        batch_start, batch_start + len(batch), exc,
                    )
                    # Fall back to marking these as searched (no match)
                    for src, _ in batch:
                        cached_results[src.source_name] = {}
                    n_searched += len(batch)
                    continue

                n_searched += len(batch)

                # Match results back to Fermi sources
                if result_table is not None and len(result_table) > 0:
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u

                    gaia_coords = SkyCoord(
                        ra=result_table["ra"],
                        dec=result_table["dec"],
                        unit="deg",
                    )

                    for src, radius_deg in batch:
                        fermi_coord = SkyCoord(
                            ra=src.ra * u.deg, dec=src.dec * u.deg,
                        )
                        seps = fermi_coord.separation(gaia_coords).deg
                        in_ellipse = seps < radius_deg

                        if np.any(in_ellipse):
                            # Among matches, take the brightest
                            match_idx = np.where(in_ellipse)[0]
                            mags = np.array([
                                float(result_table["phot_g_mean_mag"][i])
                                for i in match_idx
                            ])
                            best_i = match_idx[np.argmin(mags)]
                            row = result_table[best_i]

                            sid = str(row["source_id"])
                            parallax = float(row["parallax"])
                            distance_pc = 1000.0 / parallax

                            bp_rp_val = None
                            if (row["bp_rp"] is not None
                                    and not np.ma.is_masked(row["bp_rp"])):
                                bp_rp_val = float(row["bp_rp"])

                            target = {
                                "target_id": f"FERMI_{src.source_name}_{sid}",
                                "host_star": f"Gaia DR3 {sid}",
                                "ra": float(row["ra"]),
                                "dec": float(row["dec"]),
                                "distance_pc": distance_pc,
                                "hz_flag": False,
                                "phot_g_mean_mag": float(row["phot_g_mean_mag"]),
                                "bp_rp": bp_rp_val,
                                "fermi_source": src.source_name,
                                "fermi_class": src.source_class or "unid",
                                "discovery_channel": "fermi_neighborhood",
                                "discovery_reason": (
                                    f"Near unID Fermi {src.source_name}, "
                                    f"d={distance_pc:.1f}pc, "
                                    f"err={radius_deg:.2f}deg"
                                ),
                            }
                            targets.append(target)
                            cached_results[src.source_name] = target
                            n_with_star += 1
                        else:
                            cached_results[src.source_name] = {}
                else:
                    for src, _ in batch:
                        cached_results[src.source_name] = {}

                if n_searched % 200 == 0 or n_searched == len(to_search):
                    log.info(
                        "Fermi scan progress: %d/%d searched, "
                        "%d nearby stars found ...",
                        n_searched, len(to_search), n_with_star,
                    )

                # Save cache periodically
                if n_searched % 200 == 0:
                    save_cache(cache_key, cached_results, subfolder="targeting")

                time.sleep(0.5)  # gentle rate limit between batches

            # Final cache save
            save_cache(cache_key, cached_results, subfolder="targeting")

            log.info(
                "Fermi neighborhood scan complete: %d unID sources searched, "
                "%d with nearby star within %d pc",
                n_searched, n_with_star, self.max_distance_pc,
            )

        except Exception as exc:
            log.warning("Fermi neighborhood scan failed: %s", exc)

        return targets

    def _scan_icecube_hotspots(self) -> List[Dict[str, Any]]:
        """Find spatial hotspots in the IceCube neutrino sky map.

        Clusters high-energy neutrino events and identifies sky regions
        with excess counts above the isotropic background expectation.
        Then queries Gaia for nearby stars in those regions.

        Uses 2-degree sky bins with cos(dec) solid-angle normalization
        to correct for varying bin area across declination.  Identifies
        bins with >3-sigma density excess above the isotropic expectation.
        """
        targets = []

        try:
            from src.ingestion.icecube_catalog import get_high_energy

            # Get high-energy events (> 100 TeV — more events for statistics,
            # median angular resolution ~0.5 deg at >100 TeV)
            events = get_high_energy(min_gev=100_000)
            if not events or events[0].source == "simulated":
                log.info(
                    "IceCube: %s events available (%s source), "
                    "skipping hotspot scan",
                    len(events) if events else 0,
                    events[0].source if events else "none",
                )
                return []

            log.info(
                "Scanning %d IceCube events (>100 TeV) for spatial hotspots ...",
                len(events),
            )

            # 2-degree bins for better angular resolution
            bin_size = 2.0  # degrees
            ra_bins = np.arange(0, 360, bin_size)
            dec_bins = np.arange(-90, 90, bin_size)

            # Count events per bin (vectorized — avoids O(n_bins*n_events) loop)
            event_ra = np.array([e.ra for e in events])
            event_dec = np.array([e.dec for e in events])

            ra_edges = np.arange(0, 360 + bin_size, bin_size)
            dec_edges = np.arange(-90, 90 + bin_size, bin_size)
            counts, _, _ = np.histogram2d(
                event_ra, event_dec,
                bins=[ra_edges, dec_edges],
            )
            counts = counts.astype(int)

            # --- Solid-angle normalization ---
            # Bins near the equator cover more sky than bins at high |dec|.
            # Solid angle of a bin ∝ cos(dec_center) * Δra * Δdec.
            # Normalize counts by relative solid angle so high-|dec| bins
            # aren't artificially suppressed.
            dec_centers = dec_bins + bin_size / 2.0
            cos_dec = np.cos(np.radians(dec_centers))
            cos_dec = np.clip(cos_dec, 0.01, None)  # avoid div-by-zero at poles

            # Density = counts / relative_solid_angle
            # relative_solid_angle ∝ cos(dec)
            density = np.zeros_like(counts, dtype=float)
            for j in range(len(dec_bins)):
                density[:, j] = counts[:, j] / cos_dec[j]

            # Find bins with > 3 sigma excess in density
            nonzero_density = density[density > 0]
            if len(nonzero_density) < 10:
                log.info("IceCube: too few occupied bins for hotspot analysis")
                return []

            mean_density = np.mean(nonzero_density)
            std_density = np.std(nonzero_density)

            if std_density <= 0:
                log.info("IceCube: no significant spatial structure found")
                return []

            sigma_threshold = 3.0
            hotspot_threshold = mean_density + sigma_threshold * std_density
            hotspot_bins = []
            for i in range(len(ra_bins)):
                for j in range(len(dec_bins)):
                    if density[i, j] > hotspot_threshold:
                        sigma = (density[i, j] - mean_density) / std_density
                        hotspot_bins.append((
                            ra_bins[i] + bin_size / 2,
                            dec_bins[j] + bin_size / 2,
                            int(counts[i, j]),
                            float(sigma),
                        ))

            # Sort hotspot bins by sigma (most significant first)
            hotspot_bins.sort(key=lambda x: -x[3])

            log.info(
                "IceCube: %d hotspot bins above %.1f-sigma "
                "(density threshold=%.1f, mean=%.1f, std=%.1f)",
                len(hotspot_bins), sigma_threshold,
                hotspot_threshold, mean_density, std_density,
            )
            for hb in hotspot_bins[:5]:
                log.info(
                    "  Hotspot: RA=%.1f Dec=%.1f, %d events, %.1f sigma",
                    hb[0], hb[1], hb[2], hb[3],
                )

            # For each hotspot, query Gaia for nearby bright stars
            if hotspot_bins:
                try:
                    from src.ingestion.gaia_query import cone_search
                    min_parallax = 1000.0 / self.max_distance_pc

                    # Search radius = bin_size / 2 in arcsec (1 deg radius
                    # for 2-deg bins)
                    search_radius_arcsec = bin_size * 3600 / 2

                    for ra_center, dec_center, n_events, sigma in hotspot_bins[:30]:
                        try:
                            result = cone_search(
                                ra_center, dec_center,
                                radius_arcsec=search_radius_arcsec,
                                top_n=10,
                            )
                            if result is not None and len(result) > 0:
                                nearby = result[result["parallax"] > min_parallax]
                                if len(nearby) > 0:
                                    best = nearby.sort_values(
                                        "phot_g_mean_mag"
                                    ).iloc[0]
                                    sid = str(best["source_id"])
                                    distance_pc = 1000.0 / float(best["parallax"])

                                    targets.append({
                                        "target_id": f"ICECUBE_HOT_{sid}",
                                        "host_star": f"Gaia DR3 {sid}",
                                        "ra": float(best["ra"]),
                                        "dec": float(best["dec"]),
                                        "distance_pc": distance_pc,
                                        "hz_flag": False,
                                        "phot_g_mean_mag": float(
                                            best["phot_g_mean_mag"]
                                        ),
                                        "icecube_events": n_events,
                                        "icecube_sigma": sigma,
                                        "discovery_channel": "icecube_hotspot",
                                        "discovery_reason": (
                                            f"IceCube hotspot ({n_events} events "
                                            f">100 TeV, {sigma:.1f}σ), "
                                            f"d={distance_pc:.1f}pc"
                                        ),
                                    })
                        except Exception:
                            continue

                except ImportError:
                    log.warning("Cannot query Gaia for IceCube hotspot stars")

        except Exception as exc:
            log.warning("IceCube hotspot scan failed: %s", exc)

        return targets

    # =================================================================
    #  Merging and tier assignment
    # =================================================================

    def _merge_sources(self, all_sources: List[Dict[str, Any]]) -> None:
        """Merge sources from different channels, deduplicating by position.

        Sources within 5 arcsec of each other are considered the same target.
        When merging, the discovery channels are accumulated.
        """
        self._sources.clear()
        self._source_channels.clear()

        for src in all_sources:
            tid = src["target_id"]
            channel = src.get("discovery_channel", "unknown")

            # Check for positional duplicates
            merged = False
            for existing_id, existing in self._sources.items():
                if self._angular_sep(
                    src["ra"], src["dec"],
                    existing["ra"], existing["dec"],
                ) < 5.0 / 3600.0:  # 5 arcsec
                    # Merge channels
                    self._source_channels[existing_id].add(channel)
                    # Keep the entry with more metadata
                    if src.get("distance_pc", 999) < existing.get("distance_pc", 999):
                        self._sources[existing_id].update(src)
                        self._sources[existing_id]["target_id"] = existing_id
                    merged = True
                    break

            if not merged:
                self._sources[tid] = src
                self._source_channels[tid].add(channel)

        log.info(
            "Merged %d raw sources → %d unique targets",
            len(all_sources), len(self._sources),
        )

    def _assign_tiers(self) -> List[Dict[str, Any]]:
        """Assign quality tiers based on number of discovery channels.

        Tier A: 2+ independent channels (multi-signal coincidence)
        Tier B: 1 channel, within max_distance_pc
        Tier C: Positional coincidence only (fermi/icecube neighborhood)
        """
        targets = []
        for tid, src in self._sources.items():
            channels = self._source_channels[tid]
            n_channels = len(channels)

            if n_channels >= 2:
                tier = "A"
            elif "gaia_astrometry" in channels:
                tier = "B"
            else:
                tier = "C"

            src["source_tier"] = tier
            src["n_discovery_channels"] = n_channels
            src["discovery_channels"] = sorted(channels)
            targets.append(src)

        return targets

    def _apply_evolver_weights(
        self, targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-weight targets based on Evolver strategy preferences.

        Targets matching promoted strategies get a priority boost;
        targets matching deprioritized strategies get demoted.
        """
        try:
            from src.engines.evolver import EvolverEngine
            evolver = EvolverEngine()
            weights = evolver.get_strategy_weights()

            if not weights:
                return targets

            log.info("Applying Evolver weights: %s", weights)

            for t in targets:
                boost = 1.0
                channel = t.get("discovery_channel", "")

                # Map discovery channels to scoring strategy names
                if "gaia" in channel:
                    w = weights.get("proper_motion_anomaly", 1.0)
                    boost *= w
                if "fermi" in channel:
                    w = weights.get("gamma_crossmatch", 1.0)
                    boost *= w
                if "ir" in channel:
                    w = weights.get("ir_excess", 1.0)
                    boost *= w

                t["evolver_boost"] = boost
                if boost != 1.0:
                    t["source_tier"] = "D"  # Evolver-modified

        except Exception as exc:
            log.debug("Evolver integration skipped: %s", exc)

        return targets

    def _rank_and_truncate(
        self, targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort targets by tier priority and truncate to max_targets.

        Priority order:
        1. Tier A (multi-signal) — highest
        2. Tier D (Evolver-boosted)
        3. Tier B (single-catalog anomaly)
        4. Tier C (positional coincidence)

        Within each tier: sort by distance (nearest first) then by
        number of discovery channels.
        """
        tier_priority = {"A": 0, "D": 1, "B": 2, "C": 3}

        targets.sort(key=lambda t: (
            tier_priority.get(t.get("source_tier", "C"), 3),
            -t.get("n_discovery_channels", 1),
            t.get("distance_pc", 999),
        ))

        return targets[:self.max_targets]

    # =================================================================
    #  Utility
    # =================================================================

    @staticmethod
    def _angular_sep(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        """Approximate angular separation in degrees."""
        cos_dec = np.cos(np.radians(0.5 * (dec1 + dec2)))
        dra = (ra1 - ra2) * cos_dec
        ddec = dec1 - dec2
        return float(np.sqrt(dra**2 + ddec**2))


# =====================================================================
#  Module-level convenience
# =====================================================================

def generate_smart_targets(
    max_targets: int = 500,
    max_distance_pc: float = 50.0,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """One-call convenience function for generating smart targets.

    Parameters
    ----------
    max_targets : int
        Maximum targets to generate.
    max_distance_pc : float
        Maximum distance in parsecs.
    output_path : str, optional
        If provided, save campaign JSON to this path.

    Returns
    -------
    list of dict
        Target dictionaries compatible with target_loader.py.
    """
    st = SmartTargeter(
        max_targets=max_targets,
        max_distance_pc=max_distance_pc,
    )

    if output_path:
        st.save(output_path)
        return st.generate()
    else:
        return st.generate()


# =====================================================================
#  CLI entry point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EXODUS Smart Target Selection"
    )
    parser.add_argument(
        "--max-targets", type=int, default=500,
        help="Maximum number of targets (default: 500)",
    )
    parser.add_argument(
        "--max-distance", type=float, default=50.0,
        help="Maximum distance in parsecs (default: 50)",
    )
    parser.add_argument(
        "--min-ruwe", type=float, default=2.0,
        help="Minimum RUWE for Gaia anomalies (default: 2.0)",
    )
    parser.add_argument(
        "--output", default="data/targets/smart_targets.json",
        help="Output campaign JSON path",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate but don't save",
    )
    args = parser.parse_args()

    st = SmartTargeter(
        max_targets=args.max_targets,
        max_distance_pc=args.max_distance,
        min_ruwe=args.min_ruwe,
    )

    if args.dry_run:
        targets = st.generate()
    else:
        path = st.save(args.output)
        targets = st.generate()
        print(f"\nSaved to: {path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  EXODUS Smart Targeter — {len(targets)} targets")
    print(f"{'='*70}")

    tier_counts = defaultdict(int)
    channel_counts = defaultdict(int)
    for t in targets:
        tier_counts[t.get("source_tier", "?")] += 1
        for ch in t.get("discovery_channels", []):
            channel_counts[ch] += 1

    print(f"\nTier breakdown:")
    for tier in sorted(tier_counts):
        print(f"  Tier {tier}: {tier_counts[tier]} targets")

    print(f"\nDiscovery channels:")
    for ch, count in sorted(channel_counts.items(), key=lambda x: -x[1]):
        print(f"  {ch}: {count} targets")

    print(f"\nTop 20 targets:")
    print(f"  {'Rank':<5} {'ID':<35} {'Tier':<5} {'Dist':>6} {'Channels':>4} {'Reason'}")
    print(f"  {'-'*85}")
    for i, t in enumerate(targets[:20], 1):
        print(
            f"  {i:<5} {t['target_id'][:34]:<35} {t.get('source_tier','?'):<5} "
            f"{t.get('distance_pc',0):>6.1f} "
            f"{t.get('n_discovery_channels',1):>4} "
            f"{t.get('discovery_reason','')[:40]}"
        )
