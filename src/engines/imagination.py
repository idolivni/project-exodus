"""
Imagination Engine -- Non-Anthropocentric Hypothesis Generator for Project EXODUS.

Every detector in EXODUS assumes human-like technology.  The imagination engine
generates hypotheses about technology we HAVEN'T imagined, tests them against
data, and feeds results back to the Evolver.

Maintains a bank of 10 non-anthropocentric technosignature hypotheses, each with
a testable prediction and specific data test.  The engine:
  1. Selects the appropriate data and test for each hypothesis
  2. Runs the test through the Analyst engine
  3. Records results
  4. If interesting: generates follow-up hypotheses
  5. If null: records in hypothesis graveyard, never retests

The Evolver calls ``imagination.generate_new_hypotheses()`` every N iterations,
feeding in current results to inspire new ideas.

Public API
----------
ImaginationEngine(graveyard_path=None)
    Main class.

.run_hypothesis(hypothesis_id, data)
    Run a specific hypothesis test.

.run_all_available(available_data)
    Run all hypotheses that haven't been graveyard-ed.

.generate_new_hypotheses(current_results)
    Generate new hypotheses from observed patterns.

.get_hypothesis_bank()
    Return the current hypothesis bank.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_logger, PROJECT_ROOT

log = get_logger("engines.imagination")


# =====================================================================
#  Non-Anthropocentric Hypotheses
# =====================================================================

@dataclass
class Hypothesis:
    """A single non-anthropocentric technosignature hypothesis."""
    id: str
    name: str
    theory: str
    signature: str
    dataset: str
    test_method: str
    null_explanation: str
    status: str = "pending"      # pending, tested, interesting, graveyard
    test_result: Dict[str, Any] = field(default_factory=dict)
    tested_at: str = ""
    n_times_tested: int = 0


NON_ANTHROPOCENTRIC_HYPOTHESES = [
    Hypothesis(
        id="NAH001",
        name="Stellar Surgery",
        theory="A civilization modifies its star's fusion process to extend its lifetime or change its output",
        signature="Star in wrong position on HR diagram for its age/cluster membership",
        dataset="gaia_photometry",
        test_method="stellar_anomaly",
        null_explanation="Normal stellar evolution scatter",
    ),
    Hypothesis(
        id="NAH002",
        name="Waste Heat Minimization",
        theory="A VERY advanced civilization approaches thermodynamic limits, producing LESS waste heat than expected",
        signature="Star with LOWER IR than photospheric model predicts (negative IR 'excess')",
        dataset="gaia_wise",
        test_method="negative_ir_excess",
        null_explanation="Measurement error, unusual dust geometry",
    ),
    Hypothesis(
        id="NAH003",
        name="Computational Substrate",
        theory="A civilization uses a neutron star as a computational substrate, modifying its spin or emission",
        signature="Pulsar with anomalous spin-down, glitch patterns that carry information content",
        dataset="pulsar_timing",
        test_method="information_entropy",
        null_explanation="Normal pulsar glitch physics",
    ),
    Hypothesis(
        id="NAH004",
        name="Gravitational Communication",
        theory="A civilization modulates gravitational waves for communication by rapidly moving massive objects",
        signature="GW events with non-astrophysical waveforms or impossible mass ratios",
        dataset="gw_catalog",
        test_method="gw_waveform_anomaly",
        null_explanation="Unmodeled astrophysical sources, noise transients",
    ),
    Hypothesis(
        id="NAH005",
        name="Dark Matter Shepherding",
        theory="A civilization concentrates dark matter for energy extraction",
        signature="Gravitational lensing anomalies coincident with exoplanet hosts",
        dataset="gaia_astrometry",
        test_method="lensing_anomaly",
        null_explanation="Normal dark matter substructure",
    ),
    Hypothesis(
        id="NAH006",
        name="Neutrino Beacon",
        theory="A civilization produces directed neutrino beams as communication",
        signature="Excess directional neutrinos from a specific star system",
        dataset="icecube",
        test_method="neutrino_excess",
        null_explanation="Background fluctuation",
    ),
    Hypothesis(
        id="NAH007",
        name="Temporal Encoding in Variable Stars",
        theory="A civilization modulates a natural variable star's pulsation as a carrier signal",
        signature="Variable star whose pulsation residuals contain non-random structure",
        dataset="lightcurves",
        test_method="information_theoretic",
        null_explanation="Normal pulsation jitter",
    ),
    Hypothesis(
        id="NAH008",
        name="Void Engineering",
        theory="A Type III civilization clears regions of space for unknown purposes",
        signature="Suspiciously empty regions with sharper boundaries than expected",
        dataset="galaxy_surveys",
        test_method="void_analysis",
        null_explanation="Normal cosmic web structure",
    ),
    Hypothesis(
        id="NAH009",
        name="Directed Panspermia Evidence",
        theory="Interstellar objects are artificial probes",
        signature="ISO trajectories tracing back to known exoplanet systems with non-gravitational acceleration",
        dataset="iso_orbits",
        test_method="trajectory_backtrace",
        null_explanation="Random interstellar origin",
    ),
    Hypothesis(
        id="NAH010",
        name="Correlated Anomaly Clusters",
        theory="A spacefaring civilization's activity creates anomalies in MULTIPLE nearby star systems",
        signature="Spatial clustering of multi-modal anomalies",
        dataset="exodus_scores",
        test_method="spatial_autocorrelation",
        null_explanation="Random distribution of anomalies",
    ),
]


# =====================================================================
#  Test implementations
# =====================================================================

def _test_stellar_anomaly(data: Dict[str, Any]) -> Dict[str, Any]:
    """Test NAH001: Are there stars in the wrong HR diagram position?"""
    targets = data.get("targets", [])
    if not targets:
        return {"detection": 0.0, "n_tested": 0, "n_anomalous": 0}

    try:
        from src.detection.stellar_anomaly import batch_detect
        results = batch_detect(targets, sigma_threshold=3.0)
        anomalous = [r for r in results if r.is_anomalous]
        return {
            "detection": len(anomalous) / max(len(results), 1),
            "n_tested": len(results),
            "n_anomalous": len(anomalous),
            "top_anomalies": [r.to_dict() for r in anomalous[:5]],
        }
    except Exception as e:
        return {"detection": 0.0, "error": str(e)}


def _test_negative_ir_excess(data: Dict[str, Any]) -> Dict[str, Any]:
    """Test NAH002: Are there stars with LESS IR than expected?"""
    ir_results = data.get("ir_results", [])
    if not ir_results:
        return {"detection": 0.0, "n_tested": 0, "n_negative": 0}

    # Look for targets with significantly negative excess (dimmer in IR than predicted)
    n_negative = 0
    negative_targets = []
    for r in ir_results:
        excess_w3 = r.get("excess_W3", 0)
        sigma_w3 = r.get("sigma_W3", 0)
        # Positive excess_mag means DIMMER than model (less IR)
        if excess_w3 is not None and excess_w3 > 0 and sigma_w3 is not None and sigma_w3 > 3.0:
            n_negative += 1
            negative_targets.append(r)

    return {
        "detection": n_negative / max(len(ir_results), 1),
        "n_tested": len(ir_results),
        "n_negative": n_negative,
        "targets": negative_targets[:5],
    }


def _test_information_entropy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Test NAH003/NAH007: Do residuals contain non-random structure?"""
    residuals = data.get("residuals")
    if residuals is None or len(residuals) < 20:
        return {"detection": 0.0, "entropy": 0.0}

    residuals = np.asarray(residuals, dtype=np.float64)

    # Compute Shannon entropy of binned residuals
    hist, _ = np.histogram(residuals, bins=20, density=True)
    hist = hist[hist > 0]
    bin_width = (residuals.max() - residuals.min()) / 20
    entropy = -np.sum(hist * np.log2(hist) * bin_width)

    # Expected entropy for Gaussian distribution
    sigma = np.std(residuals)
    expected_entropy = 0.5 * np.log2(2 * np.pi * np.e * sigma**2) if sigma > 0 else 0

    # Mutual information: autocorrelation of successive residuals
    if len(residuals) > 2:
        r1 = residuals[:-1]
        r2 = residuals[1:]
        corr = np.corrcoef(r1, r2)[0, 1]
    else:
        corr = 0.0

    # Detection: significant autocorrelation suggests non-randomness
    detection = float(abs(corr)) if abs(corr) > 0.2 else 0.0

    return {
        "detection": detection,
        "entropy": float(entropy),
        "expected_entropy": float(expected_entropy),
        "autocorrelation": float(corr),
        "is_structured": abs(corr) > 0.3,
    }


def _test_spatial_autocorrelation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Test NAH010: Are anomalies spatially clustered?"""
    scores = data.get("exodus_scores", [])
    if len(scores) < 10:
        return {"detection": 0.0, "morans_i": 0.0}

    # Extract positions and scores
    ras = np.array([s.get("ra", 0) for s in scores])
    decs = np.array([s.get("dec", 0) for s in scores])
    values = np.array([s.get("total_score", 0) for s in scores])

    if np.std(values) == 0:
        return {"detection": 0.0, "morans_i": 0.0, "note": "No variance in scores"}

    # Compute Moran's I statistic
    n = len(values)
    z = values - np.mean(values)

    # Spatial weight matrix (inverse distance, using simple Euclidean approx)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((ras[i] - ras[j])**2 + (decs[i] - decs[j])**2)
            if dist > 0:
                w = 1.0 / max(dist, 0.01)
                W[i, j] = w
                W[j, i] = w

    W_sum = np.sum(W)
    if W_sum == 0:
        return {"detection": 0.0, "morans_i": 0.0, "note": "No spatial variation"}

    numerator = n * np.sum(W * np.outer(z, z))
    denominator = W_sum * np.sum(z**2)
    morans_i = float(numerator / denominator) if denominator != 0 else 0.0

    # Expected value under random distribution
    expected_i = -1.0 / (n - 1)

    # Significance (simplified)
    detection = float(max(morans_i - expected_i, 0))

    return {
        "detection": min(detection, 1.0),
        "morans_i": morans_i,
        "expected_i": expected_i,
        "is_clustered": morans_i > expected_i + 0.1,
    }


def _test_generic(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generic test for hypotheses without specific data access yet."""
    return {
        "detection": 0.0,
        "note": "Specific test not yet implemented — requires dedicated data pipeline",
        "data_available": bool(data),
    }


# Map test methods to functions
_TEST_FUNCTIONS = {
    "stellar_anomaly": _test_stellar_anomaly,
    "negative_ir_excess": _test_negative_ir_excess,
    "information_entropy": _test_information_entropy,
    "information_theoretic": _test_information_entropy,
    "spatial_autocorrelation": _test_spatial_autocorrelation,
    "neutrino_excess": _test_generic,
    "gw_waveform_anomaly": _test_generic,
    "lensing_anomaly": _test_generic,
    "void_analysis": _test_generic,
    "trajectory_backtrace": _test_generic,
}


# =====================================================================
#  Imagination Engine
# =====================================================================

class ImaginationEngine:
    """Non-anthropocentric hypothesis generator and tester.

    Parameters
    ----------
    graveyard_path : str or Path, optional
        Path to persist the hypothesis graveyard.
    """

    def __init__(self, graveyard_path: Optional[str | Path] = None) -> None:
        self._graveyard_path = (
            Path(graveyard_path) if graveyard_path
            else PROJECT_ROOT / "data" / "hypotheses" / "imagination_graveyard.json"
        )
        self._graveyard_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize hypothesis bank (deep copy)
        self._hypotheses: Dict[str, Hypothesis] = {
            h.id: Hypothesis(
                id=h.id, name=h.name, theory=h.theory, signature=h.signature,
                dataset=h.dataset, test_method=h.test_method,
                null_explanation=h.null_explanation,
            )
            for h in NON_ANTHROPOCENTRIC_HYPOTHESES
        }

        # Load graveyard state
        self._load_graveyard()

        n_active = sum(1 for h in self._hypotheses.values() if h.status != "graveyard")
        log.info(
            "ImaginationEngine initialized: %d hypotheses (%d active, %d graveyard)",
            len(self._hypotheses), n_active, len(self._hypotheses) - n_active,
        )

    # ─── Core API ─────────────────────────────────────────────────────

    def run_hypothesis(
        self, hypothesis_id: str, data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a specific hypothesis test.

        Parameters
        ----------
        hypothesis_id : str
            ID of the hypothesis to test.
        data : dict
            Data to feed into the test.

        Returns
        -------
        dict
            Test results including detection score.
        """
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None:
            log.warning("Hypothesis %s not found", hypothesis_id)
            return {"error": f"Hypothesis {hypothesis_id} not found"}

        if hyp.status == "graveyard":
            log.info("Hypothesis %s is in graveyard; skipping", hypothesis_id)
            return {"skipped": True, "reason": "In graveyard (already tested null)"}

        test_fn = _TEST_FUNCTIONS.get(hyp.test_method, _test_generic)
        log.info("Running imagination test [%s] %s: %s", hyp.id, hyp.name, hyp.test_method)

        result = test_fn(data)
        detection = result.get("detection", 0.0)

        hyp.n_times_tested += 1
        hyp.tested_at = datetime.now(timezone.utc).isoformat()
        hyp.test_result = result

        if detection > 0.1:
            hyp.status = "interesting"
            log.info(
                "INTERESTING: %s [%s] detection=%.3f",
                hyp.name, hyp.id, detection,
            )
        else:
            hyp.status = "graveyard"
            log.info(
                "Null result for %s [%s] — moving to graveyard",
                hyp.name, hyp.id,
            )

        self._save_graveyard()
        return result

    def run_all_available(
        self, available_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run all hypotheses that have matching data and aren't graveyard-ed.

        Parameters
        ----------
        available_data : dict
            Mapping of dataset_name -> data_dict.

        Returns
        -------
        list of dict
            Results for each hypothesis tested.
        """
        results = []
        for hyp_id, hyp in self._hypotheses.items():
            if hyp.status == "graveyard":
                continue

            # Check if we have data for this hypothesis
            data = available_data.get(hyp.dataset, {})
            if not data:
                # Try matching by test_method
                data = available_data.get(hyp.test_method, {})

            result = self.run_hypothesis(hyp_id, data)
            result["hypothesis_id"] = hyp_id
            result["hypothesis_name"] = hyp.name
            results.append(result)

        return results

    def generate_new_hypotheses(
        self, current_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate new hypotheses from observed patterns.

        Looks at current results and generates derivative hypotheses
        by combining existing interesting results.

        Parameters
        ----------
        current_results : dict
            Current pipeline results to inspire new hypotheses.

        Returns
        -------
        list of dict
            New hypotheses generated.
        """
        new_hypotheses = []
        interesting = [
            h for h in self._hypotheses.values()
            if h.status == "interesting"
        ]

        if not interesting:
            return new_hypotheses

        # Cross-domain combinations
        for h in interesting:
            detection = h.test_result.get("detection", 0)
            if detection > 0.2:
                # Generate a follow-up: "Does this anomaly correlate with other datasets?"
                for other_dataset in ["gaia_photometry", "wise", "radio", "icecube"]:
                    if other_dataset != h.dataset:
                        new_hyp = {
                            "id": f"NAH_DERIVED_{h.id}_{other_dataset}",
                            "name": f"Cross-domain: {h.name} + {other_dataset}",
                            "theory": (
                                f"If {h.name} is real, it should also produce "
                                f"signatures in {other_dataset} data."
                            ),
                            "source": h.id,
                            "priority": detection * 0.8,
                        }
                        new_hypotheses.append(new_hyp)

        # Look for anomalies the named detectors missed
        unsupervised_anomalies = current_results.get("unsupervised_anomalies", [])
        named_detections = current_results.get("named_detections", set())

        for anomaly in unsupervised_anomalies:
            target_id = anomaly.get("target_id", "")
            if target_id not in named_detections:
                new_hyp = {
                    "id": f"NAH_UNSUPERVISED_{target_id}",
                    "name": f"Unknown anomaly in {target_id}",
                    "theory": (
                        f"Target {target_id} flagged by unsupervised analysis "
                        f"but NOT by any named detector — this is the most "
                        f"interesting kind of anomaly."
                    ),
                    "source": "unsupervised_stacking",
                    "priority": 1.0,
                }
                new_hypotheses.append(new_hyp)

        log.info(
            "Imagination generated %d new hypotheses from %d interesting results",
            len(new_hypotheses), len(interesting),
        )

        return new_hypotheses

    def get_hypothesis_bank(self) -> List[Dict[str, Any]]:
        """Return the current hypothesis bank with status."""
        return [
            {
                "id": h.id,
                "name": h.name,
                "status": h.status,
                "n_tested": h.n_times_tested,
                "detection": h.test_result.get("detection", 0) if h.test_result else 0,
                "theory": h.theory,
                "signature": h.signature,
            }
            for h in self._hypotheses.values()
        ]

    def get_interesting(self) -> List[Hypothesis]:
        """Return hypotheses that produced interesting results."""
        return [h for h in self._hypotheses.values() if h.status == "interesting"]

    def get_graveyard(self) -> List[Hypothesis]:
        """Return graveyard-ed hypotheses (null results)."""
        return [h for h in self._hypotheses.values() if h.status == "graveyard"]

    def reset(self) -> None:
        """Reset all hypotheses to pending (for testing)."""
        for h in self._hypotheses.values():
            h.status = "pending"
            h.test_result = {}
            h.tested_at = ""
            h.n_times_tested = 0
        self._save_graveyard()

    # ─── Persistence ──────────────────────────────────────────────────

    def _save_graveyard(self) -> None:
        """Persist hypothesis states to disk."""
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "hypotheses": {
                h.id: {
                    "status": h.status,
                    "n_tested": h.n_times_tested,
                    "tested_at": h.tested_at,
                    "test_result": h.test_result,
                }
                for h in self._hypotheses.values()
            },
        }
        with open(self._graveyard_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_graveyard(self) -> None:
        """Load hypothesis states from disk."""
        if not self._graveyard_path.exists():
            return
        try:
            with open(self._graveyard_path) as f:
                data = json.load(f)
            for hyp_id, state in data.get("hypotheses", {}).items():
                if hyp_id in self._hypotheses:
                    self._hypotheses[hyp_id].status = state.get("status", "pending")
                    self._hypotheses[hyp_id].n_times_tested = state.get("n_tested", 0)
                    self._hypotheses[hyp_id].tested_at = state.get("tested_at", "")
                    self._hypotheses[hyp_id].test_result = state.get("test_result", {})
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to load graveyard: %s", e)


# =====================================================================
#  CLI demo / self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Project EXODUS -- Imagination Engine Demo")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # Use temp graveyard
    graveyard = PROJECT_ROOT / "data" / "hypotheses" / "imagination_demo.json"
    if graveyard.exists():
        graveyard.unlink()
    engine = ImaginationEngine(graveyard_path=graveyard)

    # Show initial bank
    print(f"\n  Hypothesis Bank ({len(engine.get_hypothesis_bank())} hypotheses):")
    for h in engine.get_hypothesis_bank():
        print(f"    [{h['id']}] {h['name']:<35s}  status={h['status']}")

    # Prepare mock data for testable hypotheses
    print("\n  Running testable hypotheses...")

    # NAH001: Stellar anomaly — provide mock targets
    targets = []
    for i in range(50):
        bp_rp = rng.uniform(0.3, 3.0)
        # Most normal, a few anomalous
        if i < 3:
            abs_g = float(1.0 + 4.2 * bp_rp - 5.0)  # way too bright
        else:
            abs_g = float(1.0 + 4.2 * bp_rp + rng.normal(0, 0.3))
        targets.append({
            "source_id": f"IMAG_STAR_{i:03d}",
            "bp_rp": float(bp_rp),
            "abs_g": abs_g,
        })

    # NAH002: Negative IR excess — provide mock IR results
    ir_results = []
    for i in range(30):
        ir_results.append({
            "excess_W3": float(rng.normal(0, 0.5)),
            "sigma_W3": float(abs(rng.normal(0, 1.5))),
        })

    # NAH010: Spatial autocorrelation — provide mock EXODUS scores
    exodus_scores = []
    for i in range(100):
        exodus_scores.append({
            "ra": float(rng.uniform(150, 210)),
            "dec": float(rng.uniform(30, 60)),
            "total_score": float(rng.exponential(0.3)),
        })

    available_data = {
        "gaia_photometry": {"targets": targets},
        "gaia_wise": {"ir_results": ir_results},
        "exodus_scores": {"exodus_scores": exodus_scores},
        "lightcurves": {"residuals": list(rng.normal(0, 1, 100))},
    }

    results = engine.run_all_available(available_data)

    print(f"\n  Results:")
    for r in results:
        hyp_id = r.get("hypothesis_id", "?")
        hyp_name = r.get("hypothesis_name", "?")
        detection = r.get("detection", 0)
        skipped = r.get("skipped", False)
        if skipped:
            print(f"    [{hyp_id}] {hyp_name:<35s}: SKIPPED (graveyard)")
        else:
            print(f"    [{hyp_id}] {hyp_name:<35s}: detection={detection:.3f}")

    # Show final state
    print(f"\n  Final State:")
    interesting = engine.get_interesting()
    graveyard_list = engine.get_graveyard()
    print(f"    Interesting: {len(interesting)}")
    for h in interesting:
        print(f"      [{h.id}] {h.name}")
    print(f"    Graveyard:   {len(graveyard_list)}")

    # Test that graveyard prevents re-testing
    print(f"\n  Re-running (should skip graveyard)...")
    results2 = engine.run_all_available(available_data)
    n_skipped = sum(1 for r in results2 if r.get("skipped"))
    print(f"    Skipped: {n_skipped}")

    # Generate new hypotheses
    new_hyps = engine.generate_new_hypotheses({
        "unsupervised_anomalies": [{"target_id": "MYSTERY_001"}],
        "named_detections": set(),
    })
    print(f"\n  New hypotheses from imagination: {len(new_hyps)}")
    for h in new_hyps[:5]:
        print(f"    [{h['id']}] {h['name']}")

    # Cleanup
    if graveyard.exists():
        graveyard.unlink()

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
