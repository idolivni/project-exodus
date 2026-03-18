"""
Tests for Empirical Brown's Method (EBM) implementation — Session 36.

Validates:
1. ebm_combine() with no covariance = standard Fisher
2. ebm_combine() with identity covariance = standard Fisher
3. ebm_combine() with positive correlations → less significant (more conservative)
4. ebm_combine() edge cases (empty, single p-value)
5. compute_ebm_covariance() produces valid covariance matrix
6. EBM integrated into EXODUSScore dataclass
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.statistics import (
    ebm_combine,
    compute_ebm_covariance,
    fisher_combine,
    stouffer_combine,
)


class TestEBMCombine:
    """Tests for ebm_combine()."""

    def test_no_covariance_equals_fisher(self):
        """Without covariance matrix, EBM should equal Fisher."""
        pvals = [0.01, 0.03, 0.05]
        assert abs(ebm_combine(pvals, covariance_matrix=None) - fisher_combine(pvals)) < 1e-10

    def test_identity_covariance_equals_fisher(self):
        """Identity-like covariance (independent channels) ≈ Fisher."""
        pvals = [0.01, 0.03, 0.05]
        # Under independence, each t_i = -2*log(p_i) ~ chi2(2),
        # so Var(t_i)=4 and Cov(t_i,t_j)=0 for i!=j.
        k = len(pvals)
        cov = np.eye(k) * 4.0  # diagonal = Var(chi2(2)) = 4
        ebm_p = ebm_combine(pvals, covariance_matrix=cov)
        fisher_p = fisher_combine(pvals)
        assert abs(ebm_p - fisher_p) < 1e-10

    def test_positive_correlation_less_significant(self):
        """Positively correlated channels should give LESS significant (larger) p."""
        pvals = [0.01, 0.03, 0.05]
        k = len(pvals)
        # Strong positive off-diagonal covariance
        cov = np.eye(k) * 4.0
        cov += 2.0  # off-diagonal = 2 (positive correlation)
        ebm_p = ebm_combine(pvals, covariance_matrix=cov)
        fisher_p = fisher_combine(pvals)
        # EBM should be LESS significant (larger p) than Fisher
        assert ebm_p > fisher_p, f"EBM ({ebm_p}) should be > Fisher ({fisher_p})"

    def test_negative_correlation_more_significant(self):
        """Negatively correlated channels should give MORE significant (smaller) p."""
        pvals = [0.01, 0.03, 0.05]
        k = len(pvals)
        # Negative off-diagonal (anti-correlated channels provide MORE info)
        cov = np.eye(k) * 4.0
        cov[0, 1] = cov[1, 0] = -1.0
        cov[0, 2] = cov[2, 0] = -1.0
        cov[1, 2] = cov[2, 1] = -1.0
        ebm_p = ebm_combine(pvals, covariance_matrix=cov)
        fisher_p = fisher_combine(pvals)
        assert ebm_p < fisher_p, f"EBM ({ebm_p}) should be < Fisher ({fisher_p})"

    def test_empty_returns_one(self):
        assert ebm_combine([]) == 1.0

    def test_single_pvalue_passthrough(self):
        assert ebm_combine([0.042]) == 0.042

    def test_invalid_pvalues_filtered(self):
        """Zero and negative p-values should be filtered out."""
        result = ebm_combine([0.0, -0.5, 0.05])
        assert result == 0.05  # only one valid → passthrough

    def test_small_covariance_matrix_fallback(self):
        """If covariance matrix is too small, should fallback to Fisher."""
        pvals = [0.01, 0.03, 0.05]
        small_cov = np.eye(2) * 4.0  # 2×2 but we have 3 p-values
        ebm_p = ebm_combine(pvals, covariance_matrix=small_cov)
        fisher_p = fisher_combine(pvals)
        assert abs(ebm_p - fisher_p) < 1e-10  # fallback


class TestComputeEBMCovariance:
    """Tests for compute_ebm_covariance()."""

    def test_independent_controls_diagonal(self):
        """Independent control channels → near-diagonal covariance."""
        rng = np.random.default_rng(42)
        n_controls = 200
        n_channels = 4
        # Generate independent uniform p-values (null distribution)
        p_matrix = rng.uniform(0.001, 1.0, size=(n_controls, n_channels))
        cov = compute_ebm_covariance(p_matrix)
        assert cov.shape == (n_channels, n_channels)
        # Diagonal should be ~4 (Var of chi2(2))
        for i in range(n_channels):
            assert 2.5 < cov[i, i] < 6.0, f"Diagonal [{i}] = {cov[i,i]}, expected ~4"
        # Off-diagonal should be near zero for independent channels
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                assert abs(cov[i, j]) < 1.5, f"Off-diag [{i},{j}] = {cov[i,j]}, expected ~0"

    def test_correlated_controls_offdiag(self):
        """Correlated control channels → non-zero off-diagonal."""
        rng = np.random.default_rng(123)
        n_controls = 300
        # Create correlated p-values: ch0 and ch1 share a latent variable
        latent = rng.normal(0, 1, size=n_controls)
        from scipy.stats import norm
        p0 = norm.sf(latent + rng.normal(0, 0.5, n_controls))
        p1 = norm.sf(latent + rng.normal(0, 0.5, n_controls))
        p2 = rng.uniform(0.001, 1.0, n_controls)  # independent
        p_matrix = np.column_stack([p0, p1, p2])
        p_matrix = np.clip(p_matrix, 1e-10, 1.0)
        cov = compute_ebm_covariance(p_matrix)
        # ch0-ch1 should have positive covariance
        assert cov[0, 1] > 1.0, f"Correlated channels cov={cov[0,1]}, expected >1"
        # ch0-ch2 should be near zero
        assert abs(cov[0, 2]) < 2.0, f"Independent channel cov={cov[0,2]}, expected ~0"

    def test_symmetric(self):
        """Covariance matrix must be symmetric."""
        rng = np.random.default_rng(7)
        p_matrix = rng.uniform(0.001, 1.0, size=(100, 5))
        cov = compute_ebm_covariance(p_matrix)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_positive_semidefinite(self):
        """Covariance matrix must be positive semi-definite."""
        rng = np.random.default_rng(99)
        p_matrix = rng.uniform(0.001, 1.0, size=(100, 4))
        cov = compute_ebm_covariance(p_matrix)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues}"


class TestEBMInScorer:
    """Test EBM integration in EXODUSScore dataclass."""

    def test_ebm_p_in_dataclass(self):
        from src.scoring.exodus_score import EXODUSScore
        score = EXODUSScore(
            target_id="test",
            ra=0.0, dec=0.0,
            total_score=1.0,
            channel_scores={},
            n_active_channels=0,
            convergence_bonus=1.0,
            geo_mean=0.0,
            ebm_p=0.042,
        )
        assert score.ebm_p == 0.042
        d = score.to_dict()
        assert d["ebm_p"] == 0.042
        assert any("Empirical Brown" in c for c in d.get("statistical_caveats", []))

    def test_ebm_p_default_none(self):
        from src.scoring.exodus_score import EXODUSScore
        score = EXODUSScore(
            target_id="test",
            ra=0.0, dec=0.0,
            total_score=1.0,
            channel_scores={},
            n_active_channels=0,
            convergence_bonus=1.0,
            geo_mean=0.0,
        )
        assert score.ebm_p is None

    def test_ebm_shown_in_summary(self):
        from src.scoring.exodus_score import EXODUSScore
        score = EXODUSScore(
            target_id="test",
            ra=0.0, dec=0.0,
            total_score=1.0,
            channel_scores={},
            n_active_channels=0,
            convergence_bonus=1.0,
            geo_mean=0.0,
            ebm_p=0.0042,
        )
        summary = score.summary()
        assert "ebm=" in summary
