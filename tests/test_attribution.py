"""
Tests for the attribution module.
"""

import numpy as np
import pytest

from neutrohydro.preprocessing import Preprocessor
from neutrohydro.encoder import NDGEncoder
from neutrohydro.model import PNPLS
from neutrohydro.nvip import compute_nvip
from neutrohydro.attribution import (
    compute_nsr,
    compute_sample_baseline_fraction,
    attribution_summary,
    nsr_to_dataframe,
    sample_attribution_to_dataframe,
    NSRResult,
    SampleAttributionResult,
)


class TestComputeNSR:
    """Tests for compute_nsr function."""

    @pytest.fixture
    def nvip_result(self, sample_data):
        """Get NVIP result for NSR tests."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)
        model = PNPLS(n_components=5)
        model.fit(triplets, y_std)
        return compute_nvip(model)

    def test_nsr_returns_correct_type(self, nvip_result):
        """Test that compute_nsr returns NSRResult."""
        nsr = compute_nsr(nvip_result)
        assert isinstance(nsr, NSRResult)

    def test_nsr_dimensions(self, nvip_result):
        """Test that NSR arrays have correct dimensions."""
        nsr = compute_nsr(nvip_result)
        p = nvip_result.n_features

        assert len(nsr.NSR) == p
        assert len(nsr.pi_G) == p
        assert len(nsr.pi_A) == p
        assert len(nsr.classification) == p

    def test_pi_g_bounds(self, nvip_result):
        """Test that pi_G is in [0, 1]."""
        nsr = compute_nsr(nvip_result)

        assert all(nsr.pi_G >= 0)
        assert all(nsr.pi_G <= 1)

    def test_pi_g_plus_pi_a_equals_one(self, nvip_result):
        """Test that pi_G + pi_A = 1."""
        nsr = compute_nsr(nvip_result)

        np.testing.assert_array_almost_equal(nsr.pi_G + nsr.pi_A, 1.0)

    def test_nsr_positive(self, nvip_result):
        """Test that NSR is positive."""
        nsr = compute_nsr(nvip_result)
        assert all(nsr.NSR > 0)

    def test_nsr_pi_g_relationship(self, nvip_result):
        """Test the relationship: pi_G = NSR / (1 + NSR)."""
        nsr = compute_nsr(nvip_result, epsilon=1e-10)

        # Compute expected pi_G from NSR
        expected_pi_G = nsr.NSR / (1 + nsr.NSR)

        np.testing.assert_array_almost_equal(nsr.pi_G, expected_pi_G, decimal=5)

    def test_energy_sum(self, nvip_result):
        """Test that E_T + E_P equals total energy."""
        nsr = compute_nsr(nvip_result)

        E_total = nsr.E_T + nsr.E_P
        expected_total = nvip_result.E_T + nvip_result.E_I + nvip_result.E_F

        np.testing.assert_array_almost_equal(E_total, expected_total)

    def test_classification(self, nvip_result):
        """Test classification based on gamma threshold."""
        gamma = 0.7
        nsr = compute_nsr(nvip_result, gamma=gamma)

        for j in range(len(nsr.pi_G)):
            if nsr.pi_G[j] >= gamma:
                assert nsr.classification[j] == "baseline"
            elif nsr.pi_G[j] <= 1 - gamma:
                assert nsr.classification[j] == "perturbation"
            else:
                assert nsr.classification[j] == "mixed"

    def test_different_gamma(self, nvip_result):
        """Test different gamma values."""
        nsr_low = compute_nsr(nvip_result, gamma=0.5)
        nsr_high = compute_nsr(nvip_result, gamma=0.9)

        # Higher gamma should result in fewer "baseline" classifications
        n_baseline_low = np.sum(nsr_low.classification == "baseline")
        n_baseline_high = np.sum(nsr_high.classification == "baseline")

        assert n_baseline_high <= n_baseline_low


class TestComputeSampleBaselineFraction:
    """Tests for compute_sample_baseline_fraction function."""

    @pytest.fixture
    def model_and_nsr(self, sample_data):
        """Get fitted model, triplets, and NSR for tests."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)
        model = PNPLS(n_components=5)
        model.fit(triplets, y_std)
        nvip = compute_nvip(model)
        nsr = compute_nsr(nvip)
        return model, triplets, nsr

    def test_returns_correct_type(self, model_and_nsr):
        """Test that function returns SampleAttributionResult."""
        model, triplets, nsr = model_and_nsr
        result = compute_sample_baseline_fraction(model, triplets, nsr)

        assert isinstance(result, SampleAttributionResult)

    def test_dimensions(self, model_and_nsr):
        """Test that arrays have correct dimensions."""
        model, triplets, nsr = model_and_nsr
        n = triplets.T.shape[0]
        p = triplets.T.shape[1]

        result = compute_sample_baseline_fraction(model, triplets, nsr)

        assert len(result.G) == n
        assert len(result.A) == n
        assert result.w.shape == (n, p)
        assert result.c.shape == (n, p)

    def test_g_bounds(self, model_and_nsr):
        """Test that G is in [0, 1]."""
        model, triplets, nsr = model_and_nsr
        result = compute_sample_baseline_fraction(model, triplets, nsr)

        assert all(result.G >= 0)
        assert all(result.G <= 1)

    def test_g_plus_a_equals_one(self, model_and_nsr):
        """Test that G + A = 1."""
        model, triplets, nsr = model_and_nsr
        result = compute_sample_baseline_fraction(model, triplets, nsr)

        np.testing.assert_array_almost_equal(result.G + result.A, 1.0)

    def test_w_non_negative(self, model_and_nsr):
        """Test that attribution mass w is non-negative."""
        model, triplets, nsr = model_and_nsr
        result = compute_sample_baseline_fraction(model, triplets, nsr)

        assert np.all(result.w >= 0)


class TestAttributionSummary:
    """Tests for attribution_summary function."""

    @pytest.fixture
    def full_results(self, sample_data, ion_names):
        """Get full attribution results for summary tests."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)
        model = PNPLS(n_components=5)
        model.fit(triplets, y_std)
        nvip = compute_nvip(model)
        nsr = compute_nsr(nvip)
        sample_attr = compute_sample_baseline_fraction(model, triplets, nsr)
        return nsr, sample_attr, ion_names

    def test_summary_contains_expected_keys(self, full_results):
        """Test that summary contains expected keys."""
        nsr, sample_attr, feature_names = full_results
        summary = attribution_summary(nsr, sample_attr, feature_names)

        expected_keys = [
            'n_ions', 'n_baseline_ions', 'n_perturbation_ions', 'n_mixed_ions',
            'baseline_ions', 'perturbation_ions', 'mean_pi_G', 'mean_NSR',
            'n_samples', 'G_mean', 'G_std', 'G_median',
            'frac_baseline_samples', 'frac_perturbation_samples',
            'classification_threshold',
        ]

        for key in expected_keys:
            assert key in summary


class TestDataFrameConversions:
    """Tests for dataframe conversion functions."""

    @pytest.fixture
    def results(self, sample_data):
        """Get results for dataframe tests."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)
        model = PNPLS(n_components=5)
        model.fit(triplets, y_std)
        nvip = compute_nvip(model)
        nsr = compute_nsr(nvip)
        sample_attr = compute_sample_baseline_fraction(model, triplets, nsr)
        return nsr, sample_attr

    def test_nsr_to_dataframe(self, results):
        """Test nsr_to_dataframe function."""
        nsr, _ = results
        df = nsr_to_dataframe(nsr)

        expected_columns = ['ion', 'NSR', 'pi_G', 'pi_A', 'E_T', 'E_P', 'classification']
        assert list(df.columns) == expected_columns
        assert len(df) == len(nsr.pi_G)

    def test_sample_attribution_to_dataframe(self, results):
        """Test sample_attribution_to_dataframe function."""
        _, sample_attr = results
        df = sample_attribution_to_dataframe(sample_attr)

        expected_columns = ['sample_id', 'G', 'A', 'total_attribution_mass']
        assert list(df.columns) == expected_columns
        assert len(df) == len(sample_attr.G)
