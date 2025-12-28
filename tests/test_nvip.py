"""
Tests for the NVIP module.
"""

import numpy as np
import pytest

from neutrohydro.preprocessing import Preprocessor
from neutrohydro.encoder import NDGEncoder
from neutrohydro.model import PNPLS
from neutrohydro.nvip import (
    compute_nvip,
    verify_l2_decomposition,
    nvip_to_dataframe,
    NVIPResult,
)


class TestComputeNVIP:
    """Tests for compute_nvip function."""

    @pytest.fixture
    def fitted_model(self, sample_data):
        """Get a fitted model for NVIP tests."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)
        model = PNPLS(n_components=5)
        model.fit(triplets, y_std)
        return model, triplets, X.shape[1]

    def test_nvip_returns_correct_type(self, fitted_model):
        """Test that compute_nvip returns NVIPResult."""
        model, triplets, p = fitted_model
        nvip = compute_nvip(model)

        assert isinstance(nvip, NVIPResult)

    def test_nvip_dimensions(self, fitted_model):
        """Test that NVIP arrays have correct dimensions."""
        model, triplets, p = fitted_model
        nvip = compute_nvip(model)

        assert len(nvip.VIP_T) == p
        assert len(nvip.VIP_I) == p
        assert len(nvip.VIP_F) == p
        assert len(nvip.VIP_agg) == p
        assert nvip.n_features == p

    def test_vip_non_negative(self, fitted_model):
        """Test that VIP values are non-negative."""
        model, triplets, p = fitted_model
        nvip = compute_nvip(model)

        assert all(nvip.VIP_T >= 0)
        assert all(nvip.VIP_I >= 0)
        assert all(nvip.VIP_F >= 0)
        assert all(nvip.VIP_agg >= 0)

    def test_energy_non_negative(self, fitted_model):
        """Test that energy values are non-negative."""
        model, triplets, p = fitted_model
        nvip = compute_nvip(model)

        assert all(nvip.E_T >= 0)
        assert all(nvip.E_I >= 0)
        assert all(nvip.E_F >= 0)

    def test_energy_is_vip_squared(self, fitted_model):
        """Test that E_c = VIP_c^2."""
        model, triplets, p = fitted_model
        nvip = compute_nvip(model)

        np.testing.assert_array_almost_equal(nvip.E_T, nvip.VIP_T ** 2)
        np.testing.assert_array_almost_equal(nvip.E_I, nvip.VIP_I ** 2)
        np.testing.assert_array_almost_equal(nvip.E_F, nvip.VIP_F ** 2)

    def test_l2_decomposition_theorem(self, fitted_model):
        """Test the L2 decomposition: VIP_agg^2 = VIP_T^2 + VIP_I^2 + VIP_F^2."""
        model, triplets, p = fitted_model
        nvip = compute_nvip(model)

        # Use the verification function
        assert verify_l2_decomposition(nvip)

        # Also check directly
        lhs = nvip.VIP_agg ** 2
        rhs = nvip.VIP_T ** 2 + nvip.VIP_I ** 2 + nvip.VIP_F ** 2
        np.testing.assert_array_almost_equal(lhs, rhs)

    def test_vip_sum_property(self, fitted_model):
        """Test that sum of VIP_agg^2 equals p (number of variables)."""
        model, triplets, p = fitted_model
        nvip = compute_nvip(model)

        # In standard VIP, sum of VIP^2 = p
        vip_agg_sum = np.sum(nvip.VIP_agg ** 2)
        # Should be close to p (with some tolerance due to numerical precision)
        assert abs(vip_agg_sum - p) < 1e-6

    def test_ssy_non_negative(self, fitted_model):
        """Test that SSY (response energy) is non-negative."""
        model, triplets, p = fitted_model
        nvip = compute_nvip(model)

        assert all(nvip.SSY >= 0)

    def test_unfitted_model_raises(self, sample_data):
        """Test that unfitted model raises error."""
        model = PNPLS()
        with pytest.raises(RuntimeError):
            compute_nvip(model)


class TestNVIPToDataFrame:
    """Tests for nvip_to_dataframe function."""

    @pytest.fixture
    def nvip_result(self, sample_data, ion_names):
        """Get NVIP result for dataframe tests."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)
        model = PNPLS(n_components=5)
        model.fit(triplets, y_std)
        return compute_nvip(model)

    def test_dataframe_columns(self, nvip_result, ion_names):
        """Test that dataframe has expected columns."""
        df = nvip_to_dataframe(nvip_result, ion_names)

        expected_columns = ['feature', 'VIP_T', 'VIP_I', 'VIP_F', 'VIP_agg',
                            'E_T', 'E_I', 'E_F']
        assert list(df.columns) == expected_columns

    def test_dataframe_rows(self, nvip_result, ion_names):
        """Test that dataframe has correct number of rows."""
        df = nvip_to_dataframe(nvip_result, ion_names)
        assert len(df) == nvip_result.n_features

    def test_dataframe_feature_names(self, nvip_result, ion_names):
        """Test that feature names are correct."""
        df = nvip_to_dataframe(nvip_result, ion_names)
        assert list(df['feature']) == ion_names

    def test_dataframe_default_names(self, nvip_result):
        """Test default feature names when not provided."""
        df = nvip_to_dataframe(nvip_result)
        assert df['feature'].iloc[0] == "X0"


class TestVerifyL2Decomposition:
    """Tests for verify_l2_decomposition function."""

    def test_valid_decomposition(self, sample_data):
        """Test that valid decomposition passes."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)
        model = PNPLS(n_components=5)
        model.fit(triplets, y_std)
        nvip = compute_nvip(model)

        assert verify_l2_decomposition(nvip)

    def test_invalid_decomposition(self):
        """Test that invalid decomposition fails."""
        # Create a fake NVIPResult with wrong values
        p = 5
        nvip = NVIPResult(
            VIP_T=np.array([1.0] * p),
            VIP_I=np.array([1.0] * p),
            VIP_F=np.array([1.0] * p),
            VIP_agg=np.array([2.0] * p),  # Wrong! Should be sqrt(3)
            E_T=np.array([1.0] * p),
            E_I=np.array([1.0] * p),
            E_F=np.array([1.0] * p),
            SSY=np.array([1.0]),
            n_features=p,
            n_components=1,
        )

        assert not verify_l2_decomposition(nvip)
