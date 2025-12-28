"""
Tests for the NDG encoder module.
"""

import numpy as np
import pytest

from neutrohydro.preprocessing import Preprocessor
from neutrohydro.encoder import (
    NDGEncoder,
    TripletData,
    BaselineType,
    FalsityMap,
    create_censoring_indeterminacy,
)


class TestNDGEncoder:
    """Tests for NDGEncoder class."""

    @pytest.fixture
    def standardized_data(self, sample_data):
        """Get standardized data for encoder tests."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, _ = preprocessor.fit_transform(X, y)
        return X_std

    def test_fit_basic(self, standardized_data):
        """Test basic fitting."""
        encoder = NDGEncoder()
        encoder.fit(standardized_data)

        assert encoder.is_fitted_
        assert encoder.params_ is not None
        assert len(encoder.params_.sigma) == standardized_data.shape[1]

    def test_transform_returns_triplet(self, standardized_data):
        """Test that transform returns TripletData."""
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(standardized_data)

        assert isinstance(triplets, TripletData)
        assert triplets.T.shape == standardized_data.shape
        assert triplets.I.shape == standardized_data.shape
        assert triplets.F.shape == standardized_data.shape
        assert triplets.R.shape == standardized_data.shape

    def test_triplet_bounds(self, standardized_data):
        """Test that I and F are bounded in [0, 1]."""
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(standardized_data)

        assert triplets.I.min() >= 0.0
        assert triplets.I.max() <= 1.0
        assert triplets.F.min() >= 0.0
        assert triplets.F.max() <= 1.0

    def test_residual_computation(self, standardized_data):
        """Test that residuals are X_std - T."""
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(standardized_data)

        np.testing.assert_array_almost_equal(
            triplets.R, standardized_data - triplets.T
        )

    def test_baseline_median(self, standardized_data):
        """Test median baseline."""
        encoder = NDGEncoder(baseline_type=BaselineType.MEDIAN)
        triplets = encoder.fit_transform(standardized_data)

        # T should be constant per column (replicated median)
        for j in range(standardized_data.shape[1]):
            assert len(np.unique(triplets.T[:, j])) == 1

    def test_baseline_low_rank(self, standardized_data):
        """Test low-rank baseline."""
        encoder = NDGEncoder(baseline_type=BaselineType.LOW_RANK, baseline_rank=3)
        encoder.fit(standardized_data)

        assert encoder.params_.baseline_rank == 3
        assert encoder.params_.baseline_metadata["type"] == "low_rank"

    def test_baseline_robust_pca(self, standardized_data):
        """Test robust PCA baseline."""
        encoder = NDGEncoder(baseline_type=BaselineType.ROBUST_PCA, baseline_rank=3)
        encoder.fit(standardized_data)

        assert encoder.params_.baseline_metadata["type"] == "robust_pca"

    def test_falsity_exponential(self, standardized_data):
        """Test exponential falsity map."""
        encoder = NDGEncoder(falsity_map=FalsityMap.EXPONENTIAL)
        triplets = encoder.fit_transform(standardized_data)

        # F should be 1 - exp(-|R|/sigma)
        # Just verify bounds and monotonicity indirectly
        assert triplets.F.min() >= 0.0
        assert triplets.F.max() < 1.0  # Approaches 1 but never reaches

    def test_falsity_logistic(self, standardized_data):
        """Test logistic falsity map."""
        encoder = NDGEncoder(
            falsity_map=FalsityMap.LOGISTIC,
            falsity_params={"a": 2.0, "b": 1.0}
        )
        triplets = encoder.fit_transform(standardized_data)

        assert triplets.F.min() >= 0.0
        assert triplets.F.max() <= 1.0

    def test_hydrofacies_baseline(self, standardized_data):
        """Test hydrofacies-conditioned baseline."""
        n = standardized_data.shape[0]
        groups = np.array([0] * (n // 2) + [1] * (n - n // 2))

        encoder = NDGEncoder(baseline_type=BaselineType.HYDROFACIES_MEDIAN)
        triplets = encoder.fit_transform(standardized_data, groups=groups)

        # Within each group, T should be constant per column
        for g in [0, 1]:
            mask = groups == g
            for j in range(standardized_data.shape[1]):
                assert len(np.unique(triplets.T[mask, j])) == 1

    def test_low_rank_requires_rank(self):
        """Test that low_rank baseline requires baseline_rank."""
        with pytest.raises(ValueError):
            NDGEncoder(baseline_type=BaselineType.LOW_RANK)

    def test_transform_without_fit_raises(self, standardized_data):
        """Test that transform before fit raises error."""
        encoder = NDGEncoder()
        with pytest.raises(RuntimeError):
            encoder.transform(standardized_data)

    def test_string_baseline_type(self, standardized_data):
        """Test that string baseline type works."""
        encoder = NDGEncoder(baseline_type="median")
        encoder.fit(standardized_data)
        assert encoder.baseline_type == BaselineType.MEDIAN

    def test_string_falsity_map(self, standardized_data):
        """Test that string falsity map works."""
        encoder = NDGEncoder(falsity_map="logistic")
        assert encoder.falsity_map == FalsityMap.LOGISTIC


class TestCensoringIndeterminacy:
    """Tests for create_censoring_indeterminacy function."""

    def test_basic_censoring(self):
        """Test basic censoring indeterminacy."""
        X = np.array([[0.5, 2.0], [1.5, 0.3], [2.0, 1.0]])
        detection_limits = np.array([1.0, 0.5])

        I = create_censoring_indeterminacy(X, detection_limits, iota_dl=0.5)

        assert I[0, 0] == 0.5  # Below DL
        assert I[0, 1] == 0.0  # Above DL
        assert I[1, 0] == 0.0  # Above DL
        assert I[1, 1] == 0.5  # Below DL

    def test_custom_iota_dl(self):
        """Test custom iota_dl value."""
        X = np.array([[0.5, 2.0]])
        detection_limits = np.array([1.0, 0.5])

        I = create_censoring_indeterminacy(X, detection_limits, iota_dl=0.8)

        assert I[0, 0] == 0.8
