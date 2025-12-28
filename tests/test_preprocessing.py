"""
Tests for the preprocessing module.
"""

import numpy as np
import pytest

from neutrohydro.preprocessing import Preprocessor, handle_missing_data


class TestPreprocessor:
    """Tests for Preprocessor class."""

    def test_fit_basic(self, sample_data):
        """Test basic fitting."""
        X, y = sample_data
        preprocessor = Preprocessor()
        preprocessor.fit(X, y)

        assert preprocessor.is_fitted_
        assert preprocessor.params_ is not None
        assert len(preprocessor.params_.mu) == X.shape[1]
        assert len(preprocessor.params_.s) == X.shape[1]

    def test_transform_shape(self, sample_data):
        """Test that transform preserves shape."""
        X, y = sample_data
        preprocessor = Preprocessor()
        preprocessor.fit(X, y)

        X_std, y_std = preprocessor.transform(X, y)

        assert X_std.shape == X.shape
        assert y_std.shape == y.shape

    def test_fit_transform_same_as_separate(self, sample_data):
        """Test fit_transform gives same result as fit then transform."""
        X, y = sample_data

        prep1 = Preprocessor()
        X_std1, y_std1 = prep1.fit_transform(X, y)

        prep2 = Preprocessor()
        prep2.fit(X, y)
        X_std2, y_std2 = prep2.transform(X, y)

        np.testing.assert_array_almost_equal(X_std1, X_std2)
        np.testing.assert_array_almost_equal(y_std1, y_std2)

    def test_standardization_properties(self, sample_data):
        """Test that standardized data has expected properties."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)

        # Median should be approximately 0 (within tolerance)
        medians = np.median(X_std, axis=0)
        assert np.allclose(medians, 0, atol=0.01)

    def test_log_transform(self, sample_data):
        """Test log transform option."""
        X, y = sample_data
        preprocessor = Preprocessor(log_transform=True)
        preprocessor.fit(X, y)

        assert preprocessor.params_.log_transform is True

        X_std, _ = preprocessor.transform(X)
        # Log transform should reduce the range
        assert X_std.max() - X_std.min() < X.max() - X.min()

    def test_inverse_transform_y(self, sample_data):
        """Test that inverse transform recovers original scale."""
        X, y = sample_data
        preprocessor = Preprocessor()
        preprocessor.fit(X, y)

        X_std, y_std = preprocessor.transform(X, y)
        y_recovered = preprocessor.inverse_transform_y(y_std)

        np.testing.assert_array_almost_equal(y, y_recovered, decimal=10)

    def test_feature_names(self, sample_data, ion_names):
        """Test feature names are stored."""
        X, y = sample_data
        preprocessor = Preprocessor()
        preprocessor.fit(X, y, feature_names=ion_names)

        assert preprocessor.params_.feature_names == ion_names

    def test_transform_without_fit_raises(self, sample_data):
        """Test that transform before fit raises error."""
        X, y = sample_data
        preprocessor = Preprocessor()

        with pytest.raises(RuntimeError):
            preprocessor.transform(X)

    def test_invalid_input_shape(self):
        """Test that invalid input shapes raise errors."""
        preprocessor = Preprocessor()

        # 1D array should fail
        with pytest.raises(ValueError):
            preprocessor.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))


class TestHandleMissingData:
    """Tests for handle_missing_data function."""

    def test_no_missing_data(self):
        """Test with no missing data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_filled, M = handle_missing_data(X)

        np.testing.assert_array_equal(X, X_filled)
        assert M.all()

    def test_missing_data_median_fill(self):
        """Test median filling of missing data."""
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        X_filled, M = handle_missing_data(X, fill_method="median")

        assert not np.isnan(X_filled).any()
        assert not M[0, 1]  # This was missing
        assert X_filled[0, 1] == 5.0  # Median of [4, 6]

    def test_missing_data_zero_fill(self):
        """Test zero filling of missing data."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        X_filled, M = handle_missing_data(X, fill_method="zero")

        assert X_filled[0, 1] == 0.0

    def test_detection_limit_handling(self):
        """Test detection limit handling."""
        X = np.array([[0.5, 2.0], [1.5, 0.3], [2.0, 1.0]])
        detection_limits = np.array([1.0, 0.5])

        X_filled, M = handle_missing_data(
            X, detection_limits=detection_limits, fill_method="dl_half"
        )

        # Values below DL should be marked as missing
        assert not M[0, 0]  # 0.5 < 1.0
        assert M[1, 0]      # 1.5 >= 1.0
        assert not M[1, 1]  # 0.3 < 0.5

    def test_dl_half_fill(self):
        """Test DL/2 filling method."""
        X = np.array([[0.5, 2.0], [1.5, 0.3]])
        detection_limits = np.array([1.0, 0.5])

        X_filled, M = handle_missing_data(
            X, detection_limits=detection_limits, fill_method="dl_half"
        )

        assert X_filled[0, 0] == 0.5  # DL/2 = 1.0/2
        assert X_filled[1, 1] == 0.25  # DL/2 = 0.5/2
