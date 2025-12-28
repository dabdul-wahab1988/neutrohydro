"""
Tests for the PNPLS model module.
"""

import numpy as np
import pytest

from neutrohydro.preprocessing import Preprocessor
from neutrohydro.encoder import NDGEncoder
from neutrohydro.model import PNPLS, PLSComponents


class TestPNPLS:
    """Tests for PNPLS class."""

    @pytest.fixture
    def prepared_data(self, sample_data):
        """Get preprocessed and encoded data for model tests."""
        X, y = sample_data
        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)
        return triplets, y_std

    def test_fit_basic(self, prepared_data):
        """Test basic fitting."""
        triplets, y = prepared_data
        model = PNPLS(n_components=3)
        model.fit(triplets, y)

        assert model.is_fitted_
        assert model.components_ is not None
        assert model.components_.n_components <= 3

    def test_components_structure(self, prepared_data):
        """Test that components have correct structure."""
        triplets, y = prepared_data
        n = triplets.T.shape[0]
        p = triplets.T.shape[1]

        model = PNPLS(n_components=3)
        model.fit(triplets, y)

        components = model.components_
        k = components.n_components

        assert components.T.shape == (n, k)
        assert components.W.shape == (3 * p, k)
        assert components.P.shape == (3 * p, k)
        assert len(components.q) == k
        assert len(components.beta) == 3 * p

    def test_predict_shape(self, prepared_data):
        """Test that predictions have correct shape."""
        triplets, y = prepared_data
        model = PNPLS(n_components=3)
        model.fit(triplets, y)

        y_pred = model.predict(triplets)
        assert y_pred.shape == y.shape

    def test_score_range(self, prepared_data):
        """Test that R2 score is in valid range."""
        triplets, y = prepared_data
        model = PNPLS(n_components=5)
        model.fit(triplets, y)

        r2 = model.score(triplets, y)
        # R2 should be positive for training data with reasonable fit
        assert r2 > 0.0
        assert r2 <= 1.0

    def test_get_coefficients(self, prepared_data):
        """Test coefficient partitioning."""
        triplets, y = prepared_data
        p = triplets.T.shape[1]

        model = PNPLS(n_components=3)
        model.fit(triplets, y)

        coeffs = model.get_coefficients()

        assert "beta_T" in coeffs
        assert "beta_I" in coeffs
        assert "beta_F" in coeffs
        assert len(coeffs["beta_T"]) == p
        assert len(coeffs["beta_I"]) == p
        assert len(coeffs["beta_F"]) == p

    def test_get_weights_by_channel(self, prepared_data):
        """Test weight partitioning."""
        triplets, y = prepared_data
        p = triplets.T.shape[1]

        model = PNPLS(n_components=3)
        model.fit(triplets, y)

        weights = model.get_weights_by_channel()

        assert "W_T" in weights
        assert "W_I" in weights
        assert "W_F" in weights

        k = model.components_.n_components
        assert weights["W_T"].shape == (p, k)

    def test_channel_weights(self, prepared_data):
        """Test different channel weight configurations."""
        triplets, y = prepared_data

        # Default weights
        model1 = PNPLS(n_components=3, rho_I=1.0, rho_F=1.0)
        model1.fit(triplets, y)
        r2_1 = model1.score(triplets, y)

        # Higher I weight
        model2 = PNPLS(n_components=3, rho_I=2.0, rho_F=1.0)
        model2.fit(triplets, y)
        r2_2 = model2.score(triplets, y)

        # Both should fit
        assert r2_1 > 0.0
        assert r2_2 > 0.0

    def test_lambda_f_effect(self, prepared_data):
        """Test falsity weighting strength."""
        triplets, y = prepared_data

        # Low lambda_F (less downweighting)
        model1 = PNPLS(n_components=3, lambda_F=0.1)
        model1.fit(triplets, y)

        # High lambda_F (more downweighting)
        model2 = PNPLS(n_components=3, lambda_F=5.0)
        model2.fit(triplets, y)

        # Coefficients should differ
        assert not np.allclose(model1.components_.beta, model2.components_.beta)

    def test_n_components_auto_limit(self, small_data):
        """Test that n_components is limited by data size."""
        X, y = small_data
        n, p = X.shape

        preprocessor = Preprocessor()
        X_std, y_std = preprocessor.fit_transform(X, y)
        encoder = NDGEncoder()
        triplets = encoder.fit_transform(X_std)

        # Request more components than possible
        model = PNPLS(n_components=100)
        model.fit(triplets, y_std)

        # Should be limited
        assert model.components_.n_components <= min(n, 3 * p)

    def test_predict_without_fit_raises(self, prepared_data):
        """Test that predict before fit raises error."""
        triplets, y = prepared_data
        model = PNPLS()

        with pytest.raises(RuntimeError):
            model.predict(triplets)

    def test_explained_variance(self, prepared_data):
        """Test explained variance per component."""
        triplets, y = prepared_data
        model = PNPLS(n_components=5)
        model.fit(triplets, y)

        ev = model.components_.explained_variance
        k = model.components_.n_components

        assert len(ev) == k
        assert all(e >= 0 for e in ev)
        # First components should explain more variance
        # (not always true, but generally expected)

    def test_reproducibility(self, prepared_data):
        """Test that fitting is reproducible."""
        triplets, y = prepared_data

        model1 = PNPLS(n_components=3)
        model1.fit(triplets, y)

        model2 = PNPLS(n_components=3)
        model2.fit(triplets, y)

        np.testing.assert_array_almost_equal(
            model1.components_.beta,
            model2.components_.beta
        )
