"""
Tests for the Jacobian-orthogonal encoding and exact attribution.

Each test checks one of the identities stated in
docs/jacobian_orthogonality.md.
"""

import warnings

import numpy as np
import pytest

from neutrohydro.encoder import NDGEncoder
from neutrohydro.model import PNPLS
from neutrohydro.orthogonal import (
    OrthogonalNDGEncoder,
    select_ranks,
    verify_jacobian_orthogonality,
    orthogonal_attribution,
    orthogonal_nsr,
    orthogonal_sample_attribution,
    subspace_alignment,
    channel_sensitivity,
    scenario_response,
    directions_to_standardized,
    jacobian_entanglement,
)
from neutrohydro.pipeline import NeutroHydroPipeline, PipelineConfig
from neutrohydro.preprocessing import Preprocessor


@pytest.fixture
def factor_data():
    """Two strong latent factors + noise + a few gross outliers."""
    rng = np.random.default_rng(1)
    n, p = 150, 8
    Z = rng.normal(size=(n, 2))
    L = 2.0 * rng.normal(size=(2, p))
    X = Z @ L + 0.5 * rng.normal(size=(n, p))
    X[:6] += 5.0 * rng.normal(size=(6, p))
    y = X[:, 0] - 0.5 * X[:, 4] + 0.3 * X[:, 7] + 0.3 * rng.normal(size=n)
    return X, y, L


@pytest.fixture
def fitted(factor_data):
    X, y, _ = factor_data
    enc = OrthogonalNDGEncoder(rank_T=2, rank_I=1)
    trip = enc.fit_transform(X)
    model = PNPLS(n_components=3).fit(trip, y)
    return X, y, enc, trip, model


class TestRankSelection:
    def test_recovers_two_factors(self, factor_data):
        X, _, _ = factor_data
        sel = select_ranks(X, n_permutations=100, n_bootstrap=100)
        assert sel["rank_observed"] == 2
        assert sel["rank_T"] == 2

    def test_auto_ranks_in_encoder(self, factor_data):
        X, _, _ = factor_data
        enc = OrthogonalNDGEncoder(n_permutations=100, n_bootstrap=100).fit(X)
        assert enc.basis_.rank_T == 2
        assert enc.basis_.rank_I >= 0
        assert enc.basis_.rank_selection is not None

    def test_invalid_ranks_raise(self, factor_data):
        X, _, _ = factor_data
        with pytest.raises(ValueError):
            OrthogonalNDGEncoder(rank_T=6, rank_I=5).fit(X)


class TestJacobianOrthogonality:
    def test_projectors(self, fitted):
        X, _, enc, _, _ = fitted
        chk = verify_jacobian_orthogonality(enc, X)
        assert chk["jacobian_cross"] < 1e-12
        assert chk["completeness"] < 1e-12
        assert chk["idempotence"] < 1e-12

    def test_pythagoras_and_memberships(self, fitted):
        X, _, enc, trip, _ = fitted
        chk = verify_jacobian_orthogonality(enc, X)
        assert chk["pythagoras"] < 1e-9
        m = trip.memberships
        np.testing.assert_allclose(m["tau"] + m["iota"] + m["phi"], 1.0, atol=1e-12)
        assert m["mu_F"].min() >= 0 and m["mu_F"].max() <= 1

    def test_components_reconstruct(self, fitted):
        X, _, enc, trip, _ = fitted
        np.testing.assert_allclose(
            trip.T + trip.I + trip.F, X - enc.basis_.center, atol=1e-12
        )

    def test_sample_space_orthogonality(self, fitted):
        X, _, enc, _, _ = fitted
        assert verify_jacobian_orthogonality(enc, X)["sample_cross_covariance"] < 1e-12

    def test_finite_difference_entanglement(self, fitted):
        X, _, enc, _, _ = fitted
        ent = jacobian_entanglement(enc, X, n_samples=5)
        assert ent["T-F"] < 1e-6 and ent["T-I"] < 1e-6 and ent["I-F"] < 1e-6

    def test_legacy_model_channels_are_entangled(self, fitted):
        X, y, _, _, _ = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc = NDGEncoder("low_rank", baseline_rank=2)
            trip = enc.fit_transform(X)
        model = PNPLS(n_components=3).fit(trip, y)
        ent = jacobian_entanglement(enc, X, model=model, n_samples=5)
        assert ent["T-F"] > 0.1

    def test_median_baseline_is_insensitive(self, fitted):
        X, _, _, _, _ = fitted
        enc = NDGEncoder("median").fit(X)
        assert jacobian_entanglement(enc, X, n_samples=3)["norms"]["T"] == 0.0


class TestExactAttribution:
    def test_prediction_invariance(self, fitted):
        """PNPLS on [T, I, F] == weighted PLS on X (rho = 1)."""
        X, y, enc, trip, model = fitted
        w = trip.sample_weights
        sw = np.sqrt(w)
        mx = (w[:, None] * X).sum(0) / w.sum()
        my = (w * y).sum() / w.sum()
        ref = PNPLS(n_components=3)._nipals_pls1(sw[:, None] * (X - mx), sw * (y - my), 3)
        np.testing.assert_allclose(
            model.predict(trip), (X - mx) @ ref.beta + my, atol=1e-10
        )

    def test_exact_r2_split(self, fitted):
        X, y, _, trip, model = fitted
        at = orthogonal_attribution(model, trip, y)
        assert at.R2 > 0.8
        assert abs(at.R2 - at.R2_channels.sum()) < 1e-10
        assert np.all(at.R2_channels >= 0)

    def test_completeness_and_ion_additivity(self, fitted):
        X, y, _, trip, model = fitted
        at = orthogonal_attribution(model, trip, y)
        np.testing.assert_allclose(
            at.contributions.sum(axis=(1, 2)) + at.intercept,
            model.predict(trip),
            atol=1e-10,
        )
        assert at.checks["ion_additivity"] < 1e-10
        assert at.checks["prediction_cross_energy"] < 1e-10
        np.testing.assert_allclose(at.pi_T + at.pi_I + at.pi_F, 1.0, atol=1e-12)

    def test_sample_G_aggregates_to_r2_share(self, fitted):
        X, y, _, trip, model = fitted
        at = orthogonal_attribution(model, trip, y)
        w = at.sample_weights
        sq = at.channel_predictions ** 2
        agg = np.sum(w * at.sample_G * sq.sum(1)) / np.sum(w * sq.sum(1))
        assert abs(agg - at.R2_channels[0] / at.R2) < 1e-10

    def test_orthogonality_survives_channel_weights(self, factor_data):
        X, y, _ = factor_data
        trip = OrthogonalNDGEncoder(rank_T=2, rank_I=1).fit_transform(X)
        model = PNPLS(n_components=3, rho_I=3.0, rho_F=0.3).fit(trip, y)
        at = orthogonal_attribution(model, trip, y)
        assert abs(at.R2 - at.R2_channels.sum()) < 1e-10

    def test_drop_in_results(self, fitted):
        X, y, _, trip, model = fitted
        at = orthogonal_attribution(model, trip, y)
        nsr = orthogonal_nsr(at, gamma=0.7)
        sa = orthogonal_sample_attribution(at)
        assert nsr.pi_G.shape == (X.shape[1],)
        assert sa.G.shape == (X.shape[0],)
        assert np.all((sa.G >= 0) & (sa.G <= 1))

    def test_requires_orthogonal_triplets(self, fitted):
        X, y, _, _, _ = fitted
        trip = NDGEncoder("low_rank", baseline_rank=2).fit_transform(X)
        model = PNPLS(n_components=2).fit(trip, y)
        with pytest.raises(ValueError):
            orthogonal_attribution(model, trip, y)


class TestOutOfSampleAndAlignment:
    def test_transform_uses_fitted_state(self, fitted):
        X, _, enc, _, _ = fitted
        shift = 3.0 * enc.basis_.basis("F")[:, 0]
        trip = enc.transform(X[:10] + shift)
        base = enc.transform(X[:10])
        np.testing.assert_allclose(trip.F - base.F, np.tile(shift, (10, 1)), atol=1e-10)
        np.testing.assert_allclose(trip.T, base.T, atol=1e-10)

    def test_permutation_equivariance(self, fitted):
        X, _, enc, _, _ = fitted
        perm = np.random.default_rng(0).permutation(len(X))
        np.testing.assert_allclose(enc.transform(X[perm]).F, enc.transform(X).F[perm])

    def test_alignment_sums_to_one(self, fitted):
        X, _, enc, _, _ = fitted
        V = enc.basis_.eigenvectors
        df = subspace_alignment(enc, {"t": V[:, 0], "f": V[:, -1], "mix": V[:, 0] + V[:, -1]})
        np.testing.assert_allclose(df[["T", "I", "F"]].sum(axis=1), 1.0)
        assert df.loc[0, "T"] == pytest.approx(1.0)
        assert df.loc[1, "F"] == pytest.approx(1.0)
        assert df.loc[2, "T"] == pytest.approx(0.5)
        assert list(df["dominant"])[:2] == ["T", "F"]

    def test_sensitivity_is_orthogonal_split(self, fitted):
        X, y, enc, trip, model = fitted
        b = channel_sensitivity(enc, model)
        assert abs(b["T"] @ b["F"]) < 1e-12
        assert np.dot(b["total"], b["total"]) == pytest.approx(
            sum(np.dot(b[c], b[c]) for c in "TIF")
        )
        # finite-difference gradient of the prediction matches
        x0 = X[:1].copy()
        eps = 1e-6
        g = np.array([
            (model.predict(enc.transform(x0 + eps * e[None])) -
             model.predict(enc.transform(x0 - eps * e[None])))[0] / (2 * eps)
            for e in np.eye(X.shape[1])
        ])
        np.testing.assert_allclose(g, b["total"], atol=1e-6)

    def test_scenario_response(self, fitted):
        X, y, enc, trip, model = fitted
        delta = np.eye(X.shape[1])[:2]
        sc = scenario_response(enc, model, delta)
        np.testing.assert_allclose(sc["by_channel"].sum(axis=1), sc["response"])
        pred = model.predict(enc.transform(X[:1] + delta[:1])) - model.predict(enc.transform(X[:1]))
        assert sc["response"][0] == pytest.approx(pred[0])

    def test_directions_to_standardized_log_jacobian(self):
        rng = np.random.default_rng(0)
        X = rng.lognormal(1.0, 0.5, size=(50, 4))
        y = rng.normal(size=50)
        pre = Preprocessor(log_transform=True).fit(X, y)
        c0 = np.exp(pre.params_.mu) - pre.params_.delta_x
        d = np.array([[1.0, 0.0, 0.5, 0.0]])
        eps = 1e-6
        fd = (pre.transform((c0 + eps * d))[0] - pre.transform(c0[None])[0]) / eps
        np.testing.assert_allclose(directions_to_standardized(d, pre), fd, rtol=1e-4)


class TestPipelineIntegration:
    def test_orthogonal_pipeline(self, sample_data, ion_names):
        X, y = sample_data
        pipe = NeutroHydroPipeline(PipelineConfig(
            baseline_type="orthogonal", baseline_rank=None, n_components=3,
        ))
        res = pipe.fit(X, y, feature_names=ion_names)
        assert res.orthogonal is not None
        assert abs(res.orthogonal.R2 - res.orthogonal.R2_channels.sum()) < 1e-10
        assert res.nsr.pi_G.shape == (len(ion_names),)
        y_new = pipe.predict(X[:5])
        np.testing.assert_allclose(y_new, res.y_pred_original[:5], atol=1e-8)
        summary = pipe.get_summary()
        assert "orthogonal" in summary
        dfs = pipe.to_dataframes()
        assert "orthogonal_ions" in dfs and "orthogonal_samples" in dfs


def test_nvip_bootstrap_keeps_orthogonal_encoding(fitted, monkeypatch):
    from neutrohydro.nvip import compute_nvip_bootstrap

    X, y, _, trip, model = fitted
    seen = []
    original_fit = PNPLS.fit

    def spy(self, triplets, y, *args, **kwargs):
        seen.append((triplets.encoding, triplets.sample_weights is not None))
        return original_fit(self, triplets, y, *args, **kwargs)

    monkeypatch.setattr(PNPLS, "fit", spy)
    out = compute_nvip_bootstrap(model, trip, y, n_bootstrap=5, random_state=0)
    assert len(seen) == 5
    assert all(enc == "orthogonal" and has_w for enc, has_w in seen)
    assert np.all(np.isfinite(out["VIP_agg_mean"]))
