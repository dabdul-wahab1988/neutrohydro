"""
Tests for the Jacobian-orthogonal attribution plots in neutrohydro.visualization.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402

from neutrohydro.pipeline import NeutroHydroPipeline, PipelineConfig  # noqa: E402
from neutrohydro.orthogonal import subspace_alignment  # noqa: E402
from neutrohydro.visualization import (  # noqa: E402
    plot_neutrosophic_ternary,
    plot_exact_r2_split,
    plot_ion_channel_shares,
    plot_process_alignment,
)


@pytest.fixture
def ortho_results(sample_data, ion_names):
    X, y = sample_data
    pipe = NeutroHydroPipeline(PipelineConfig(
        baseline_type="orthogonal", baseline_rank=2, indeterminacy_rank=1, n_components=3,
    ))
    return pipe.fit(X, y, feature_names=ion_names)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _n_points(ax):
    return sum(len(c.get_offsets()) for c in ax.collections if isinstance(c, PathCollection))


class TestTernary:
    def test_from_attribution(self, ortho_results):
        at = ortho_results.orthogonal
        fig = plot_neutrosophic_ternary(at)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert _n_points(fig.axes[0]) == len(at.tau)

    def test_color_by_and_labels(self, ortho_results):
        at = ortho_results.orthogonal
        n = len(at.tau)
        fig = plot_neutrosophic_ternary(
            at, color_by=np.arange(n, dtype=float), color_label="index",
            labels=[f"S{i}" for i in range(n)], max_labels=5,
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert _n_points(fig.axes[0]) == n
        assert len(fig.axes) == 2  # main axes + colorbar
        annotations = [t for t in fig.axes[0].texts if t.get_text().startswith("S")]
        assert len(annotations) == 5

    def test_explicit_memberships_and_ax(self):
        fig, ax = plt.subplots()
        tau = np.array([1.0, 0.0, 0.0, 0.2])
        iota = np.array([0.0, 1.0, 0.0, 0.3])
        phi = np.array([0.0, 0.0, 1.0, 0.5])
        out = plot_neutrosophic_ternary(tau=tau, iota=iota, phi=phi, ax=ax)
        assert out is fig
        assert _n_points(ax) == 4

    def test_missing_memberships_raise(self):
        with pytest.raises(ValueError):
            plot_neutrosophic_ternary(tau=np.ones(3))

    def test_saves_file(self, ortho_results, tmp_path):
        path = tmp_path / "ternary.png"
        plot_neutrosophic_ternary(ortho_results.orthogonal, output_path=str(path))
        assert path.exists() and path.stat().st_size > 0


class TestBars:
    def test_exact_r2_split(self, ortho_results):
        fig = plot_exact_r2_split(ortho_results.orthogonal)
        assert isinstance(fig, matplotlib.figure.Figure)
        # three channels + the unexplained remainder
        assert len(fig.axes[0].patches) == 4

    def test_r2_split_requires_target(self, ortho_results):
        from neutrohydro.orthogonal import orthogonal_attribution

        at = orthogonal_attribution(ortho_results.model, ortho_results.triplets)  # no y
        with pytest.raises(ValueError):
            plot_exact_r2_split(at)

    def test_ion_channel_shares(self, ortho_results, ion_names):
        fig = plot_ion_channel_shares(ortho_results.orthogonal)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes[0].patches) == 3 * len(ion_names)

    def test_ion_channel_shares_on_given_ax(self, ortho_results):
        fig, ax = plt.subplots()
        assert plot_ion_channel_shares(ortho_results.orthogonal, ax=ax, gamma=None) is fig


class TestProcessAlignment:
    def test_process_alignment(self, ortho_results):
        enc = ortho_results.encoder
        p = enc.basis_.center.shape[0]
        V = enc.basis_.eigenvectors
        directions = {
            "baseline-like": V[:, 0],
            "perturbation-like": V[:, -1],
            "ion 0": np.eye(p)[0],
        }
        df = subspace_alignment(enc, directions)
        fig = plot_process_alignment(df)
        assert isinstance(fig, matplotlib.figure.Figure)
        ax = fig.axes[0]
        assert len(ax.patches) == 3
        widths = {lab.get_text(): patch.get_width()
                  for lab, patch in zip(ax.get_yticklabels(), ax.patches)}
        assert widths["baseline-like"] > 0 > widths["perturbation-like"]

    def test_metric_T(self, ortho_results):
        enc = ortho_results.encoder
        df = subspace_alignment(enc, {"a": enc.basis_.eigenvectors[:, 0]})
        fig = plot_process_alignment(df, metric="T")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_bad_metric_raises(self, ortho_results):
        enc = ortho_results.encoder
        df = subspace_alignment(enc, {"a": enc.basis_.eigenvectors[:, 0]})
        with pytest.raises(ValueError):
            plot_process_alignment(df, metric="nope")
