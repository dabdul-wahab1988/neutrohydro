"""
Tests for the visualization module (requires the optional ``viz`` extras).
"""

import dataclasses

import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from neutrohydro.attribution import nsr_to_dataframe  # noqa: E402
from neutrohydro.pipeline import NeutroHydroPipeline, PipelineConfig  # noqa: E402
from neutrohydro.visualization import (  # noqa: E402
    create_figure,
    plot_vip_decomposition,
)


@pytest.fixture(autouse=True)
def _isolate_matplotlib():
    """Undo the global style changes the plotting functions make, and free figures."""
    with matplotlib.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def results(sample_data, ion_names):
    """Pipeline results fitted like examples/run_pipeline_with_plots.py."""
    X, y = sample_data
    pipeline = NeutroHydroPipeline(PipelineConfig(n_components=3))
    return pipeline.fit(X, y, feature_names=ion_names)


def _panel_texts(ax):
    return [t.get_text() for t in ax.texts]


def _bar_values(ax):
    """Return the tick labels and bar heights of a bar panel, left to right."""
    bars = sorted(ax.patches, key=lambda bar: bar.get_x())
    labels = [label.get_text() for label in ax.get_xticklabels()]
    return labels, [bar.get_height() for bar in bars]


def test_plot_vip_decomposition_renders_pipeline_result(results, tmp_path):
    output = tmp_path / "vip_decomposition.png"

    plot_vip_decomposition(results.nvip, output_path=str(output))

    assert output.stat().st_size > 0


# fig4_attribution is '[v1|v2][v3|v4]', so the baseline fraction panel (v3) is third.
@pytest.mark.parametrize(
    "layout, baseline_panel", [("[v3]", 0), ("fig4_attribution", 2)]
)
def test_vip_figures_with_only_nvip_result_do_not_raise(
    results, layout, baseline_panel, tmp_path
):
    """Regression: NVIPResult has feature names but no pi_G, which made the
    baseline-fraction panel raise "All arrays must be of the same length"."""
    output = tmp_path / "figure.png"

    fig = create_figure(layout, {"nvip_result": results.nvip}, output_path=str(output))

    assert output.stat().st_size > 0
    assert _panel_texts(fig.axes[baseline_panel]) == ["No pi_G data"]


@pytest.mark.parametrize("source", ["nsr_result", "pipeline_results", "nsr_dataframe"])
def test_baseline_fraction_panel_plots_pi_G_per_ion(results, source):
    nsr_input = {
        "nsr_result": results.nsr,
        "pipeline_results": results,
        "nsr_dataframe": nsr_to_dataframe(results.nsr, results.feature_names),
    }[source]

    fig = create_figure("[v3]", {"nvip_result": results.nvip, "nsr_result": nsr_input})

    labels, heights = _bar_values(fig.axes[0])
    expected = dict(zip(results.feature_names, results.nsr.pi_G))
    assert dict(zip(labels, heights)) == pytest.approx(expected)
    assert heights == sorted(heights, reverse=True)


def test_baseline_fraction_panel_with_mismatched_names_shows_placeholder(results):
    nvip_short_names = dataclasses.replace(
        results.nvip, feature_names=results.feature_names[:2]
    )

    fig = create_figure(
        "[v3]", {"nvip_result": nvip_short_names, "nsr_result": results.nsr}
    )

    ax = fig.axes[0]
    assert _panel_texts(ax) == ["No pi_G data"]
    assert not ax.patches


def test_attribution_figure_renders_with_full_pipeline_results(results, tmp_path):
    output = tmp_path / "attribution.png"
    data = {
        "nvip_result": results.nvip,
        "nsr_result": results.nsr,
        "samples_result": results.sample_attribution,
    }

    fig = create_figure("fig4_attribution", data, output_path=str(output))

    assert output.stat().st_size > 0
    for ax in fig.axes:
        assert ax.patches, f"panel '{ax.get_title()}' drew no bars"
        assert not _panel_texts(ax), f"panel '{ax.get_title()}' shows a placeholder"
