# Visualization Module

The `neutrohydro` package includes a comprehensive visualization module designed to generate publication-quality hydrogeochemical plots. This module is built on Matplotlib and Seaborn and is designed to be accessible even for users with limited coding experience.

## Key Features

- **Gibbs Diagrams**: For identifying dominant hydrogeochemical processes (precipitation, rock weathering, evaporation).
- **ILR Water Classification**: A 2x2 panel plot for compositional data visualization and water type classification using Isometric Log-Ratios.
- **Correlation Matrices**: Professional heatmaps for visualizing ion relationships.
- **Mineral Analysis Plots**: Stacked bar charts for mineral fractions and horizontal bar charts for Saturation Indices (SI).
- **VIP Decomposition**: Visualizing the contribution of Truth (T), Indeterminacy (I), and Falsity (F) channels to overall variable importance.

## Usage

### Simple Report Generation

For non-coders, the `generate_report` function is the easiest way to generate all standard plots with a single command.

```python
import pandas as pd
from neutrohydro import generate_report

# Load your data (mg/L)
df = pd.read_csv("your_data.csv")

# Generate all standard plots
generate_report(df, output_dir="results/figures")
```

### Individual Plots

You can also generate individual plots for more customization.

```python
from neutrohydro import plot_gibbs, plot_ilr_classification, plot_correlation_matrix

# Gibbs Plot
fig_gibbs = plot_gibbs(df, output_path="gibbs.png")

# ILR Classification
fig_ilr = plot_ilr_classification(df, output_path="ilr.png")

# Correlation Matrix
fig_corr = plot_correlation_matrix(df, output_path="correlation.png")
```

## Plot Registry and Custom Layouts

NeutroHydro uses a **Plot Registry** system that allows you to compose custom multi-panel figures using a simple naming convention.

| Code | Name | Description |
| :--- | :--- | :--- |
| `g1` | Gibbs Cations | Na vs TDS |
| `g2` | Gibbs Anions | Cl vs TDS |
| `i1` | ILR Class. | Full 2x2 water classification figure |
| `h1` | Correlation | lower-triangle correlation heatmap |
| `m1` | Min. Fractions | Stacked bar chart of mineral contributions |
| `m2` | Sat. Indices | Bar chart of mineral saturation indices |
| `v1` | VIP Decomp. | T/I/F channel decomposition for NVIP |

### Custom Composition

You can use the `create_figure` function with a grid-based string syntax. Brackets `[]` define rows, and pipes `|` separate columns.

```python
from neutrohydro import create_figure

# Create a figure with Gibbs Anions on the left and Correlation Matrix on the right
data = {'df': df}
fig = create_figure(layout="[g2|h1]", data=data, output_path="custom_fig.png")
```

## Data Requirements

For most plots, your DataFrame should contain columns for major ions in **mg/L**:
- `Ca`, `Mg`, `Na`, `K`, `HCO3`, `Cl`, `SO4`
- `TDS` (Total Dissolved Solids)
- Optional: `NO3`, `F`, `pH`, `Eh`

The module automatically converts units to **meq/L** where necessary for calculations like ionic balance and ratios.

## Exporting for Publication

All plots are generated with publication-quality defaults:
- **DPI**: 300
- **Format**: PNG (default), supports PDF, SVG, etc.
- **Styling**: Consistent fonts, proper LaTeX-style labels (e.g., $Ca^{2+}$), and clean grids.

To apply the NeutroHydro style to your own Matplotlib plots:

```python
from neutrohydro.visualization import apply_publication_style
apply_publication_style()
```

## Jacobian-Orthogonal Attribution Plots

When the pipeline runs with `baseline_type="orthogonal"`, four plots show the exact
attribution described in [Jacobian Orthogonality](jacobian_orthogonality.md). All four use
the same channel colours: **T (baseline)** blue, **I (indeterminate)** aqua and
**F (perturbation)** orange. These are the first three slots of a palette checked for
colour-vision deficiency. The aqua has low contrast on white, so each plot also carries
legends or direct labels. Every function returns a Matplotlib `Figure`, accepts an
optional `ax` to draw into an existing figure, and saves at 300 DPI when given
`output_path`.

```python
from neutrohydro import NeutroHydroPipeline
from neutrohydro.pipeline import PipelineConfig
from neutrohydro.orthogonal import subspace_alignment, directions_to_standardized
from neutrohydro.visualization import (
    plot_neutrosophic_ternary,
    plot_exact_r2_split,
    plot_ion_channel_shares,
    plot_process_alignment,
)

res = NeutroHydroPipeline(PipelineConfig(baseline_type="orthogonal", baseline_rank=None)).fit(
    X, y, feature_names=ion_names
)
at = res.orthogonal

# 1. Each sample's (tau, iota, phi) memberships in a ternary diagram. tau + iota + phi = 1
#    exactly, so every sample is a point. Corners: T bottom-left, F bottom-right,
#    I at the top. Points are coloured by dominant channel, or by a numeric variable
#    on a one-hue scale with a colorbar.
plot_neutrosophic_ternary(at, labels=sample_ids, max_labels=10, output_path="fig/ternary.png")
plot_neutrosophic_ternary(at, color_by=tds, color_label="TDS (mg/L)")

# 2. The exact R^2 split, R^2 = R^2_T + R^2_I + R^2_F, as one stacked bar.
#    The unexplained part is shown in grey.
plot_exact_r2_split(at, output_path="fig/r2_split.png")

# 3. Channel shares of each ion's predictive energy (pi_T, pi_I, pi_F; each row sums
#    to 1). Ions are sorted by pi_T; dashed guides mark gamma and 1 - gamma.
plot_ion_channel_shares(at, gamma=0.7, output_path="fig/ion_shares.png")

# 4. Process alignment: log2(T enrichment / F enrichment) for mineral or end-member
#    directions. Bars to the right lean towards the baseline subspace, bars to the left
#    towards the perturbation subspace, and 0 means no preference compared with a random
#    direction. Processes whose dominant channel is I are tagged "(I)".
#    Use metric="T" to plot log2(T enrichment) against chance instead.
directions = directions_to_standardized(A.T, res.preprocessor)  # A: (ions x processes)
alignment = subspace_alignment(res.encoder, directions, names=process_names)
plot_process_alignment(alignment, output_path="fig/process_alignment.png")
```

| Function | Input | Shows |
|----------|-------|-------|
| `plot_neutrosophic_ternary` | `OrthogonalAttribution` or `tau`, `iota`, `phi` arrays | Per-sample memberships on the T–I–F simplex |
| `plot_exact_r2_split` | `OrthogonalAttribution` computed with the target | $R^2_T + R^2_I + R^2_F$ and the unexplained remainder |
| `plot_ion_channel_shares` | `OrthogonalAttribution` | Exact per-ion channel shares $\pi_T, \pi_I, \pi_F$ |
| `plot_process_alignment` | DataFrame from `subspace_alignment` | Which processes align with the baseline or perturbation subspace |
