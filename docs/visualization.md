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
