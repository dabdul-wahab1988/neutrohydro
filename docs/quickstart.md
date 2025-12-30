# Quick Start Guide

This guide will help you run your first analysis using NeutroHydro.

## 1. Basic Workflow

The core of NeutroHydro is the `NeutroHydroPipeline`. It handles preprocessing, encoding, model training, and mineral inversion in a single step.

### Step 1: Prepare Your Data

Prepare a CSV file (e.g., `data.csv`) with your ion concentrations. The columns should match standard chemical symbols (e.g., `Ca`, `Mg`, `Na`, `HCO3`, `Cl`, `SO4`).

| SampleID | Ca | Mg | Na | K | HCO3 | Cl | SO4 | NO3 |
|----------|----|----|----|---|------|----|-----|-----|
| S1       | 45 | 12 | 25 | 3 | 150  | 30 | 40  | 5   |
| S2       | 80 | 25 | 60 | 5 | 200  | 85 | 90  | 12  |

### Step 2: Run the Pipeline

Create a Python script (e.g., `analysis.py`):

```python
import pandas as pd
from neutrohydro import NeutroHydroPipeline

# 1. Load Data
df = pd.read_csv("data.csv")

# 2. Initialize Pipeline
pipeline = NeutroHydroPipeline()

# 3. Fit the Model with Thermodynamic Validation
from neutrohydro.pipeline import PipelineConfig

# Prepare data
feature_cols = ["Ca", "Mg", "Na", "K", "HCO3", "Cl", "SO4", "NO3"]
X = df[feature_cols].values
y = df["TDS"].values  # Example target
pH = df["pH"].values
Eh = df["Eh"].values

# Optional: Configure thermodynamic validation
config = PipelineConfig(run_thermodynamic_validation=True)
pipeline = NeutroHydroPipeline(config=config)

# Prepare ion data for inversion (in meq/L)
c_meq = df[feature_cols].values # Simplified

# Fit with ion data and redox parameters
pipeline.fit(X, y, c_meq=c_meq, pH=pH, Eh=Eh)

# 4. Get Results
results = pipeline.analyze(df)

# 5. Access Specific Outputs
print("Variable Importance (VIP):")
print(results["vip_scores"])

print("\nMineral Contributions (First Sample):")
print(results["mineral_fractions"].iloc[0])

print("\nWater Quality Flags:")
print(results["quality_flags"].iloc[0])
```

## 2. Advanced Features

### Mineral Inversion with Quality Constraints

NeutroHydro can use water quality flags (like WHO exceedances) to constrain the mineral inversion.

```python
# The pipeline does this automatically if you use the .analyze() method.
# If thermodynamic validation is enabled, 'Thermo-Plausible' flags are added.

quality_df = results["quality_flags"]
print(results["mineral_fractions"]) # Includes SI and Thermo-Plausibility
```

### Hydrogeochemical Indices

The analysis also calculates standard indices automatically:

```python
indices = results["indices"]
print(indices[["Simpson_Class", "Freshening_Ratio", "BEX", "Gibbs_Ratio_1"]].head())
```

## 3. Visualization

You can quickly visualize the results using standard libraries.

```python
import matplotlib.pyplot as plt

# Plot Mineral Fractions for the first 5 samples
minerals = results["mineral_fractions"].head(5)
minerals.plot(kind="bar", stacked=True)
plt.title("Mineral Composition")
plt.ylabel("Fraction")
plt.show()
```

## Next Steps

- Learn about the [Mathematical Framework](mathematical_framework.md).
- Explore [Mineral Inversion](minerals.md) details.
- Check the [API Reference](api_pipeline.md) for full documentation.
