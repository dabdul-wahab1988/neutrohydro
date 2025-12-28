# NeutroHydro

## Neutrosophic Chemometrics for Groundwater Analysis

NeutroHydro implements a mathematically well-posed workflow for groundwater chemometrics in absolute concentration space:

**NDG encoder → PNPLS regression → NVIP decomposition → NSR/$π_G$ attribution → mineral plausibility via stoichiometric inversion**

## Features

- **NDG Encoder**: Maps scalar ion concentrations to neutrosophic triplets (T, I, F)
  - T (Truth): Baseline component via robust operators (median, low-rank, robust PCA)
  - I (Indeterminacy): Uncertainty/ambiguity channel
  - F (Falsity): Perturbation likelihood from residuals

- **PNPLS**: Probabilistic Neutrosophic PLS regression in augmented Hilbert space
  - Combines T, I, F channels with configurable weights
  - Elementwise precision weighting from falsity
  - EM-like imputation for missing data

- **NVIP**: Neutrosophic Variable Importance in Projection
  - Channel-wise VIP decomposition ($VIP_T, VIP_I, VIP_F$)
  - L2 decomposition theorem: $VIP_{agg}^2 = VIP_T^2 + VIP_I^2 + VIP_F^2$

- **NSR/$π_G$ Attribution**: Baseline vs perturbation analysis
  - Neutrosophic Source Ratio (odds)
  - Baseline fraction per ion and per sample

- **Mineral Inference**: Stoichiometric inversion via weighted NNLS
  - Standard mineral library included
  - Custom mineral definitions supported

## Installation

```bash
pip install neutrohydro
```

Or install from source:

```bash
git clone https://github.com/dabdul-wahab1988/neutrohydro.git
cd neutrohydro
pip install -e .
```

## Quick Start

```python
import numpy as np
from neutrohydro import NeutroHydroPipeline

# Prepare your data
X = ...  # Ion concentrations (n_samples, n_ions)
y = ...  # Target (e.g., log TDS)
ion_names = ["Ca2+", "Mg2+", "Na+", "K+", "HCO3-", "Cl-", "SO42-"]

# Run the pipeline
pipeline = NeutroHydroPipeline()
results = pipeline.fit(X, y, feature_names=ion_names)

# Access results
print(f"R²: {results.r2_train:.4f}")
print(f"VIP_agg: {results.nvip.VIP_agg}")
print(f"π_G: {results.nsr.pi_G}")
print(f"G (sample baseline): {results.sample_attribution.G}")
```

## Command Line Interface

```bash
# Run pipeline on CSV data
neutrohydro run data.csv --target TDS --output results/

# Interactive mode
neutrohydro-pipeline

# Show package info
neutrohydro info
```

## Detailed Usage

### Step-by-Step Workflow

```python
from neutrohydro import (
    Preprocessor,
    NDGEncoder,
    PNPLS,
    compute_nvip,
    compute_nsr,
    compute_sample_baseline_fraction,
)

# 1. Preprocessing
preprocessor = Preprocessor(log_transform=False)
X_std, y_std = preprocessor.fit_transform(X, y)

# 2. NDG Encoding
encoder = NDGEncoder(baseline_type="median")
triplets = encoder.fit_transform(X_std)

# 3. PNPLS Regression
model = PNPLS(n_components=5)
model.fit(triplets, y_std)

# 4. NVIP Computation
nvip = compute_nvip(model)

# 5. NSR / π_G
nsr = compute_nsr(nvip, gamma=0.7)

# 6. Sample Attribution
sample_attr = compute_sample_baseline_fraction(model, triplets, nsr)
```

### Mineral Inference

```python
from neutrohydro import MineralInverter

# Create inverter with standard minerals
inverter = MineralInverter()

# Invert ion data (in meq/L)
c_meq = ...  # (n_samples, 9) for standard ions
result = inverter.invert(c_meq, pi_G=nsr.pi_G)

# Access results
print(result.mineral_fractions)  # Normalized contributions
print(result.plausible)          # Plausibility mask
```

### Configuration Options

```python
from neutrohydro.pipeline import PipelineConfig

config = PipelineConfig(
    # Preprocessing
    log_transform=False,

    # NDG Encoder
    baseline_type="median",  # or "low_rank", "robust_pca"
    baseline_rank=None,      # Required for low_rank/robust_pca
    falsity_map="exponential",  # or "logistic"

    # PNPLS
    n_components=5,
    rho_I=1.0,  # Indeterminacy channel weight
    rho_F=1.0,  # Falsity channel weight
    lambda_F=1.0,  # Falsity weighting strength

    # Attribution
    gamma=0.7,  # Classification threshold

    # Minerals
    run_mineral_inference=False,
)
```

## Mathematical Foundation

### NDG Triplets

For standardized predictor $X^{(std)}$:

- **Truth**: $X_T = \mathcal{B}(X^{(std)})$ (baseline operator)
- **Residual**: $R = X^{(std)} - X_T$
- **Falsity**: $F = 1 - \exp(-|R|/\sigma)$ (perturbation likelihood)
- **Indeterminacy**: $I$ (uncertainty/ambiguity)

### Augmented Space

$$X^{aug} = [X_T \quad \sqrt{\rho_I}X_I \quad \sqrt{\rho_F}X_F]$$

### NVIP L2 Decomposition

$$VIP_{agg}^2(j) = VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)$$

### Baseline Fraction

$$\pi_G(j) = \frac{E_T(j)}{E_T(j) + E_P(j)}$$

where $E_T = VIP_T^2$ and $E_P = VIP_I^2 + VIP_F^2$.

## Testing

```bash
pip install -e .[dev]
pytest tests/ -v
```
# NeutroHydro Documentation

## Neutrosophic Chemometrics for Groundwater Analysis

## Table of Contents

### Getting Started

- [Quick Start Guide](quickstart.md)
- [Installation](installation.md)

### Mathematical Foundations

- [Mathematical Framework Overview](mathematical_framework.md)
- [Preprocessing & Robust Scaling](preprocessing.md)
- [NDG Encoder: Neutrosophic Triplets](encoder.md)
- [PNPLS: Probabilistic Neutrosophic PLS](model.md)
- [NVIP: Variable Importance Decomposition](nvip.md)
- [Attribution: NSR and Baseline Fractions](attribution.md)
- [Mineral Stoichiometric Inversion](minerals.md)
- [Water Quality Assessment](quality_check.md)
- [Model Limitations & Validity](limitations.md)
- [Hydrogeochemical Processes](hydrogeochemical_processes.md): Mixing, Exchange, Redox
- [Mathematical Critique](mathematical_critique.md): Rigorous review of potential issues
- [Final Critical Review](final_critical_review.md): "Red Team" analysis of validity

### API Reference

- [Pipeline API](api_pipeline.md)
- [Core Modules API](api_modules.md)

### Examples & Tutorials

- [Basic Usage Example](examples_basic.md)
- [Advanced Workflows](examples_advanced.md)
- [Interpreting Results](interpreting_results.md)

## Overview

NeutroHydro implements a mathematically well-posed workflow for groundwater chemometrics in **absolute concentration space** (non-compositional):

```text
Raw Ion Data
     ↓
Preprocessing (Robust centering/scaling)
     ↓
NDG Encoder (T, I, F triplets)
     ↓
PNPLS Regression (Augmented Hilbert space)
     ↓
NVIP (Channel-wise variable importance)
     ↓
NSR/π_G (Baseline vs perturbation attribution)
     ↓
Mineral Inference (Stoichiometric inversion)
```

## Core Mathematical Innovations

### 1. Neutrosophic Data Representation

Maps each ion concentration to a triplet **(T, I, F)**:

- **T (Truth)**: Baseline/reference component
- **I (Indeterminacy)**: Uncertainty/ambiguity
- **F (Falsity)**: Perturbation likelihood

### 2. L2-Additive VIP Decomposition

**Theorem**: Variable importance decomposes additively across channels:

$$VIP_{agg}^2(j) = VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)$$

This allows **unambiguous attribution** of prediction importance to baseline vs. perturbation sources.

### 3. Non-Compositional Framework

Unlike compositional data analysis (CoDa), NeutroHydro operates in **absolute concentration space**, preserving:

- Physical interpretability
- Additive mixing models
- Direct stoichiometric constraints

### 4. Hybrid Geochemical-Statistical Engine

Combines rigorous mathematical optimization with expert hydrogeochemical heuristics:

- **Context-Aware Inversion**: Uses **WHO Quality Flags** and **Gibbs Diagrams** to dynamically constrain the mineral solver.
- **Redox Detection**: Explicitly solves for mass loss (e.g., Denitrification) using negative stoichiometry.
- **Advanced Indices**: Integrated **Simpson's Ratio** (Standard & Inverse) for precise salinity diagnosis (Seawater vs. Recharge).

## Quick Navigation

**For Users:**

- New to NeutroHydro? → [Quick Start Guide](quickstart.md)
- Need to understand results? → [Interpreting Results](interpreting_results.md)
- Looking for examples? → [Basic Examples](examples_basic.md)

**For Researchers:**

- Mathematical theory? → [Mathematical Framework](mathematical_framework.md)
- Specific module details? → See individual module docs
- Implementation details? → [API Reference](api_modules.md)

**For Developers:**

- Contributing? → See `CONTRIBUTING.md` in repo root
- Testing? → See `tests/` directory

## Citation

If you use NeutroHydro in your research, please cite:

```bibtex
@software{neutrohydro,
  title = {NeutroHydro: Neutrosophic Chemometrics for Groundwater Analysis},
  year = {2024},
     url = {https://github.com/dabdul-wahab1988/neutrohydro}
}
```

## License

MIT License - see LICENSE file for details.

## License

MIT License (see [LICENSE](LICENSE)).

## Authors

- **Dickson Abdul-Wahab**, University of Ghana, Ghana
  - Email: <mailto:dabdul-wahab@live.com>
  - ORCID: <https://orcid.org/0000-0001-7446-5909>
  - LinkedIn: <https://www.linkedin.com/in/dickson-abdul-wahab-0764a1a9/>
  - ResearchGate: <https://www.researchgate.net/profile/Dickson-Abdul-Wahab>

- **Ebenezer Aquisman Asare**, Organic Laboratory Research, Atomic Energy Commission (GAEC), Nuclear Chemistry and Environmental Research Centre, National Nuclear Research Institute (NNRI), Legon-Accra, Ghana
  - Email: <mailto:aquisman1989@gmail.com>
  - ORCID: <https://orcid.org/0000-0003-1185-1479>
  - ResearchGate: <https://www.researchgate.net/profile/Ebenezer-Aquisman-Asare>

## Citation

If you use NeutroHydro in your research, please cite:

```bibtex
@software{neutrohydro,
  title = {NeutroHydro: Neutrosophic Chemometrics for Groundwater Analysis},
  year = {2025},
  url = {https://github.com/dabdul-wahab1988/neutrohydro}
}
```
