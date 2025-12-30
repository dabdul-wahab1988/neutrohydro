# NeutroHydro: Complete Technical Documentation

**Neutrosophic Chemometrics for Groundwater Analysis**

Generated: December 28, 2025
---

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

<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

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
# 'target_ions' are the ions you want to model/predict (usually all major ions)
pipeline = NeutroHydroPipeline(
    target_ions=["Ca", "Mg", "Na", "K", "HCO3", "Cl", "SO4", "NO3"]
)

# 3. Fit the Model
# The pipeline automatically handles preprocessing and encoding
pipeline.fit(df)

# 4. Get Results
# This returns a dictionary containing all analysis outputs
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
# You can access the quality assessment directly:

quality_df = results["quality_flags"]
print(quality_df[["Exceedances", "Inferred_Sources"]].head())
```

### Hydrogeochemical Indices

The analysis also calculates standard indices automatically:

```python
indices = results["indices"]
print(indices[["Simpson_Class", "Simpson_Ratio_Inverse", "Gibbs_Ratio_1"]].head())
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


---

<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```
# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Installation from Source

NeutroHydro is currently available as a source distribution. To install it, clone the repository and install using `pip`.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/neutrohydro.git
    cd neutrohydro
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\Activate.ps1

    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the package:**

    ```bash
    pip install .
    ```

    For development (including testing and documentation tools):

    ```bash
    pip install -e .[dev]
    ```

## Verifying Installation

To verify that NeutroHydro is installed correctly, you can run the following command in your terminal:

```bash
python -c "import neutrohydro; print(neutrohydro.__version__)"
```

If installed correctly, this should print the version number without errors.

## Dependencies

The core dependencies are automatically installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib` (for plotting)
- `seaborn` (for advanced visualization)


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Mathematical Framework Overview

## 1. Introduction

NeutroHydro implements a **neutrosophic chemometric framework** for groundwater analysis that operates in **absolute concentration space**. This document provides a high-level overview of the mathematical theory underpinning the package.

## 2. Problem Statement

### 2.1 Input Data

- **Predictor matrix**: $X \in \mathbb{R}^{n \times p}$, where:
  - $n$ = number of water samples
  - $p$ = number of ion species (e.g., Ca²⁺, Mg²⁺, Na⁺, Cl⁻, etc.)
  - $X_{ij}$ = concentration of ion $j$ in sample $i$ (units: mg/L, meq/L, etc.)

- **Target vector**: $y \in \mathbb{R}^n$
  - Typically: log TDS, log EC, or log ionic strength
  - Scalar response for each sample

### 2.2 Objectives

1. **Predict** target $y$ from ion concentrations $X$
2. **Decompose** prediction importance into:
   - Baseline/reference component (geogenic/natural)
   - Perturbation component (anthropogenic/anomalous)
3. **Infer** plausible mineral sources via stoichiometry

## 3. Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PREPROCESSING (Section 2)                         │
├─────────────────────────────────────────────────────────────┤
│ Input:  X, y (raw concentrations)                          │
│ Output: X_std, y_std (standardized)                        │
│                                                             │
│ Operations:                                                 │
│ • Optional log transform: X_log = log(X + δ_x)            │
│ • Robust centering: μ_j = median(X_j)                      │
│ • Robust scaling: s_j = 1.4826 × MAD(X_j)                 │
│ • Standardize: X_std = (X - μ) / (s + δ_s)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: NDG ENCODING (Section 3)                          │
├─────────────────────────────────────────────────────────────┤
│ Input:  X_std                                               │
│ Output: Triplets (X_T, X_I, X_F)                          │
│                                                             │
│ Truth Channel (T):                                          │
│   X_T = B(X_std)    [baseline operator]                   │
│                                                             │
│ Residuals:                                                  │
│   R = X_std - X_T                                          │
│   σ_j = 1.4826 × MAD(R_j)                                 │
│                                                             │
│ Falsity Channel (F):                                        │
│   u_ij = |R_ij| / σ_j                                      │
│   F_ij = 1 - exp(-u_ij)    [perturbation likelihood]      │
│                                                             │
│ Indeterminacy Channel (I):                                 │
│   I_ij ∈ [0,1]    [uncertainty/ambiguity]                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: PNPLS REGRESSION (Section 4)                      │
├─────────────────────────────────────────────────────────────┤
│ Input:  Triplets (X_T, X_I, X_F), y_std                   │
│ Output: PLS components (T, W, P, q, β)                    │
│                                                             │
│ Augmented Space:                                            │
│   X_aug = [X_T  √ρ_I·X_I  √ρ_F·X_F] ∈ ℝ^(n×3p)          │
│                                                             │
│ Precision Weights:                                          │
│   W_ij = exp(-λ_F · F_ij)                                  │
│   X̃_aug = W ⊙ X_aug    [elementwise product]             │
│                                                             │
│ PLS Decomposition (NIPALS):                                │
│   Extract k latent components                               │
│   Prediction: ŷ = X̃_aug · β                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: NVIP DECOMPOSITION (Section 5)                    │
├─────────────────────────────────────────────────────────────┤
│ Input:  PLS weights W, scores T, loadings q                │
│ Output: VIP_T, VIP_I, VIP_F, VIP_agg                      │
│                                                             │
│ Channel Energies:                                           │
│   E_T(j) = VIP_T²(j)                                       │
│   E_I(j) = VIP_I²(j)                                       │
│   E_F(j) = VIP_F²(j)                                       │
│                                                             │
│ L2 Decomposition Theorem:                                   │
│   VIP_agg²(j) = VIP_T²(j) + VIP_I²(j) + VIP_F²(j)        │
│                                                             │
│ Interpretation:                                             │
│   VIP_agg(j) ≥ 1  ⟹  ion j is important for prediction  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: ATTRIBUTION (Sections 6-7)                        │
├─────────────────────────────────────────────────────────────┤
│ Input:  NVIP energies, regression coefficients             │
│ Output: NSR, π_G (ion-level), G_i (sample-level)          │
│                                                             │
│ Ion-Level Attribution:                                      │
│   E_P(j) = E_I(j) + E_F(j)    [perturbation energy]       │
│   π_G(j) = E_T(j) / [E_T(j) + E_P(j)]    ∈ [0,1]         │
│   NSR(j) = E_T(j) / E_P(j)    [odds ratio]                │
│                                                             │
│ Classification (threshold γ = 0.7):                         │
│   π_G(j) ≥ γ       ⟹  baseline-dominant                  │
│   π_G(j) ≤ 1-γ     ⟹  perturbation-dominant              │
│   otherwise        ⟹  mixed                                │
│                                                             │
│ Sample-Level Attribution:                                   │
│   w_ij = |contribution of ion j in sample i|              │
│   G_i = Σ_j π_G(j)·w_ij / Σ_j w_ij    ∈ [0,1]           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: MINERAL INFERENCE (Section 8) [Optional]          │
├─────────────────────────────────────────────────────────────┤
│ Input:  Ion concentrations c (meq/L), π_G                  │
│ Output: Mineral contributions s, plausibility              │
│                                                             │
│ Stoichiometric Model:                                       │
│   c ≈ A·s    where s ≥ 0                                   │
│   A ∈ ℝ^(m×K): stoichiometric matrix                      │
│   s ∈ ℝ^K: mineral contributions                          │
│                                                             │
│ Weighted NNLS:                                              │
│   D = diag(π_G(1)^η, ..., π_G(m)^η)                       │
│   ŝ = argmin_{s≥0} ‖D(c - As)‖²                           │
│                                                             │
│ Plausibility Criteria:                                      │
│   Mineral k plausible if:                                   │
│     • ŝ_k > τ_s    (sufficient contribution)               │
│     • ‖D(c - Aŝ)‖ ≤ τ_r    (good fit)                     │
└─────────────────────────────────────────────────────────────┘
```

## 4. Key Mathematical Guarantees

### 4.1 Euclidean Structure

All operations occur in true Euclidean spaces:
- Preprocessing: $\mathbb{R}^p$
- Augmented space: $\mathbb{R}^{3p}$ with inner product
  $$\langle u, v \rangle_{\mathcal{N}} = u_T^\top v_T + \rho_I u_I^\top v_I + \rho_F u_F^\top v_F$$
- Well-defined projections, deflations, and orthogonality

### 4.2 L2 Additivity (Core Theorem)

**Theorem** (NVIP L2 Decomposition):

For each variable $j$:
$$VIP_{agg}^2(j) = VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)$$

**Proof Sketch**:
The augmented weight vector for component $h$ is:
$$w_h = [w_{T,h}^\top \quad w_{I,h}^\top \quad w_{F,h}^\top]^\top$$

VIP is computed as:
$$VIP_c^2(j) = p \cdot \frac{\sum_{h=1}^k SSY_h \cdot \frac{\omega_{c,h}(j)}{\Omega_h}}{\sum_{h=1}^k SSY_h}$$

where $\omega_{c,h}(j) = w_{c,h}^2(j)$ and $\Omega_h = \|w_h\|^2$.

Since $\Omega_h = \sum_m [\omega_{T,h}(m) + \omega_{I,h}(m) + \omega_{F,h}(m)]$, the normalized weights sum additively, yielding the L2 decomposition. □

### 4.3 Conservation Laws

**Unity constraints**:
1. $\pi_G(j) + \pi_A(j) = 1$ for all ions $j$
2. $G_i + A_i = 1$ for all samples $i$

**Bounds**:
1. $VIP_c(j) \geq 0$ for all channels $c \in \{T, I, F\}$
2. $\pi_G(j), G_i \in [0, 1]$
3. $I_{ij}, F_{ij} \in [0, 1]$

### 4.4 Stability

- **Robust statistics**: Median and MAD resist outliers
- **Precision weighting**: High-falsity observations downweighted
- **Regularization**: Optional ridge/elastic net in PLS

## 5. Operational vs. Causal Interpretation

### 5.1 Operational Definitions

The framework defines **baseline** and **perturbation** **operationally**:

- **Baseline** = component captured by $\mathcal{B}(X^{(std)})$
  - Median ⟹ central tendency
  - Low-rank ⟹ common geochemical manifold
  - Robust PCA ⟹ low-rank + sparse decomposition

- **Perturbation** = deviations from baseline, quantified by falsity $F$

### 5.2 External Validation Required

Attribution to **physical sources** (geogenic vs. anthropogenic) requires **external evidence**:
- Spatial patterns (urban vs. rural)
- Temporal trends (pre/post contamination event)
- Isotopic tracers
- Land use correlations

The framework provides **consistent mathematical attribution**; causal interpretation depends on domain knowledge.

## 6. Comparison to Other Methods

| Feature | NeutroHydro | Standard PLS | CoDa Methods |
|---------|-------------|--------------|--------------|
| **Space** | Absolute concentrations | Absolute/relative | Compositional (simplex) |
| **VIP decomposition** | L2-additive (T, I, F) | Single VIP | Not applicable |
| **Uncertainty** | Explicit (I channel) | Implicit | Not standard |
| **Robustness** | Falsity weighting | Optional | Depends on method |
| **Missing data** | EM imputation | Varies | Special handling |
| **Stoichiometry** | Direct (NNLS) | Post-hoc | Difficult |
| **Interpretability** | High (3 channels) | Moderate | Low (log-ratios) |

## 7. Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Preprocessing | $O(np)$ | Median/MAD per column |
| NDG Encoding (median) | $O(np)$ | Per-column operations |
| NDG Encoding (low-rank) | $O(np \min(n,p))$ | SVD cost |
| PNPLS (k components) | $O(knp^2)$ | NIPALS iterations |
| NVIP | $O(kp)$ | Post-PLS computation |
| Attribution | $O(np)$ | Linear pass |
| Mineral NNLS (per sample) | $O(m^2 K)$ | Per-sample inversion |

**Total pipeline**: $O(knp^2 + nm^2K)$ where typically $k \ll n, p$ and $m, K \lesssim 10$.

## 8. Hyperparameter Selection

### 8.1 Critical Parameters

1. **Number of components** $k$:
   - Cross-validation (minimize RMSE on held-out data)
   - Elbow method (explained variance)
   - Typical range: 3–10

2. **Baseline type**:
   - `median`: Fast, robust, interpretable (default)
   - `low_rank`: Captures geochemical manifold
   - `robust_pca`: Handles sparse anomalies

3. **Channel weights** $\rho_I, \rho_F$:
   - Default: $\rho_I = \rho_F = 1$ (equal)
   - Increase $\rho_I$ to emphasize uncertainty
   - Increase $\rho_F$ to emphasize perturbations

4. **Falsity strength** $\lambda_F$:
   - Default: $\lambda_F = 1$
   - Higher values ⟹ stronger downweighting of anomalies
   - Typical range: 0.5–5

5. **Classification threshold** $\gamma$:
   - Default: $\gamma = 0.7$
   - Higher $\gamma$ ⟹ stricter baseline classification
   - Typical range: 0.6–0.8

### 8.2 Sensitivity Analysis

Recommended workflow:
1. Fit with default hyperparameters
2. Compute predictions and VIPs
3. Vary one hyperparameter at a time
4. Check stability of:
   - Top-ranked ions (VIP)
   - Classification (baseline vs. perturbation)
   - Prediction accuracy ($R^2$)

## 9. Limitations and Assumptions

### 9.1 Assumptions

1. **Linearity**: PLS assumes (approximately) linear relationships
   - Mitigated by: log transforms, polynomial features
2. **Stationarity**: Baseline operator assumes some regularity
   - Mitigated by: hydrofacies conditioning, spatial stratification
3. **Independence**: Samples assumed independent
   - May violate: temporal autocorrelation, spatial clustering

### 9.2 Limitations

1. **Scalar response**: Current implementation supports single target
   - Extension to multivariate $Y$ possible (PLS2/NPLS)
2. **Stoichiometry**: Mineral inference assumes simple dissolution
   - Does not model: ion exchange, redox, precipitation, kinetics
3. **Computational**: Large $n, p$ may be slow
   - Mitigations: use `baseline_type='median'`, reduce $k$

## 10. Software Implementation

The mathematical framework is implemented in Python with:
- **NumPy/SciPy**: Core numerical operations
- **scikit-learn**: PLS utilities (validation, not direct use)
- **Custom NIPALS**: Channel-aware PLS implementation
- **SciPy.optimize.nnls**: Non-negative least squares for minerals

All operations preserve **numerical stability** via:
- Robust scaling (MAD, not std)
- Small $\delta$ constants for division safety
- SVD for pseudo-inverses
- Convergence tolerances in iterative methods

---

**Next**: See individual module documentation for detailed equations and algorithms.


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```
docs/api_modules.md docs/api_pipeline.md docs/attribution.md docs/combined_docs.md docs/encoder.md docs/examples_advanced.md docs/examples_basic.md docs/final_critical_review.md docs/hydrogeochemical_processes.md docs/index.md docs/installation.md docs/interpreting_results.md docs/limitations.md docs/mathematical_critique.md docs/mathematical_framework.md docs/minerals.md docs/model.md docs/NeutroHydro_Complete_Documentation.html docs/NeutroHydro_Complete_Documentation.md docs/nvip.md docs/preprocessing.md docs/quality_check.md docs/quickstart.md

<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# NDG Encoder: Neutrosophic Triplets

**Module**: `neutrohydro.encoder`

## Overview

The NDG (Neutrosophic Data Generator) Encoder maps each standardized ion concentration to a **neutrosophic triplet** $(T, I, F)$:

- **T (Truth)**: Baseline/reference component
- **I (Indeterminacy)**: Uncertainty/ambiguity
- **F (Falsity)**: Perturbation/anomaly likelihood

This representation enables **explicit decomposition** of prediction importance into baseline and perturbation sources.

## Mathematical Foundation

### 1. Input

Standardized predictor matrix from preprocessing:
$$X^{(\text{std})} \in \mathbb{R}^{n \times p}$$

### 2. Output

Three channel matrices:
$$X_T, X_I, X_F \in \mathbb{R}^{n \times p}$$

Plus residuals and metadata:
$$R \in \mathbb{R}^{n \times p}, \quad \sigma \in \mathbb{R}^p$$

## Truth Channel (T): Baseline Operator

### 1. Definition

$$X_T = \mathcal{B}(X^{(\text{std})})$$

where $\mathcal{B}: \mathbb{R}^{n \times p} \to \mathbb{R}^{n \times p}$ is a **baseline operator**.

### 2. Baseline Operator Options

#### Option 1: Robust Columnwise Median (Default)

$$(X_T)_{ij} = \text{median}_{i'}(X^{(\text{std})}_{i'j})$$

**Replicated per row** - constant baseline for each ion.

**Advantages**:
- Fast: $O(np)$
- Robust to outliers
- Interpretable: "central tendency"

**When to use**: Default choice for most applications.

#### Option 2: Hydrofacies-Conditioned Median

$$(X_T)_{ij} = \text{median}_{i': g(i')=g(i)}(X^{(\text{std})}_{i'j})$$

where $g(i) \in \{1, \ldots, G\}$ assigns sample $i$ to hydrofacies group $g(i)$.

**Advantages**:
- Accounts for spatial/lithological heterogeneity
- Different baselines for different aquifer units

**When to use**: Known hydrofacies/spatial stratification exists.

#### Option 3: Low-Rank Baseline

$$X_T = \arg\min_{L: \text{rank}(L) \leq r} \|X^{(\text{std})} - L\|_F^2$$

Solution via **truncated SVD**:

$$X^{(\text{std})} = U \Sigma V^\top, \quad X_T = U_r \Sigma_r V_r^\top$$

where subscript $r$ denotes top-$r$ singular values/vectors.

**Advantages**:
- Captures **geochemical manifold** (common patterns)
- Smooth, low-dimensional baseline

**When to use**: Strong correlations among ions, clear manifold structure.

**Hyperparameter**: Rank $r$ (typical: 2–5)

#### Option 4: Robust PCA Baseline

Decomposition:
$$X^{(\text{std})} = L + S$$

where:
- $L$ is **low-rank** (baseline)
- $S$ is **sparse** (outliers/anomalies)

Set $X_T := L$.

**Iterative algorithm** (simplified):
```
Initialize: L = 0, S = 0
For iteration = 1, ..., max_iter:
    # Update L (low-rank via SVD)
    L ← TruncatedSVD(X - S, rank=r)

    # Update S (sparse via soft thresholding)
    S ← SoftThreshold(X - L, λ)

    # Check convergence
    If ||L - L_old||_F / ||L_old||_F < tol:
        break
```

**Advantages**:
- Robust to sparse outliers
- Separates systematic (L) from anomalous (S) patterns

**When to use**: Known sparse contamination events.

**Hyperparameters**: Rank $r$, sparsity parameter $\lambda$

### 3. Residuals

$$R = X^{(\text{std})} - X_T$$

**Interpretation**: Deviations from baseline.

### 4. Robust Residual Scale

For each ion $j$:

$$\sigma_j = 1.4826 \times \text{MAD}_i(R_{ij})$$

where:
$$\text{MAD}_i(R_{ij}) = \text{median}_i(|R_{ij}|)$$

**Purpose**: Normalize residuals to comparable units.

## Falsity Channel (F): Perturbation Likelihood

### 1. Normalized Residuals

$$u_{ij} = \frac{|R_{ij}|}{\sigma_j + \delta}$$

where $\delta > 0$ (default: $10^{-10}$) prevents division by zero.

**Interpretation**: $u_{ij}$ measures how many "robust standard deviations" sample $i$ deviates from baseline for ion $j$.

### 2. Falsity Map

$$F_{ij} = g_F(u_{ij})$$

where $g_F: \mathbb{R}_{\geq 0} \to [0, 1]$ is a **monotone increasing** map.

#### Option 1: Exponential Saturation (Default)

$$F_{ij} = 1 - \exp(-u_{ij})$$

**Properties**:
- $F_{ij} \to 0$ as $u_{ij} \to 0$ (small deviations → low falsity)
- $F_{ij} \to 1$ as $u_{ij} \to \infty$ (large deviations → high falsity)
- Smooth, differentiable
- No additional hyperparameters

**Interpretation**: Probability of "falsity" (deviation from baseline) grows exponentially with residual magnitude.

#### Option 2: Logistic

$$F_{ij} = \frac{1}{1 + \exp(-a(u_{ij} - b))}$$

**Hyperparameters**:
- $a > 0$: Steepness (default: 2.0)
- $b > 0$: Inflection point (default: 1.0)

**Properties**:
- Sigmoid shape
- $F_{ij} \approx 0.5$ when $u_{ij} \approx b$
- Sharper transition than exponential

**When to use**: Want explicit threshold for "significant" deviations.

### 3. Bounds

By construction: $F_{ij} \in [0, 1]$.

## Indeterminacy Channel (I): Uncertainty

### 1. Purpose

Capture **ambiguity** not purely due to residual magnitude:
- Measurement uncertainty
- Censoring (below detection limit)
- Spatial/temporal variability
- Bootstrap instability

### 2. Methods

#### Method 1: Local Heterogeneity (Default)

For spatial/temporal data with neighborhood structure:

$$I_{ij} = 1 - \exp\left(-\frac{\text{Var}(\mathcal{N}(i), j)}{\tau_j + \delta}\right)$$

where:
- $\mathcal{N}(i)$ = neighborhood of sample $i$ (e.g., $k$ nearest spatial/temporal neighbors)
- $\text{Var}(\mathcal{N}(i), j)$ = variance of ion $j$ within neighborhood
- $\tau_j$ = scale parameter (e.g., median variance)

**Interpretation**: High local variability → high indeterminacy.

**Limitation**: Requires spatial/temporal structure.

#### Method 2: Censoring/Detection Limit

$$I_{ij} = \begin{cases}
\iota_{\text{DL}} & \text{if } X_{ij} < \text{DL}_j \\
0 & \text{otherwise}
\end{cases}$$

where $\iota_{\text{DL}} \in (0, 1)$ (default: 0.5).

**Advantages**:
- Simple, interpretable
- Directly encodes measurement uncertainty

**When to use**: Data has known detection limits.

#### Method 3: Bootstrap Instability

1. Generate $B$ bootstrap samples of the data
2. Fit encoder on each bootstrap sample
3. Compute:
   $$I_{ij} = \frac{\text{sd}_b(\widehat{X}^{(b)}_{Tij})}{\text{scale}}$$

   where scale maps to $[0, 1]$ (e.g., via percentile).

**Advantages**:
- Data-driven uncertainty quantification
- No domain assumptions

**Disadvantages**:
- Computationally expensive ($B \times$ encoding cost)

#### Method 4: Uniform Small Indeterminacy

$$I_{ij} = \epsilon$$

for some small $\epsilon > 0$ (e.g., 0.01).

**When to use**: No strong uncertainty information; want placeholder.

### 3. Custom Indeterminacy

User can provide custom function:

```python
def my_indeterminacy(X_std):
    # X_std: (n, p) standardized data
    # Return: (n, p) indeterminacy in [0, 1]
    I = ...
    return I

triplets = encoder.transform(X_std, indeterminacy_func=my_indeterminacy)
```

## Triplet Contract

The encoder **must guarantee**:

1. **Shape**: $X_T, X_I, X_F$ have shape $(n, p)$
2. **Bounds**: $I_{ij}, F_{ij} \in [0, 1]$ for all $i, j$
3. **Residuals**: $R = X^{(\text{std})} - X_T$
4. **Reproducibility**: Given same $X^{(\text{std})}$ and hyperparameters, output is deterministic

## Algorithm

### Fitting

**Input**: $X^{(\text{std})} \in \mathbb{R}^{n \times p}$, baseline type, hyperparameters

**Output**: Encoder parameters $\theta = (\sigma, \text{metadata})$

```
1. Compute baseline: X_T ← B(X_std)
2. Compute residuals: R ← X_std - X_T
3. Compute robust scales:
     For j = 1, ..., p:
         σ_j ← 1.4826 × median(|R[:,j]|)
4. Store metadata (baseline type, rank, etc.)
5. Return θ = (σ, metadata)
```

### Transformation

**Input**: $X^{(\text{std})} \in \mathbb{R}^{n \times p}$, parameters $\theta$, optional indeterminacy function

**Output**: Triplets $(X_T, X_I, X_F)$ and residuals $R$

```
1. Compute baseline: X_T ← B(X_std)
2. Compute residuals: R ← X_std - X_T
3. Compute falsity:
     u ← |R| / (σ + δ)
     F ← g_F(u)  # Apply falsity map
4. Compute indeterminacy:
     If indeterminacy_func provided:
         I ← indeterminacy_func(X_std)
     Else:
         I ← default_indeterminacy(X_std)  # e.g., local variance
5. Clip to bounds: I ← clip(I, 0, 1), F ← clip(F, 0, 1)
6. Return (X_T, X_I, X_F, R)
```

## Properties

### 1. Baseline Projection

For median baseline:
$$\text{median}_i(X_{Tij}) = \text{median}_i(X^{(\text{std})}_{ij})$$

For low-rank baseline:
$$\text{rank}(X_T) \leq r$$

### 2. Residual Orthogonality (Low-Rank)

For low-rank baseline via SVD:
$$\langle X_T, R \rangle_F = 0$$

where $\langle A, B \rangle_F = \text{tr}(A^\top B)$ is the Frobenius inner product.

#### Proof

**Theorem**: For low-rank baseline via truncated SVD, $\langle X_T, R \rangle_F = 0$.

**Proof**:

Let $X^{(\text{std})} = U\Sigma V^\top$ be the full SVD where:
- $U \in \mathbb{R}^{n \times n}$ (left singular vectors)
- $\Sigma \in \mathbb{R}^{n \times p}$ (diagonal singular values)
- $V \in \mathbb{R}^{p \times p}$ (right singular vectors)

The rank-$r$ truncated SVD baseline is:
$$X_T = U_r \Sigma_r V_r^\top$$

where subscript $r$ denotes the top $r$ components.

The residual is:
$$R = X^{(\text{std})} - X_T = U\Sigma V^\top - U_r \Sigma_r V_r^\top$$

Partition the SVD:
$$U = [U_r \mid U_{r+1:n}], \quad \Sigma = \begin{bmatrix} \Sigma_r & 0 \\ 0 & \Sigma_{r+1:n} \end{bmatrix}, \quad V = [V_r \mid V_{r+1:p}]$$

Then:
$$X^{(\text{std})} = U_r \Sigma_r V_r^\top + U_{r+1:n} \Sigma_{r+1:n} V_{r+1:p}^\top$$

So:
$$R = U_{r+1:n} \Sigma_{r+1:n} V_{r+1:p}^\top$$

The Frobenius inner product is:
$$\langle X_T, R \rangle_F = \text{tr}(X_T^\top R)$$

$$= \text{tr}\left[(V_r \Sigma_r U_r^\top)^\top (U_{r+1:n} \Sigma_{r+1:n} V_{r+1:p}^\top)\right]$$

$$= \text{tr}\left[V_r \Sigma_r U_r^\top U_{r+1:n} \Sigma_{r+1:n} V_{r+1:p}^\top\right]$$

Since $U$ is orthogonal: $U_r^\top U_{r+1:n} = 0$

Therefore:
$$\langle X_T, R \rangle_F = 0$$

$\square$

**Interpretation**: The low-rank baseline and residual are **orthogonal in Frobenius space**, implying no correlation between baseline patterns and deviations.

### 3. Falsity Monotonicity

$$|R_{ij}| < |R_{kj}| \implies F_{ij} < F_{kj}$$

Larger deviations always have higher falsity.

#### Proof

**Theorem**: For exponential falsity map $F_{ij} = 1 - \exp(-u_{ij})$ where $u_{ij} = |R_{ij}|/(\sigma_j + \delta)$, falsity is strictly monotone increasing in $|R_{ij}|$.

**Proof**:

Consider the falsity map:
$$F_{ij} = g_F(u_{ij}) = 1 - \exp(-u_{ij})$$

where:
$$u_{ij} = \frac{|R_{ij}|}{\sigma_j + \delta}$$

For fixed ion $j$, we have:
$$\frac{\partial F_{ij}}{\partial u_{ij}} = \frac{\partial}{\partial u}[1 - \exp(-u)] = \exp(-u) > 0 \quad \forall u \geq 0$$

Since $\partial F/\partial u > 0$, the map is **strictly increasing**.

Furthermore:
$$\frac{\partial u_{ij}}{\partial |R_{ij}|} = \frac{1}{\sigma_j + \delta} > 0$$

By chain rule:
$$\frac{\partial F_{ij}}{\partial |R_{ij}|} = \frac{\partial F_{ij}}{\partial u_{ij}} \cdot \frac{\partial u_{ij}}{\partial |R_{ij}|} > 0$$

Therefore:
$$|R_{ij}| < |R_{kj}| \implies u_{ij} < u_{kj} \implies F_{ij} < F_{kj}$$

$\square$

**Note**: For logistic falsity map $F_{ij} = 1/(1 + \exp(-a(u_{ij} - b)))$, the same result holds since:
$$\frac{\partial F}{\partial u} = \frac{a \exp(-a(u-b))}{[1 + \exp(-a(u-b))]^2} > 0 \quad \forall u$$

### 4. Independence of Channels

$(T, I, F)$ are **conceptually independent**:
- $T$ captures baseline
- $I$ captures uncertainty (may be independent of baseline)
- $F$ captures perturbations (derived from $R = X^{(\text{std})} - T$)

## Hyperparameter Selection

| Parameter | Values | Selection Criterion |
|-----------|--------|---------------------|
| `baseline_type` | median, low_rank, robust_pca | Domain knowledge, exploratory analysis |
| `baseline_rank` | 2–10 | Scree plot, explained variance |
| `falsity_map` | exponential, logistic | Default: exponential (no tuning) |
| `falsity_params` (logistic) | $a \in [1, 5]$, $b \in [0.5, 2]$ | Cross-validation on downstream task |

## Usage Example

```python
from neutrohydro.encoder import NDGEncoder, BaselineType, FalsityMap

# Option 1: Simple median baseline
encoder = NDGEncoder(baseline_type="median")
encoder.fit(X_std)
triplets = encoder.transform(X_std)

# Option 2: Low-rank baseline
encoder = NDGEncoder(
    baseline_type=BaselineType.LOW_RANK,
    baseline_rank=3
)
triplets = encoder.fit_transform(X_std)

# Option 3: Custom indeterminacy
def detection_limit_I(X_std):
    # Assume we have original X and DL
    I = np.zeros_like(X_std)
    I[X < DL] = 0.5
    return I

triplets = encoder.transform(X_std, indeterminacy_func=detection_limit_I)

# Access components
print(f"Truth shape: {triplets.T.shape}")
print(f"Indeterminacy range: [{triplets.I.min():.3f}, {triplets.I.max():.3f}]")
print(f"Falsity range: [{triplets.F.min():.3f}, {triplets.F.max():.3f}]")
```

## Diagnostics

### 1. Baseline Quality

For low-rank baseline, check **explained variance**:

$$\text{EV}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(n,p)} \sigma_i^2}$$

where $\sigma_i$ are singular values.

**Rule of thumb**: $\text{EV}(r) > 0.7$ for good baseline.

### 2. Residual Distribution

Plot histogram of normalized residuals $u_{ij}$:
- **Unimodal near 0**: Baseline captures central tendency well
- **Bimodal**: May have distinct populations (consider hydrofacies)
- **Heavy tails**: Many outliers (robust PCA may help)

### 3. Falsity Patterns

Plot $F_{ij}$ vs. $i$ (sample index):
- **Random scatter**: No systematic pattern (good)
- **Clusters**: May indicate contamination events or spatial patterns
- **All high/low**: Baseline may be poor fit

### 4. Channel Correlations

Compute:
$$\text{corr}(F_j, I_j) = \text{corr}(F_{:,j}, I_{:,j})$$

**Expected**: Low correlation (channels capture different aspects)

**High correlation**: May indicate redundancy or poor indeterminacy definition

## Interpretation

### Truth Channel $T$

- **Physical meaning**: "Expected" or "reference" concentration
- **Baseline = median**: Central tendency across all samples
- **Baseline = low-rank**: Typical geochemical pattern
- **Values**: Typically close to 0 (since data is standardized)

### Indeterminacy Channel $I$

- **Physical meaning**: Degree of uncertainty/ambiguity
- **High $I$**: Measurement uncertain, censored, or highly variable
- **Low $I$**: Well-measured, confident value
- **Range**: $[0, 1]$ by construction

### Falsity Channel $F$

- **Physical meaning**: Likelihood of deviation from baseline
- **High $F$**: Anomalous, contaminated, or perturbed
- **Low $F$**: Consistent with baseline
- **Range**: $[0, 1]$ by construction

## References

1. Smarandache, F. (1998). *Neutrosophy: Neutrosophic probability, set, and logic*. American Research Press.

2. Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis? *Journal of the ACM*, 58(3), 1-37.

3. Reimann, C., Filzmoser, P., Garrett, R. G., & Dutter, R. (2008). *Statistical data analysis explained: Applied environmental statistics with R*. John Wiley & Sons.

---

**Next**: [PNPLS Model](model.md) - Regression in augmented neutrosophic space.


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```
# PNPLS: Probabilistic Neutrosophic PLS

**Module**: `neutrohydro.model`

## Overview

PNPLS extends Partial Least Squares (PLS) regression to **neutrosophic triplet data** $(T, I, F)$ by:
1. Constructing an **augmented predictor space** combining the three channels
2. Applying **elementwise precision weights** based on falsity
3. Fitting PLS via the **NIPALS algorithm** in this augmented Hilbert space

## Mathematical Foundation

### 1. Input

From NDG encoder:
- Triplet channels: $X_T, X_I, X_F \in \mathbb{R}^{n \times p}$
- Standardized target: $y^{(\text{std})} \in \mathbb{R}^n$

### 2. Augmented Predictor Space

#### 2.1 Channel Concatenation

$$X^{(\text{aug})} = \left[\, X_T \quad \sqrt{\rho_I} X_I \quad \sqrt{\rho_F} X_F \,\right] \in \mathbb{R}^{n \times 3p}$$

**Channel weights**:
- $\rho_T = 1$ (Truth channel, always included)
- $\rho_I \geq 0$ (Indeterminacy weight, default: 1)
- $\rho_F \geq 0$ (Falsity weight, default: 1)

#### 2.2 Induced Inner Product

The augmented space $\mathbb{R}^{3p}$ has inner product:

$$\langle u, v \rangle_{\mathcal{N}} = u_T^\top v_T + \rho_I u_I^\top v_I + \rho_F u_F^\top v_F$$

where $u = [u_T^\top, u_I^\top, u_F^\top]^\top$ and similarly for $v$.

**Properties**:
- Positive definite (since $\rho_I, \rho_F \geq 0$)
- Euclidean structure preserved
- Well-defined projections and orthogonality

### 3. Precision Weighting

#### 3.1 Elementwise Weights from Falsity

$$W_{ij} = \exp(-\lambda_F \cdot F_{ij})$$

where $\lambda_F > 0$ controls downweighting strength.

**Interpretation**:
- High falsity $F_{ij} \approx 1$ → low weight $W_{ij} \approx \exp(-\lambda_F)$
- Low falsity $F_{ij} \approx 0$ → high weight $W_{ij} \approx 1$

**Effect**: Observations with large deviations from baseline contribute less to covariance estimation.

#### 3.2 Extension to Augmented Space

Repeat weights for all three channels:

$$W^{(\text{aug})} = \left[\, W \quad W \quad W \,\right] \in \mathbb{R}^{n \times 3p}$$

#### 3.3 Weighted Predictors

$$\widetilde{X}^{(\text{aug})} = W^{(\text{aug})} \odot X^{(\text{aug})}$$

where $\odot$ is elementwise (Hadamard) product.

**Result**: $\widetilde{X}^{(\text{aug})}_{ij}$ is downweighted if sample $i$ has high falsity for ion $j$.

## PLS1 via NIPALS

### 4. NIPALS Algorithm

**Input**: $\widetilde{X}^{(\text{aug})} \in \mathbb{R}^{n \times 3p}$, $y^{(\text{std})} \in \mathbb{R}^n$, $k$ components

**Output**: Latent components $(T, W, P, q, \beta)$

#### Initialization

```
X_deflated ← X̃^(aug)
y_deflated ← y^(std)
T ← zeros(n, k)  # Scores
W ← zeros(3p, k)  # Weights
P ← zeros(3p, k)  # Loadings
q ← zeros(k)      # Response loadings
```

#### Component Extraction (for h = 1, ..., k)

**Step 1: Initialize weight vector**

$$w_h = \frac{X_{\text{deflated}}^\top y_{\text{deflated}}}{\|X_{\text{deflated}}^\top y_{\text{deflated}}\|}$$

**Step 2: Iterative refinement** (until convergence)


Repeat:
  $$w_{\text{old}} \leftarrow w_h$$
    # Score
  $$t_h \leftarrow X_{\text{deflated}} \cdot w_h$$
  $$t_h \leftarrow \frac{t_h}{\|t_h\|}$$
    # Weight update
  $$w_h \leftarrow X_{\text{deflated}}^\top \cdot t_h$$
    $$w_h \leftarrow \frac{w_h}{\|w_h\|}$$
Until $$\|w_h - w_{\text{old}}\| < \text{tol}$$


**Step 3: Final score and loadings**

$$t_h = X_{\text{deflated}} w_h$$

$$p_h = \frac{X_{\text{deflated}}^\top t_h}{t_h^\top t_h}$$

$$q_h = \frac{y_{\text{deflated}}^\top t_h}{t_h^\top t_h}$$

**Step 4: Deflation**

$$X_{\text{deflated}} \leftarrow X_{\text{deflated}} - t_h p_h^\top$$

$$y_{\text{deflated}} \leftarrow y_{\text{deflated}} - t_h q_h$$

**Step 5: Store**

$$T[:,h] \leftarrow t_h$$
$$W[:,h] \leftarrow w_h$$
$$P[:,h] \leftarrow p_h$$
$$q[h] \leftarrow q_h$$

#### Regression Coefficients

After extracting $k$ components:

$$\beta = W (P^\top W)^{-1} q$$

If $P^\top W$ is singular, use pseudo-inverse:

$$\beta = W (P^\top W)^+ q$$

### 5. Prediction

For new triplet data $(\widetilde{X}_{\text{new}}^{(\text{aug})})$:

$$\hat{y}_{\text{new}} = \widetilde{X}_{\text{new}}^{(\text{aug})} \beta$$

Transform back to original scale using preprocessing parameters.

## Handling Missing Data

### 6. EM-Like Imputation

For missing values in original $X$, extended to augmented space:

#### E-Step: Imputation

Current reconstruction from $k$ components:

$$\widehat{X}^{(\text{aug})} = T P^\top$$

Impute missing entries:

$$X^{(\text{aug})}_{\text{complete}} = M^{(\text{aug})} \odot X^{(\text{aug})} + (1 - M^{(\text{aug})}) \odot \widehat{X}^{(\text{aug})}$$

where $M^{(\text{aug})} \in \{0, 1\}^{n \times 3p}$ is the missingness mask.

#### M-Step: Refit PLS

Run NIPALS on $X^{(\text{aug})}_{\text{complete}}$ to update $(T, W, P, q, \beta)$.

#### Convergence

Iterate E-step and M-step until:

$$\frac{\|\beta^{(t+1)} - \beta^{(t)}\|}{\|\beta^{(t)}\|} < \text{tol}_{\text{EM}}$$

Typically: 10–20 iterations suffice.

## Model Diagnostics

### 7. Explained Variance

For component $h$:

$$\text{SSY}_h = q_h^2 \cdot (t_h^\top t_h)$$

**Cumulative explained variance**:

$$R^2_{\text{cumul}}(k) = \frac{\sum_{h=1}^k \text{SSY}_h}{\|y^{(\text{std})}\|^2}$$

**Typical behavior**: First few components explain most variance; diminishing returns after.

### 8. Q² (Predictive Quality)

Use cross-validation:

$$Q^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i^{(\text{CV})})^2}{\sum_i (y_i - \bar{y})^2}$$

where $\hat{y}_i^{(\text{CV})}$ is prediction with sample $i$ excluded.

### 9. Coefficient Partitioning

Partition $\beta \in \mathbb{R}^{3p}$ into channels:

$$\beta = \begin{bmatrix} \beta_T \\ \beta_I \\ \beta_F \end{bmatrix}, \quad \beta_T, \beta_I, \beta_F \in \mathbb{R}^p$$

**Interpretation**:
- $\beta_T(j)$: Contribution of baseline ion $j$
- $\beta_I(j)$: Contribution of indeterminacy for ion $j$
- $\beta_F(j)$: Contribution of falsity for ion $j$

**Net contribution** for sample $i$, ion $j$:

$$c_{ij} = (\widetilde{X}_T)_{ij} \beta_T(j) + (\widetilde{X}_I)_{ij} \beta_I(j) + (\widetilde{X}_F)_{ij} \beta_F(j)$$

## Properties

### 10. Optimality (Standard PLS)

Standard PLS maximizes covariance between scores and response:

$$\max_{w} \text{Cov}(Xw, y) \quad \text{s.t.} \quad \|w\| = 1$$

**PNPLS extension**: Operates in augmented space $\widetilde{X}^{(\text{aug})}$, inheriting PLS optimality.

### 11. Multicollinearity Handling

PLS naturally handles **multicollinearity** (unlike ordinary least squares):
- Extracts latent components capturing correlated structure
- Stable even when $p \gg n$

### 12. Relationship to PCR

**PCR** (Principal Component Regression):
1. Compute PCA of $X$
2. Regress $y$ on principal components

**PLS** (includes PNPLS):
1. Extract components **maximizing covariance with $y$**
2. Directly optimizes prediction

**Result**: PLS typically outperforms PCR for prediction.

### 13. NIPALS Convergence

The NIPALS algorithm (Section 4) iteratively refines weight vectors. We prove convergence.

#### Theorem: NIPALS Convergence

**Theorem**: For non-zero data matrix $X$ and target $y$, the NIPALS iterative update:

  $$w^{(t+1)} = X^T t^{(t)} / ||X^T t^{(t)}||$$
  $$t^{(t+1)} = X w^{(t+1)} / ||X w^{(t+1)}||$$

converges to the **leading left and right singular vectors** of the matrix $X^T y y^T X$ (scaled by norms).

**Proof**:

Consider the NIPALS iteration for component $h$:

**Initialization**: $w_h^{(0)} = X^T y / \|X^T y\|$

**Iteration** $t = 0, 1, 2, \ldots$:

1. $t_h^{(t)} = X w_h^{(t)} / \|X w_h^{(t)}\|$
2. $w_h^{(t+1)} = X^T t_h^{(t)} / \|X^T t_h^{(t)}\|$

**Combine** steps:
$$w_h^{(t+1)} = \frac{X^T (X w_h^{(t)} / \|X w_h^{(t)}\|)}{\|X^T (X w_h^{(t)} / \|X w_h^{(t)}\|)\|}$$

Let $M = X^T X$. Then (ignoring normalization momentarily):
$$w_h^{(t+1)} \propto X^T X w_h^{(t)} = M w_h^{(t)}$$

$$w_h^{(t)} \propto M^t w_h^{(0)}$$

This is the **power method** applied to matrix $M = X^T X$.

**Convergence of Power Method**:

Let $M = X^T X$ have eigendecomposition:
$$M = V \Lambda V^T$$

where $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_p \geq 0$ and $V = [v_1 \mid v_2 \mid \ldots \mid v_p]$ are orthonormal eigenvectors.

Expand initial vector:
$$w_h^{(0)} = \sum_{i=1}^p \alpha_i v_i$$

After $t$ iterations:
$$M^t w_h^{(0)} = \sum_{i=1}^p \alpha_i \lambda_i^t v_i = \lambda_1^t \left(\alpha_1 v_1 + \sum_{i=2}^p \alpha_i \left(\frac{\lambda_i}{\lambda_1}\right)^t v_i\right)$$

Since $|\lambda_i/\lambda_1| < 1$ for $i > 1$:
$$\lim_{t \to \infty} \frac{M^t w_h^{(0)}}{\|M^t w_h^{(0)}\|} = \pm v_1$$

(assuming $\alpha_1 \neq 0$, which holds generically).

**Convergence Rate**:
$$\|w_h^{(t)} - v_1\| = O\left(\left|\frac{\lambda_2}{\lambda_1}\right|^t\right)$$

**Practical implication**: NIPALS converges **exponentially fast** when the leading eigenvalue is well-separated from the second eigenvalue.

$\square$

#### Corollary: Convergence Tolerance

**Typical tolerance**: $\|w_h^{(t+1)} - w_h^{(t)}\| < \epsilon$ with $\epsilon = 10^{-6}$.

**Expected iterations**: $t \approx \log(\epsilon) / \log(|\lambda_2/\lambda_1|)$.

For typical data with $\lambda_1/\lambda_2 \approx 2$–5, convergence in **5–15 iterations**.

## Hyperparameters

| Parameter | Range | Default | Selection Method |
|-----------|-------|---------|------------------|
| $k$ (components) | 1–20 | 5 | Cross-validation (minimize RMSE) |
| $\rho_I$ | [0.1, 5] | 1.0 | Grid search or domain knowledge |
| $\rho_F$ | [0.1, 5] | 1.0 | Grid search or domain knowledge |
| $\lambda_F$ | [0.1, 10] | 1.0 | Cross-validation |

### Component Selection

**Elbow method**: Plot explained variance vs. $k$; choose $k$ at "elbow".

**Cross-validation**:
- For $k = 1, \ldots, k_{\max}$:
  - Compute $Q^2(k)$ via cross-validation
- Select $k$ maximizing $Q^2$

**Heuristic**: Start with $k = 5$; increase if underfitting, decrease if overfitting.

### Channel Weights

**Equal weights** ($\rho_I = \rho_F = 1$): Default, balanced approach.

**Emphasize uncertainty** ($\rho_I > 1$): If uncertainty is critical (e.g., censored data).

**Emphasize perturbations** ($\rho_F > 1$): If anomalies are primary interest.

### Falsity Weighting Strength

**Low $\lambda_F$ (≈ 0.5)**: Mild downweighting, inclusive of outliers.

**High $\lambda_F$ (≈ 5)**: Strong downweighting, robust to anomalies.

**Optimal**: Often around $\lambda_F \in [1, 2]$.

## Algorithm Complexity

| Operation | Complexity |
|-----------|------------|
| Single NIPALS iteration | $O(np)$ |
| Single component (converged) | $O(I_{conv} \cdot np)$ where $I_{conv} \approx 5$–10 |
| Full PLS ($k$ components) | $O(knp)$ |
| EM imputation ($I_{EM}$ iterations) | $O(I_{EM} \cdot knp)$ where $I_{EM} \approx 10$–20 |

**Total**: $O(knp)$ for complete data, $O(I_{EM} \cdot knp)$ with missing data.

## Usage Example

```python
from neutrohydro.model import PNPLS

# Initialize
model = PNPLS(
    n_components=5,
    rho_I=1.0,
    rho_F=1.0,
    lambda_F=1.0,
    tol=1e-6,
    max_iter=500
)

# Fit on triplets and target
model.fit(triplets, y_std)

# Predictions
y_pred_std = model.predict(triplets)

# Score (R²)
r2 = model.score(triplets, y_std)
print(f"R² = {r2:.4f}")

# Get coefficients by channel
coeffs = model.get_coefficients()
print(f"Truth coefficients: {coeffs['beta_T']}")

# Get weights by channel (for NVIP)
weights = model.get_weights_by_channel()
W_T = weights['W_T']  # Shape: (p, k)
```

## Comparison to Standard PLS

| Feature | Standard PLS | PNPLS |
|---------|--------------|-------|
| **Predictor space** | $\mathbb{R}^p$ | $\mathbb{R}^{3p}$ (augmented) |
| **Robustness** | Optional | Built-in (falsity weighting) |
| **Uncertainty** | Not explicit | Explicit (I channel) |
| **Baseline** | Not explicit | Explicit (T channel) |
| **Missing data** | Various methods | EM imputation |
| **Complexity** | $O(knp)$ | $O(3knp) = O(knp)$ (same order) |

## References

1. Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: A basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109-130.

2. Geladi, P., & Kowalski, B. R. (1986). Partial least-squares regression: A tutorial. *Analytica Chimica Acta*, 185, 1-17.

3. Mehmood, T., Liland, K. H., Snipen, L., & Sæbø, S. (2012). A review of variable selection methods in partial least squares regression. *Chemometrics and Intelligent Laboratory Systems*, 118, 62-69.

---

**Next**: [NVIP](nvip.md) - Decomposing variable importance across neutrosophic channels.


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# NVIP: Neutrosophic Variable Importance in Projection

**Module**: `neutrohydro.nvip`

## Overview

NVIP extends the classical VIP (Variable Importance in Projection) metric to **neutrosophic triplet data**, enabling **L2-additive decomposition** of variable importance across Truth, Indeterminacy, and Falsity channels.

**Core Innovation**: Variable importance can be **unambiguously partitioned** into baseline and perturbation components.

## Mathematical Foundation

### 1. Classical VIP (Standard PLS)

For standard PLS with $k$ components, the VIP for variable $j$ is:

$$\text{VIP}(j) = \sqrt{p \cdot \frac{\sum_{h=1}^k \text{SSY}_h \cdot \frac{w_h^2(j)}{\|w_h\|^2}}{\sum_{h=1}^k \text{SSY}_h}}$$

where:
- $p$ = number of variables
- $\text{SSY}_h = q_h^2 \cdot (t_h^\top t_h)$ = response variance explained by component $h$
- $w_h(j)$ = weight for variable $j$ in component $h$

**Interpretation**: VIP$(j) \geq 1$ indicates variable $j$ is **important** for prediction.

**Property**: $\sum_{j=1}^p \text{VIP}^2(j) = p$

### 2. NVIP Extension

For PNPLS with augmented space $\mathbb{R}^{3p}$:

#### Weight Partitioning

$$w_h = \begin{bmatrix} w_{T,h} \\ w_{I,h} \\ w_{F,h} \end{bmatrix} \in \mathbb{R}^{3p}$$

where $w_{T,h}, w_{I,h}, w_{F,h} \in \mathbb{R}^p$ are weights for each channel.

#### Per-Variable, Per-Channel Squared Weights

$$\omega_{T,h}(j) = w_{T,h}^2(j)$$
$$\omega_{I,h}(j) = w_{I,h}^2(j)$$
$$\omega_{F,h}(j) = w_{F,h}^2(j)$$

#### Total Squared Weight (Normalization)

$$\Omega_h = \sum_{m=1}^p \left[\omega_{T,h}(m) + \omega_{I,h}(m) + \omega_{F,h}(m)\right] = \|w_h\|^2$$

#### Channel-Wise VIP Definitions

**Truth VIP**:

$$\text{VIP}_T(j) = \sqrt{p \cdot \frac{\sum_{h=1}^k \text{SSY}_h \cdot \frac{\omega_{T,h}(j)}{\Omega_h}}{\sum_{h=1}^k \text{SSY}_h}}$$

**Indeterminacy VIP**:

$$\text{VIP}_I(j) = \sqrt{p \cdot \frac{\sum_{h=1}^k \text{SSY}_h \cdot \frac{\omega_{I,h}(j)}{\Omega_h}}{\sum_{h=1}^k \text{SSY}_h}}$$

**Falsity VIP**:

$$\text{VIP}_F(j) = \sqrt{p \cdot \frac{\sum_{h=1}^k \text{SSY}_h \cdot \frac{\omega_{F,h}(j)}{\Omega_h}}{\sum_{h=1}^k \text{SSY}_h}}$$

**Aggregated VIP**:

$$\text{VIP}_{\text{agg}}(j) = \sqrt{p \cdot \frac{\sum_{h=1}^k \text{SSY}_h \cdot \frac{\omega_{T,h}(j) + \omega_{I,h}(j) + \omega_{F,h}(j)}{\Omega_h}}{\sum_{h=1}^k \text{SSY}_h}}$$

## L2 Decomposition Theorem

### 3. Main Result

**Theorem** (NVIP L2 Additivity):

For each variable $j = 1, \ldots, p$:

$$\boxed{\text{VIP}_{\text{agg}}^2(j) = \text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j)}$$

### 4. Proof

**Define energies**:

$$E_T(j) = \text{VIP}_T^2(j), \quad E_I(j) = \text{VIP}_I^2(j), \quad E_F(j) = \text{VIP}_F^2(j)$$

**Expand**:

$$E_T(j) = p \cdot \frac{\sum_{h=1}^k \text{SSY}_h \cdot \frac{\omega_{T,h}(j)}{\Omega_h}}{\sum_{h=1}^k \text{SSY}_h}$$

Similarly for $E_I(j)$ and $E_F(j)$.

**Sum**:

$$E_T(j) + E_I(j) + E_F(j) = p \cdot \frac{\sum_{h=1}^k \text{SSY}_h \cdot \frac{\omega_{T,h}(j) + \omega_{I,h}(j) + \omega_{F,h}(j)}{\Omega_h}}{\sum_{h=1}^k \text{SSY}_h}$$

**By definition** of $\text{VIP}_{\text{agg}}$:

$$E_T(j) + E_I(j) + E_F(j) = \text{VIP}_{\text{agg}}^2(j)$$

Therefore:

$$\text{VIP}_{\text{agg}}^2(j) = \text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j) \quad \square$$

### 5. Corollary: Energy Conservation

$$\sum_{j=1}^p \text{VIP}_{\text{agg}}^2(j) = p$$

and:

$$\sum_{j=1}^p E_T(j) + \sum_{j=1}^p E_I(j) + \sum_{j=1}^p E_F(j) = p$$

## Interpretation

### 6. Channel Energies

- **$E_T(j)$**: Importance of **baseline** (Truth) for ion $j$
- **$E_I(j)$**: Importance of **uncertainty** (Indeterminacy) for ion $j$
- **$E_F(j)$**: Importance of **perturbation** (Falsity) for ion $j$

**Perturbation energy**:

$$E_P(j) = E_I(j) + E_F(j)$$

### 7. Variable Selection

Use $\text{VIP}_{\text{agg}}(j)$ for overall importance:

- $\text{VIP}_{\text{agg}}(j) \geq 1$: Ion $j$ is **important** for prediction
- $\text{VIP}_{\text{agg}}(j) < 1$: Ion $j$ is **less important**

**Rationale**: Threshold of 1 comes from the conservation property $\sum_j \text{VIP}^2 = p$. If all ions equally important, each would have $\text{VIP} = 1$.

### 8. Attribution Fractions

For ion $j$, **baseline fraction**:

$$\pi_G(j) = \frac{E_T(j)}{E_T(j) + E_P(j)} \in [0, 1]$$

**Perturbation fraction**:

$$\pi_A(j) = 1 - \pi_G(j) = \frac{E_P(j)}{E_T(j) + E_P(j)}$$

See [Attribution](attribution.md) for detailed discussion.

## Algorithm

### Input

From fitted PNPLS model:
- Weight matrix $W \in \mathbb{R}^{3p \times k}$
- Score matrix $T \in \mathbb{R}^{n \times k}$
- Response loadings $q \in \mathbb{R}^k$

### Computation

```python
# 1. Partition weights by channel
W_T = W[:p, :]      # (p, k)
W_I = W[p:2p, :]    # (p, k)
W_F = W[2p:, :]     # (p, k)

# 2. Squared weights
omega_T = W_T ** 2  # (p, k)
omega_I = W_I ** 2  # (p, k)
omega_F = W_F ** 2  # (p, k)

# 3. Total squared weight per component
Omega = np.sum(W ** 2, axis=0)  # (k,)

# 4. Response energy per component
t_sq = np.sum(T ** 2, axis=0)  # (k,)
SSY = q ** 2 * t_sq            # (k,)
total_SSY = np.sum(SSY)

# 5. Weighted ratio
weight_ratio = SSY / (Omega + eps)  # (k,)

# 6. Energies
E_T = p * (omega_T @ weight_ratio) / total_SSY  # (p,)
E_I = p * (omega_I @ weight_ratio) / total_SSY  # (p,)
E_F = p * (omega_F @ weight_ratio) / total_SSY  # (p,)

# 7. VIPs
VIP_T = np.sqrt(np.maximum(E_T, 0))
VIP_I = np.sqrt(np.maximum(E_I, 0))
VIP_F = np.sqrt(np.maximum(E_F, 0))
VIP_agg = np.sqrt(E_T + E_I + E_F)
```

## Properties

### 9. Non-Negativity

$$\text{VIP}_c(j) \geq 0 \quad \forall j, c \in \{T, I, F, \text{agg}\}$$

**Proof**: Squared weights $\omega_{c,h}(j) \geq 0$ and SSY$_h \geq 0$.

### 10. Scale Invariance

If variables are standardized, VIP is **dimensionless** and comparable across ions.

### 11. Component Monotonicity

Adding more components (increasing $k$) generally:
- Increases $\text{VIP}_{\text{agg}}$ (more variance explained)
- May shift balance among $\text{VIP}_T, \text{VIP}_I, \text{VIP}_F$

**Implication**: Use same $k$ when comparing across analyses.

## Bootstrap Confidence Intervals

### 12. Procedure

1. Generate $B$ bootstrap samples (resampling with replacement)
2. For each bootstrap sample $b = 1, \ldots, B$:
   - Fit PNPLS
   - Compute NVIP
   - Store $\text{VIP}^{(b)}_T, \text{VIP}^{(b)}_I, \text{VIP}^{(b)}_F, \text{VIP}^{(b)}_{\text{agg}}$
3. Compute statistics:
   - Mean: $\bar{\text{VIP}}_c = \frac{1}{B} \sum_b \text{VIP}^{(b)}_c$
   - Std: $\text{sd}(\text{VIP}_c) = \sqrt{\frac{1}{B-1} \sum_b (\text{VIP}^{(b)}_c - \bar{\text{VIP}}_c)^2}$
   - 95% CI: $[\text{percentile}_{2.5}, \text{percentile}_{97.5}]$

### 13. Interpretation of Uncertainty

- **Wide CI**: Variable importance is **unstable** across resamples
  - May indicate: overfitting, insufficient data, high noise
- **Narrow CI**: Variable importance is **stable**
  - Reliable ranking

**Use**: Rank variables by mean VIP; use CI to assess uncertainty.

## Diagnostics

### 14. Verification of L2 Decomposition

After computing NVIP, **always verify**:

$$|\text{VIP}_{\text{agg}}^2(j) - [\text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j)]| < \epsilon$$

for all $j$, where $\epsilon = 10^{-10}$ (numerical tolerance).

**If violated**: Implementation error (should never happen with correct code).

```python
from neutrohydro.nvip import verify_l2_decomposition

assert verify_l2_decomposition(nvip_result, tol=1e-10)
```

### 15. Component Energy Distribution

Plot SSY$_h$ vs. $h$:
- **Steep drop**: First few components dominate
- **Gradual decline**: Many components contribute

**Heuristic**: If SSY$_1 \gg$ SSY$_2 \gg \ldots$, few components suffice.

### 16. Channel Dominance

For each ion, classify:
- **Truth-dominant**: $E_T(j) > E_I(j) + E_F(j)$
- **Perturbation-dominant**: $E_T(j) < E_I(j) + E_F(j)$

Plot histogram of $\pi_G(j) = \frac{E_T(j)}{E_T(j) + E_P(j)}$:
- **Unimodal near 0**: Most ions perturbation-driven
- **Unimodal near 1**: Most ions baseline-driven
- **Bimodal**: Mixed population

## Visualization

### 17. Stacked Bar Plot

For each ion $j$, plot stacked bars:
- Bottom: $E_T(j)$ (Truth, blue)
- Middle: $E_I(j)$ (Indeterminacy, orange)
- Top: $E_F(j)$ (Falsity, red)

**Height** = $\text{VIP}_{\text{agg}}^2(j)$

**Horizontal line** at $y = 1$ marks importance threshold.

### 18. Scatter Plot: $\pi_G$ vs. $\text{VIP}_{\text{agg}}$

Each ion is a point $(x, y) = (\pi_G(j), \text{VIP}_{\text{agg}}(j))$:
- **Upper left** ($\pi_G \approx 0, \text{VIP} > 1$): Important, perturbation-driven
- **Upper right** ($\pi_G \approx 1, \text{VIP} > 1$): Important, baseline-driven
- **Lower** ($\text{VIP} < 1$): Less important

**Quadrants**:
- Q1 ($\pi_G > 0.5, \text{VIP} > 1$): Important baseline ions
- Q2 ($\pi_G < 0.5, \text{VIP} > 1$): Important perturbation ions

### 19. Heatmap: Channel VIPs

Heatmap with:
- Rows: Ions
- Columns: $\text{VIP}_T, \text{VIP}_I, \text{VIP}_F, \text{VIP}_{\text{agg}}$
- Color: VIP magnitude

Sort rows by $\text{VIP}_{\text{agg}}$ (descending) to highlight most important ions.

## Usage Example

```python
from neutrohydro import PNPLS, compute_nvip, verify_l2_decomposition
from neutrohydro.nvip import nvip_to_dataframe

# Fit model
model = PNPLS(n_components=5)
model.fit(triplets, y_std)

# Compute NVIP
nvip = compute_nvip(model)

# Verify L2 decomposition
assert verify_l2_decomposition(nvip)

# Identify important ions
important = nvip.VIP_agg >= 1
print(f"Important ions: {np.where(important)[0]}")

# Convert to DataFrame
df = nvip_to_dataframe(nvip, feature_names=ion_names)
print(df.sort_values('VIP_agg', ascending=False))

# Plot stacked VIP energies
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(range(p), nvip.E_T, label='Truth')
ax.bar(range(p), nvip.E_I, bottom=nvip.E_T, label='Indeterminacy')
ax.bar(range(p), nvip.E_F, bottom=nvip.E_T + nvip.E_I, label='Falsity')
ax.axhline(1, color='k', linestyle='--', label='Threshold')
ax.set_xlabel('Ion')
ax.set_ylabel('VIP² Energy')
ax.legend()
plt.show()
```

## References

1. Wold, S., Johansson, E., & Cocchi, M. (1993). PLS—partial least-squares projections to latent structures. In *3D QSAR in drug design* (pp. 523-550).

2. Mehmood, T., Liland, K. H., Snipen, L., & Sæbø, S. (2012). A review of variable selection methods in partial least squares regression. *Chemometrics and Intelligent Laboratory Systems*, 118, 62-69.

3. Farrés, M., Platikanov, S., Tsakovski, S., & Tauler, R. (2015). Comparison of the variable importance in projection (VIP) and of the selectivity ratio (SR) methods for variable selection and interpretation. *Journal of Chemometrics*, 29(10), 528-536.

---

**Next**: [Attribution Metrics](attribution.md) - NSR, $π_G$, and sample-level baseline fractions.


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Attribution Metrics: NSR and Baseline Fractions

**Module**: `neutrohydro.attribution`

## Overview

Attribution metrics quantify the **baseline vs. perturbation** character of ions and samples based on NVIP energies. Two levels of attribution:

1. **Ion-level**: NSR and π_G quantify baseline fraction per ion
2. **Sample-level**: G_i quantifies baseline fraction per water sample

## Mathematical Foundation

### 1. Ion-Level Attribution

#### 1.1 Energy Partition

From NVIP (see [nvip.md](nvip.md)):

**Truth energy** (baseline):
$$E_T(j) = \text{VIP}_T^2(j)$$

**Perturbation energy**:
$$E_P(j) = E_I(j) + E_F(j) = \text{VIP}_I^2(j) + \text{VIP}_F^2(j)$$

#### 1.2 Baseline Fraction π_G

$$\boxed{\pi_G(j) = \frac{E_T(j)}{E_T(j) + E_P(j)} \in [0, 1]}$$

**Interpretation**:
- $\pi_G(j) \approx 1$: Ion $j$ prediction driven by **baseline**
- $\pi_G(j) \approx 0$: Ion $j$ prediction driven by **perturbation**
- $\pi_G(j) \approx 0.5$: Mixed contribution

**Perturbation fraction**:
$$\pi_A(j) = 1 - \pi_G(j) = \frac{E_P(j)}{E_T(j) + E_P(j)}$$

**Conservation**:
$$\pi_G(j) + \pi_A(j) = 1$$

#### 1.3 Neutrosophic Source Ratio (NSR)

$$\boxed{\text{NSR}(j) = \frac{E_T(j) + \epsilon}{E_P(j) + \epsilon}}$$

where $\epsilon > 0$ (default: $10^{-10}$) prevents division by zero.

**Interpretation**:
- NSR$(j) \gg 1$: Baseline-dominant
- NSR$(j) \approx 1$: Balanced
- NSR$(j) \ll 1$: Perturbation-dominant

**Relationship to π_G**:

$$\pi_G(j) = \frac{\text{NSR}(j)}{1 + \text{NSR}(j)}$$

$$\text{NSR}(j) = \frac{\pi_G(j)}{1 - \pi_G(j)}$$

NSR is the **odds ratio** version of the fraction π_G.

#### 1.4 Classification

Choose threshold $\gamma \in (0.5, 1)$ (default: $\gamma = 0.7$):

- **Baseline-dominant**: $\pi_G(j) \geq \gamma$
- **Perturbation-dominant**: $\pi_G(j) \leq 1 - \gamma$
- **Mixed**: $1 - \gamma < \pi_G(j) < \gamma$

**Rationale for γ = 0.7**:
- Corresponds to NSR $\approx 2.33$ (odds of 2.33:1)
- Strong but not extreme threshold
- Balances sensitivity and specificity

## 2. Sample-Level Attribution

### 2.1 Net Contributions

For sample $i$, ion $j$, the **net contribution** to prediction is:

$$c_{ij} = (\widetilde{X}_T)_{ij} \beta_T(j) + (\widetilde{X}_I)_{ij} \beta_I(j) + (\widetilde{X}_F)_{ij} \beta_F(j)$$

where:
- $\widetilde{X}_T, \widetilde{X}_I, \widetilde{X}_F$ are weighted channel matrices
- $\beta_T, \beta_I, \beta_F$ are regression coefficients partitioned by channel

### 2.2 Attribution Mass

To avoid sign cancellation, use **absolute contributions**:

$$w_{ij} = |c_{ij}|$$

**Interpretation**: $w_{ij}$ quantifies how much ion $j$ contributes (in magnitude) to predicting sample $i$.

### 2.3 Sample Baseline Fraction G_i

$$\boxed{G_i = \frac{\sum_{j=1}^p \pi_G(j) \cdot w_{ij}}{\sum_{j=1}^p w_{ij}} \in [0, 1]}$$

**Weighted average** of ion-level baseline fractions, using attribution masses as weights.

**Interpretation**:
- $G_i \approx 1$: Sample $i$ prediction driven by baseline-dominant ions
- $G_i \approx 0$: Sample $i$ prediction driven by perturbation-dominant ions
- $G_i \approx 0.5$: Mixed

**Sample perturbation fraction**:
$$A_i = 1 - G_i$$

**Conservation**:
$$G_i + A_i = 1$$

### 2.4 Precise Interpretation

$G_i$ is **not** the fraction of mass in baseline sources. Rather:

> $G_i$ is the fraction of the model's **absolute predictive attribution mass** carried by baseline-dominant ions.

Physical interpretation of baseline vs. perturbation requires **external validation** (see Section 5).

## 3. Algorithm

### Ion-Level

**Input**: NVIP result $(E_T, E_I, E_F)$, threshold $\gamma$, epsilon $\epsilon$

**Output**: NSR, $π_G$, $π_A$, classification

```python
# Perturbation energy
E_P = E_I + E_F

# NSR (odds ratio)
NSR = (E_T + epsilon) / (E_P + epsilon)

# Baseline fraction
total_energy = E_T + E_P
pi_G = np.where(total_energy > epsilon, E_T / total_energy, 0.5)
pi_G = np.clip(pi_G, 0.0, 1.0)

# Perturbation fraction
pi_A = 1.0 - pi_G

# Classification
classification = np.empty(p, dtype='U12')
classification[:] = "mixed"
classification[pi_G >= gamma] = "baseline"
classification[pi_G <= 1 - gamma] = "perturbation"
```

### Sample-Level

**Input**: Fitted model (coefficients), triplet data, NSR result ($π_G$)

**Output**: G, A, attribution masses w, contributions c

```python
# Get coefficients by channel
beta_T, beta_I, beta_F = get_coefficients_by_channel(model)

# Weighted data
W_precision = np.exp(-lambda_F * triplets.F)
X_T_w = W_precision * triplets.T
X_I_w = W_precision * sqrt(rho_I) * triplets.I
X_F_w = W_precision * sqrt(rho_F) * triplets.F

# Net contributions
c = X_T_w * beta_T + X_I_w * beta_I + X_F_w * beta_F  # (n, p)

# Attribution mass
w = np.abs(c)

# Sample baseline fraction
numerator = w @ pi_G          # (n,)
denominator = w.sum(axis=1)   # (n,)
G = np.where(denominator > eps, numerator / denominator, 0.5)
G = np.clip(G, 0.0, 1.0)

# Sample perturbation fraction
A = 1.0 - G
```

## 4. Properties

### 4.1 Bounds

1. $\pi_G(j), \pi_A(j) \in [0, 1]$ for all $j$
2. $G_i, A_i \in [0, 1]$ for all $i$
3. NSR$(j) \in [0, \infty]$

### 4.2 Conservation

1. $\pi_G(j) + \pi_A(j) = 1$ for all $j$
2. $G_i + A_i = 1$ for all $i$

#### Proof of Ion-Level Conservation

**Theorem**: $\pi_G(j) + \pi_A(j) = 1$ for all ions $j$.

**Proof**:

By definition (Section 1.2):
$$\pi_G(j) = \frac{E_T(j)}{E_T(j) + E_P(j)}$$

$$\pi_A(j) = \frac{E_P(j)}{E_T(j) + E_P(j)}$$

Sum:
$$\pi_G(j) + \pi_A(j) = \frac{E_T(j)}{E_T(j) + E_P(j)} + \frac{E_P(j)}{E_T(j) + E_P(j)}$$

$$= \frac{E_T(j) + E_P(j)}{E_T(j) + E_P(j)} = 1$$

$\square$

**Note**: This holds even when $E_T(j) + E_P(j) = 0$ by the convention $\pi_G(j) = 0.5$ in this degenerate case (see algorithm).

#### Proof of Sample-Level Conservation

**Theorem**: $G_i + A_i = 1$ for all samples $i$.

**Proof**:

By definition (Section 2.3):
$$G_i = \frac{\sum_{j=1}^p \pi_G(j) \cdot w_{ij}}{\sum_{j=1}^p w_{ij}}$$

where $w_{ij} = |c_{ij}|$ are attribution masses.

By definition of $A_i$:
$$A_i = 1 - G_i$$

Substituting:
$$A_i = 1 - \frac{\sum_{j=1}^p \pi_G(j) \cdot w_{ij}}{\sum_{j=1}^p w_{ij}}$$

$$= \frac{\sum_{j=1}^p w_{ij} - \sum_{j=1}^p \pi_G(j) \cdot w_{ij}}{\sum_{j=1}^p w_{ij}}$$

$$= \frac{\sum_{j=1}^p w_{ij}(1 - \pi_G(j))}{\sum_{j=1}^p w_{ij}}$$

Using ion-level conservation $1 - \pi_G(j) = \pi_A(j)$:
$$A_i = \frac{\sum_{j=1}^p w_{ij} \cdot \pi_A(j)}{\sum_{j=1}^p w_{ij}}$$

This is the **weighted average** of perturbation fractions, so:
$$G_i + A_i = \frac{\sum_{j=1}^p w_{ij}(\pi_G(j) + \pi_A(j))}{\sum_{j=1}^p w_{ij}} = \frac{\sum_{j=1}^p w_{ij}}{\sum_{j=1}^p w_{ij}} = 1$$

$\square$

### 4.3 Monotonicity

If $E_T(j)$ increases (holding $E_P(j)$ fixed), then:
- $\pi_G(j)$ increases
- NSR$(j)$ increases
- Classification may change to "baseline"

### 4.4 Extremes

**All baseline** ($E_P(j) = 0$):
- $\pi_G(j) = 1$
- NSR$(j) \to \infty$

**All perturbation** ($E_T(j) = 0$):
- $\pi_G(j) = 0$
- NSR$(j) = 0$

**Equal** ($E_T(j) = E_P(j)$):
- $\pi_G(j) = 0.5$
- NSR$(j) = 1$

## 5. Interpretation and Validation

### 5.1 Operational Definition

π_G and G are **mathematically well-defined** operational quantities. However, interpreting them as **physical source fractions** (e.g., geogenic vs. anthropogenic) requires:

1. **External evidence**:
   - Spatial patterns (urban vs. rural)
   - Temporal trends (pre/post contamination)
   - Isotopic tracers
   - Land use correlations

2. **Baseline validation**:
   - Does the chosen baseline operator (median, low-rank, etc.) capture the "natural" state?
   - Are perturbations truly anomalies, not natural variability?

3. **Domain expertise**:
   - Hydrogeological context
   - Known contamination sources
   - Historical water quality data

### 5.2 Common Pitfalls

**Pitfall 1**: Assuming $π_G$ = fraction of mass from geogenic sources

**Reality**: $π_G$ quantifies contribution to **prediction importance**, not mass balance.

**Pitfall 2**: Ignoring baseline choice

**Reality**: Different baseline operators (median vs. low-rank) yield different $π_G$.

**Pitfall 3**: Treating classification as hard truth

**Reality**: Threshold $γ$ is somewhat arbitrary; use as guide, not absolute.

### 5.3 Recommended Workflow

1. Compute $π_G$ and G with **multiple baseline types** (median, low-rank)
2. Check **robustness**: Do classifications agree?
3. **Cross-validate** with external data:
   - Do "baseline-dominant" samples cluster spatially in pristine areas?
   - Do "perturbation-dominant" samples correspond to known contamination?
4. **Sensitivity analysis**: Vary $γ$; check if rankings change drastically

## 6. Summary Statistics

### 6.1 Ion-Level Summary

For a dataset:

- **Mean $π_G$**: Average baseline fraction across all ions
- **N baseline**: Number of baseline-dominant ions
- **N perturbation**: Number of perturbation-dominant ions
- **N mixed**: Number of mixed ions
- **Most baseline ion**: arg max$_j$ $π_G(j)$
- **Most perturbation ion**: arg min$_j$ $π_G(j)$

### 6.2 Sample-Level Summary

- **Mean G**: Average baseline fraction across all samples
- **Std G**: Variability in baseline character
- **Frac baseline samples**: Fraction with $G \geq \gamma$
- **Frac perturbation samples**: Fraction with $G \leq 1 - \gamma$

### 6.3 Example Summary

```
Attribution Summary
-------------------
Ion-Level:
  Total ions: 7
  Baseline-dominant: 2 (Ca²⁺, HCO₃⁻)
  Perturbation-dominant: 3 (Na⁺, Cl⁻, NO₃⁻)
  Mixed: 2 (Mg²⁺, SO₄²⁻)
  Mean π_G: 0.42

Sample-Level:
  Total samples: 100
  Mean G: 0.38 (±0.15)
  Baseline samples (G ≥ 0.7): 18 (18%)
  Perturbation samples (G ≤ 0.3): 45 (45%)
  Mixed samples: 37 (37%)
```

## 7. Visualization

### 7.1 Bar Plot: $π_G$ by Ion

Horizontal bar plot:
- X-axis: $π_G$ (0 to 1)
- Y-axis: Ion names
- Color: By classification (blue=baseline, red=perturbation, gray=mixed)
- Vertical lines at $γ$ and $1-γ$

### 7.2 Histogram: G Distribution

Histogram of sample baseline fractions:
- X-axis: $G_i$ values
- Y-axis: Frequency
- Vertical lines at $γ$ and $1-γ$
- Annotate fractions in each region

### 7.3 Spatial Map: G (if spatial data)

If samples have coordinates $(x_i, y_i)$:
- Scatter plot with color = $G_i$
- Colormap: Blue (low G, perturbation) to Red (high G, baseline)
- Overlay known contamination sources

### 7.4 2D Plot: $π_G$ vs. $VIP_{agg}$
See [nvip.md](nvip.md) Section 18.

## 8. Usage Example

```python
from neutrohydro import compute_nvip, compute_nsr, compute_sample_baseline_fraction
from neutrohydro.attribution import attribution_summary, nsr_to_dataframe

# After fitting model
nvip = compute_nvip(model)

# Ion-level attribution
nsr = compute_nsr(nvip, epsilon=1e-10, gamma=0.7)

print(f"Baseline ions: {sum(nsr.classification == 'baseline')}")
print(f"Perturbation ions: {sum(nsr.classification == 'perturbation')}")

# Sample-level attribution
sample_attr = compute_sample_baseline_fraction(model, triplets, nsr)

print(f"Mean G: {sample_attr.G.mean():.3f}")
print(f"Std G: {sample_attr.G.std():.3f}")

# Summary statistics
summary = attribution_summary(nsr, sample_attr, ion_names)
print(summary)

# DataFrame export
df_nsr = nsr_to_dataframe(nsr, ion_names)
print(df_nsr.sort_values('pi_G', ascending=False))
```

## 9. Sensitivity to Hyperparameters

| Parameter | Effect on $π_G$ | Recommendation |
|-----------|---------------|----------------|
| Baseline type | **High** | Test multiple; check robustness |
| $\rho_I$ | Medium (changes $E_I$ weight) | Default 1.0 unless strong prior |
| $\rho_F$ | Medium (changes $E_F$ weight) | Default 1.0 unless strong prior |
| $\lambda_F$ | Low (affects weights, not VIP) | Default 1.0 |
| $k$ (components) | Medium (more components → finer decomposition) | Cross-validate |
| $\gamma$ (threshold) | **Classification only** | 0.7 reasonable; test 0.6-0.8 |

## 10. Comparison to Other Methods

| Method | Outputs | Interpretation | Limitations |
|--------|---------|----------------|-------------|
| **NVIP/NSR** | $π_G$ per ion, G per sample | Baseline vs. perturbation fraction | Requires validation for physical meaning |
| **PMF** | Source contributions | Mass apportionment | Compositional, non-negative |
| **PCA loadings** | Variable weights | Pattern strength | Not directly about sources |
| **APCS** | Source contributions | Absolute mass | Requires source profiles |

**Advantages of NSR**:
- **Non-compositional**: Works in absolute space
- **Explicit uncertainty**: I channel captures ambiguity
- **L2 decomposition**: Unambiguous partitioning

## References

1. Paatero, P., & Tapper, U. (1994). Positive matrix factorization: A non-negative factor model with optimal utilization of error estimates of data values. *Environmetrics*, 5(2), 111-126.

2. Thurston, G. D., & Spengler, J. D. (1985). A quantitative assessment of source contributions to inhalable particulate matter pollution in metropolitan Boston. *Atmospheric Environment*, 19(1), 9-25.

3. Henry, R. C. (1987). Current factor analysis receptor models are ill-posed. *Atmospheric Environment*, 21(8), 1815-1820.

---

**Next**: [Mineral Stoichiometric Inversion](minerals.md) - Inferring plausible mineral sources.

## Author

- **Dickson Abdul-Wahab**, University of Ghana, Ghana
- Email: <mailto:dabdul-wahab@live.com>
- ORCID: <https://orcid.org/0000-0001-7446-5909>
- LinkedIn: <https://www.linkedin.com/in/dickson-abdul-wahab-0764a1a9/>
- ResearchGate: <https://www.researchgate.net/profile/Dickson-Abdul-Wahab>


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Mineral Stoichiometric Inversion

**Module**: `neutrohydro.minerals`

## Overview

The mineral inference module uses **stoichiometric inversion** to estimate plausible mineral contributions from ion concentration data. This provides a **geochemical interpretation** of water composition in terms of **mineral dissolution/weathering** sources.

**Key innovation**: Uses baseline fractions $π_G$ to **weight ions**, emphasizing baseline-dominant ions in the inversion.

## Mathematical Foundation

### 1. Stoichiometric Model

#### 1.1 Forward Model

Ion concentrations arise from **mineral dissolution**:

$$c = A s + r$$

where:
- $c \in \mathbb{R}^m$: Observed ion concentrations (in **meq/L**, recommended)
- $A \in \mathbb{R}^{m \times K}$: **Stoichiometric matrix** (mineral compositions)
- $s \in \mathbb{R}^K_{\geq 0}$: **Mineral contributions** (non-negative)
- $r \in \mathbb{R}^m$: **Residual** (unmodeled processes)

**Units**:
- $c_\ell$ in meq/L (milli-equivalents per liter) ensures **charge balance**
- $s_k$ in arbitrary units (relative contributions)
- $A_{\ell k}$ = meq of ion $\ell$ per unit mineral $k$

#### 1.2 Stoichiometric Matrix

Column $k$ of $A$ encodes the composition of mineral $k$:

$$A_{:,k} = \begin{bmatrix} a_{1k} \\ \vdots \\ a_{mk} \end{bmatrix}$$

where $a_{\ell k}$ = equivalents of ion $\ell$ produced per unit dissolution of mineral $k$.

**Example** (Halite, NaCl):

$$\text{NaCl} \to \text{Na}^+ + \text{Cl}^-$$

In meq/L, dissolution of 1 mole NaCl produces:
- 1 meq Na⁺
- 1 meq Cl⁻

So for halite column:
```
Ion order: [Ca²⁺, Mg²⁺, Na⁺, K⁺, HCO₃⁻, Cl⁻, SO₄²⁻, NO₃⁻, F⁻]
Halite:    [0,    0,    1.0, 0,   0,      1.0, 0,     0,     0  ]
```

See Section 4 for standard mineral library.

### 2. Inverse Problem

#### 2.1 Unweighted NNLS

$$\hat{s} = \arg\min_{s \geq 0} \|c - As\|^2$$

**Non-negative least squares (NNLS)**:
- Ensures $s_k \geq 0$ (minerals can only dissolve, not precipitate)
- Convex optimization (unique solution)
- Fast algorithms available (SciPy, active set method)

#### 2.2 Weighted NNLS (NeutroHydro Innovation)

Use baseline fractions $π_G$ to **emphasize baseline-dominant ions**:

$$\hat{s} = \arg\min_{s \geq 0} \|D(c - As)\|^2$$

where $D = \text{diag}(d_1, \ldots, d_m)$ and:

$$d_\ell = \pi_G(\text{ion}_\ell)^\eta$$

**Hyperparameters**:
- $\eta \geq 1$: Weighting exponent (default: 1.0)
  - $\eta = 1$: Linear weighting
  - $\eta > 1$: Stronger emphasis on baseline

**Rationale**: Baseline-dominant ions reflect **natural geochemical processes** (mineral weathering), while perturbation-dominant ions may reflect **anthropogenic inputs** (fertilizers, contamination, ion exchange) not modeled by simple dissolution.

#### 2.3 Equivalence to Standard NNLS

Weighted NNLS is equivalent to:

$$\hat{s} = \arg\min_{s \geq 0} \|c_w - A_w s\|^2$$

where:
$$c_w = D c, \quad A_w = D A$$

**Implementation**: Transform data, solve standard NNLS.

##### Proof of Equivalence

**Theorem**: The weighted NNLS problem:
$$\min_{s \geq 0} \|D(c - As)\|^2$$

is equivalent to the standard NNLS problem:
$$\min_{s \geq 0} \|c_w - A_w s\|^2$$

where $c_w = Dc$ and $A_w = DA$.

**Proof**:

Expand the weighted objective:
$$J(s) = \|D(c - As)\|^2$$

$$= (c - As)^T D^T D (c - As)$$

Since $D = \text{diag}(d_1, \ldots, d_m)$ is diagonal:
$$D^T D = D^2 = \text{diag}(d_1^2, \ldots, d_m^2)$$

Therefore:
$$J(s) = (c - As)^T D^2 (c - As)$$

$$= \sum_{\ell=1}^m d_\ell^2 (c_\ell - (As)_\ell)^2$$

Now consider the transformed problem with $c_w = Dc$ and $A_w = DA$:
$$J_w(s) = \|c_w - A_w s\|^2$$

$$= \|Dc - DAs\|^2$$

$$= \|D(c - As)\|^2$$

$$= J(s)$$

Since the objectives are **identical** for all $s$, they have the same minimizer:
$$\arg\min_{s \geq 0} J(s) = \arg\min_{s \geq 0} J_w(s)$$

$\square$

**Computational advantage**: Standard NNLS solvers (e.g., `scipy.optimize.nnls`) can be used directly after pre-multiplying by $D$.

### 3. Residual Diagnostics

#### 3.1 Weighted Residual

$$r = c - A\hat{s}$$

$$\|r\|_D = \|D r\| = \sqrt{\sum_{\ell=1}^m d_\ell^2 r_\ell^2}$$

**Interpretation**: Goodness of fit, accounting for ion weights.

#### 3.2 Plausibility Criteria

Mineral $k$ is **plausible** in sample $i$ if:

1. **Sufficient contribution**: $\hat{s}_{ik} > \tau_s$ (default: 0.01)
2. **Good fit**: $\|r_i\|_D \leq \tau_r$ (default: 1.0)

**Both** criteria must be satisfied.

**Rationale**:
- Criterion 1: Avoids spurious tiny contributions
- Criterion 2: Ensures model fits the data

### 4. Standard Mineral Library

NeutroHydro includes an expanded "Scientific Research Grade" library of **24 minerals/endmembers**, covering silicates, carbonates, evaporites, and specific anthropogenic markers.

#### 4.1 Natural Minerals (Geogenic)

| Mineral | Formula | Key Ions | Description |
|---------|---------|----------|-------------|
| **Calcite** | CaCO₃ | Ca²⁺, HCO₃⁻ | Carbonate dissolution |
| **Dolomite** | CaMg(CO₃)₂ | Ca²⁺, Mg²⁺, HCO₃⁻ | Carbonate dissolution |
| **Magnesite** | MgCO₃ | Mg²⁺, HCO₃⁻ | Magnesium carbonate |
| **Gypsum** | CaSO₄·2H₂O | Ca²⁺, SO₄²⁻ | Sulfate dissolution |
| **Anhydrite** | CaSO₄ | Ca²⁺, SO₄²⁻ | Anhydrous sulfate |
| **Halite** | NaCl | Na⁺, Cl⁻ | Saline deposits/intrusion |
| **Sylvite** | KCl | K⁺, Cl⁻ | Potash deposits |
| **Mirabilite** | Na₂SO₄·10H₂O | Na⁺, SO₄²⁻ | Sodium sulfate |
| **Thenardite** | Na₂SO₄ | Na⁺, SO₄²⁻ | Anhydrous sodium sulfate |
| **Glauberite** | Na₂Ca(SO₄)₂ | Na⁺, Ca²⁺, SO₄²⁻ | Mixed sulfate |
| **Epsomite** | MgSO₄·7H₂O | Mg²⁺, SO₄²⁻ | Magnesium sulfate |
| **Fluorite** | CaF₂ | Ca²⁺, F⁻ | Fluoride source |
| **Albite** | NaAlSi₃O₈ | Na⁺, HCO₃⁻ | Plagioclase weathering |
| **Anorthite** | CaAl₂Si₂O₈ | Ca²⁺, HCO₃⁻ | Plagioclase weathering |
| **K-feldspar** | KAlSi₃O₈ | K⁺, HCO₃⁻ | Orthoclase weathering |
| **Biotite** | K(Mg,Fe)₃... | K⁺, Mg²⁺, HCO₃⁻ | Mica weathering |

#### 4.2 Anthropogenic Markers (Pollution Proxies)

These phases are used to fingerprint specific contamination sources.

| Marker | Formula | Key Ions | Interpretation |
|--------|---------|----------|----------------|
| **Niter** | KNO₃ | K⁺, NO₃⁻ | Potassium-based fertilizers |
| **Soda Niter** | NaNO₃ | Na⁺, NO₃⁻ | Sodium-based fertilizers or wastewater |
| **Nitrocalcite** | Ca(NO₃)₂ | Ca²⁺, NO₃⁻ | Calcium nitrate fertilizers |
| **Otavite** | CdCO₃ | Cd²⁺, HCO₃⁻ | Cadmium impurity in phosphate fertilizers |
| **Smithsonite** | ZnCO₃ | Zn²⁺, HCO₃⁻ | Industrial zinc or sewage sludge |
| **Cerussite** | PbCO₃ | Pb²⁺, HCO₃⁻ | Industrial lead or road runoff |
| **Borax** | Na₂B₄O₇ | Na⁺, B | Detergents (wastewater) |
| **Malachite** | Cu₂CO₃(OH)₂ | Cu²⁺, HCO₃⁻ | Pesticides/Fungicides |

#### 4.3 Redox Phases (Optional)

These phases represent biogeochemical sinks (mass loss) or sources (mass gain) driven by redox reactions. They are not enabled by default but can be added for advanced modeling.

| Phase | Formula | Key Ions | Interpretation |
|-------|---------|----------|----------------|
| **Sink_Denitrification** | NO₃⁻ → N₂ | NO₃⁻ (-1), HCO₃⁻ (+1) | Nitrate reduction (Sink) |
| **Sink_SulfateReduction** | SO₄²⁻ → H₂S | SO₄²⁻ (-1), HCO₃⁻ (+1) | Sulfate reduction (Sink) |
| **Source_Nitrification** | NH₄⁺ → NO₃⁻ | NO₃⁻ (+1) | Nitrification (Source) |

**Note**: Negative stoichiometry allows the NNLS solver (which finds positive coefficients) to account for mass loss (negative residuals).

**Note on Trace Metals**: The model includes markers for Cd, Zn, Pb, B, Cu, As, Cr, and U. However, these are **only evaluated if the corresponding ion data is provided**. If metal concentrations are missing, these minerals are automatically excluded from the inversion to prevent "ghost" assignments.

**Full Ion Order** (m = 17):
```
[Ca²⁺, Mg²⁺, Na⁺, K⁺, HCO₃⁻, Cl⁻, SO₄²⁻, NO₃⁻, F⁻, 
 Zn²⁺, Cd²⁺, Pb²⁺, B, Cu²⁺, As, Cr, U]
```

### 5. Adaptive Ion Handling

The `MineralInverter` and pipeline are designed to handle datasets with varying ion availability:

1.  **Data-Driven Mineral Selection**: The model checks which ions are present in the input dataset.
2.  **Automatic Filtering**: Any mineral requiring a missing ion is removed from the candidate list.
    *   *Example*: If `Cd` is not measured, `Otavite` is removed.
    *   *Example*: If `NO3` is not measured, `Niter`, `SodaNiter`, and `Nitrocalcite` are removed.
3.  **Robustness**: This ensures that the model never "hallucinates" a mineral contribution based on missing data, while still allowing for sophisticated forensic analysis when comprehensive data is available.

### 6. Algorithm

#### Fitting (Per Sample)

**Input**:
- Ion vector $c_i \in \mathbb{R}^m$ (in meq/L)
- Stoichiometric matrix $A \in \mathbb{R}^{m \times K}$
- Baseline fractions $\pi_G \in \mathbb{R}^m$ (optional)
- Hyperparameters $\eta, \tau_s, \tau_r$

**Output**:
- Mineral contributions $\hat{s}_i \in \mathbb{R}^K$
- Residual $r_i \in \mathbb{R}^m$
- Plausibility mask $P_i \in \{0, 1\}^K$

```python
# 1. Compute weights
if pi_G is None:
    D = np.ones(m)
else:
    D = pi_G ** eta

# 2. Transform system
c_w = D * c_i
A_w = D[:, np.newaxis] * A

# 3. Solve NNLS
s_hat, rnorm = scipy.optimize.nnls(A_w, c_w)

# 4. Compute residual (original space)
r = c_i - A @ s_hat
r_norm_weighted = np.linalg.norm(D * r)

# 5. Plausibility
plausible = (s_hat > tau_s) & (r_norm_weighted <= tau_r)

# 6. Normalized fractions
s_total = s_hat.sum() + eps
mineral_fractions = s_hat / s_total

return s_hat, r, r_norm_weighted, plausible, mineral_fractions
```

### 6. Custom Minerals

Users can define custom mineral dictionaries:

```python
custom_minerals = {
    "MyMineral": {
        "formula": "XYZ",
        "stoichiometry": {
            "Ca2+": 2.0,  # meq per unit dissolution
            "SO42-": 2.0,
        },
        "description": "Custom mineral description"
    },
}

from neutrohydro.minerals import MineralInverter

inverter = MineralInverter(minerals=custom_minerals)
result = inverter.invert(c_meq, pi_G)
```

**Requirements**:
- Keys must match **ion order** (see `STANDARD_IONS`)
- Values in **meq per unit** (charge-consistent)
- Unlisted ions assumed zero contribution

### 7. Unit Conversion

#### 7.1 mg/L to meq/L

$$\text{meq/L} = \frac{\text{mg/L}}{M} \times |z|$$

where:
- $M$ = molar mass (g/mol)
- $|z|$ = absolute charge

**Example** (Ca²⁺):
- Molar mass = 40.078 g/mol
- Charge = +2
- 100 mg/L Ca²⁺ = $\frac{100}{40.078} \times 2 = 4.99$ meq/L

**Function provided**:

```python
from neutrohydro.minerals import convert_to_meq, ION_MASSES, ION_CHARGES

c_meq = convert_to_meq(
    concentrations_mg,
    ion_charges=ION_CHARGES,
    ion_masses=ION_MASSES,
    from_unit="mg/L"
)
```

#### 7.2 mmol/L to meq/L

$$\text{meq/L} = \text{mmol/L} \times |z|$$

**Example** (SO₄²⁻):
- 5 mmol/L SO₄²⁻ = $5 \times 2 = 10$ meq/L

### 8. Limitations and Caveats

#### 8.1 Model Assumptions

The stoichiometric model assumes:
1. **Simple dissolution**: Minerals dissolve incongruently
2. **No ion exchange**: Cations don't swap on clays
3. **No redox**: Oxidation states don't change (e.g., Fe²⁺ → Fe³⁺)
4. **No precipitation**: Minerals only dissolve, not precipitate
5. **Equilibrium**: Instantaneous reactions

**Violations common in real systems**:
- Cation exchange (Ca²⁺ ↔ 2Na⁺ on clays)
- Redox (O₂, NO₃⁻ reduction)
- Kinetics (slow weathering)
- Mixing (groundwater from multiple sources)

#### 8.2 Residual Sources

Large residuals ($\|r\|_D > \tau_r$) may indicate:
- **Missing endmembers**: Important minerals not in library
- **Non-mineral sources**: Fertilizers (NO₃⁻, K⁺), wastewater (Na⁺, Cl⁻)
- **Ion exchange**: Alters Ca/Na ratio without mineral dissolution
- **Atmospheric inputs**: Sea salt spray (Na⁺, Cl⁻, Mg²⁺, SO₄²⁻)
- **Data quality**: Measurement errors, charge imbalance

#### 8.3 Non-Uniqueness

NNLS solution may not be **unique** if:
- Minerals have similar stoichiometry (e.g., gypsum vs. anhydrite)
- System is underdetermined ($K > m$)

**Implication**: Interpret $\hat{s}$ as **one plausible decomposition**, not absolute truth.

### 9. Diagnostics

#### 9.1 Charge Balance

Check total cations vs. anions:

$$\sum_{\ell \in \text{cations}} c_\ell \approx \sum_{\ell \in \text{anions}} c_\ell$$

**Acceptable error**: $< 10\%$

**If violated**: Data quality issue, missing major ions.

#### 9.2 Residual Patterns

Plot $r_\ell$ vs. $\ell$ (ion index):
- **Random scatter**: Good fit
- **Systematic bias** (all positive or negative): Missing endmember
- **Outliers**: Specific ions poorly fit

#### 9.3 Mineral Fractions

Normalized fractions sum to 1:

$$\sum_{k=1}^K \frac{\hat{s}_k}{\sum_{k'} \hat{s}_{k'}} = 1$$

**Interpretation**: Relative importance of each mineral.

**Caution**: Fractions depend on chosen library; different library → different fractions.

#### 9.4 Sensitivity to $π_G$

Compare inversions with and without $π_G$ weighting:
- Large difference → Attribution matters
- Small difference → Stoichiometry dominates

### 10. Visualization

#### 10.1 Stacked Bar Chart: Mineral Fractions

For each sample $i$, stacked bar with height = 1:
- Segments = mineral fractions
- Color by mineral type (carbonates, sulfates, chlorides, etc.)

#### 10.2 Heatmap: Mineral Plausibility

Heatmap with:
- Rows: Samples
- Columns: Minerals
- Color: Plausibility (binary) or contribution $\hat{s}_{ik}$

#### 10.3 Residual Magnitude

Histogram of $\|r_i\|_D$ across all samples:
- Threshold $\tau_r$ marked
- Fraction above threshold = poor fits

### 11. Hydrogeochemical Constraints & Indices

NeutroHydro integrates standard hydrogeochemical indices to constrain the mineral inversion and provide automatic classification.

#### 11.1 Chloro-Alkaline Indices (CAI)

Used to identify **Ion Exchange** processes.

- **CAI-1**: $[Cl^- - (Na^+ + K^+)] / Cl^-$
- **CAI-2**: $[Cl^- - (Na^+ + K^+)] / (SO_4^{2-} + HCO_3^- + NO_3^-)$

**Constraints**:
- **CAI < 0** (Freshening): Implies $Ca^{2+} \to Na^+$ exchange. The model **bans** `Clay_ReleaseCa`.
- **CAI > 0** (Intrusion): Implies $Na^+ \to Ca^{2+}$ exchange. The model **bans** `Clay_ReleaseNa`.

*Note*: These constraints are disabled if the Cl/Br ratio suggests anthropogenic chloride (which makes CAI unreliable).

#### 11.2 Gibbs Diagram Ratios

Used to classify the dominant hydrogeochemical process.

- **Anion Ratio**: $Cl^- / (Cl^- + HCO_3^-)$
- **Cation Ratio**: $Na^+ / (Na^+ + Ca^{2+})$

**Constraints**:
- **Rock Dominance** (Both ratios < 0.5): The model **penalizes/bans** Evaporite minerals (e.g., Halite, Mirabilite) to prevent overfitting noise as saline deposits in fresh water.

#### 11.3 Simpson's Ratio (Salinity & Intrusion)

Both the **Standard** and **Inverse** ratios are implemented to provide a complete diagnosis of salinity. They are designed to be used together:

1.  **Step 1: Assess Severity (Standard Ratio)**
    *   Formula: $Cl^- / (HCO_3^- + CO_3^{2-})$
    *   **Purpose**: Classifies the water from "Fresh" to "Extremely Saline".
    *   **Thresholds**:
        - **< 0.5**: Fresh (Low Salinity)
        - **0.5 - 1.3**: Slightly Saline
        - **1.3 - 2.8**: Moderately Saline
        - **2.8 - 6.6**: Highly Saline
        - **6.6 - 15.5**: Severely Saline
        - **> 15.5**: Extremely Saline (Seawater)

2.  **Step 2: Confirm Mechanism (Inverse Ratio)**
    *   Formula: $(HCO_3^- + CO_3^{2-}) / Cl^-$
    *   **Purpose**: Distinguishes between freshwater recharge and saline intrusion.
    *   **Interpretation**:
        - **> 1**: Freshwater Recharge (Dominant Bicarbonate)
        - **< 0.5**: Seawater Influence (Dominant Chloride)

**Combined Diagnosis**: If the Standard Ratio indicates "Extremely Saline" (> 15.5) AND the Inverse Ratio is low (< 0.5), it confirms **Seawater Intrusion** as the specific cause.

#### 11.4 WHO Quality Integration

The inversion can be "helped" by the **Quality Assessment** module (`neutrohydro.quality_check`).

- If **Saline Intrusion** is detected (High Na + Cl), the model **overrides** Gibbs constraints to allow Halite/Sylvite.
- If **Gypsum** source is inferred (High Ca + SO4), the model ensures Sulfates are plausible.

This allows the model to be **context-aware**, adapting its constraints based on the specific pollution signature of each sample.

### 12. Usage Example

```python
from neutrohydro.minerals import MineralInverter, convert_to_meq
from neutrohydro.quality_check import add_quality_flags
import numpy as np
import pandas as pd

# 1. Load and Assess Quality
df = pd.read_csv("data.csv")
df_quality = add_quality_flags(df)
quality_flags = df_quality.to_dict('records')

# 2. Convert ion data to meq/L
from neutrohydro.minerals import ION_MASSES, ION_CHARGES
# ... (conversion logic) ...

# 3. Create inverter
inverter = MineralInverter()

# 4. Invert with Quality Constraints
result = inverter.invert(
    c_meq, 
    use_cai_constraints=True,
    use_gibbs_constraints=True,
    quality_flags=quality_flags  # <--- Context-aware override
)

# 5. Access results & Indices
print(f"Simpson Class: {result.indices['Simpson_Class']}")
print(f"Inferred Sources: {df_quality['Inferred_Sources']}")

# 6. Export to DataFrame
df_res = inverter.results_to_dataframe(result, sample_ids=df['Code'])
```

### 13. Integration with Pipeline

The full pipeline can run mineral inference automatically:

```python
from neutrohydro import NeutroHydroPipeline
from neutrohydro.pipeline import PipelineConfig

config = PipelineConfig(
    run_mineral_inference=True,
    mineral_eta=1.0,
    mineral_tau_s=0.01,
    mineral_tau_r=1.0
)

pipeline = NeutroHydroPipeline(config)
results = pipeline.fit(X, y, c_meq=c_meq)  # Pass ion data in meq/L

# Mineral results available
if results.mineral_result is not None:
    print(results.mineral_result.mineral_fractions)
```

### 13. Extensions

#### 13.1 Bayesian Inversion

Replace NNLS with **Bayesian NNLS**:
- Prior distributions on $s$ (e.g., log-normal)
- Posterior samples via MCMC
- Uncertainty quantification

#### 13.2 Sparse Regularization

Add L1 penalty to encourage **sparse solutions**:

$$\hat{s} = \arg\min_{s \geq 0} \|D(c - As)\|^2 + \lambda \|s\|_1$$

**Effect**: Favors fewer minerals with larger contributions.

#### 13.3 Hierarchical Models

Account for spatial/temporal structure:

$$s_i \sim \text{Distribution}(\theta)$$

where $\theta$ are shared parameters across samples.

## References

1. Parkhurst, D. L., & Appelo, C. A. J. (2013). Description of input and examples for PHREEQC version 3. *US Geological Survey Techniques and Methods*, 6(A43), 497.

2. Güler, C., Thyne, G. D., McCray, J. E., & Turner, A. K. (2002). Evaluation of graphical and multivariate statistical methods for classification of water chemistry data. *Hydrogeology Journal*, 10(4), 455-474.

3. Plummer, L. N., & Back, W. (1980). The mass balance approach: Application to interpreting the chemical evolution of hydrologic systems. *American Journal of Science*, 280(2), 130-142.

---

**End of Mathematical Documentation**

For practical usage examples, see [Quick Start Guide](quickstart.md) and [Examples](examples_basic.md).


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Water Quality Assessment

**Module**: `neutrohydro.quality_check`

## Overview

The Quality Assessment module provides an automated system for evaluating groundwater samples against **WHO (World Health Organization)** drinking water guidelines. Beyond simple compliance checking, it implements an **intelligent source inference** engine that interprets combinations of exceedances to suggest potential pollution origins.

## Features

1.  **WHO Compliance Check**: Automatically flags parameters exceeding standard limits.
2.  **Source Inference**: Uses hydrogeochemical logic to infer the likely cause of contamination (e.g., Saline Intrusion vs. Anthropogenic Pollution).
3.  **Integration**: Can be used as a standalone tool or to provide **context-aware constraints** for the Mineral Inversion model.

## Mathematical Logic

### 1. Thresholds

The module uses standard WHO guideline values (mg/L):

| Parameter | Limit |
| :--- | :--- |
| **TDS** | 1000 |
| **pH** | 6.5 - 8.5 |
| **Na** | 200 |
| **Cl** | 250 |
| **SO4** | 250 |
| **NO3** | 50 |
| **F** | 1.5 |
| **Heavy Metals** | Various (e.g., Pb 0.01, As 0.01) |

### 2. Source Inference Rules

The module applies a set of heuristic rules to infer sources based on specific combinations of exceedances:

#### 2.1 Saline Intrusion
*   **Trigger**: High Chloride ($Cl > 250$) **AND** High Sodium ($Na > 200$).
*   **Inference**: "Saline Intrusion/Brine".
*   **Implication**: Suggests seawater mixing or deep brine upwelling.

#### 2.2 Anthropogenic Pollution
*   **Trigger**: High Nitrate ($NO_3 > 50$).
*   **Inference**: "Anthropogenic (Agri/Sewage)".
*   **Implication**: Surface contamination from fertilizers or wastewater.

#### 2.3 Industrial/Mining
*   **Trigger**: High Sulfate ($SO_4 > 250$) **WITHOUT** High Calcium (which would suggest Gypsum).
*   **Inference**: "Industrial/Mining".
*   **Implication**: Acid mine drainage or industrial effluent.

#### 2.4 Geogenic (Rock-Water Interaction)
*   **Trigger**: High Fluoride ($F > 1.5$) or High Calcium/Sulfate (Gypsum).
*   **Inference**: "Geogenic (Rock-Water)".
*   **Implication**: Natural weathering of specific mineral formations.

## Usage

### Standalone Assessment

```python
import pandas as pd
from neutrohydro.quality_check import add_quality_flags

# Load Data
df = pd.read_csv("data.csv")

# Run Assessment
df_report = add_quality_flags(df)

# View Results
print(df_report[['Code', 'Exceedances', 'Inferred_Sources']])
```

### Integration with Mineral Inversion

The inferred sources can be passed to the `MineralInverter` to override standard constraints. For example, if "Saline Intrusion" is detected, the inverter will force **Halite** to be considered plausible, even if other indices (like Gibbs) suggest otherwise.

```python
# 1. Get Quality Flags
quality_flags = df_report.to_dict('records')

# 2. Run Inversion with Flags
result = inverter.invert(
    c=concentrations,
    quality_flags=quality_flags
)
```


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Model Limitations and Validity

**Module**: `neutrohydro`

## Overview

While NeutroHydro provides a mathematically rigorous framework for groundwater chemometrics, it is subject to inherent limitations common to inverse geochemical modeling and statistical learning. Understanding these limitations is crucial for the scientific defensibility of the results.

## 1. Stoichiometric Assumptions

### Limitation
The **Mineral Inversion** module relies on a library of fixed stoichiometric formulas (e.g., Calcite = $CaCO_3$).
*   **Reality**: Natural minerals often exist as solid solutions (e.g., Magnesian Calcite $Ca_{(1-x)}Mg_xCO_3$) or have impurities.
*   **Impact**: The model may slightly over- or under-estimate the mass of a specific phase if the real-world mineral deviates from the ideal formula.

### Addressing Validity
*   **Residual Analysis**: The model calculates a `residual_norm` for every sample. A high residual indicates that the standard library cannot fully explain the water chemistry, prompting a check for non-standard minerals.
*   **Endmember Expansion**: The "Scientific" library includes pure endmembers (e.g., Albite and Anorthite) rather than a generic "Plagioclase," allowing the model to mix them to approximate the real solid solution.

## 2. Non-Uniqueness of Inversion

### Limitation
The problem of reconstructing mineral assemblages from dissolved ions is mathematically **underdetermined** (more minerals than ions) or **non-unique**.
*   **Example**: Dissolved $Ca^{2+}$ and $SO_4^{2-}$ could come from Gypsum ($CaSO_4 \cdot 2H_2O$) or Anhydrite ($CaSO_4$). Chemically, they produce the exact same ions.

### Addressing Validity
*   **Parsimony Principle**: The Non-Negative Least Squares (NNLS) algorithm inherently favors "sparse" solutions, selecting the fewest minerals needed to explain the data.
*   **Contextual Validation**: The model outputs "Plausible Minerals." The user must validate if these make geological sense (e.g., Anhydrite is rare in shallow aquifers; Gypsum is more likely).
*   **Thermodynamic Consistency**: Results should be cross-referenced with Saturation Indices (SI) if pH and temperature data are available (external validation).

## 3. Linearity of Baseline (PCA/PLS)

### Limitation
The **NDG Encoder** and **PNPLS** use linear projections (PCA-based) to define the "Baseline" (Truth).
*   **Reality**: Natural geochemical evolution (e.g., redox fronts, sorption isotherms) can be highly non-linear.

### Addressing Validity
*   **Neutrosophic Compensation**: This is the core strength of NeutroHydro. Unlike standard PCA which forces non-linear outliers into the model (distorting the baseline), NeutroHydro captures non-linear deviations in the **Indeterminacy ($I$)** and **Falsity ($F$)** channels.
*   **Interpretation**: A high $I$ or $F$ score doesn't just mean "error"; it often signifies a non-linear geochemical process (like denitrification) that deviates from the linear mixing baseline.

## 4. Data Completeness (Missing Ions)

### Limitation
Geochemical inversion requires charge balance. Missing major ions (especially Alkalinity/HCO3) makes it impossible to distinguish certain minerals (e.g., Calcite vs. Gypsum).

### Addressing Validity
*   **Adaptive Filtering**: The model dynamically scans the input dataset. If a critical ion (e.g., Nitrate) is missing, all minerals requiring that ion (e.g., Nitrocalcite) are **automatically removed** from the candidate list.
*   **Benefit**: This prevents "ghost" minerals (hallucinations) and ensures that any identified mineral is positively supported by the available data.

## 5. "Closed System" Assumption

### Limitation
Mass balance inversion assumes that the dissolved ions come solely from mineral dissolution/precipitation within the control volume.
*   **Reality**: Groundwater is an open system. Ions can be added via rainfall, evaporation, or anthropogenic inputs.

### Addressing Validity
*   **Anthropogenic Markers**: We explicitly include "Pollution Proxies" (e.g., Niter, Nitrocalcite) in the library to account for non-geogenic inputs.
*   **Evaporation Handling**: Conservative ions (Cl, Br) are used in the baseline to track physical concentration effects (evaporation) separate from chemical reactions.

## 6. Heuristic Constraints (Gibbs, Simpson, WHO)

### Limitation
The model now incorporates empirical hydrogeochemical rules (Gibbs Diagram, Simpson's Ratio, WHO Guidelines) to constrain the mathematical inversion.
*   **Empirical Nature**: These rules are generalizations derived from global datasets. They may not hold in specific, complex local geologies (e.g., a "Rock Dominance" zone that happens to have a local salt deposit).
*   **Discontinuity**: The use of thresholds (e.g., Simpson Ratio > 15.5) creates discrete classification bins for what is physically a continuous variable.

### Addressing Validity
*   **Context-Aware Overrides**: The "Quality Flag" integration allows the model to override general rules (like Gibbs) if specific evidence (like WHO exceedances) points to a contradiction (e.g., Saline Intrusion).
*   **Transparency**: The model reports the calculated indices (`Simpson_Ratio`, `CAI`) alongside the mineral results, allowing the user to see *why* a constraint was applied.


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Hydrogeochemical Processes in NeutroHydro

**Module**: `neutrohydro`

This document explains how specific hydrogeochemical processes (Mixing, Salinization, Ion Exchange) are mathematically represented within the NeutroHydro framework and how the model accounts for them.

## 1. Mixing and Salinization

**Process**: The physical mixing of two or more distinct water bodies (e.g., fresh recharge + saline connate water).
*   **Mathematical Nature**: Linear. $C_{mix} = f_1 C_1 + f_2 C_2$.
*   **Model Representation**: **Truth ($T$) Channel**.

### How it works
The **NDG Encoder** uses Robust PCA (or Low-Rank Approximation) to define the "Truth" baseline.
*   The Principal Components (PCs) of the $T$ channel naturally align with the **Mixing Lines**.
*   **Salinization** (e.g., Seawater Intrusion) typically appears as the **First Principal Component (PC1)** because it explains the largest variance in total dissolved solids (TDS).
*   **Validity**: Since mixing is a linear operation, the linear algebra underlying the $T$ channel is the mathematically valid term for these processes.

## 2. Ion Exchange

**Process**: The adsorption of one ion onto a clay surface and the release of another (e.g., $Ca^{2+}$ adsorbs, $2Na^+$ releases).
*   **Mathematical Nature**: Non-linear / Non-conservative relative to the mixing line.
*   **Model Representation**: **Indeterminacy ($I$) and Falsity ($F$) Channels**.

### How it works
Ion exchange creates a deviation from the linear mixing trend defined in $T$.
*   *Example*: In simple mixing, if $Cl^-$ increases, $Na^+$ should increase proportionally (Halite ratio).
*   *Effect*: If Ion Exchange occurs, $Na^+$ increases *more* than expected, while $Ca^{2+}$ increases *less* (or decreases).
*   **The "Term"**: The model captures this deviation in the **Falsity ($F$)** matrix.
    *   $F_{Na} > 0$ (Positive perturbation: Excess Na)
    *   $F_{Ca} > 0$ (Negative perturbation: Deficit Ca - note $F$ is magnitude, sign is in residual)

### Explicit Modeling (The "Exchanger Term")
To explicitly solve for the mass of ions exchanged during **Mineral Inversion**, we can introduce **Pseudo-Minerals** with negative stoichiometric coefficients.

**Mathematically Valid Term**:
$$ \text{Clay}_{Na \to Ca} : \quad +1 \text{ Ca}^{2+} \quad -2 \text{ Na}^+ $$

*   **Interpretation**: This "mineral" adds Calcium to the water and *removes* Sodium.
*   **Constraint**: Since the NNLS solver requires positive mass ($x \ge 0$), we define two directional exchangers:
    1.  **Direct Exchange** (Freshening): $Ca^{2+} \to 2Na^+$ (Release Na, Remove Ca)
    2.  **Reverse Exchange** (Intrusion): $2Na^+ \to Ca^{2+}$ (Release Ca, Remove Na)

## 3. Redox Processes (Denitrification, Sulfate Reduction)

**Process**: Biogeochemical removal of species (e.g., $NO_3^- \to N_2(g)$) or addition (Nitrification).
*   **Mathematical Nature**: Mass loss (Sink) or Gain (Source).
*   **Model Representation**: **Falsity ($F$)** and **Redox Phases**.

### How it works
These processes look like "missing mass" relative to the conservative baseline.
*   **The "Term"**: A high Falsity score for Nitrate ($F_{NO3}$) combined with a low Truth value indicates depletion.
*   **Validity**: The $F$ channel provides the statistical evidence of the process.

### Explicit Modeling (The "Sink Term")
Similar to Ion Exchange, we can explicitly solve for the mass lost to redox processes by introducing **Redox Phases** with negative stoichiometry.

**Mathematically Valid Term**:
$$ \text{Sink}_{Denit} : \quad -1 \text{ NO}_3^- \quad +1 \text{ HCO}_3^- $$

*   **Interpretation**: This "mineral" removes Nitrate and adds Alkalinity (bicarbonate).
*   **Constraint**: Since the NNLS solver requires positive mass ($x \ge 0$), a positive value for this sink phase ($x_{denit} > 0$) mathematically accounts for the *negative* residual of Nitrate.
    *   Equation: $C_{final} = C_{mix} + x_{denit} \times (-1)$
    *   If $C_{final} < C_{mix}$ (Depletion), then $x_{denit}$ must be positive.

**Available Redox Phases**:
1.  **Denitrification**: Removes $NO_3^-$, adds $HCO_3^-$.
2.  **Sulfate Reduction**: Removes $SO_4^{2-}$, adds $HCO_3^-$.
3.  **Nitrification**: Adds $NO_3^-$ (Source).

### Detection vs. Assumption

The model **detects** the process; it does not assume it is present.

*   **Candidate Approach**: The Redox phases are provided as *candidates* to the solver.
*   **Selection Logic**: The NNLS solver will only assign a positive value to `Sink_Denitrification` if there is a **mass deficit** in Nitrate that cannot be explained by mixing or other minerals.
    *   If the observed Nitrate matches the expected background, the solver sets the Denitrification term to **0**.
    *   If the observed Nitrate is *lower* than expected (a deficit), the solver increases the Denitrification term to minimize the error.
*   **Result**: The magnitude of the term ($x_{denit}$) represents the **calculated mass** of Nitrate lost to the process.

## Summary of Terms

| Process | Mathematical Term in Model | Validity |
| :--- | :--- | :--- |
| **Mixing** | $T$ (Truth Matrix) | Valid (Linear Algebra) |
| **Salinization** | $T$ (PC1) + `Halite` (Mineral) | Valid (Stoichiometry) |
| **Ion Exchange** | $F$ (Falsity Matrix) + `Exchanger` (Pseudo-Mineral) | Valid (Perturbation Theory) |
| **Redox** | $F$ (Falsity Matrix) | Valid (Outlier Detection) |


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Mathematical Critique of NeutroHydro Model

**Date**: December 28, 2025
**Module**: `neutrohydro`

This document provides a critical mathematical review of the NeutroHydro framework, identifying potential limitations and proposing rigorous solutions.

## 1. The "Weighting Paradox" in Mineral Inversion

### The Issue
The current `MineralInverter` minimizes the weighted norm:
$$ \min_{s \ge 0} \| D \cdot (c - A s) \|_2^2 $$
where the weights $D$ are derived from the Baseline Fraction $\pi_G$:
$$ D_{jj} \approx (\pi_G(j))^\eta $$

*   **Logic**: Trust the "Baseline" ions more; downweight the "Perturbed/Noisy" ions.
*   **Consequence**: Anthropogenic markers (e.g., Nitrate from fertilizer) are often **perturbations** (High $F$, Low $\pi_G$).
*   **The Paradox**: By downweighting the perturbation, the solver is effectively told "It is okay to ignore Nitrate."
    *   **Result**: The model may underestimate the mass of `Nitrocalcite` or `Niter` because the penalty for missing the Nitrate target is small.

### Mathematical Solution
For **Forensic Analysis** (identifying pollution), the weighting scheme should be inverted or removed:
1.  **Unweighted Inversion**: Set $D = I$ (Identity). This forces the model to explain *all* ions, including pollutants.
2.  **Targeted Weighting**: Explicitly set high weights for suspected markers (e.g., $D_{NO3} = 1.0$) regardless of their $\pi_G$ score.

## 2. Mixing vs. Mineral Dissolution

### The Issue
The NNLS solver assumes:
$$ c_{total} = \sum (\text{Mineral}_k \times \text{Mass}_k) $$
This assumes all solutes come from dissolving solid phases.
*   **Reality**: Groundwater often involves **Mixing** with a pre-existing brine (e.g., Seawater).
*   **Approximation**: The model approximates Seawater as a sum of `Halite` + `Sylvite` + `Gypsum` + ...
*   **Critique**: This loses the **Constant Proportion** constraint of Seawater (e.g., $Cl/Br$ ratio). It allows the model to "break" Seawater into separate salts, which is physically impossible in simple mixing.

### Mathematical Solution
Add **Fluid Endmembers** to the Stoichiometric Matrix $A$:
*   Define a "mineral" called `Seawater` with the exact ionic composition of standard seawater.
*   $$ A_{Seawater} = [Na=468, Mg=53, Ca=10, Cl=545, SO4=28, ...] $$
*   This forces the solver to use the *exact* seawater ratio, improving validity for salinization studies.

## 3. Non-Uniqueness of Ion Exchange

### The Issue
I introduced `Exchanger` phases (e.g., $Ca \to 2Na$) to model ion exchange.
*   **Mathematical Risk**: This increases **Multicollinearity**.
    *   *Scenario*: High Na, Low Ca.
    *   *Explanation A*: Dissolve Halite ($Na, Cl$) + Precipitate Calcite ($-Ca, -CO3$).
    *   *Explanation B*: Ion Exchange ($Ca \to 2Na$).
*   **Solver Behavior**: NNLS will pick the path of least resistance (lowest residual). It cannot distinguish between these mechanisms without isotopic data.

### Mathematical Solution
**Regularization**: Apply L2 (Ridge) or L1 (Lasso) penalties to the Exchanger terms to ensure they are only selected when standard minerals *cannot* explain the data (i.e., when Cl is conservative but Na is not).

## 4. Error Propagation in NDG

### The Issue
The NDG Encoder calculates $T, I, F$ sequentially.
*   $T$ = Robust PCA.
*   $I$ = PCA on Residuals.
*   $F$ = Distance to $T+I$.
*   **Critique**: Errors in the estimation of $T$ (e.g., wrong rank) propagate to $I$ and $F$. If $T$ overfits, $I$ and $F$ vanish.

### Mathematical Solution
**Cross-Validation**: The rank of $T$ (number of components) must be selected via Cross-Validation ($Q^2$ metric) to ensure $T$ only captures the stable baseline, leaving the true noise/perturbation for $I$ and $F$.

## 5. The "Rule-Based" Override Problem

### The Issue
The integration of **WHO Quality Flags** and **Gibbs Constraints** introduces a rule-based logic layer on top of the optimization layer.
*   **Scenario**: The NNLS solver wants to fit `Halite` to explain Cl. The Gibbs constraint says "Rock Dominance" and bans `Halite`. The WHO flag says "Saline Intrusion" and forces `Halite` back in.
*   **Critique**: This creates a **Hybrid System** where the objective function is dynamically modified by discrete logic gates. This makes the model behavior non-smooth and potentially sensitive to the specific thresholds used in the rules.

### Mathematical Solution
**Soft Constraints / Priors**: Instead of binary Banning/Forcing, these heuristics should be implemented as **Bayesian Priors**.
*   Gibbs "Rock Dominance" $\to$ Low Prior Probability for Halite.
*   WHO "Saline Intrusion" $\to$ High Prior Probability for Halite.
*   This allows the data (Likelihood) to still have a say, rather than being overruled by a hard logic gate.

## 6. Simpson's Ratio Discretization

### The Issue
The model uses discrete bins for Simpson's Ratio (e.g., "Moderately Saline" vs "Highly Saline").
*   **Critique**: Discretization throws away information. A sample at 2.7 is labeled "Moderately", while 2.9 is "Highly", despite being chemically nearly identical.

### Mathematical Solution
**Continuous Scoring**: Use the raw ratio values (Standard and Inverse) for any downstream statistical analysis (like correlation or clustering), and reserve the discrete classes only for the final human-readable report.


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Final Critical Review: Mathematical & Hydrogeochemical Integrity

**Date**: December 28, 2025
**Module**: `neutrohydro`

This document serves as the final "Red Team" critique of the NeutroHydro framework, evaluating its scientific validity after the inclusion of advanced features like Chloro-Alkaline Indices (CAI) and Ion Exchange phases.

## 1. The "Hard Threshold" Problem in CAI Constraints

### Critique
The current implementation uses a hard threshold (e.g., `CAI > 0.05`) to switch between "Freshening" and "Intrusion" modes.
*   **Mathematical Issue**: This introduces a **discontinuity** in the model function. A sample with CAI=0.049 allows one set of minerals, while CAI=0.051 allows another.
*   **Hydrogeochemical Reality**: Natural systems are continuous. A sample near equilibrium (CAI ≈ 0) might experience minor fluctuations.
*   **Risk**: Small measurement errors in Na or Cl could flip the switch, causing a sudden jump in the predicted mineral assemblage (Instability).

### Recommendation
*   **Soft Gating**: Instead of binary removal (0 or 1), use a **Sigmoid Weighting** function.
    *   Weight for `ReleaseNa` = $\sigma(-k \cdot \text{CAI})$
    *   This smoothly transitions the allowed mass of the exchanger phase to zero as the index moves against it.

## 2. The "Sink" Asymmetry

### Critique
The Non-Negative Least Squares (NNLS) algorithm ($s \ge 0$) is excellent for **Dissolution** (Source) but struggles with **Precipitation** (Sink).
*   **Scenario**: Calcite precipitation ($Ca^{2+} + CO_3^{2-} \to CaCO_3$). This removes ions.
*   **Model Behavior**: The model cannot assign a negative mass to "Calcite". It can only model this if we explicitly define a "Precipitation" phase with negative stoichiometry.
*   **Current State**: We added `Exchanger` phases with negative terms, but we do not have "Calcite Precipitation" phases.
*   **Consequence**: If the water is supersaturated and precipitating calcite, the model will simply have a large **Residual** (it will overestimate the Ca/HCO3 that *should* be there based on other minerals).

### Recommendation
*   **Residual Interpretation**: Explicitly document that **Negative Residuals** (Observed < Predicted) imply precipitation or biological uptake.

## 3. Thermodynamic Blindness

### Critique
NeutroHydro is a **Mass Balance** model, not a **Thermodynamic** model.
*   **Issue**: It can mathematically propose a mineral assemblage that is thermodynamically impossible (e.g., dissolving Anhydrite in a water that is undersaturated with respect to Gypsum but supersaturated with Anhydrite - rare but possible).
*   **Missing Link**: The model does not check **Saturation Indices (SI)**. It doesn't know if the water *can* dissolve the mineral, only that the ions *fit* the pattern.

### Recommendation
*   **External Validation**: For publication, results *must* be cross-referenced with PHREEQC or similar codes to ensure the identified phases are not supersaturated (which would imply precipitation, not dissolution).

## 4. The "Conservative Chloride" Assumption

### Critique
The CAI calculation and many mixing models assume Chloride ($Cl^-$) is perfectly conservative.
*   **Reality**: In some arid environments or specific geologies, Cl can be added via Halite dissolution or removed via salt precipitation.
*   **Impact on CAI**: If Halite dissolves, Cl increases. CAI = $(Cl - (Na+K))/Cl$. If Na and Cl increase equally, CAI stays near 0. But if Cl comes from another source (e.g., volcanism, anthropogenic), CAI is skewed.

### Recommendation
*   **Source Verification**: Ensure $Cl/Br$ ratios (if available) confirm the marine/halite origin of Chloride before trusting CAI blindly.

## 5. The Hybrid Model: Optimization + Heuristics

### Critique
The latest version of NeutroHydro has evolved into a **Hybrid System**. It combines:
1.  **Rigorous Optimization**: Weighted NNLS for mineral apportionment.
2.  **Heuristic Logic**: Gibbs Diagrams, Simpson's Ratios, and WHO Guidelines to constrain the search space.

**Strength**: This makes the model "Expert-Guided." It prevents mathematically optimal but geologically foolish solutions (like finding Halite in a fresh mountain spring).
**Weakness**: It relies on the validity of the heuristics. If the Gibbs diagram is wrong for a specific unusual aquifer, the model will be constrained incorrectly.

### Verdict
The integration of **WHO Quality Flags** is a significant robustness improvement. By allowing the "Pollution Context" (e.g., Saline Intrusion) to override the "Geological Context" (Gibbs), the model avoids the common pitfall of forcing anthropogenic/intrusion signals into natural weathering patterns.

## 6. Conclusion: Is it Defensible?

**Yes**, with caveats.

The model is now **mathematically superior** to standard inverse models because:
1.  **It handles Uncertainty**: The Neutrosophic ($I, F$) logic captures the "noise" that breaks other models.
2.  **It is Constrained**: The CAI and Gibbs logic removes the most egregious non-uniqueness errors.
3.  **It is Context-Aware**: The WHO integration ensures pollution sources are respected.

**Final Verdict**: The model is valid for **Hypothesis Generation** and **Forensic Fingerprinting**. It should not be used as a replacement for thermodynamic equilibrium modeling (PHREEQC) but as a complementary tool to identify *sources* and *processes* that thermodynamic models assume as inputs.


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Pipeline API

## NeutroHydroPipeline

The `NeutroHydroPipeline` class orchestrates the entire workflow, from data loading to result generation.

### Class Reference

```python
class NeutroHydroPipeline:
    def __init__(self, target_ions: list[str], ...):
        """
        Initialize the pipeline.
        
        Args:
            target_ions: List of ions to model (e.g., ['Ca', 'Mg', ...])
        """
        ...

    def fit(self, df: pd.DataFrame):
        """
        Fit the internal models (Scaler, Encoder, PNPLS) to the data.
        """
        ...

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Run the full analysis on the provided data.
        
        Returns:
            dict: A dictionary containing:
                - 'vip_scores': Variable Importance
                - 'mineral_fractions': Mineral inversion results
                - 'quality_flags': WHO assessment
                - 'indices': Hydrogeochemical indices
        """
        ...
```


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Core Modules API

## `neutrohydro.encoder`

Handles the Neutrosophic Data Transformation.

- **`NDGEncoder`**: Transforms raw concentrations into Truth (T), Indeterminacy (I), and Falsity (F) components.

## `neutrohydro.minerals`

Handles Stoichiometric Inversion.

- **`MineralInverter`**: Performs weighted NNLS inversion to estimate mineral contributions.
- **`calculate_simpson_ratio`**: Computes Standard and Inverse Simpson's Ratios.

## `neutrohydro.quality_check`

Handles Water Quality Assessment.

- **`assess_water_quality`**: Checks samples against WHO guidelines.
- **`add_quality_flags`**: Adds quality columns to a DataFrame.

## `neutrohydro.nvip`

Handles Variable Importance.

- **`calculate_nvip`**: Computes Neutrosophic Variable Importance in Projection.


<!-- examples_basic.md -->

# Basic Examples

For a complete walkthrough of the basic usage, please refer to the [Quick Start Guide](quickstart.md).

## Running the Example Script

The repository includes a ready-to-run example script:

```bash
python examples/basic_example.py
```

This script demonstrates:
1.  Creating synthetic data.
2.  Initializing the pipeline.
3.  Running the analysis.
4.  Printing the results.

<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```
# Advanced Workflows

## Custom Mineral Libraries

You can define custom minerals for the inversion engine:

```python
from neutrohydro.minerals import MineralInverter

custom_minerals = {
    "MyMineral": {
        "formula": "X2Y",
        "stoichiometry": {"X": 2.0, "Y": 1.0},
        "description": "A custom phase"
    }
}

inverter = MineralInverter(minerals=custom_minerals)
```

## Handling Redox Processes

To explicitly model redox sinks (like Denitrification), include the `REDOX_PHASES` in your mineral library.

```python
from neutrohydro.minerals import STANDARD_MINERALS, REDOX_PHASES

combined_minerals = {**STANDARD_MINERALS, **REDOX_PHASES}
inverter = MineralInverter(minerals=combined_minerals)
```


<div style="page-break-after: always;"></div>

```{=latex}
\newpage
```

# Interpreting Results

## Understanding the Outputs

### 1. Mineral Fractions
These represent the relative contribution of each mineral to the total dissolved solids (TDS) of the sample.
- **Sum to 1**: The fractions are normalized.
- **Interpretation**: A high "Calcite" fraction indicates carbonate weathering is the dominant process.

### 2. Quality Flags
- **Exceedances**: Lists parameters that violate WHO guidelines.
- **Inferred Sources**: Suggests potential origins (e.g., "Saline Intrusion", "Agricultural").

### 3. Simpson's Ratio
- **Standard Ratio**: Indicates severity of salinity.
- **Inverse Ratio**: Distinguishes mechanism (Recharge vs. Intrusion).
    - **> 1**: Recharge (Fresh)
    - **< 0.5**: Intrusion (Seawater)

### 4. VIP Scores (Variable Importance)
- **T-VIP**: Importance of the baseline trend (Mixing).
- **F-VIP**: Importance of perturbations (Pollution/Exchange).