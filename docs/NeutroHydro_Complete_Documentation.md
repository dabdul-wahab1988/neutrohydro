# NeutroHydro: Complete Technical Documentation

**Neutrosophic Chemometrics for Groundwater Analysis**

Generated: December 28, 2025

---

## Table of Contents

1. [Getting Started](#getting-started)
   - Quick Start Guide
   - Installation Guide

2. [Mathematical Foundations](#mathematical-foundations)
   - Mathematical Framework Overview
   - Preprocessing & Robust Scaling
   - NDG Encoder: Neutrosophic Triplets
   - PNPLS: Probabilistic Neutrosophic PLS
   - NVIP: Neutrosophic Variable Importance in Projection
   - Attribution: NSR and Baseline Fractions
   - Mineral Stoichiometric Inversion
   - Water Quality Assessment
   - Model Limitations & Validity
   - Hydrogeochemical Processes
   - Mathematical Critique
   - Final Critical Review

3. [API Reference](#api-reference)
   - Pipeline API
   - Core Modules API

4. [Examples & Tutorials](#examples--tutorials)
   - Basic Usage Example
   - Advanced Workflows
   - Interpreting Results

---

# GETTING STARTED

## Quick Start Guide

This guide will help you run your first analysis using NeutroHydro.

### 1. Basic Workflow

The core of NeutroHydro is the `NeutroHydroPipeline`. It handles preprocessing, encoding, model training, and mineral inversion in a single step.

#### Step 1: Prepare Your Data

Prepare a CSV file (e.g., `data.csv`) with your ion concentrations. The columns should match standard chemical symbols (e.g., `Ca`, `Mg`, `Na`, `HCO3`, `Cl`, `SO4`).

| SampleID | Ca | Mg | Na | K | HCO3 | Cl | SO4 | NO3 |
|----------|----|----|----|---|------|----|-----|-----|
| S1       | 45 | 12 | 25 | 3 | 150  | 30 | 40  | 5   |
| S2       | 80 | 25 | 60 | 5 | 200  | 85 | 90  | 12  |

#### Step 2: Run the Pipeline

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

### 2. Advanced Features

#### Mineral Inversion with Quality Constraints

NeutroHydro can use water quality flags (like WHO exceedances) to constrain the mineral inversion.

```python
# The pipeline does this automatically if you use the .analyze() method.
# You can access the quality assessment directly:

quality_df = results["quality_flags"]
print(quality_df[["Exceedances", "Inferred_Sources"]].head())
```

#### Hydrogeochemical Indices

The analysis also calculates standard indices automatically:

```python
indices = results["indices"]
print(indices[["Simpson_Class", "Simpson_Ratio_Inverse", "Gibbs_Ratio_1"]].head())
```

### 3. Visualization

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

### Next Steps

- Learn about the [Mathematical Framework](#mathematical-framework-overview).
- Explore [Mineral Inversion](#mineral-stoichiometric-inversion) details.
- Check the [API Reference](#pipeline-api) for full documentation.

---

## Installation Guide

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Installation from Source

NeutroHydro is currently available as a source distribution. To install it, clone the repository and install using `pip`.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/neutrohydro.git
    cd neutrohydro
    ```

2. **Create a virtual environment (Recommended):**

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\Activate.ps1

    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install the package:**

    ```bash
    pip install .
    ```

    For development (including testing and documentation tools):

    ```bash
    pip install -e .[dev]
    ```

### Verifying Installation

To verify that NeutroHydro is installed correctly, you can run the following command in your terminal:

```bash
python -c "import neutrohydro; print(neutrohydro.__version__)"
```

If installed correctly, this should print the version number without errors.

### Dependencies

The core dependencies are automatically installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib` (for plotting)
- `seaborn` (for advanced visualization)

---

# MATHEMATICAL FOUNDATIONS

## Mathematical Framework Overview

### 1. Introduction

NeutroHydro implements a **neutrosophic chemometric framework** for groundwater analysis that operates in **absolute concentration space**. This document provides a high-level overview of the mathematical theory underpinning the package.

### 2. Problem Statement

#### 2.1 Input Data

- **Predictor matrix**: $X \in \mathbb{R}^{n \times p}$, where:
  - $n$ = number of water samples
  - $p$ = number of ion species (e.g., Ca²⁺, Mg²⁺, Na⁺, Cl⁻, etc.)
  - $X_{ij}$ = concentration of ion $j$ in sample $i$ (units: mg/L, meq/L, etc.)

- **Target vector**: $y \in \mathbb{R}^n$
  - Typically: log TDS, log EC, or log ionic strength
  - Scalar response for each sample

#### 2.2 Objectives

1. **Predict** target $y$ from ion concentrations $X$
2. **Decompose** prediction importance into:
   - Baseline/reference component (geogenic/natural)
   - Perturbation component (anthropogenic/anomalous)
3. **Infer** plausible mineral sources via stoichiometry

### 3. Workflow Overview

```
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

### 4. Key Mathematical Guarantees

#### 4.1 Euclidean Structure

All operations occur in true Euclidean spaces:
- Preprocessing: $\mathbb{R}^p$
- Augmented space: $\mathbb{R}^{3p}$ with inner product
  $$\langle u, v \rangle_{\mathcal{N}} = u_T^\top v_T + \rho_I u_I^\top v_I + \rho_F u_F^\top v_F$$
- Well-defined projections, deflations, and orthogonality

#### 4.2 L2 Additivity (Core Theorem)

**Theorem** (NVIP L2 Decomposition):

For each variable $j$:
$$VIP_{agg}^2(j) = VIP_T^2(j) + VIP_I^2(j) + VIP_F^2(j)$$

#### 4.3 Conservation Laws

**Unity constraints**:
1. $\pi_G(j) + \pi_A(j) = 1$ for all ions $j$
2. $G_i + A_i = 1$ for all samples $i$

**Bounds**:
1. $VIP_c(j) \geq 0$ for all channels $c \in \{T, I, F\}$
2. $\pi_G(j), G_i \in [0, 1]$
3. $I_{ij}, F_{ij} \in [0, 1]$

#### 4.4 Stability

- **Robust statistics**: Median and MAD resist outliers
- **Precision weighting**: High-falsity observations downweighted
- **Regularization**: Optional ridge/elastic net in PLS

### 5. Operational vs. Causal Interpretation

#### 5.1 Operational Definitions

The framework defines **baseline** and **perturbation** **operationally**:

- **Baseline** = component captured by $\mathcal{B}(X^{(std)})$
  - Median ⟹ central tendency
  - Low-rank ⟹ common geochemical manifold
  - Robust PCA ⟹ low-rank + sparse decomposition

- **Perturbation** = deviations from baseline, quantified by falsity $F$

#### 5.2 External Validation Required

Attribution to **physical sources** (geogenic vs. anthropogenic) requires **external evidence**:
- Spatial patterns (urban vs. rural)
- Temporal trends (pre/post contamination event)
- Isotopic tracers
- Land use correlations

### 6. Comparison to Other Methods

| Feature | NeutroHydro | Standard PLS | CoDa Methods |
|---------|-------------|--------------|--------------|
| **Space** | Absolute concentrations | Absolute/relative | Compositional (simplex) |
| **VIP decomposition** | L2-additive (T, I, F) | Single VIP | Not applicable |
| **Uncertainty** | Explicit (I channel) | Implicit | Not standard |
| **Robustness** | Falsity weighting | Optional | Depends on method |
| **Missing data** | EM imputation | Varies | Special handling |
| **Stoichiometry** | Direct (NNLS) | Post-hoc | Difficult |
| **Interpretability** | High (3 channels) | Moderate | Low (log-ratios) |

### 7. Computational Complexity

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

### 8. Hyperparameter Selection

#### 8.1 Critical Parameters

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

---

## Preprocessing & Robust Scaling

**Module**: `neutrohydro.preprocessing`

### Overview

The preprocessing module implements **robust, non-compositional standardization** of ion concentration data. Unlike compositional data analysis (CoDa), we operate in absolute concentration space, preserving physical interpretability and additive mixing models.

### Mathematical Foundation

#### 1. Input Data

**Predictor matrix**:
$$X \in \mathbb{R}^{n \times p}, \quad X_{ij} \geq 0$$

where:
- $n$ = number of water samples
- $p$ = number of ion species
- $X_{ij}$ = concentration of ion $j$ in sample $i$ (units: mg/L, mmol/L, or meq/L)

**Target vector**:
$$y \in \mathbb{R}^n$$

Typically: $y_i = \log(\text{TDS}_i)$, $\log(\text{EC}_i)$, or $\log(I_i)$ where $I$ is ionic strength.

#### 2. Optional Log Transform

Ion concentrations often span **several orders of magnitude**. For such data:

$$X^{(\log)}_{ij} := \log(X_{ij} + \delta_x)$$

where $\delta_x > 0$ (default: $10^{-12}$) ensures numerical stability for near-zero values.

**When to use**:
- Concentrations vary by $> 10\times$ across ions
- Want to reduce heteroscedasticity
- Multiplicative mixing models are appropriate

**When NOT to use**:
- Need direct physical interpretability
- Stoichiometric constraints must be preserved exactly
- Data already in log scale

#### 3. Robust Centering

For each ion $j$:

$$\mu_j = \text{median}_i(X_{ij})$$

**Why median, not mean?**
- **Robust to outliers**: Median has 50% breakdown point
- **Stable**: Unaffected by single extreme values
- **Appropriate for skewed distributions**: Groundwater data often log-normal

#### 4. Robust Scaling

For each ion $j$:

$$s_j = 1.4826 \times \text{MAD}_i(X_{ij})$$

where **MAD** (Median Absolute Deviation):

$$\text{MAD}_i(X_{ij}) = \text{median}_i \left( |X_{ij} - \mu_j| \right)$$

**Scaling factor 1.4826**:
- For normal distribution: $\text{MAD} \times 1.4826 \approx \sigma$ (standard deviation)
- Makes MAD-based scale **consistent** with standard deviation

#### 5. Standardization

$$X^{(\text{std})}_{ij} = \frac{X_{ij} - \mu_j}{s_j + \delta_s}$$

where $\delta_s > 0$ (default: $10^{-10}$) prevents division by zero for constant columns.

**Similarly for target**:

$$y^{(\text{std})}_i = \frac{y_i - \mu_y}{s_y + \delta_s}$$

where:
$$\mu_y = \text{median}(y), \quad s_y = 1.4826 \times \text{MAD}(y)$$

#### 6. Inverse Transform

To recover predictions in original scale:

$$\hat{y}_i = \hat{y}^{(\text{std})}_i \cdot (s_y + \delta_s) + \mu_y$$

### Usage Example

```python
from neutrohydro.preprocessing import Preprocessor

# Initialize
preprocessor = Preprocessor(
    log_transform=False,  # Use if data spans orders of magnitude
    delta_x=1e-12,
    delta_s=1e-10
)

# Fit on training data
preprocessor.fit(X_train, y_train, feature_names=ion_names)

# Transform training data
X_train_std, y_train_std = preprocessor.transform(X_train, y_train)

# Transform test data
X_test_std, y_test_std = preprocessor.transform(X_test, y_test)

# Inverse transform predictions
y_pred_original = preprocessor.inverse_transform_y(y_pred_std)

# Access parameters
params = preprocessor.get_params()
print(f"Centers: {params.mu}")
print(f"Scales: {params.s}")
```

---

## NDG Encoder: Neutrosophic Triplets

**Module**: `neutrohydro.encoder`

### Overview

The NDG (Neutrosophic Data Generator) Encoder maps each standardized ion concentration to a **neutrosophic triplet** $(T, I, F)$:

- **T (Truth)**: Baseline/reference component
- **I (Indeterminacy)**: Uncertainty/ambiguity
- **F (Falsity)**: Perturbation/anomaly likelihood

This representation enables **explicit decomposition** of prediction importance into baseline and perturbation sources.

### Truth Channel (T): Baseline Operator

#### 1. Definition

$$X_T = \mathcal{B}(X^{(\text{std})})$$

where $\mathcal{B}: \mathbb{R}^{n \times p} \to \mathbb{R}^{n \times p}$ is a **baseline operator**.

#### 2. Baseline Operator Options

**Option 1: Robust Columnwise Median (Default)**

$$(X_T)_{ij} = \text{median}_{i'}(X^{(\text{std})}_{i'j})$$

**Advantages**:
- Fast: $O(np)$
- Robust to outliers
- Interpretable: "central tendency"

**When to use**: Default choice for most applications.

**Option 2: Low-Rank Baseline**

$$X_T = \arg\min_{L: \text{rank}(L) \leq r} \|X^{(\text{std})} - L\|_F^2$$

Solution via **truncated SVD**:

$$X^{(\text{std})} = U \Sigma V^\top, \quad X_T = U_r \Sigma_r V_r^\top$$

**Advantages**:
- Captures **geochemical manifold** (common patterns)
- Smooth, low-dimensional baseline

**When to use**: Strong correlations among ions, clear manifold structure.

### Falsity Channel (F): Perturbation Likelihood

#### 1. Normalized Residuals

$$u_{ij} = \frac{|R_{ij}|}{\sigma_j + \delta}$$

where $\delta > 0$ (default: $10^{-10}$) prevents division by zero.

**Interpretation**: $u_{ij}$ measures how many "robust standard deviations" sample $i$ deviates from baseline for ion $j$.

#### 2. Falsity Map

$$F_{ij} = g_F(u_{ij})$$

where $g_F: \mathbb{R}_{\geq 0} \to [0, 1]$ is a **monotone increasing** map.

**Option 1: Exponential Saturation (Default)**

$$F_{ij} = 1 - \exp(-u_{ij})$$

**Properties**:
- $F_{ij} \to 0$ as $u_{ij} \to 0$ (small deviations → low falsity)
- $F_{ij} \to 1$ as $u_{ij} \to \infty$ (large deviations → high falsity)
- Smooth, differentiable
- No additional hyperparameters

### Indeterminacy Channel (I): Uncertainty

#### 1. Purpose

Capture **ambiguity** not purely due to residual magnitude:
- Measurement uncertainty
- Censoring (below detection limit)
- Spatial/temporal variability
- Bootstrap instability

#### 2. Methods

**Method 1: Local Heterogeneity (Default)**

For spatial/temporal data with neighborhood structure:

$$I_{ij} = 1 - \exp\left(-\frac{\text{Var}(\mathcal{N}(i), j)}{\tau_j + \delta}\right)$$

**Method 2: Censoring/Detection Limit**

$$I_{ij} = \begin{cases}
\iota_{\text{DL}} & \text{if } X_{ij} < \text{DL}_j \\
0 & \text{otherwise}
\end{cases}$$

where $\iota_{\text{DL}} \in (0, 1)$ (default: 0.5).

**Method 3: Uniform Small Indeterminacy**

$$I_{ij} = \epsilon$$

for some small $\epsilon > 0$ (e.g., 0.01).

### Usage Example

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

# Access components
print(f"Truth shape: {triplets.T.shape}")
print(f"Indeterminacy range: [{triplets.I.min():.3f}, {triplets.I.max():.3f}]")
print(f"Falsity range: [{triplets.F.min():.3f}, {triplets.F.max():.3f}]")
```

---

## PNPLS: Probabilistic Neutrosophic PLS

**Module**: `neutrohydro.model`

### Overview

PNPLS extends Partial Least Squares (PLS) regression to **neutrosophic triplet data** $(T, I, F)$ by:
1. Constructing an **augmented predictor space** combining the three channels
2. Applying **elementwise precision weights** based on falsity
3. Fitting PLS via the **NIPALS algorithm** in this augmented Hilbert space

### Augmented Predictor Space

#### Channel Concatenation

$$X^{(\text{aug})} = \left[\, X_T \quad \sqrt{\rho_I} X_I \quad \sqrt{\rho_F} X_F \,\right] \in \mathbb{R}^{n \times 3p}$$

**Channel weights**:
- $\rho_T = 1$ (Truth channel, always included)
- $\rho_I \geq 0$ (Indeterminacy weight, default: 1)
- $\rho_F \geq 0$ (Falsity weight, default: 1)

#### Induced Inner Product

The augmented space $\mathbb{R}^{3p}$ has inner product:

$$\langle u, v \rangle_{\mathcal{N}} = u_T^\top v_T + \rho_I u_I^\top v_I + \rho_F u_F^\top v_F$$

### Precision Weighting

#### Elementwise Weights from Falsity

$$W_{ij} = \exp(-\lambda_F \cdot F_{ij})$$

where $\lambda_F > 0$ controls downweighting strength.

**Interpretation**:
- High falsity $F_{ij} \approx 1$ → low weight $W_{ij} \approx \exp(-\lambda_F)$
- Low falsity $F_{ij} \approx 0$ → high weight $W_{ij} \approx 1$

### NIPALS Algorithm

**Input**: $\widetilde{X}^{(\text{aug})} \in \mathbb{R}^{n \times 3p}$, $y^{(\text{std})} \in \mathbb{R}^n$, $k$ components

**Output**: Latent components $(T, W, P, q, \beta)$

#### Component Extraction (for h = 1, ..., k)

1. Initialize weight vector: $w_h = \frac{X_{\text{deflated}}^\top y_{\text{deflated}}}{\|X_{\text{deflated}}^\top y_{\text{deflated}}\|}$

2. Iterative refinement (until convergence):
   - Compute score: $t_h = X_{\text{deflated}} w_h / \|X_{\text{deflated}} w_h\|$
   - Update weight: $w_h = X_{\text{deflated}}^\top t_h / \|X_{\text{deflated}}^\top t_h\|$

3. Final score and loadings:
   - $t_h = X_{\text{deflated}} w_h$
   - $p_h = X_{\text{deflated}}^\top t_h / (t_h^\top t_h)$
   - $q_h = y_{\text{deflated}}^\top t_h / (t_h^\top t_h)$

4. Deflation:
   - $X_{\text{deflated}} \leftarrow X_{\text{deflated}} - t_h p_h^\top$
   - $y_{\text{deflated}} \leftarrow y_{\text{deflated}} - t_h q_h$

### Regression Coefficients

After extracting $k$ components:

$$\beta = W (P^\top W)^{-1} q$$

### Usage Example

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
```

---

## NVIP: Neutrosophic Variable Importance in Projection

**Module**: `neutrohydro.nvip`

### Overview

NVIP extends the classical VIP (Variable Importance in Projection) metric to **neutrosophic triplet data**, enabling **L2-additive decomposition** of variable importance across Truth, Indeterminacy, and Falsity channels.

**Core Innovation**: Variable importance can be **unambiguously partitioned** into baseline and perturbation components.

### L2 Decomposition Theorem

**Theorem** (NVIP L2 Additivity):

For each variable $j = 1, \ldots, p$:

$$\boxed{\text{VIP}_{\text{agg}}^2(j) = \text{VIP}_T^2(j) + \text{VIP}_I^2(j) + \text{VIP}_F^2(j)}$$

### Channel Energies

- **$E_T(j)$**: Importance of **baseline** (Truth) for ion $j$
- **$E_I(j)$**: Importance of **uncertainty** (Indeterminacy) for ion $j$
- **$E_F(j)$**: Importance of **perturbation** (Falsity) for ion $j$

**Perturbation energy**:

$$E_P(j) = E_I(j) + E_F(j)$$

### Attribution Fractions

For ion $j$, **baseline fraction**:

$$\pi_G(j) = \frac{E_T(j)}{E_T(j) + E_P(j)} \in [0, 1]$$

**Perturbation fraction**:

$$\pi_A(j) = 1 - \pi_G(j) = \frac{E_P(j)}{E_T(j) + E_P(j)}$$

### Usage Example

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
```

---

## Attribution Metrics: NSR and Baseline Fractions

**Module**: `neutrohydro.attribution`

### Overview

Attribution metrics quantify the **baseline vs. perturbation** character of ions and samples based on NVIP energies. Two levels of attribution:

1. **Ion-level**: NSR and π_G quantify baseline fraction per ion
2. **Sample-level**: G_i quantifies baseline fraction per water sample

### Mathematical Foundation

#### Ion-Level Attribution

**Energy Partition**:

**Truth energy** (baseline):
$$E_T(j) = \text{VIP}_T^2(j)$$

**Perturbation energy**:
$$E_P(j) = E_I(j) + E_F(j) = \text{VIP}_I^2(j) + \text{VIP}_F^2(j)$$

#### Baseline Fraction π_G

$$\boxed{\pi_G(j) = \frac{E_T(j)}{E_T(j) + E_P(j)} \in [0, 1]}$$

**Interpretation**:
- $\pi_G(j) \approx 1$: Ion $j$ prediction driven by **baseline**
- $\pi_G(j) \approx 0$: Ion $j$ prediction driven by **perturbation**
- $\pi_G(j) \approx 0.5$: Mixed contribution

#### Neutrosophic Source Ratio (NSR)

$$\boxed{\text{NSR}(j) = \frac{E_T(j) + \epsilon}{E_P(j) + \epsilon}}$$

where $\epsilon > 0$ (default: $10^{-10}$) prevents division by zero.

**Interpretation**:
- NSR$(j) \gg 1$: Baseline-dominant
- NSR$(j) \approx 1$: Balanced
- NSR$(j) \ll 1$: Perturbation-dominant

**Relationship to π_G**:

$$\pi_G(j) = \frac{\text{NSR}(j)}{1 + \text{NSR}(j)}$$

#### Sample-Level Attribution

For sample $i$, the **net contribution** to prediction is:

$$c_{ij} = (\widetilde{X}_T)_{ij} \beta_T(j) + (\widetilde{X}_I)_{ij} \beta_I(j) + (\widetilde{X}_F)_{ij} \beta_F(j)$$

#### Sample Baseline Fraction G_i

$$\boxed{G_i = \frac{\sum_{j=1}^p \pi_G(j) \cdot w_{ij}}{\sum_{j=1}^p w_{ij}} \in [0, 1]}$$

**Weighted average** of ion-level baseline fractions, using attribution masses as weights.

**Interpretation**:
- $G_i \approx 1$: Sample $i$ prediction driven by baseline-dominant ions
- $G_i \approx 0$: Sample $i$ prediction driven by perturbation-dominant ions
- $G_i \approx 0.5$: Mixed

---

## Mineral Stoichiometric Inversion

**Module**: `neutrohydro.minerals`

### Overview

The mineral inference module uses **stoichiometric inversion** to estimate plausible mineral contributions from ion concentration data. This provides a **geochemical interpretation** of water composition in terms of **mineral dissolution/weathering** sources.

**Key innovation**: Uses baseline fractions $π_G$ to **weight ions**, emphasizing baseline-dominant ions in the inversion.

### Stoichiometric Model

#### Forward Model

Ion concentrations arise from **mineral dissolution**:

$$c = A s + r$$

where:
- $c \in \mathbb{R}^m$: Observed ion concentrations (in **meq/L**, recommended)
- $A \in \mathbb{R}^{m \times K}$: **Stoichiometric matrix** (mineral compositions)
- $s \in \mathbb{R}^K_{\geq 0}$: **Mineral contributions** (non-negative)
- $r \in \mathbb{R}^m$: **Residual** (unmodeled processes)

### Weighted NNLS (NeutroHydro Innovation)

Use baseline fractions $π_G$ to **emphasize baseline-dominant ions**:

$$\hat{s} = \arg\min_{s \geq 0} \|D(c - As)\|^2$$

where $D = \text{diag}(d_1, \ldots, d_m)$ and:

$$d_\ell = \pi_G(\text{ion}_\ell)^\eta$$

**Hyperparameters**:
- $\eta \geq 1$: Weighting exponent (default: 1.0)

**Rationale**: Baseline-dominant ions reflect **natural geochemical processes** (mineral weathering), while perturbation-dominant ions may reflect **anthropogenic inputs** (fertilizers, contamination, ion exchange) not modeled by simple dissolution.

### Standard Mineral Library

NeutroHydro includes an expanded "Scientific Research Grade" library of **24 minerals/endmembers**, covering silicates, carbonates, evaporites, and specific anthropogenic markers.

#### Natural Minerals (Geogenic)

| Mineral | Formula | Key Ions | Description |
|---------|---------|----------|-------------|
| **Calcite** | CaCO₃ | Ca²⁺, HCO₃⁻ | Carbonate dissolution |
| **Dolomite** | CaMg(CO₃)₂ | Ca²⁺, Mg²⁺, HCO₃⁻ | Carbonate dissolution |
| **Gypsum** | CaSO₄·2H₂O | Ca²⁺, SO₄²⁻ | Sulfate dissolution |
| **Halite** | NaCl | Na⁺, Cl⁻ | Saline deposits/intrusion |
| **Sylvite** | KCl | K⁺, Cl⁻ | Potash deposits |
| **K-feldspar** | KAlSi₃O₈ | K⁺, HCO₃⁻ | Orthoclase weathering |
| **Albite** | NaAlSi₃O₈ | Na⁺, HCO₃⁻ | Plagioclase weathering |
| **Anorthite** | CaAl₂Si₂O₈ | Ca²⁺, HCO₃⁻ | Plagioclase weathering |

#### Anthropogenic Markers (Pollution Proxies)

| Marker | Formula | Key Ions | Interpretation |
|--------|---------|----------|----------------|
| **Niter** | KNO₃ | K⁺, NO₃⁻ | Potassium-based fertilizers |
| **Soda Niter** | NaNO₃ | Na⁺, NO₃⁻ | Sodium-based fertilizers or wastewater |
| **Nitrocalcite** | Ca(NO₃)₂ | Ca²⁺, NO₃⁻ | Calcium nitrate fertilizers |

### Unit Conversion

#### mg/L to meq/L

$$\text{meq/L} = \frac{\text{mg/L}}{M} \times |z|$$

where:
- $M$ = molar mass (g/mol)
- $|z|$ = absolute charge

**Example** (Ca²⁺):
- Molar mass = 40.078 g/mol
- Charge = +2
- 100 mg/L Ca²⁺ = $\frac{100}{40.078} \times 2 = 4.99$ meq/L

### Usage Example

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

# 5. Access results
print(f"Simpson Class: {result.indices['Simpson_Class']}")
print(f"Inferred Sources: {df_quality['Inferred_Sources']}")
```

---

## Water Quality Assessment

**Module**: `neutrohydro.quality_check`

### Overview

The Quality Assessment module provides an automated system for evaluating groundwater samples against **WHO (World Health Organization)** drinking water guidelines. Beyond simple compliance checking, it implements an **intelligent source inference** engine that interprets combinations of exceedances to suggest potential pollution origins.

### Features

1. **WHO Compliance Check**: Automatically flags parameters exceeding standard limits.
2. **Source Inference**: Uses hydrogeochemical logic to infer the likely cause of contamination (e.g., Saline Intrusion vs. Anthropogenic Pollution).
3. **Integration**: Can be used as a standalone tool or to provide **context-aware constraints** for the Mineral Inversion model.

### Thresholds

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

### Source Inference Rules

#### Saline Intrusion
- **Trigger**: High Chloride ($Cl > 250$) **AND** High Sodium ($Na > 200$).
- **Inference**: "Saline Intrusion/Brine".
- **Implication**: Suggests seawater mixing or deep brine upwelling.

#### Anthropogenic Pollution
- **Trigger**: High Nitrate ($NO_3 > 50$).
- **Inference**: "Anthropogenic (Agri/Sewage)".
- **Implication**: Surface contamination from fertilizers or wastewater.

#### Geogenic (Rock-Water Interaction)
- **Trigger**: High Fluoride ($F > 1.5$) or High Calcium/Sulfate (Gypsum).
- **Inference**: "Geogenic (Rock-Water)".
- **Implication**: Natural weathering of specific mineral formations.

### Usage

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

---

## Model Limitations and Validity

**Module**: `neutrohydro`

### Overview

While NeutroHydro provides a mathematically rigorous framework for groundwater chemometrics, it is subject to inherent limitations common to inverse geochemical modeling and statistical learning.

### Stoichiometric Assumptions

**Limitation**: The Mineral Inversion module relies on a library of fixed stoichiometric formulas (e.g., Calcite = CaCO₃).

**Reality**: Natural minerals often exist as solid solutions or have impurities.

**Addressing Validity**:
- **Residual Analysis**: High residuals indicate standard library cannot fully explain the water chemistry.
- **Endmember Expansion**: Include pure endmembers (e.g., Albite and Anorthite) to approximate solid solutions.

### Non-Uniqueness of Inversion

**Limitation**: The problem of reconstructing mineral assemblages from dissolved ions is mathematically **underdetermined** or **non-unique**.

**Example**: Dissolved Ca²⁺ and SO₄²⁻ could come from Gypsum or Anhydrite. Chemically, they produce identical ions.

**Addressing Validity**:
- **Parsimony Principle**: NNLS favors sparse solutions.
- **Contextual Validation**: User must validate if identified minerals make geological sense.
- **Thermodynamic Consistency**: Cross-reference with Saturation Indices (external validation).

### Linearity of Baseline (PCA/PLS)

**Limitation**: The NDG Encoder and PNPLS use linear projections to define the "Baseline".

**Reality**: Natural geochemical evolution can be highly non-linear (redox fronts, sorption isotherms).

**Addressing Validity**:
- **Neutrosophic Compensation**: The $I$ and $F$ channels capture non-linear deviations from the linear baseline.
- **Interpretation**: High $I$ or $F$ score often signifies a non-linear geochemical process.

### Data Completeness (Missing Ions)

**Limitation**: Geochemical inversion requires charge balance. Missing major ions makes it impossible to distinguish certain minerals.

**Addressing Validity**:
- **Adaptive Filtering**: The model dynamically removes minerals requiring missing ions from the candidate list.

### "Closed System" Assumption

**Limitation**: Mass balance inversion assumes ions come solely from mineral dissolution/precipitation.

**Reality**: Groundwater is an open system. Ions can be added via rainfall, evaporation, or anthropogenic inputs.

**Addressing Validity**:
- **Anthropogenic Markers**: Explicitly include "Pollution Proxies" (e.g., Niter, Nitrocalcite) in the library.
- **Evaporation Handling**: Use conservative ions (Cl, Br) in the baseline to track physical concentration effects.

---

## Hydrogeochemical Processes in NeutroHydro

**Module**: `neutrohydro`

This document explains how specific hydrogeochemical processes (Mixing, Salinization, Ion Exchange, Redox) are mathematically represented within the NeutroHydro framework.

### 1. Mixing and Salinization

**Process**: The physical mixing of two or more distinct water bodies (e.g., fresh recharge + saline connate water).

**Mathematical Nature**: Linear.

**Model Representation**: **Truth ($T$) Channel**.

The **NDG Encoder** uses Robust PCA (or Low-Rank Approximation) to define the "Truth" baseline.
- The Principal Components (PCs) of the $T$ channel naturally align with the **Mixing Lines**.
- **Salinization** (e.g., Seawater Intrusion) typically appears as the **First Principal Component (PC1)** because it explains the largest variance in total dissolved solids (TDS).
- **Validity**: Since mixing is a linear operation, the linear algebra underlying the $T$ channel is mathematically valid for these processes.

### 2. Ion Exchange

**Process**: The adsorption of one ion onto a clay surface and the release of another (e.g., $Ca^{2+}$ adsorbs, $2Na^+$ releases).

**Mathematical Nature**: Non-linear / Non-conservative relative to the mixing line.

**Model Representation**: **Indeterminacy ($I$) and Falsity ($F$) Channels**.

Ion exchange creates a deviation from the linear mixing trend defined in $T$.
- In simple mixing, if $Cl^-$ increases, $Na^+$ should increase proportionally.
- If Ion Exchange occurs, $Na^+$ increases more than expected, while $Ca^{2+}$ increases less.
- The model captures this deviation in the **Falsity ($F$)** matrix.

### 3. Redox Processes (Denitrification, Sulfate Reduction)

**Process**: Biogeochemical removal of species (e.g., $NO_3^- \to N_2(g)$) or addition (Nitrification).

**Mathematical Nature**: Mass loss (Sink) or Gain (Source).

**Model Representation**: **Falsity ($F$)** and **Redox Phases**.

These processes look like "missing mass" relative to the conservative baseline.
- A high Falsity score for Nitrate ($F_{NO3}$) combined with a low Truth value indicates depletion.
- The $F$ channel provides the statistical evidence of the process.

---

## Mathematical Critique of NeutroHydro Model

**Date**: December 28, 2025

This document provides a critical mathematical review of the NeutroHydro framework, identifying potential limitations and proposing rigorous solutions.

### 1. The "Weighting Paradox" in Mineral Inversion

**Issue**: The current `MineralInverter` minimizes the weighted norm:

$$\min_{s \geq 0} \| D \cdot (c - A s) \|_2^2$$

where the weights $D$ are derived from the Baseline Fraction $\pi_G$.

**Logic**: Trust the "Baseline" ions more; downweight the "Perturbed/Noisy" ions.

**Consequence**: Anthropogenic markers (e.g., Nitrate from fertilizer) are often **perturbations** (High $F$, Low $\pi_G$).

**The Paradox**: By downweighting the perturbation, the solver is told "It is okay to ignore Nitrate."

**Result**: The model may underestimate anthropogenic pollution.

**Mathematical Solution**:
For **Forensic Analysis** (identifying pollution), the weighting scheme should be inverted or removed:
1. **Unweighted Inversion**: Set $D = I$ (Identity). Force the model to explain *all* ions.
2. **Targeted Weighting**: Set high weights for suspected markers (e.g., $D_{NO3} = 1.0$) regardless of their $\pi_G$ score.

### 2. Mixing vs. Mineral Dissolution

**Issue**: The NNLS solver assumes all solutes come from dissolving solid phases.

**Reality**: Groundwater often involves **Mixing** with a pre-existing brine (e.g., Seawater).

**Approximation**: Seawater is approximated as a sum of `Halite` + `Sylvite` + `Gypsum` + ...

**Critique**: This loses the **Constant Proportion** constraint of Seawater (e.g., $Cl/Br$ ratio).

**Mathematical Solution**:
Add **Fluid Endmembers** to the Stoichiometric Matrix $A$:
- Define a "mineral" called `Seawater` with exact ionic composition of standard seawater.
- $$A_{Seawater} = [Na=468, Mg=53, Ca=10, Cl=545, SO4=28, ...]$$
- This forces the solver to use the *exact* seawater ratio.

### 3. Non-Uniqueness of Ion Exchange

**Issue**: Ion Exchange phases (e.g., $Ca \to 2Na$) increase **Multicollinearity**.

**Scenario**: 
- High Na, Low Ca.
- **Explanation A**: Dissolve Halite + Precipitate Calcite.
- **Explanation B**: Ion Exchange.

**Solver Behavior**: NNLS picks the path of least resistance. Cannot distinguish without isotopic data.

**Mathematical Solution**:
Apply **L2 (Ridge) or L1 (Lasso) penalties** to Exchanger terms to ensure they are only selected when standard minerals cannot explain the data.

### 4. Simpson's Ratio Discretization

**Issue**: The model uses discrete bins for Simpson's Ratio (e.g., "Moderately Saline" vs "Highly Saline").

**Critique**: Discretization throws away information. A sample at 2.7 is labeled "Moderately", while 2.9 is "Highly", despite being nearly identical.

**Mathematical Solution**:
Use raw ratio values for downstream statistical analysis (correlation, clustering), and reserve discrete classes only for the final human-readable report.

---

## Final Critical Review: Mathematical & Hydrogeochemical Integrity

**Date**: December 28, 2025

This document serves as the final "Red Team" critique of the NeutroHydro framework, evaluating its scientific validity.

### 1. The "Hard Threshold" Problem in CAI Constraints

**Critique**: The implementation uses a hard threshold (e.g., `CAI > 0.05`) to switch between "Freshening" and "Intrusion" modes.

**Mathematical Issue**: This introduces a **discontinuity** in the model function. A sample with CAI=0.049 allows one set of minerals, while CAI=0.051 allows another.

**Hydrogeochemical Reality**: Natural systems are continuous. A sample near equilibrium might experience minor fluctuations.

**Risk**: Small measurement errors in Na or Cl could flip the switch, causing sudden jumps in predicted mineral assemblages (Instability).

**Recommendation**:
- **Soft Gating**: Use a **Sigmoid Weighting** function.
- Weight for `ReleaseNa` = $\sigma(-k \cdot \text{CAI})$
- This smoothly transitions the allowed mass of the exchanger phase to zero.

### 2. The "Sink" Asymmetry

**Critique**: NNLS ($s \ge 0$) is excellent for **Dissolution** but struggles with **Precipitation**.

**Scenario**: Calcite precipitation ($Ca^{2+} + CO_3^{2-} \to CaCO_3$). This removes ions.

**Model Behavior**: The model cannot assign negative mass to "Calcite". It can only model this if we explicitly define a "Precipitation" phase with negative stoichiometry.

**Consequence**: If water is supersaturated and precipitating calcite, the model will have a large **Residual**.

**Recommendation**:
- **Residual Interpretation**: Explicitly document that **Negative Residuals** (Observed < Predicted) imply precipitation or biological uptake.

### 3. Thermodynamic Blindness

**Critique**: NeutroHydro is a **Mass Balance** model, not a **Thermodynamic** model.

**Issue**: It can mathematically propose an impossible mineral assemblage (e.g., dissolved Anhydrite in water that is undersaturated with Gypsum but supersaturated with Anhydrite).

**Missing Link**: The model does not check **Saturation Indices (SI)**. It doesn't know if water *can* dissolve the mineral.

**Recommendation**:
- **External Validation**: For publication, results *must* be cross-referenced with PHREEQC or similar codes to ensure identified phases are not supersaturated.

### 4. Conclusion: Is it Defensible?

**Yes**, with caveats.

The model is now **mathematically superior** to standard inverse models because:
1. **It handles Uncertainty**: The Neutrosophic ($I, F$) logic captures noise that breaks other models.
2. **It is Constrained**: The CAI and Gibbs logic removes egregious non-uniqueness errors.
3. **It is Context-Aware**: WHO integration ensures pollution sources are respected.

**Final Verdict**: The model is valid for **Hypothesis Generation** and **Forensic Fingerprinting**. It should not replace thermodynamic equilibrium modeling (PHREEQC) but complement it.

---

# API REFERENCE

## Pipeline API

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

---

## Core Modules API

### `neutrohydro.encoder`

Handles the Neutrosophic Data Transformation.

- **`NDGEncoder`**: Transforms raw concentrations into Truth (T), Indeterminacy (I), and Falsity (F) components.

### `neutrohydro.minerals`

Handles Stoichiometric Inversion.

- **`MineralInverter`**: Performs weighted NNLS inversion to estimate mineral contributions.
- **`calculate_simpson_ratio`**: Computes Standard and Inverse Simpson's Ratios.

### `neutrohydro.quality_check`

Handles Water Quality Assessment.

- **`assess_water_quality`**: Checks samples against WHO guidelines.
- **`add_quality_flags`**: Adds quality columns to a DataFrame.

### `neutrohydro.nvip`

Handles Variable Importance.

- **`calculate_nvip`**: Computes Neutrosophic Variable Importance in Projection.

---

# EXAMPLES & TUTORIALS

## Basic Examples

For a complete walkthrough of the basic usage, please refer to the [Quick Start Guide](#quick-start-guide).

### Running the Example Script

The repository includes a ready-to-run example script:

```bash
python examples/basic_example.py
```

This script demonstrates:
1. Creating synthetic data.
2. Initializing the pipeline.
3. Running the analysis.
4. Printing the results.

---

## Advanced Workflows

### Custom Mineral Libraries

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

### Handling Redox Processes

To explicitly model redox sinks (like Denitrification), include the `REDOX_PHASES` in your mineral library.

```python
from neutrohydro.minerals import STANDARD_MINERALS, REDOX_PHASES

combined_minerals = {**STANDARD_MINERALS, **REDOX_PHASES}
inverter = MineralInverter(minerals=combined_minerals)
```

---

## Interpreting Results

### Understanding the Outputs

#### 1. Mineral Fractions

These represent the relative contribution of each mineral to the total dissolved solids (TDS) of the sample.

- **Sum to 1**: The fractions are normalized.
- **Interpretation**: A high "Calcite" fraction indicates carbonate weathering is the dominant process.

#### 2. Quality Flags

- **Exceedances**: Lists parameters that violate WHO guidelines.
- **Inferred Sources**: Suggests potential origins (e.g., "Saline Intrusion", "Agricultural").

#### 3. Simpson's Ratio

- **Standard Ratio**: Indicates severity of salinity.
- **Inverse Ratio**: Distinguishes mechanism (Recharge vs. Intrusion).
  - **> 1**: Recharge (Fresh)
  - **< 0.5**: Intrusion (Seawater)

#### 4. VIP Scores (Variable Importance)

- **T-VIP**: Importance of the baseline trend (Mixing).
- **F-VIP**: Importance of perturbations (Pollution/Exchange).

---

## End of Documentation

**Generated**: December 28, 2025

For more information, visit the GitHub repository or consult the individual module documentation.

