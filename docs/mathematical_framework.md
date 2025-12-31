# Mathematical Framework Overview

## 1. Introduction

NeutroHydro implements the **Neutralization-Displacement Geosystem (NDG) theory with Stoichiometric Inversion**, a neutrosophic chemometric framework for groundwater analysis that operates in **absolute concentration space**. This document provides a high-level overview of the mathematical theory underpinning the package.

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
│ Output: Mineral contributions s, plausibility, indices (SR, BEX)  │
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
│                                                             │
│ Diagnostic Indices:                                         │
│   • Simpson Ratio (SR): Cl / (HCO3 + CO3)                  │
│   • Base Exchange Index (BEX): Na + K + Mg - 1.0716·Cl     │
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
