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
