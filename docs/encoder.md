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
