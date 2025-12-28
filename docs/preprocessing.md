# Preprocessing & Robust Scaling

**Module**: `neutrohydro.preprocessing`

## Overview

The preprocessing module implements **robust, non-compositional standardization** of ion concentration data. Unlike compositional data analysis (CoDa), we operate in absolute concentration space, preserving physical interpretability and additive mixing models.

## Mathematical Foundation

### 1. Input Data

**Predictor matrix**:
$$X \in \mathbb{R}^{n \times p}, \quad X_{ij} \geq 0$$

where:
- $n$ = number of water samples
- $p$ = number of ion species
- $X_{ij}$ = concentration of ion $j$ in sample $i$

**Units**: Typically mg/L, mmol/L, or meq/L (recommended for stoichiometry)

**Target vector**:
$$y \in \mathbb{R}^n$$

Typically: $y_i = \log(\text{TDS}_i)$, $\log(\text{EC}_i)$, or $\log(I_i)$ where $I$ is ionic strength.

### 2. Optional Log Transform

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

### 3. Robust Centering

For each ion $j$:

$$\mu_j = \text{median}_i(X_{ij})$$

**Why median, not mean?**
- **Robust to outliers**: Median has 50% breakdown point
- **Stable**: Unaffected by single extreme values
- **Appropriate for skewed distributions**: Groundwater data often log-normal

### 4. Robust Scaling

For each ion $j$:

$$s_j = 1.4826 \times \text{MAD}_i(X_{ij})$$

where **MAD** (Median Absolute Deviation):

$$\text{MAD}_i(X_{ij}) = \text{median}_i \left( |X_{ij} - \mu_j| \right)$$

**Scaling factor 1.4826**:
- For normal distribution: $\text{MAD} \times 1.4826 \approx \sigma$ (standard deviation)
- Makes MAD-based scale **consistent** with standard deviation

#### Derivation of 1.4826 Factor

**Theorem**: For $Z \sim \mathcal{N}(0, \sigma^2)$, $\text{MAD}(Z) = \sigma \cdot \Phi^{-1}(0.75)$ where $\Phi^{-1}$ is the inverse standard normal CDF.

**Proof**:

Let $Z \sim \mathcal{N}(0, \sigma^2)$. The median of $Z$ is 0 (by symmetry).

The MAD is:
$$\text{MAD}(Z) = \text{median}(|Z - 0|) = \text{median}(|Z|)$$

For $|Z|$, we need the median of the **folded normal distribution**.

The CDF of $|Z|$ is:
$$P(|Z| \leq t) = P(-t \leq Z \leq t) = \Phi(t/\sigma) - \Phi(-t/\sigma) = 2\Phi(t/\sigma) - 1$$

where $\Phi$ is the standard normal CDF.

The median $m$ of $|Z|$ satisfies:
$$P(|Z| \leq m) = 0.5$$

Therefore:
$$2\Phi(m/\sigma) - 1 = 0.5$$
$$\Phi(m/\sigma) = 0.75$$
$$m/\sigma = \Phi^{-1}(0.75)$$

From standard normal tables: $\Phi^{-1}(0.75) \approx 0.6745$

Thus:
$$\text{MAD}(Z) = 0.6745 \sigma$$

To make MAD consistent with $\sigma$, we use:
$$\text{scaling factor} = \frac{1}{0.6745} = 1.4826$$

Therefore:
$$\boxed{s_j = 1.4826 \times \text{MAD}_j \approx \sigma_j \quad \text{for normal data}}$$

$\square$

**Why MAD, not standard deviation?**
- **Robust to outliers**: 50% breakdown point
- **Stable**: Unaffected by extreme values
- **Appropriate for heavy tails**: Common in environmental data

### 5. Standardization

$$X^{(\text{std})}_{ij} = \frac{X_{ij} - \mu_j}{s_j + \delta_s}$$

where $\delta_s > 0$ (default: $10^{-10}$) prevents division by zero for constant columns.

**Similarly for target**:

$$y^{(\text{std})}_i = \frac{y_i - \mu_y}{s_y + \delta_s}$$

where:
$$\mu_y = \text{median}(y), \quad s_y = 1.4826 \times \text{MAD}(y)$$

### 6. Inverse Transform

To recover predictions in original scale:

$$\hat{y}_i = \hat{y}^{(\text{std})}_i \cdot (s_y + \delta_s) + \mu_y$$

## Handling Missing Data

### 1. Missingness Mask

$$M \in \{0, 1\}^{n \times p}$$

where $M_{ij} = 1$ if observed, $M_{ij} = 0$ if missing.

### 2. Detection Limits

For left-censored data (below detection limit):

$$\text{DL} \in \mathbb{R}^p_{\geq 0}$$

where $\text{DL}_j$ is the detection limit for ion $j$.

Values $X_{ij} < \text{DL}_j$ are flagged as censored.

### 3. Imputation Methods

**Method 1: Median Fill**

$$X_{ij} \leftarrow \text{median}_{i': M_{i'j} = 1}(X_{i'j}) \quad \text{if } M_{ij} = 0$$

**Method 2: Zero Fill**
$$X_{ij} \leftarrow 0 \quad \text{if } M_{ij} = 0$$

**Method 3: DL/2 Fill** (standard for environmental data)
$$X_{ij} \leftarrow \frac{\text{DL}_j}{2} \quad \text{if } X_{ij} < \text{DL}_j$$

### 4. Propagation to Indeterminacy

Censored observations should be marked with **high indeterminacy** $I_{ij}$ in the NDG encoder (see [encoder.md](encoder.md)).

## Algorithm

### Fitting (Training Data)

**Input**: $X \in \mathbb{R}^{n \times p}$, $y \in \mathbb{R}^n$

**Output**: Preprocessing parameters $\theta = (\mu, s, \mu_y, s_y)$

```
1. If log_transform:
     X_work ← log(X + δ_x)
   else:
     X_work ← X

2. For each column j = 1, ..., p:
     μ_j ← median(X_work[:,j])
     deviations ← |X_work[:,j] - μ_j|
     MAD_j ← median(deviations)
     s_j ← 1.4826 × MAD_j

3. For target:
     μ_y ← median(y)
     MAD_y ← median(|y - μ_y|)
     s_y ← 1.4826 × MAD_y

4. Store θ = (μ, s, μ_y, s_y, log_transform, δ_x, δ_s)
```

### Transformation (New Data)

**Input**: $X_{\text{new}} \in \mathbb{R}^{m \times p}$, parameters $\theta$

**Output**: $X^{(\text{std})}_{\text{new}} \in \mathbb{R}^{m \times p}$

```
1. If log_transform:
     X_work ← log(X_new + δ_x)
   else:
     X_work ← X_new

2. For each column j:
     X_std_new[:,j] ← (X_work[:,j] - μ_j) / (s_j + δ_s)

3. Return X_std_new
```

## Properties

### 1. Location-Scale Family

Standardization is **affine-equivariant**:

If $\tilde{X}_j = a_j X_j + b_j$, then after standardization:
$$\tilde{X}^{(\text{std})}_j = \text{sign}(a_j) \cdot X^{(\text{std})}_j$$

### 2. Robustness

**Breakdown point**: Proportion of outliers that can be tolerated before estimate degrades arbitrarily.

- Median: 50% breakdown point
- MAD: 50% breakdown point
- Mean: 0% breakdown point
- Standard deviation: 0% breakdown point

#### Proof: 50% Breakdown Point of Median

**Definition**: The **finite-sample breakdown point** of an estimator $T$ on a sample $X = \{x_1, \ldots, x_n\}$ is:
$$\epsilon^*_n(T, X) = \min\left\{\frac{m}{n} : \sup_{\tilde{X}} |T(\tilde{X})| = \infty\right\}$$

where $\tilde{X}$ is obtained by replacing $m$ values in $X$.

**Theorem**: The median has breakdown point $\epsilon^* = \lfloor n/2 \rfloor / n \approx 0.5$.

**Proof**:

Let $X = \{x_1, \ldots, x_n\}$ be ordered: $x_1 \leq x_2 \leq \ldots \leq x_n$.

The median is:

$$\text{median}(X) = \begin{cases} x_{(n+1)/2} & \text{if } n \text{ odd} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ even} \end{cases}$$

**Case 1**: Replace $m < n/2$ values with arbitrary outliers $\to \infty$.

Even if we replace the $m$ largest values with $+\infty$, the median is determined by the middle position(s). Since $m < n/2$, the middle position(s) are **not affected**, so:
$$|\text{median}(\tilde{X})| < \infty$$

**Case 2**: Replace $m \geq n/2$ values with outliers $\to \infty$.

If we replace the largest $\lceil n/2 \rceil$ values with $+\infty$, then the median position itself is $+\infty$:
$$\text{median}(\tilde{X}) = +\infty$$

Therefore:
$$\epsilon^*_n(\text{median}) = \frac{\lfloor n/2 \rfloor}{n}$$

For large $n$: $\epsilon^* \to 0.5$.

$\square$

#### Proof: 50% Breakdown Point of MAD

**Theorem**: MAD has breakdown point $\epsilon^* \approx 0.5$.

**Proof**:

The MAD is:
$$\text{MAD}(X) = \text{median}(|X - \text{median}(X)|)$$

This is a **composition** of two medians:
1. Outer median (of absolute deviations)
2. Inner median (center)

**Claim**: If the inner median is finite and $m < n/2$ values are replaced, the MAD remains finite.

Let $\mu = \text{median}(X)$ (finite if $m < n/2$ by previous proof).

Even if $m < n/2$ values are replaced with outliers, the deviations $|x_i - \mu|$ for the **remaining** $> n/2$ values are finite.

The median of absolute deviations is determined by the middle position among $n$ values. Since $> n/2$ deviations are finite, the median of deviations is finite:
$$\text{MAD}(\tilde{X}) < \infty \quad \text{if } m < n/2$$

If $m \geq n/2$, we can make the median itself $\to \infty$, then:
$$|x_i - \mu| \to \infty$$

and thus:
$$\text{MAD}(\tilde{X}) \to \infty$$

Therefore:
$$\epsilon^*_n(\text{MAD}) \approx \frac{\lfloor n/2 \rfloor}{n} \to 0.5$$

$\square$

**Implication**: Median and MAD are **maximally robust** among equivariant estimators, tolerating up to 50% contamination.

**Implication**: Up to 50% of data can be outliers without affecting $\mu$ or $s$.

### 3. Standardized Distribution

After standardization:
- **Median** ≈ 0 for each column
- **MAD-based scale** ≈ 1 for each column
- **Distribution shape** preserved (no normality assumption)

### 4. Comparison to Z-Score

| Feature | Robust (MAD) | Standard (Z-score) |
|---------|--------------|-------------------|
| Center | Median | Mean |
| Scale | MAD × 1.4826 | Std deviation |
| Breakdown point | 50% | 0% |
| Efficiency (normal data) | ~95% | 100% |
| Outlier sensitivity | Low | High |

## Implementation Notes

### 1. Computational Complexity

- Centering: $O(np)$ for sorting each column
- Scaling: $O(np)$ for MAD computation
- Total: $O(np \log n)$ due to median computation

### 2. Numerical Stability

- Small constants $\delta_x, \delta_s$ prevent:
  - $\log(0)$ errors
  - Division by zero for constant columns
  - Underflow in downstream computations

### 3. Storage

Store preprocessing parameters for reproducibility:
```python
params = {
    'mu': μ,           # shape (p,)
    's': s,            # shape (p,)
    'mu_y': μ_y,       # scalar
    's_y': s_y,        # scalar
    'log_transform': bool,
    'delta_x': δ_x,
    'delta_s': δ_s,
    'feature_names': [list of ion names]
}
```

## Usage Example

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

## Diagnostics

### 1. Check for Constant Columns

If $s_j \approx 0$ after scaling, ion $j$ has **near-constant concentration**:
- May indicate measurement error
- Or truly invariant ion (e.g., background level)
- Consider removing from analysis

### 2. Check Skewness

If data is highly skewed ($> 2$), consider:
- Log transform
- Checking for outliers/data quality
- Using more robust baseline in NDG encoder

### 3. Correlation Structure

Standardization preserves correlation structure:
$$\text{corr}(X^{(\text{std})}_j, X^{(\text{std})}_k) = \text{corr}(X_j, X_k)$$

High correlations (e.g., $> 0.9$) may indicate:
- Stoichiometric relationships (e.g., Na-Cl from halite)
- Multicollinearity (not a problem for PLS)

## References

1. Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute deviation. *Journal of the American Statistical Association*, 88(424), 1273-1283.

2. Huber, P. J., & Ronchetti, E. M. (2009). *Robust statistics* (2nd ed.). John Wiley & Sons.

3. Filzmoser, P., Hron, K., & Reimann, C. (2009). Univariate statistical analysis of environmental (compositional) data: Problems and possibilities. *Science of the Total Environment*, 407(23), 6100-6108.

---

**Next**: [NDG Encoder](encoder.md) - Converting standardized data to neutrosophic triplets.
