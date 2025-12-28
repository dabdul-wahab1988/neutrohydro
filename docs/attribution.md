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
