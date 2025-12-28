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
