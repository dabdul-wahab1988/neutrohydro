# Jacobian Orthogonality: Exact Baseline/Perturbation Attribution

**Module**: `neutrohydro.orthogonal` (encoder, attribution, alignment, diagnostics) and
`neutrohydro.minerals` (identifiability, error propagation)

## Overview

NeutroHydro answers attribution questions such as: how much of the variation in TDS is
carried by the geogenic baseline, and how much by perturbations? The answer is unique
only if the Truth, Indeterminacy and Falsity channels respond to **different** directions
of ion space. This page:

1. defines that requirement (Jacobian orthogonality),
2. shows that the legacy NDG encoder does not meet it,
3. constructs an encoder that meets it exactly (`baseline_type="orthogonal"`),
4. lists the identities this buys (an exact $R^2$ split, exact per-ion shares,
   prediction invariance), each of which is checked in `tests/test_orthogonal.py`,
5. applies the same Jacobian reasoning to the mineral inversion (which minerals are
   identifiable, and with what error bars).

---

## 1. Why Attribution Needs Non-Overlapping Channels

### 1.1 Definition

An encoder maps a standardized ion vector $x \in \mathbb{R}^p$ to channels
$c(x) \in \{T(x), I(x), F(x)\}$. The Jacobian of channel $c$ is

$$J_c(x) = \frac{\partial c(x)}{\partial x} \in \mathbb{R}^{p \times p}.$$

A small change $\delta x$ moves channel $c$ by $J_c \delta x$. The rows of $J_c$ span the
input directions the channel **sees**; directions in $\ker J_c$ are invisible to it.
Two channels see disjoint (orthogonal) sets of directions iff

$$\boxed{J_a(x)\, J_b(x)^\top = 0 \qquad (a \neq b)}$$

since $(J_a J_b^\top)_{kl} = \langle \text{row}_k(J_a), \text{row}_l(J_b) \rangle$.
This is **Jacobian orthogonality**. When it holds, any change of composition is split
uniquely into a part only the baseline sees and parts only the other channels see. When
it fails, the same change of composition is visible to several channels, and how its
effect on the prediction is divided among them is not determined by the data.

### 1.2 The legacy encoder violates it

| Source of overlap | Effect |
|---|---|
| $I$ (local variance of the raw $x$) is a function of all of $x$ | $I$ sees the directions $T$ sees |
| $F = g(\lvert x - T(x)\rvert/\sigma)$ | orthogonal to $T$ only if $T$ is an orthogonal projection (the low-rank baseline); not for robust PCA |
| Elementwise precision weights: the model uses $e^{-\lambda_F F} \odot T$ | $J_{T'} = \operatorname{diag}(e^{-\lambda_F F}) J_T + \operatorname{diag}(T \odot \partial_x e^{-\lambda_F F})$, so the effective $T$ sees $F$'s directions |
| Robust-PCA baseline with soft thresholding | the baseline is a nonlinear function of $x$ whose Jacobian is not a projector |
| Median baseline | $J_T = 0$: the baseline sees nothing, so it cannot explain any sample-to-sample variation ($\pi_G \equiv 0$) |

`jacobian_entanglement` measures the overlap with finite differences:

$$E_{ab} = \frac{\|J_a J_b^\top\|_F}{\|J_a\|_F\,\|J_b\|_F} \in [0, 1],$$

0 meaning orthogonal and 1 meaning both channels see the same directions. On `data.csv`
(9 ions in meq/L, target $\log_{10}$ TDS, log transform, 3 components), the effective
channels of the legacy model (robust PCA, rank 2) give

| Pair | Legacy model channels | Orthogonal encoder |
|---|---|---|
| T–I | 0.513 | 0 |
| T–F | 0.550 | 0 |
| I–F | 0.652 | 0 |

For the median baseline the T-channel Jacobian norm is exactly 0.

A related caution about NVIP: the identity $VIP_{agg}^2 = VIP_T^2 + VIP_I^2 + VIP_F^2$
holds by construction, since it sums squared blocks of one weight vector. It does not
show that the channels carry separable information. That property is what this page
supplies.

---

## 2. Construction

### 2.1 Complementary orthogonal projectors

Let $\mu_\omega$ and $\Sigma_\omega$ be the weighted mean and covariance of the
standardized training data with sample weights $\omega_i$ (Section 2.2), and let
$\Sigma_\omega = V \Lambda V^\top$ with eigenvalues in decreasing order. Split the
eigenvectors into three consecutive blocks of sizes $r_T$, $r_I$ and
$r_F = p - r_T - r_I$:

$$V = [\,V_T \mid V_I \mid V_F\,], \qquad P_c = V_c V_c^\top .$$

The projectors satisfy

$$P_T + P_I + P_F = I_p, \qquad P_a P_b = 0 \;(a \neq b), \qquad P_c^2 = P_c = P_c^\top .$$

Each sample is encoded by its **signed** components

$$T_i = P_T(x_i - \mu_\omega), \quad I_i = P_I(x_i - \mu_\omega), \quad F_i = P_F(x_i - \mu_\omega),$$

so $T_i + I_i + F_i = x_i - \mu_\omega$ exactly, and $R_i = I_i + F_i$ is the off-baseline
residual. The channel Jacobians are the constant projectors, $J_c = P_c$, and
$J_a J_b^\top = P_a P_b = 0$. Unlike the legacy $F$, the components keep the sign of the
deviation: enrichment and depletion are distinguished.

- **T**: the dominant, statistically significant covariation (regional geogenic
  processes, mixing).
- **I**: directions whose membership in the significant structure cannot be decided from
  the data (Section 2.3).
- **F**: the remaining off-baseline directions (local anomalies, contamination,
  analytical noise).

### 2.2 Falsity-reweighted spectral baseline (`robust=True`)

Outlying samples should not define the baseline. The weights are estimated by fixed-point
iteration (`max_iter=100`, `tol=1e-8` on the weights):

```
w ← 1
repeat
    μ_w, Σ_w ← weighted mean and covariance;  V ← eigenvectors of Σ_w (descending)
    d_i ← ||P_F (x_i − μ_w)||                 # perturbation distance
    f_i ← g(d_i / median_k d_k)               # falsity membership in [0, 1)
    w_i ← exp(−λ_F f_i)
until max_i |Δw_i| < tol
μ_w, Σ_w, V ← recomputed with the final stored w
```

$g$ is the falsity map: $1 - e^{-u}$ (default) or logistic. The last step guarantees that
the stored weights are exactly the ones that define the subspaces, which is what makes the
sample-space identities of Section 3 exact. The ranks are fixed before the reweighting
starts. The same weights are passed to PNPLS as row weights (weighted least squares),
replacing the elementwise precision weighting, which would mix the channels.

Elementwise memberships, for display and diagnostics, are functions of a single channel:
$\mu_{F,ij} = g(\lvert F_{ij}\rvert / \sigma_{F,j})$ and
$\mu_{I,ij} = 1 - e^{-\lvert I_{ij}\rvert/\sigma_{I,j}}$, with
$\sigma_{c,j} = 1.4826\,\operatorname{median}_i \lvert c_{ij} \rvert$.

### 2.3 Indeterminacy as undecidability of the baseline rank

`select_ranks` combines Horn's parallel analysis with a bootstrap:

1. **Null thresholds.** Permute each ion column independently (`n_permutations=200`).
   This destroys covariation but keeps each ion's distribution. The threshold for the
   $k$-th eigenvalue is the 0.95 quantile (`null_quantile`) of the $k$-th null eigenvalue.
2. **Significant rank.** $r(\lambda)$ is the number of leading eigenvalues above their
   thresholds, counted until the first failure.
3. **Bootstrap.** Resample the samples (`n_bootstrap=200`), recompute the eigenvalues and
   $r_b$ against the same thresholds.
4. **Bands.** With $\alpha = 0.1$:

$$r_T = \operatorname{clip}\big(\min(q_{\alpha}(r_b),\, r_{\text{obs}}),\ r_{\min},\ p-1\big),
\qquad r_T + r_I = \operatorname{clip}\big(\max(q_{1-\alpha}(r_b),\, r_{\text{obs}},\, r_T),\ r_T,\ p-1\big).$$

Directions significant in essentially every resample form the baseline; directions
significant in some resamples but not others are **indeterminate**; the rest are
perturbation. At least one perturbation direction is always kept. If the rank is stable,
$r_I = 0$. Explicit ranks can be given with `rank_T` and `rank_I`.

**Why the correlation matrix.** Permutation keeps each column's variance. On a covariance
matrix with unequal column variances, the null eigenvalues reproduce the largest column
variances, and real structure can fall below them. In a two-factor test case, the
covariance eigenvalues $(24.41, 6.73, \dots)$ against null thresholds
$(13.08, 10.30, \dots)$ gave rank 1. The correlation eigenvalues $(3.69, 1.80, 0.90, \dots)$
against $(1.49, 1.30, 1.18, \dots)$ gave the correct rank 2. Horn's test is defined on the
correlation matrix, and `select_ranks` uses it by default (`scale="correlation"`). The
subspaces themselves are estimated in the model's standardized coordinates.

On `data.csv` the bootstrap rank counts for ranks 0, 1, 2, 3 were $[0, 7, 158, 35]$,
which gives $r_T = 2$, $r_I = 1$ and $r_F = 6$.

---

## 3. Guarantees

Notation: $\tilde X$ is the centered training matrix, $\Omega = \operatorname{diag}(\omega)$,
$C_c = \tilde X P_c$ is the channel-$c$ block, and $M = [P_T, P_I, P_F] \in \mathbb{R}^{p \times 3p}$,
so that $X^{(\text{aug})} = \tilde X M$.

### Theorem 1 (Pythagoras per sample)

$$\|x_i - \mu_\omega\|^2 = \|T_i\|^2 + \|I_i\|^2 + \|F_i\|^2 .$$

*Proof.* The cross terms are $\langle P_a v, P_b v\rangle = v^\top P_a P_b v = 0$. $\square$

Hence $(\tau_i, \iota_i, \varphi_i) = (\|T_i\|^2, \|I_i\|^2, \|F_i\|^2)/\|x_i - \mu_\omega\|^2$
is a neutrosophic triplet with $\tau_i + \iota_i + \varphi_i = 1$, **derived** from the
data rather than posited. A sample exactly at the center is assigned $(1, 0, 0)$.
*Tests*: `test_projectors`, `test_pythagoras_and_memberships`, `test_components_reconstruct`.

### Theorem 2 (memberships inherit the Jacobian row space)

For any differentiable $h$, $J_{h \circ c}(x) = Dh(c(x))\, J_c(x)$, so the rows of
$J_{h\circ c}$ lie in the row space of $J_c$. Therefore
$J_T J_{h\circ F}^\top = J_T J_F^\top Dh^\top = 0$.

The falsity memberships $\mu_F$ and $f_i$ are functions of $F$ only (differentiable
almost everywhere). They are therefore Jacobian-orthogonal to $T$. In the legacy encoder
the same argument holds for $F$ with the low-rank baseline, where $x - T(x)$ is a
projection. It fails for the soft-thresholded robust-PCA baseline, for the I channel, and
for the precision-weighted channels the model actually uses.
This follows directly from the chain rule and is not tested separately; the channel
statement is tested by finite differences in `test_finite_difference_entanglement`.

### Theorem 3 (channels are uncorrelated across samples)

$$C_a^\top \Omega C_b = 0 \qquad (a \neq b).$$

*Proof.* $C_a^\top \Omega C_b = P_a \tilde X^\top \Omega \tilde X P_b
= (\textstyle\sum_i \omega_i)\, V_a V_a^\top V \Lambda V^\top V_b V_b^\top$. Since
$V_a^\top V \Lambda V^\top V_b$ is the off-diagonal $(a, b)$ block of the diagonal matrix
$\Lambda$, it is zero. $\square$

PLS therefore cannot trade weight between collinear channels: the split of weights across
channels is identifiable. *Test*: `test_sample_space_orthogonality`.

### Theorem 4 (prediction invariance)

With $\rho_I = \rho_F = 1$, PNPLS on $[T, I, F]$ gives the same scores, fitted values and
predictions for new samples as PLS on $\tilde x$, for any row weights.

*Proof.* $M M^\top = P_T + P_I + P_F = I_p$, so
$X^{(\text{aug})} X^{(\text{aug})\top} = \tilde X \tilde X^\top$. Row weighting and deflation
both act from the left, $Q\,\Omega^{1/2} \tilde X M$, so every Gram matrix NIPALS uses
equals that of the plain problem. The PLS1 weights are $w^{(\text{aug})}_h = M^\top w_h$,
with the same norm since $\|M^\top v\|^2 = v^\top M M^\top v$. The loadings are
$P^{(\text{aug})} = M^\top P$, hence
$\beta^{(\text{aug})} = M^\top W (P^\top M M^\top W)^{-1} q = M^\top \beta$ and
$x^{(\text{aug})\top}\beta^{(\text{aug})} = \tilde x^\top M M^\top \beta = \tilde x^\top \beta$. $\square$

The columns of $M$ form a Parseval frame of ion space. The encoding is therefore a pure
**attribution device**: tripling the number of predictor columns cannot add fit or
overfit, and the channels only redistribute what the plain model already explains.
*Test*: `test_prediction_invariance`. That PNPLS computes true PLS is tested against
scikit-learn in `tests/test_model.py::TestPLS1Correctness::test_matches_sklearn_pls`.

### Theorem 5 (exact attribution)

Let $\beta_c$ be the channel blocks of $\beta^{(\text{aug})}$, $c_{ijc} = (C_c)_{ij}(\beta_c)_j$
the contribution of ion $j$ through channel $c$ to sample $i$, and
$\hat y_{c} = C_c \beta_c$ the channel prediction.

1. **Completeness**: $\sum_{j,c} c_{ijc} = \hat y_i - \bar y_\omega$ for every sample.
2. **Orthogonal channel predictions**: $\hat y_a^\top \Omega \hat y_b = \beta_a^\top C_a^\top \Omega C_b \beta_b = 0$ (Theorem 3).
3. **Exact $R^2$ split**: the PLS1 fitted vector is the $\Omega$-orthogonal projection of
   $y - \bar y_\omega$ onto the span of the scores, so on the training data
   $$R^2_\omega = \frac{\|\hat y - \bar y_\omega\|_\Omega^2}{\|y - \bar y_\omega\|_\Omega^2}
   = R^2_T + R^2_I + R^2_F, \qquad R^2_c = \frac{\|\hat y_c\|_\Omega^2}{\|y - \bar y_\omega\|_\Omega^2}.$$
4. **Exact per-ion split**: with $E_c(j) = \sum_i \omega_i c_{ijc}^2$, the cross terms
   $\sum_i \omega_i c_{ijT} c_{ijF} = (\beta_T)_j (\beta_F)_j\,(P_T \tilde X^\top \Omega \tilde X P_F)_{jj} = 0$
   vanish, so $\sum_i \omega_i (\sum_c c_{ijc})^2 = E_T(j) + E_I(j) + E_F(j)$ and
   $\pi_c(j) = E_c(j) / \sum_{c'} E_{c'}(j)$ are exact shares that sum to 1.
5. **Sample level**: $G_i = \hat y_{iT}^2 / \sum_c \hat y_{ic}^2$. Its average with weights
   $\omega_i \sum_c \hat y_{ic}^2$ equals $R^2_T / R^2_\omega$ exactly.

*Tests*: `test_exact_r2_split`, `test_completeness_and_ion_additivity`,
`test_sample_G_aggregates_to_r2_share`. On `data.csv`:
$R^2_\omega = 0.892 = 0.753\,(T) + 0.056\,(I) + 0.084\,(F)$, with all identity residuals
below $10^{-12}$.

`orthogonal_nsr` and `orthogonal_sample_attribution` return these quantities in the
existing `NSRResult` / `SampleAttributionResult` containers ($\pi_G = \pi_T$,
$NSR = E_T/(E_I + E_F)$, $G = G_i$). Mineral weighting, visualization and export work
unchanged.

### Theorem 6 (channel weights)

With $\rho_I, \rho_F \neq 1$ the blocks become $\sqrt{\rho_c} C_c$. Theorem 3, and with it
all parts of Theorem 5, still hold. Only prediction invariance is lost: the $\rho$'s then
act as a prior favouring some subspaces. *Test*: `test_orthogonality_survives_channel_weights`.

### Theorem 7 (sensitivity split)

For a fitted model, $\hat y(x) = \bar y + \sum_c \sqrt{\rho_c}\,\big((x - \mu_\omega)^\top P_c - m_c\big)\beta_c$, so

$$\nabla_x \hat y = \sum_c b_c, \qquad b_c = \sqrt{\rho_c}\, P_c \beta_c \in \operatorname{range}(P_c),
\qquad \|\nabla_x \hat y\|^2 = \sum_c \|b_c\|^2 .$$

The predicted response to a composition change $\delta$ splits exactly as
$\Delta \hat y = \sum_c b_c^\top \delta$ (`channel_sensitivity`, `scenario_response`).
*Tests*: `test_sensitivity_is_orthogonal_split`, `test_scenario_response`.

### Theorem 8 (chain rule through preprocessing)

Process directions are known in concentration units (meq/L). Standardization is
$z = (g(c) - m)/(s + \delta_s)$ with $g$ the identity or $\log(c + \delta_x)$, so

$$\frac{\partial z}{\partial c} = \operatorname{diag}\!\left(\frac{1}{s + \delta_s}\right) \text{ (linear)}, \qquad
\frac{\partial z}{\partial c}\bigg|_{c_{\text{ref}}} = \operatorname{diag}\!\left(\frac{1}{(c_{\text{ref}} + \delta_x)(s + \delta_s)}\right) \text{ (log)}.$$

`directions_to_standardized` applies this map. With a log transform the map is local; the
default reference $c_{\text{ref}}$ is the training median composition.
*Test*: `test_directions_to_standardized_log_jacobian`.

Out-of-sample behaviour: the fitted projectors and center are applied unchanged to new
data (`test_transform_uses_fitted_state`), and the encoding is permutation-equivariant
(`test_permutation_equivariance`).

---

## 4. Process Alignment

For a process direction $a$ in standardized units (a mineral's stoichiometric column,
seawater, a fertiliser signature), `subspace_alignment` returns

$$s_c(a) = \frac{\|P_c a\|^2}{\|a\|^2}, \qquad s_T + s_I + s_F = 1 ,$$

which is exact by Theorem 1. A random direction places $r_c/p$ of its energy in subspace
$c$, so the table also reports the **enrichment** $s_c(a)/(r_c/p)$ and labels each process
by its most enriched channel.

How to read it:

- A geogenic weathering vector enriched in $T$ supports reading the baseline as geogenic.
- An anthropogenic vector (nitrate, fertiliser, wastewater) enriched in $T$ warns that the
  "baseline" itself contains a widespread contamination signal.
- A process enriched in $F$ moves water off the regional structure: a candidate
  perturbation mechanism.

`data.csv`, at the median composition (log transform), chance levels T 0.22, I 0.11, F 0.67:

| Process | $s_T$ | Enrichment T | Enrichment F | Label |
|---|---|---|---|---|
| Fluorite | 0.55 | 2.48 | 0.29 | T |
| K-feldspar | 0.38 | 1.72 | 0.91 | T |
| Dolomite | 0.36 | 1.64 | 0.77 | T |
| Biotite | 0.34 | 1.51 | 0.89 | T |
| Calcite, Halite | 0.23 | 1.03–1.06 | 1.11 | ≈ chance |
| Seawater | 0.17 | 0.78 | 1.24 | F |
| NH₄NO₃ fertiliser (nitrified) | 0.11 | 0.51 | 1.31 | F |
| Gypsum | 0.08 | 0.38 | 1.37 | F |
| Epsomite | 0.01 | 0.06 | 1.48 | F |
| SodaNiter | 0.01 | 0.05 | 1.37 | F |

Silicate and carbonate weathering align with the baseline subspace; sulfate, nitrate,
fertiliser and seawater directions align with the perturbation subspace. Niter (KNO₃) is
also T-enriched (1.65). A nitrate-bearing direction in the baseline is the kind of result
the second rule asks you to check before reading the baseline as geogenic.

---

## 5. Identifiability of the Mineral Inversion

### 5.1 Forward Jacobian

Mass balance writes $c = A s$, so $\partial c / \partial s = A$. Two minerals can be told
apart only to the extent that their (weighted) stoichiometric columns point in different
directions. `stoichiometric_identifiability` (also `MineralInverter.identifiability`) reports:

- **Hadamard orthogonality defect**
  $$\delta(A) = \sum_k \log \|a_k\| - \tfrac12 \log\det(A^\top A) \;\ge\; 0 ,$$
  which is 0 iff the columns are orthogonal and $+\infty$ if they are linearly dependent
  (Hadamard's inequality). It is the IMA contrast that Gresele et al. (2021) use for the
  identifiability of mixing Jacobians, applied here to $\partial c/\partial s$.
- **Variance inflation factors**: $VIF_k = 1/(1 - R_k^2)$, regressing mineral $k$'s
  normalized column on the others ($\infty$: exactly reproducible by the others).
- **Alias groups**: minerals with parallel columns. No data on the measured ions can
  separate them; NNLS picks one member arbitrarily.
- **Underdetermined flag**: more minerals than the rank of $A$.

For the standard library on the 9 major ions (Ca²⁺, Mg²⁺, Na⁺, K⁺, HCO₃⁻, Cl⁻, SO₄²⁻,
NO₃⁻, F⁻) there are 19 minerals and rank 8, so the library is underdetermined. The
alias groups are:

- **Calcite ≡ Anorthite**: with no dissolved silica among the measured ions, anorthite
  weathering releases Ca²⁺ and HCO₃⁻ in the same 1:1 equivalent ratio as calcite.
- **Gypsum ≡ Anhydrite**.
- **Mirabilite ≡ Thenardite**.

There are also non-pairwise dependences such as Dolomite = Calcite + Magnesite, which
VIF detects and pairwise cosines do not. Report alias groups, not their arbitrary
representative.

### 5.2 Per-sample error bars

With `invert(c, rel_error=0.05, abs_error=0.0)`, on each sample's NNLS support
$S = \{k : s_k > 0\}$ the solution is locally $s_S = (D A_S)^{+} D c$, so

$$J = \frac{\partial s_S}{\partial c} = (D A_S)^{+} D, \qquad
\operatorname{Cov}(s_S) = J \operatorname{diag}(\sigma_c^2) J^\top, \qquad
\sigma_c = \text{rel\_error}\cdot|c| + \text{abs\_error}.$$

The result fields `s_std`, `support_condition` ($\operatorname{cond}(D A_S)$) and
`support_defect` ($\delta(D A_S)$) hold the outcome. Example, sample MAM7 in `data.csv`
at 5 % analytical error: Halite $95.2 \pm 3.5$ meq/L (well determined), Sylvite
$2.1 \pm 2.3$ meq/L (not distinguishable from zero); support condition number 4.5, defect
0.73. The linearization is valid only while the support is stable, i.e. while the error
does not switch minerals in or out of the solution. *Tests*:
`tests/test_minerals.py::TestIdentifiabilityAndUncertainty`.

---

## 6. Usage

### Pipeline

```python
from neutrohydro import NeutroHydroPipeline
from neutrohydro.pipeline import PipelineConfig

config = PipelineConfig(
    baseline_type="orthogonal",
    baseline_rank=None,        # None: select r_T by bootstrap parallel analysis
    indeterminacy_rank=None,   # None: select r_I from the bootstrap rank interval
    orthogonal_robust=True,    # falsity-reweighted covariance (uses lambda_F)
    log_transform=True,
    n_components=3,
)
results = NeutroHydroPipeline(config).fit(X, y, feature_names=ions)

ortho = results.orthogonal           # OrthogonalAttribution
ortho.R2, ortho.R2_channels          # exact split (T, I, F)
ortho.ions_dataframe()               # pi_T, pi_I, pi_F, E_T, E_I, E_F per ion
ortho.samples_dataframe()            # tau, iota, phi, G, yhat_T/I/F, weight
ortho.checks                         # residuals of the exact identities
results.encoder.basis_.rank_T, results.encoder.basis_.rank_I
```

`results.nsr` and `results.sample_attribution` hold the exact quantities. `save_results`
additionally writes `orthogonal_ions.csv`, `orthogonal_samples.csv` and an `"orthogonal"`
section in `summary.json`. `PipelineConfig.baseline_rank` defaults to 2, so pass
`baseline_rank=None` explicitly to select the baseline rank automatically.

### Command line

```bash
neutrohydro run data.csv --target TDS --baseline orthogonal \
    --features Na K Mg Ca Cl SO4 HCO3 NO3 F
```

`--baseline-rank` and `--indeterminacy-rank` fix the ranks. Without `--features` the CLI
uses every other numeric column as a predictor. On `data.csv` that includes EC, which
predicts TDS almost trivially.

### Direct API

```python
from neutrohydro.preprocessing import Preprocessor
from neutrohydro.model import PNPLS
from neutrohydro.orthogonal import (
    OrthogonalNDGEncoder, orthogonal_attribution, verify_jacobian_orthogonality,
    subspace_alignment, directions_to_standardized, scenario_response,
    jacobian_entanglement,
)

pre = Preprocessor(log_transform=True)
X_std, y_std = pre.fit_transform(X, y)

enc = OrthogonalNDGEncoder()              # rank_T=None, rank_I=None: automatic
trip = enc.fit_transform(X_std)           # T, I, F components + memberships + weights
model = PNPLS(n_components=3).fit(trip, y_std)   # uses trip.sample_weights as row weights
att = orthogonal_attribution(model, trip, y_std, feature_names=ions)

verify_jacobian_orthogonality(enc, X_std)          # all entries ~1e-12 or smaller
jacobian_entanglement(enc, X_std)                  # compare with a legacy NDGEncoder

D = directions_to_standardized(A.T, pre)           # A: (n_ions, n_processes) in meq/L
subspace_alignment(enc, D, names=process_names)
scenario_response(enc, model, D, s_y=pre.params_.s_y)
```

### Mineral identifiability and error bars

```python
from neutrohydro.minerals import MineralInverter, STANDARD_IONS

inv = MineralInverter(ion_order=STANDARD_IONS[:9])
rep = inv.identifiability()               # or identifiability(pi_G=results.nsr.pi_G)
rep["alias_groups"], rep["vif"], rep["orthogonality_defect"], rep["underdetermined"]

res = inv.invert(c_meq, rel_error=0.05)
res.s, res.s_std, res.support_condition, res.support_defect
```

A complete worked example on `data.csv` is `examples/jacobian_orthogonality_demo.py`.

---

## 7. Interpretation Guide

| Quantity | Meaning |
|---|---|
| $R^2_T$ | Share of the target's (weighted) variance explained through the baseline subspace |
| $R^2_F$ | Share explained through off-baseline directions: perturbations drive the target |
| $R^2_I$ | Share carried by directions whose baseline status is statistically undecidable; a large $R^2_I$ means the attribution is not settled by these data |
| $\pi_T(j)$ | Fraction of ion $j$'s predictive contribution energy that flows through the baseline subspace |
| $(\tau_i, \iota_i, \varphi_i)$ | Fractions of sample $i$'s deviation from the center lying in each subspace (input side, model-free) |
| $G_i$ | Baseline share of sample $i$'s squared channel predictions (model side) |
| Alignment enrichment | Whether a known process is preferentially aligned with T, I or F |

On `data.csv`: Na⁺ ($\pi_T = 0.99$), Ca²⁺ (0.91), F⁻ (0.93), Mg²⁺ (0.89) and Cl⁻ (0.86)
act mainly through the baseline. K⁺ ($\pi_F = 0.98$), SO₄²⁻ (0.96) and NO₃⁻ (0.95) act
mainly through perturbations. HCO₃⁻ is split, with $\pi_I = 0.47$: its role is
undecided.

---

## 8. Limitations

- **Linear subspace baseline.** The baseline is a linear subspace of standardized (or log)
  ion space. Strongly nonlinear evolution appears as I or F.
- **Widespread contamination can enter T.** A process affecting many samples is high
  variance and is selected into the baseline. Process alignment (Section 4) is the check.
- **Bounded robustness.** Weights are bounded below by $e^{-\lambda_F}$ (0.37 for
  $\lambda_F = 1$). Increase `lambda_F` for stronger down-weighting of gross outliers.
- **Exactness on training data.** The $R^2$ split and per-ion additivity are exact on the
  data the encoder and model were fitted to. On new data, Pythagoras per sample,
  completeness and the sensitivity split remain exact, but the cross terms between
  channels need not vanish.
- **Fit.** By Theorem 4 the orthogonal model fits like weighted PLS on the ions. The legacy
  model uses nonlinear features (memberships, local variance), so its $R^2$ can be slightly
  higher: 0.905 vs 0.892 on `data.csv` (3 components).
- **Local alignment under log transform.** With a log transform, alignment and scenario
  responses are first-order statements at the reference composition.
- **NNLS error bars** are linearizations on a fixed support.

---

## 9. What Is New Here

The ingredients are standard: orthogonal projectors and their Pythagorean identities,
parallel analysis, the bootstrap, PLS, and Hadamard's inequality. What this module adds
to neutrosophic chemometrics is their combination:

1. a neutrosophic triplet defined by complementary orthogonal projectors, so that
   $(\tau, \iota, \varphi)$ are derived quantities summing to 1;
2. indeterminacy defined as the band of directions whose significance the bootstrap
   cannot decide;
3. an exact partition of $R^2$ and of each ion's predictive energy into baseline,
   indeterminate and perturbation parts, with prediction invariance guaranteeing that
   the encoding adds no fit;
4. alignment of mineral and end-member stoichiometry with the statistical subspaces;
5. Jacobian-based identifiability diagnostics and error propagation for the
   stoichiometric inversion.

**Related work (quick search, November 2026).**

- *Neutrosophic water quality.* Neutrosophic sets are used for water-quality indices
  and rankings, e.g. a neutrosophic WQI for the Gomati River (Das et al., 2024) and
  neutrosophic-entropy fluoride indices for groundwater (Parkash et al., 2022). The
  search found no neutrosophic PLS or regression model applied to hydrochemistry.
- *Geogenic versus anthropogenic separation.* PCA and factor analysis are routinely used
  to label components as geogenic or anthropogenic and to screen samples for natural
  background levels (e.g. Chen et al., 2024). The search found no exact projector-based
  split of a regression's $R^2$ with prediction invariance.
- *Jacobian orthogonality.* Independent mechanism analysis asks for orthogonal *columns*
  of a mixing Jacobian (Gresele et al., 2021), and OroJaR regularises generators towards
  orthogonal Jacobian columns (Wei et al., 2021). The block condition
  $J_a J_b^\top = 0$ used here is related but different; the Hadamard defect in §5 is
  the IMA contrast applied directly to $\partial c/\partial s$.
- *Number of components.* Confidence intervals for it already exist, e.g. bootstrap
  parallel analysis. Using that interval to *define* an indeterminate subspace was not
  found.
- *Mass-balance uncertainty.* Inverse mole-balance modelling with uncertain data is
  established (Parkhurst, 1997), and so is the non-uniqueness of reaction sets
  (Gómez et al., 2008). The closed-form support Jacobian, VIF and alias groups here are
  standard linear-inverse tools applied to the NNLS mineral library.

These statements are made to the best of our knowledge; novelty should be confirmed by a
fuller literature search before it is claimed in a publication.

---

## References

- Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis?
  *Journal of the ACM*, 58(3), 11.
- de Jong, S. (1993). SIMPLS: an alternative approach to partial least squares regression.
  *Chemometrics and Intelligent Laboratory Systems*, 18(3), 251–263.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Gresele, L., von Kügelgen, J., Stimper, V., Schölkopf, B., & Besserve, M. (2021).
  Independent mechanism analysis, a new concept? *Advances in Neural Information
  Processing Systems 34 (NeurIPS 2021)*, 28233–28248.
- Hadamard's inequality: $\lvert\det A\rvert \le \prod_k \|a_k\|$ for a square matrix
  $A$ with columns $a_k$, with equality iff the columns are orthogonal (or one is zero).
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis.
  *Psychometrika*, 30(2), 179–185. https://doi.org/10.1007/BF02289447
- Chen et al. (2024). A new approach combining principal component factor analysis and
  K-means for identifying natural background levels of NO3-N in shallow groundwater of
  the Huaihe River Basin. *Science of the Total Environment*.
- Das et al. (2024). Neutrosophic approach to water quality assessment: a case study of
  Gomati River, the largest river in Tripura, India. *Neutrosophic Systems with
  Applications*.
- Gómez et al. (2008). Sensitivity and uncertainty analysis of mixing and mass balance
  calculations with standard and PCA-based geochemical codes. *Applied Geochemistry*.
- Parkash et al. (2022). Neutrosophic entropy based fluoride contamination indices for
  community health risk assessment from groundwater of Kangra County, North India.
- Parkhurst, D. L. (1997). Geochemical mole-balance modeling with uncertain data.
  *Water Resources Research*.
- Wei, Y., et al. (2021). Orthogonal Jacobian regularization for unsupervised
  disentanglement in image generation. *ICCV 2021*, 6721–6730.
  https://doi.org/10.1109/ICCV48922.2021.00665
