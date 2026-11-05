# Mathematical Critique of NeutroHydro Model

**Date**: December 31, 2025
**Updated**: November 2026 (sections 7–14)
**Module**: `neutrohydro`

This document provides a critical mathematical review of the NeutroHydro framework, identifying potential limitations and proposing rigorous solutions.

## 1. The "Weighting Paradox" in Mineral Inversion (Addressed)

### The Issue

The original `MineralInverter` minimized the weighted norm:
$$ \min_{s \ge 0} \| D \cdot (c - A s) \|_2^2 $$
where the weights $D$ were derived from the Baseline Fraction $\pi_G$. This risked downweighting anthropogenic markers (perturbations).

### Resolution

The current implementation allows for **Unweighted Inversion** or **Targeted Weighting**. By default, the model now uses a balanced approach where critical ions (like NO3) are not penalized, ensuring that pollution signals are not ignored.

## 2. Mixing vs. Mineral Dissolution

### The Issue

The NNLS solver assumes:
$$ c_{total} = \sum (\text{Mineral}_k \times \text{Mass}_k) $$
This assumes all solutes come from dissolving solid phases.

* **Reality**: Groundwater often involves **Mixing** with a pre-existing brine (e.g., Seawater).
* **Approximation**: The model approximates Seawater as a sum of `Halite` + `Sylvite` + `Gypsum` + ...
* **Critique**: This loses the **Constant Proportion** constraint of Seawater (e.g., $Cl/Br$ ratio). It allows the model to "break" Seawater into separate salts, which is physically impossible in simple mixing.

### Mathematical Solution

Add **Fluid Endmembers** to the Stoichiometric Matrix $A$:

* Define a "mineral" called `Seawater` with the exact ionic composition of standard seawater.
* $$ A_{Seawater} = [Na=468, Mg=53, Ca=10, Cl=545, SO4=28, ...] $$
* This forces the solver to use the *exact* seawater ratio, improving validity for salinization studies.

## 3. Non-Uniqueness of Ion Exchange (Addressed via Thermodynamics)

### The Issue

I introduced `Exchanger` phases (e.g., $Ca \to 2Na$) to model ion exchange.

* **Mathematical Risk**: This increases **Multicollinearity**.
  * *Scenario*: High Na, Low Ca.
  * *Explanation A*: Dissolve Halite ($Na, Cl$) + Precipitate Calcite ($-Ca, -CO3$).
  * *Explanation B*: Ion Exchange ($Ca \to 2Na$).
* **Solver Behavior**: NNLS will pick the path of least resistance (lowest residual). It cannot distinguish between these mechanisms without isotopic data.

### Resolution

**Thermodynamic Validation**: The integration of PHREEQC-based Saturation Indices (SI) acts as a physical constraint. If Calcite is undersaturated, precipitation is disallowed, forcing the model to choose Ion Exchange if that is the only viable path. This significantly reduces non-uniqueness.

## 4. Error Propagation in NDG (Addressed via Robust PCA)

### The Issue

The NDG Encoder calculates $T, I, F$ sequentially.

* $T$ = Robust PCA.
* $I$ = PCA on Residuals.
* $F$ = Distance to $T+I$.
* **Critique**: Errors in the estimation of $T$ (e.g., wrong rank) propagate to $I$ and $F$. If $T$ overfits, $I$ and $F$ vanish.

### Resolution

**Robust PCA Default**: The framework now defaults to **Robust PCA** (RPCA) for baseline estimation. RPCA is mathematically designed to separate a low-rank matrix ($T$) from sparse errors ($F$) without overfitting, making the decomposition stable even without extensive cross-validation for rank.

## 5. The "Rule-Based" Override Problem (Addressed via Context-Awareness)

### The Issue

The integration of **WHO Quality Flags** and **Gibbs Constraints** introduces a rule-based logic layer on top of the optimization layer.

* **Scenario**: The NNLS solver wants to fit `Halite` to explain Cl. The Gibbs constraint says "Rock Dominance" and bans `Halite`. The WHO flag says "Saline Intrusion" and forces `Halite` back in.
* **Critique**: This creates a **Hybrid System** where the objective function is dynamically modified by discrete logic gates. This makes the model behavior non-smooth and potentially sensitive to the specific thresholds used in the rules.

### Resolution

**Context-Aware Inversion**: The system now prioritizes **Pollution Context** (WHO Flags) over **Geological Context** (Gibbs) when they conflict. This is a deliberate design choice: if a sample is demonstrably polluted (e.g., Cl > 1000 mg/L), the "Rock Dominance" assumption of the Gibbs diagram is invalid. This hierarchy resolves the conflict deterministically.

## 6. Simpson's Ratio (Revelle Coefficient) Discretization

### The Issue

The model uses discrete bins for Simpson's Ratio (e.g., "Moderately contaminated" vs "Injuriously contaminated").

* **Critique**: Discretization throws away information. A sample at an SR of 2.7 is labeled "Moderately", while 2.9 is "Injuriously", despite being chemically nearly identical.

### Mathematical Solution

**Continuous Scoring**: Use the raw ratio values (Standard and Freshening) for any downstream statistical analysis (like correlation or clustering), and reserve the discrete classes only for the final human-readable report.

## 7. PLS1 Inner Iteration Computed PCA, Not PLS (Fixed)

### Issue

The NIPALS loop refined each weight with $w \leftarrow X_h^\top X_h w / \|\cdot\|$. That is the
power method on $X_h^\top X_h$: it converges to the first principal component whatever the
response is. The model was principal component regression. `docs/model.md` §13 proved this
convergence but presented it as a property of PLS.

### Evidence

On a test problem whose response lies along a low-variance direction, the first weight had
$|\cos| = 0.131$ with the PLS direction $X^\top y$ and $|\cos| = 1.0$ with PC1. The
1-component $R^2$ was $0.0007$.

### Resolution

For PLS1 the weight has the closed form $w_h = X_h^\top y_h / \|X_h^\top y_h\|$, with no inner
loop. The 1-component $R^2$ on the same problem is $0.68$. Predictions match
`sklearn.cross_decomposition.PLSRegression` (`tests/test_model.py::TestPLS1Correctness`).
`docs/model.md` §4 and §13 are corrected. All earlier PNPLS results, and the VIP, NSR and
$\pi_G$ values derived from them, should be recomputed.

## 8. No Centering: Hidden Intercept (Fixed)

### Issue

The augmented matrix was not centered and the model had no intercept. Channels with non-zero
means (the constant median baseline, memberships in $[0, 1]$) acted as an intercept, so part
of "VIP_T" measured the intercept rather than a baseline effect.

### Evidence

With the median baseline the $T$ columns are constant. Their covariance with an uncentered
target is $\operatorname{median}_j \cdot n \bar y \neq 0$, which gives them weight in the
first PLS component.

### Resolution

PNPLS mean-centers predictors and target (weighted means when sample weights are used) and
predicts $\hat y = (x^{(\text{aug})} - \bar x^{(\text{aug})})^\top \beta + \bar y$
(`center=True`). Per-ion contributions are computed on the centered design, so
$\sum_j c_{ij} = \hat y_i - \bar y$ exactly.

## 9. Batch-Dependent `transform` (Fixed)

### Issue

`NDGEncoder.transform` re-estimated the baseline (medians, SVD, RPCA) on whatever data it was
given. A new sample's encoding, and hence its prediction, depended on the other samples in the
batch, and a uniformly shifted batch looked unperturbed.

### Evidence

Fitting on standardized data and transforming the first 10 samples shifted by $+5$ gave a mean
residual of $0.047$, where $\approx 5$ was expected.

### Resolution

`fit` stores the baseline operator (medians, group medians, the low-rank basis, or the RPCA
basis and threshold) and `transform` applies it. New samples are robustly projected onto the
fitted RPCA subspace. The same test now gives a mean residual of $4.951$, and encodings no
longer depend on the batch (`tests/test_encoder.py::TestOutOfSampleTransform`).

## 10. Row-Order-Dependent Indeterminacy (Fixed)

### Issue

The default $I$ channel was a rolling-window variance over **row order**. For unordered
groundwater samples, row order is the order of lines in the input file.

### Evidence

Permuting the rows changed $I$ (after undoing the permutation) by up to $0.798$, on a $[0, 1]$
scale.

### Resolution

The default is now the local variance among the `n_neighbors` nearest **training** samples in
standardized ion space (`indeterminacy="knn"`). It is order-invariant (change $0$) and applies
unchanged to new samples. The rolling window is kept as `indeterminacy="rolling"` for genuinely
ordered data (time series, transects).

## 11. Robust PCA Stopped Before Convergence (Fixed)

### Issue

The RPCA baseline was capped at 100 iterations without a convergence check, so the returned
$L$ was generally not a fixed point of the decomposition.

### Evidence

Convergence to the stated tolerance takes 114 iterations on `data.csv` and 270 on a synthetic
table. After the cap, $L$ was still changing.

### Resolution

The cap is 1000 iterations, convergence is recorded in the baseline metadata, and a warning is
issued if it is not reached. With a converged basis, the out-of-sample robust projection
reproduces the training $L$ to about $10^{-4}$.

## 12. Median Baseline Is Constant; Falsity Discards Sign (Documented)

### Issue

1. The median baseline is the same vector for every sample. The $T$ channel carries no
   sample-to-sample information and its Jacobian is zero, so it cannot explain variation in
   the target.
2. $F = g(\lvert x - T\rvert / \sigma)$ uses the absolute residual, so enrichment and
   depletion of an ion give the same falsity.

### Evidence

In the synthetic probe of this review the median baseline gave $\mathrm{std}(T) = 0$ in every
column and $\pi_G = 0$ for every ion. The correlation between $F$ and the signed residual was
between $-0.2$ and $0.1$ for the ions checked.

### Resolution

The median-baseline limitation is stated in the `NDGEncoder` docstring. For baseline
attribution, use a low-rank baseline or the orthogonal encoding (section 13). The orthogonal
encoding carries **signed** components, so it also resolves the sign loss.

## 13. Channel Entanglement: the T/I/F Split Is Not Identifiable (Addressed)

### Issue

Attribution between channels is unique only if the channels respond to different input
directions: Jacobian orthogonality, $J_a J_b^\top = 0$. In the legacy encoder, $I$ and $F$ are
functions of the same $x$ as $T$, and the elementwise precision weights multiply $T$ by a
function of $F$. The NVIP identity $VIP_{agg}^2 = VIP_T^2 + VIP_I^2 + VIP_F^2$ holds by
construction and does not establish separability.

### Evidence

The Jacobian entanglement index of the legacy model's channels on `data.csv` is T–I $0.513$,
T–F $0.550$, I–F $0.652$, on a scale where 0 means disentangled and 1 fully entangled.

### Resolution

`baseline_type="orthogonal"` encodes samples by their components in three complementary
orthogonal subspaces of a falsity-reweighted covariance, with ranks chosen by parallel analysis
and a bootstrap. Entanglement is exactly $0$, and the following hold: Pythagoras per sample,
uncorrelated channels, prediction invariance, an exact split
$R^2 = R^2_T + R^2_I + R^2_F$ ($0.892 = 0.753 + 0.056 + 0.084$ on `data.csv`), and exact
per-ion shares. See [Jacobian Orthogonality](jacobian_orthogonality.md) and
`tests/test_orthogonal.py`.

## 14. Mineral Aliasing and Missing Uncertainty (Addressed)

### Issue

The NNLS inversion reported single mineral amounts with no uncertainty and no indication that
some minerals cannot be told apart by the measured ions.

### Evidence

The forward Jacobian of $c = A s$ is $A$. On the 9 major ions the standard library has 19
minerals and rank 8. Calcite ≡ Anorthite (no silica measured), Gypsum ≡ Anhydrite and
Mirabilite ≡ Thenardite have parallel columns, and Dolomite = Calcite + Magnesite. NNLS picks
an arbitrary member of each alias group.

### Resolution

- `stoichiometric_identifiability` / `MineralInverter.identifiability` report pairwise
  cosines, VIFs, alias groups, the Hadamard orthogonality defect and an underdetermined flag.
- `invert(..., rel_error=...)` propagates analytical error through the support Jacobian
  $(D A_S)^+ D$ to give `s_std`, `support_condition` and `support_defect`. Example: sample
  MAM7 at 5 % error gives Halite $95.2 \pm 3.5$ meq/L and Sylvite $2.1 \pm 2.3$ meq/L.
- The linearization holds while the NNLS support is stable.
