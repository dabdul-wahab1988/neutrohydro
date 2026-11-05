"""
Jacobian-orthogonal neutrosophic encoding (JO-NDG) for NeutroHydro.

Idea
----
The NDG encoder maps an ion vector x to three channels (T, I, F) and the
attribution layer asks "how much of the prediction comes from the baseline
channel and how much from the perturbation channels?". That question has a
unique answer only if the channels respond to *different* directions of
ion space. Writing J_c(x) = dc/dx for the Jacobian of channel c, the
condition is

    J_a(x) J_b(x)^T = 0   for a != b            (Jacobian orthogonality)

i.e. the input directions seen by one channel are invisible to the others.
The legacy encoder violates this (F and I are functions of the same
residuals as T, and the elementwise precision weights multiply T by a
function of F), so the T/I/F split of importance is not identifiable.

This module builds the channels from three complementary orthogonal
projectors of the (robust, weighted) ion covariance:

    x - mu = P_T (x - mu) + P_I (x - mu) + P_F (x - mu),
    P_T + P_I + P_F = Id,   P_a P_b = 0 (a != b).

* T = baseline subspace (dominant, statistically significant covariation),
* I = indeterminate subspace (directions whose baseline/perturbation status
  is statistically undecidable: the bootstrap cannot tell whether they
  belong to the significant structure),
* F = perturbation subspace (the remaining, off-baseline directions).

Consequences (proved in docs/jacobian_orthogonality.md, checked in tests):

1. Pythagoras per sample: ||x - mu||^2 = ||T||^2 + ||I||^2 + ||F||^2, so
   (tau, iota, phi) = energy fractions are a neutrosophic triplet with
   tau + iota + phi = 1 *derived* from the data rather than posited.
2. The channels are uncorrelated across samples (spectral projectors), so
   PLS cannot trade weight between collinear channels.
3. Prediction invariance: with rho_I = rho_F = 1, PNPLS on [T, I, F] gives
   exactly the predictions of PLS on x. The neutrosophic encoding is a pure
   attribution device and cannot manufacture fit.
4. Exact attribution: the fitted values split as y_hat - y_bar = y_T + y_I +
   y_F with mutually orthogonal parts, so R^2 = R^2_T + R^2_I + R^2_F, and
   each ion's predictive energy splits exactly into the three channels.
5. Any membership computed from a channel (e.g. falsity 1 - exp(-|F|/s))
   inherits that channel's Jacobian row space, so it stays orthogonal to T.
6. Known process vectors (mineral stoichiometry, seawater, fertiliser) can
   be projected onto the subspaces: an exact T/I/F split of each process,
   which labels the statistical subspaces with geochemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union
import warnings

import numpy as np
from numpy.typing import NDArray

from neutrohydro.encoder import TripletData, FalsityMap

CHANNELS = ("T", "I", "F")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _falsity(u: NDArray[np.floating], fmap: FalsityMap, params: dict) -> NDArray[np.floating]:
    """Monotone map from a normalised distance u >= 0 to [0, 1]."""
    if fmap == FalsityMap.EXPONENTIAL:
        return 1.0 - np.exp(-u)
    a = params.get("a", 2.0)
    b = params.get("b", 1.0)
    return 1.0 / (1.0 + np.exp(-a * (u - b)))


def _weighted_spectrum(
    X: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Weighted mean, eigenvalues (descending) and eigenvectors of X."""
    wsum = weights.sum()
    mu = (weights[:, None] * X).sum(axis=0) / wsum
    Xc = X - mu
    C = (Xc * weights[:, None]).T @ Xc / wsum
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    return mu, np.maximum(evals[order], 0.0), evecs[:, order]


def _spectrum(X: NDArray[np.floating], scale: str) -> NDArray[np.floating]:
    """Descending eigenvalues of the covariance or correlation of X."""
    if scale == "covariance":
        return _weighted_spectrum(X, np.ones(X.shape[0]))[1]
    sd = X.std(axis=0)
    Xs = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    C = Xs.T @ Xs / X.shape[0]
    return np.sort(np.maximum(np.linalg.eigvalsh(C), 0.0))[::-1]


def _count_significant(evals: NDArray[np.floating], thresholds: NDArray[np.floating]) -> int:
    """Number of leading eigenvalues above their null thresholds (sequential)."""
    above = evals > thresholds
    if np.all(above):
        return len(evals)
    return int(np.argmin(above))


def _fill_nan(X: NDArray[np.floating], fill: NDArray[np.floating]) -> NDArray[np.floating]:
    if not np.any(np.isnan(X)):
        return X
    X = X.copy()
    r, c = np.where(np.isnan(X))
    X[r, c] = fill[c]
    return X


# ----------------------------------------------------------------------
# Rank selection: indeterminacy as statistical undecidability
# ----------------------------------------------------------------------

def select_ranks(
    X: NDArray[np.floating],
    n_permutations: int = 200,
    n_bootstrap: int = 200,
    alpha: float = 0.1,
    null_quantile: float = 0.95,
    min_rank_T: int = 1,
    random_state: Optional[int] = 0,
    scale: str = "correlation",
) -> dict:
    """
    Choose the baseline rank and the indeterminate band.

    Horn's parallel analysis gives, for each ordered eigenvalue of the ion
    covariance, a null threshold (``null_quantile`` of the eigenvalues of
    column-permuted data, which destroys covariation but keeps marginals).
    The number of leading eigenvalues above their thresholds is the
    significant rank. Bootstrapping the samples gives a distribution of that
    rank; directions that are significant in every plausible resample form
    the baseline, directions significant in some resamples but not others
    are *indeterminate* (their status cannot be decided from the data), and
    the rest are perturbation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Standardized data.
    n_permutations : int, default=200
        Permutations for the parallel-analysis null.
    n_bootstrap : int, default=200
        Bootstrap resamples of the samples.
    alpha : float, default=0.1
        Two-sided level for the rank interval: rank_T is the alpha-quantile
        of the bootstrap rank, rank_T + rank_I the (1 - alpha)-quantile.
    null_quantile : float, default=0.95
        Quantile of the permutation null used as threshold.
    min_rank_T : int, default=1
        Lower bound for the baseline rank.
    random_state : int, optional
        Seed.
    scale : {"correlation", "covariance"}, default="correlation"
        Matrix whose eigenvalues are tested. Horn's test is defined on the
        correlation matrix; on a covariance matrix with unequal column
        variances the null eigenvalues mirror the largest variances and the
        rank is underestimated.

    Returns
    -------
    selection : dict
        ``rank_T``, ``rank_I``, ``rank_observed``, ``bootstrap_ranks``,
        ``eigenvalues``, ``null_thresholds`` and the settings used.
    """
    X = np.asarray(X, dtype=np.float64)
    X = _fill_nan(X, np.nan_to_num(np.nanmedian(X, axis=0)))
    n, p = X.shape
    if scale not in ("correlation", "covariance"):
        raise ValueError("scale must be 'correlation' or 'covariance'")
    rng = np.random.default_rng(random_state)

    evals = _spectrum(X, scale)

    null = np.empty((n_permutations, p))
    for b in range(n_permutations):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(p)])
        null[b] = _spectrum(Xp, scale)
    thresholds = np.quantile(null, null_quantile, axis=0)

    r_obs = _count_significant(evals, thresholds)

    r_boot = np.empty(n_bootstrap, dtype=int)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r_boot[b] = _count_significant(_spectrum(X[idx], scale), thresholds)

    lo = int(np.quantile(r_boot, alpha, method="lower"))
    hi = int(np.quantile(r_boot, 1.0 - alpha, method="higher"))
    max_rank = max(p - 1, 1)  # keep at least one perturbation direction
    rank_T = int(np.clip(min(lo, r_obs), min_rank_T, max_rank))
    rank_TI = int(np.clip(max(hi, r_obs, rank_T), rank_T, max_rank))

    return {
        "method": "parallel_analysis_bootstrap",
        "scale": scale,
        "rank_T": rank_T,
        "rank_I": rank_TI - rank_T,
        "rank_observed": r_obs,
        "bootstrap_ranks": np.bincount(r_boot, minlength=p + 1),
        "eigenvalues": evals,
        "null_thresholds": thresholds,
        "alpha": alpha,
        "null_quantile": null_quantile,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
    }


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------

@dataclass
class OrthogonalBasis:
    """Fitted state of the Jacobian-orthogonal encoder."""

    center: NDArray[np.floating]        # (p,) weighted mean
    eigenvectors: NDArray[np.floating]  # (p, p), columns, descending eigenvalues
    eigenvalues: NDArray[np.floating]   # (p,)
    rank_T: int
    rank_I: int
    sample_weights: NDArray[np.floating]  # (n,) training row weights
    sigma_I: NDArray[np.floating]       # (p,) robust scales of the I component
    sigma_F: NDArray[np.floating]       # (p,) robust scales of the F component
    distance_scale: float               # robust scale of ||F_i||
    n_iter: int
    converged: bool
    rank_selection: Optional[dict] = None

    def basis(self, channel: str) -> NDArray[np.floating]:
        """Orthonormal basis (p, r_c) of a channel subspace."""
        rT, rI = self.rank_T, self.rank_I
        V = self.eigenvectors
        return {"T": V[:, :rT], "I": V[:, rT:rT + rI], "F": V[:, rT + rI:]}[channel]

    def projector(self, channel: str) -> NDArray[np.floating]:
        """Orthogonal projector (p, p) onto a channel subspace."""
        B = self.basis(channel)
        return B @ B.T

    @property
    def explained_variance_ratio(self) -> dict:
        tot = self.eigenvalues.sum() + 1e-300
        rT, rI = self.rank_T, self.rank_I
        ev = self.eigenvalues
        return {
            "T": float(ev[:rT].sum() / tot),
            "I": float(ev[rT:rT + rI].sum() / tot),
            "F": float(ev[rT + rI:].sum() / tot),
        }


class OrthogonalNDGEncoder:
    """
    Jacobian-orthogonal NDG encoder.

    Splits standardized ion space into three complementary orthogonal
    subspaces (baseline T, indeterminate I, perturbation F) of a robust,
    falsity-reweighted covariance and encodes each sample by its signed
    components in those subspaces. See the module docstring for the
    guarantees this buys.

    Parameters
    ----------
    rank_T : int, optional
        Baseline rank. ``None`` selects it by bootstrap parallel analysis.
    rank_I : int, optional
        Dimension of the indeterminate band. ``None`` selects it from the
        bootstrap rank interval (0 when the rank is stable).
    robust : bool, default=True
        Iteratively down-weight samples with large perturbation energy when
        estimating the covariance (falsity-reweighted spectral baseline).
    lambda_F : float, default=1.0
        Strength of the down-weighting: w_i = exp(-lambda_F * f_i), with
        f_i in [0, 1) the falsity membership of sample i.
    falsity_map : str or FalsityMap, default="exponential"
        Map from normalised perturbation distance to falsity membership.
    falsity_params : dict, optional
        Parameters for the logistic map ({"a": 2.0, "b": 1.0}).
    max_iter : int, default=100
        Maximum reweighting iterations.
    tol : float, default=1e-8
        Convergence tolerance on the weights.
    n_permutations, n_bootstrap, alpha, null_quantile, min_rank_T :
        Passed to :func:`select_ranks` when a rank is selected automatically.
    random_state : int, optional
        Seed for rank selection.
    delta : float, default=1e-10
        Numerical floor for scales.

    Attributes
    ----------
    basis_ : OrthogonalBasis
        Fitted subspaces, weights and scales.
    is_fitted_ : bool
    """

    def __init__(
        self,
        rank_T: Optional[int] = None,
        rank_I: Optional[int] = None,
        robust: bool = True,
        lambda_F: float = 1.0,
        falsity_map: Union[str, FalsityMap] = FalsityMap.EXPONENTIAL,
        falsity_params: Optional[dict] = None,
        max_iter: int = 100,
        tol: float = 1e-8,
        n_permutations: int = 200,
        n_bootstrap: int = 200,
        alpha: float = 0.1,
        null_quantile: float = 0.95,
        min_rank_T: int = 1,
        random_state: Optional[int] = 0,
        delta: float = 1e-10,
    ):
        if isinstance(falsity_map, str):
            falsity_map = FalsityMap(falsity_map)
        self.rank_T = rank_T
        self.rank_I = rank_I
        self.robust = robust
        self.lambda_F = lambda_F
        self.falsity_map = falsity_map
        self.falsity_params = falsity_params or (
            {"a": 2.0, "b": 1.0} if falsity_map == FalsityMap.LOGISTIC else {}
        )
        self.max_iter = max_iter
        self.tol = tol
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.null_quantile = null_quantile
        self.min_rank_T = min_rank_T
        self.random_state = random_state
        self.delta = delta

        self.basis_: Optional[OrthogonalBasis] = None
        self.is_fitted_: bool = False
        self._fill: Optional[NDArray[np.floating]] = None

    # -- fitting ---------------------------------------------------------

    def _resolve_ranks(self, X: NDArray[np.floating]) -> tuple[int, int, Optional[dict]]:
        p = X.shape[1]
        selection = None
        rank_T, rank_I = self.rank_T, self.rank_I
        if rank_T is None or rank_I is None:
            selection = select_ranks(
                X,
                n_permutations=self.n_permutations,
                n_bootstrap=self.n_bootstrap,
                alpha=self.alpha,
                null_quantile=self.null_quantile,
                min_rank_T=self.min_rank_T,
                random_state=self.random_state,
            )
            upper = selection["rank_T"] + selection["rank_I"]
            if rank_T is None:
                rank_T = selection["rank_T"]
            if rank_I is None:
                rank_I = max(0, upper - rank_T)
        rank_T, rank_I = int(rank_T), int(rank_I)
        if rank_T < 0 or rank_I < 0 or rank_T + rank_I > p:
            raise ValueError(
                f"Need 0 <= rank_T, rank_I and rank_T + rank_I <= {p}; "
                f"got rank_T={rank_T}, rank_I={rank_I}"
            )
        return rank_T, rank_I, selection

    def _sample_falsity(
        self,
        F: NDArray[np.floating],
        scale: float,
    ) -> NDArray[np.floating]:
        d = np.linalg.norm(F, axis=1)
        return _falsity(d / (scale + self.delta), self.falsity_map, self.falsity_params)

    def fit(self, X_std: NDArray[np.floating], groups=None) -> "OrthogonalNDGEncoder":
        """
        Fit the orthogonal subspaces.

        Parameters
        ----------
        X_std : ndarray of shape (n_samples, n_features)
            Standardized predictor matrix.
        groups : ignored
            Accepted for API compatibility with :class:`NDGEncoder`.
        """
        if groups is not None:
            warnings.warn(
                "groups are ignored by the orthogonal encoder.", stacklevel=2
            )
        X = np.asarray(X_std, dtype=np.float64)
        self._fill = np.nan_to_num(np.nanmedian(X, axis=0))
        X = _fill_nan(X, self._fill)
        n, p = X.shape

        rank_T, rank_I, selection = self._resolve_ranks(X)
        r_TI = rank_T + rank_I

        weights = np.ones(n)
        converged = not self.robust
        n_iter = 0
        if self.robust:
            for n_iter in range(1, self.max_iter + 1):
                mu, _, V = _weighted_spectrum(X, weights)
                V_F = V[:, r_TI:]
                F = (X - mu) @ V_F @ V_F.T
                d = np.linalg.norm(F, axis=1)
                scale = float(np.median(d)) if np.any(d > 0) else 1.0
                f = _falsity(d / (scale + self.delta), self.falsity_map, self.falsity_params)
                new_weights = np.exp(-self.lambda_F * f)
                change = np.max(np.abs(new_weights - weights))
                weights = new_weights
                if change < self.tol:
                    converged = True
                    break
            if not converged:
                warnings.warn(
                    f"Orthogonal encoder reweighting did not converge in "
                    f"{self.max_iter} iterations.",
                    stacklevel=2,
                )

        # Final spectrum with exactly the stored weights: this is what makes
        # the sample-space orthogonality of the channels exact.
        mu, evals, V = _weighted_spectrum(X, weights)
        Xc = X - mu
        comps = {
            "I": Xc @ V[:, rank_T:r_TI] @ V[:, rank_T:r_TI].T,
            "F": Xc @ V[:, r_TI:] @ V[:, r_TI:].T,
        }
        sigma_I = 1.4826 * np.median(np.abs(comps["I"]), axis=0)
        sigma_F = 1.4826 * np.median(np.abs(comps["F"]), axis=0)
        d = np.linalg.norm(comps["F"], axis=1)
        scale = float(np.median(d)) if np.any(d > 0) else 1.0

        self.basis_ = OrthogonalBasis(
            center=mu,
            eigenvectors=V,
            eigenvalues=evals,
            rank_T=rank_T,
            rank_I=rank_I,
            sample_weights=weights,
            sigma_I=sigma_I,
            sigma_F=sigma_F,
            distance_scale=scale,
            n_iter=n_iter,
            converged=converged,
            rank_selection=selection,
        )
        self.is_fitted_ = True
        return self

    # -- transforming ----------------------------------------------------

    def _check(self):
        if not self.is_fitted_:
            raise RuntimeError("Encoder must be fitted before transform")

    @property
    def projectors(self) -> dict:
        """Orthogonal projectors {"T": P_T, "I": P_I, "F": P_F}."""
        self._check()
        return {c: self.basis_.projector(c) for c in CHANNELS}

    def jacobians(self) -> dict:
        """
        Channel Jacobians dc/dx. They are constant (the projectors).

        Jacobian orthogonality reads ``J["T"] @ J["F"].T == 0`` etc.
        """
        return self.projectors

    def components(self, X_std: NDArray[np.floating]) -> dict:
        """Signed channel components {"T", "I", "F"} of X_std (n, p) each."""
        self._check()
        X = _fill_nan(np.asarray(X_std, dtype=np.float64), self._fill)
        Xc = X - self.basis_.center
        return {c: Xc @ self.basis_.projector(c) for c in CHANNELS}

    def memberships(self, X_std: NDArray[np.floating]) -> dict:
        """
        Neutrosophic memberships derived from the orthogonal components.

        Returns
        -------
        dict with
            ``tau``, ``iota``, ``phi`` : (n,) energy fractions of each sample
                in the T, I and F subspaces; they sum to 1 exactly.
            ``mu_I``, ``mu_F`` : (n, p) elementwise indeterminacy / falsity
                memberships in [0, 1] (functions of the I / F components only).
            ``f`` : (n,) sample falsity membership, the basis of the robust
                weights ``w = exp(-lambda_F * f)``.
        """
        comps = self.components(X_std)
        e = {c: np.sum(comps[c] ** 2, axis=1) for c in CHANNELS}
        tot = e["T"] + e["I"] + e["F"]
        safe = np.where(tot > 0, tot, 1.0)
        at_center = tot <= 0
        tau = np.where(at_center, 1.0, e["T"] / safe)
        iota = np.where(at_center, 0.0, e["I"] / safe)
        phi = np.where(at_center, 0.0, e["F"] / safe)

        mu_I = 1.0 - np.exp(-np.abs(comps["I"]) / (self.basis_.sigma_I + self.delta))
        mu_F = _falsity(
            np.abs(comps["F"]) / (self.basis_.sigma_F + self.delta),
            self.falsity_map,
            self.falsity_params,
        )
        f = self._sample_falsity(comps["F"], self.basis_.distance_scale)
        return {"tau": tau, "iota": iota, "phi": phi, "mu_I": mu_I, "mu_F": mu_F, "f": f}

    def transform(
        self,
        X_std: NDArray[np.floating],
        groups=None,
        _sample_weights: Optional[NDArray[np.floating]] = None,
    ) -> TripletData:
        """
        Encode X_std as orthogonal triplets.

        ``T + I + F = X_std - center`` exactly; ``R = I + F`` is the
        off-baseline residual. Sample weights for new data follow the fitted
        rule ``exp(-lambda_F * f_i)``.
        """
        comps = self.components(X_std)
        mem = self.memberships(X_std)
        if _sample_weights is None:
            weights = np.exp(-self.lambda_F * mem["f"]) if self.robust else np.ones(len(mem["f"]))
        else:
            weights = _sample_weights
        return TripletData(
            T=comps["T"],
            I=comps["I"],
            F=comps["F"],
            R=comps["I"] + comps["F"],
            encoding="orthogonal",
            sample_weights=weights,
            memberships=mem,
        )

    def fit_transform(self, X_std: NDArray[np.floating], groups=None) -> TripletData:
        """Fit and encode the training data (with the fitted training weights)."""
        self.fit(X_std, groups)
        return self.transform(X_std, _sample_weights=self.basis_.sample_weights)


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------

def verify_jacobian_orthogonality(
    encoder: OrthogonalNDGEncoder,
    X_std: Optional[NDArray[np.floating]] = None,
    sample_weights: Optional[NDArray[np.floating]] = None,
) -> dict:
    """
    Numerical check of the orthogonality guarantees.

    Returns max-abs errors for: projector cross products (Jacobian
    orthogonality), completeness (sum of projectors = identity),
    idempotence, and, if data are given, the Pythagorean identity per
    sample and the relative weighted cross-covariance between channels.
    """
    P = encoder.projectors
    p = P["T"].shape[0]
    out = {
        "jacobian_cross": max(
            np.abs(P[a] @ P[b].T).max() for a, b in [("T", "I"), ("T", "F"), ("I", "F")]
        ),
        "completeness": float(np.abs(P["T"] + P["I"] + P["F"] - np.eye(p)).max()),
        "idempotence": max(float(np.abs(P[c] @ P[c] - P[c]).max()) for c in CHANNELS),
    }
    if X_std is not None:
        comps = encoder.components(X_std)
        Xc = _fill_nan(np.asarray(X_std, float), encoder._fill) - encoder.basis_.center
        total = np.sum(Xc ** 2, axis=1)
        parts = sum(np.sum(comps[c] ** 2, axis=1) for c in CHANNELS)
        out["pythagoras"] = float(np.abs(total - parts).max())
        w = encoder.basis_.sample_weights if sample_weights is None else sample_weights
        if len(w) == Xc.shape[0]:
            scale = np.sum(w[:, None] * Xc ** 2) + 1e-300
            out["sample_cross_covariance"] = max(
                float(np.abs(comps[a].T @ (w[:, None] * comps[b])).max() / scale)
                for a, b in [("T", "I"), ("T", "F"), ("I", "F")]
            )
    return out


# ----------------------------------------------------------------------
# Exact attribution
# ----------------------------------------------------------------------

@dataclass
class OrthogonalAttribution:
    """
    Exact channel attribution of a PNPLS model fitted on orthogonal triplets.

    Attributes
    ----------
    contributions : (n, p, 3)
        c[i, j, c]: contribution of ion j through channel c to sample i's
        prediction; ``contributions.sum((1, 2)) == y_hat - intercept``.
    channel_predictions : (n, 3)
        y_hat split by channel (T, I, F).
    intercept : float
    R2 : float or None
        Weighted R^2 (the model's own weights) on the given data.
    R2_channels : (3,) or None
        Exact split R2 = R2_T + R2_I + R2_F (exact on the training data).
    ion_energy : (p, 3)
        Weighted predictive energy of each ion in each channel; the three
        columns add up exactly to the ion's total contribution energy.
    pi_T, pi_I, pi_F : (p,)
        Channel shares of each ion's predictive energy (sum to 1).
    sample_G : (n,)
        Baseline share of each sample's squared channel predictions; its
        weighted average over samples equals R2_T / R2 on the training data.
    tau, iota, phi : (n,)
        Input-side memberships (energy fractions of the sample's deviation
        from the center lying in each subspace).
    sample_weights : (n,)
    checks : dict
        Numerical residuals of the exact identities.
    feature_names : list of str or None
    """

    contributions: NDArray[np.floating]
    channel_predictions: NDArray[np.floating]
    intercept: float
    R2: Optional[float]
    R2_channels: Optional[NDArray[np.floating]]
    ion_energy: NDArray[np.floating]
    pi_T: NDArray[np.floating]
    pi_I: NDArray[np.floating]
    pi_F: NDArray[np.floating]
    sample_G: NDArray[np.floating]
    tau: NDArray[np.floating]
    iota: NDArray[np.floating]
    phi: NDArray[np.floating]
    sample_weights: NDArray[np.floating]
    checks: dict = field(default_factory=dict)
    feature_names: Optional[list] = None

    def ions_dataframe(self):
        """Per-ion exact attribution as a DataFrame."""
        import pandas as pd

        p = self.ion_energy.shape[0]
        names = self.feature_names or [f"Ion_{j}" for j in range(p)]
        return pd.DataFrame({
            "ion": names,
            "pi_T": self.pi_T,
            "pi_I": self.pi_I,
            "pi_F": self.pi_F,
            "E_T": self.ion_energy[:, 0],
            "E_I": self.ion_energy[:, 1],
            "E_F": self.ion_energy[:, 2],
        })

    def samples_dataframe(self, sample_ids: Optional[Sequence] = None):
        """Per-sample memberships and exact prediction split as a DataFrame."""
        import pandas as pd

        n = len(self.tau)
        return pd.DataFrame({
            "sample_id": list(sample_ids) if sample_ids is not None else list(range(n)),
            "tau": self.tau,
            "iota": self.iota,
            "phi": self.phi,
            "G": self.sample_G,
            "yhat_T": self.channel_predictions[:, 0],
            "yhat_I": self.channel_predictions[:, 1],
            "yhat_F": self.channel_predictions[:, 2],
            "weight": self.sample_weights,
        })


def orthogonal_attribution(
    model,
    triplets: TripletData,
    y: Optional[NDArray[np.floating]] = None,
    feature_names: Optional[list] = None,
) -> OrthogonalAttribution:
    """
    Exact attribution for a PNPLS model fitted on orthogonal triplets.

    Parameters
    ----------
    model : PNPLS
        Model fitted on ``triplets`` (or on the training triplets of the
        same encoder).
    triplets : TripletData
        Orthogonal triplets (``encoding == "orthogonal"``).
    y : ndarray of shape (n_samples,), optional
        Standardized target; enables R^2 and its exact channel split.
    feature_names : list of str, optional

    Returns
    -------
    OrthogonalAttribution
    """
    if getattr(triplets, "encoding", "ndg") != "orthogonal":
        raise ValueError("orthogonal_attribution requires orthogonal triplets")
    if not model.is_fitted_:
        raise RuntimeError("Model must be fitted")

    n, p = triplets.T.shape
    X_design = model.centered_design(triplets)
    beta = model.components_.beta

    contributions = np.stack(
        [X_design[:, k * p:(k + 1) * p] * beta[k * p:(k + 1) * p] for k in range(3)],
        axis=2,
    )  # (n, p, 3)
    channel_pred = contributions.sum(axis=1)  # (n, 3)
    y_hat = channel_pred.sum(axis=1) + model.y_mean_

    w = model.sample_weights_
    if w is None or len(w) != n:
        w = triplets.sample_weights if triplets.sample_weights is not None else np.ones(n)
    w = np.asarray(w, dtype=np.float64)

    ion_energy = np.einsum("i,ijc->jc", w, contributions ** 2)
    total = ion_energy.sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    pi = np.where(total[:, None] > 0, ion_energy / safe[:, None], 1.0 / 3.0)

    sq = channel_pred ** 2
    sq_tot = sq.sum(axis=1)
    sample_G = np.where(sq_tot > 0, sq[:, 0] / np.where(sq_tot > 0, sq_tot, 1.0), 0.5)

    # Exactness residuals
    net = contributions.sum(axis=2)  # (n, p)
    ion_total_energy = np.einsum("i,ij->j", w, net ** 2)
    scale_e = ion_total_energy.max() + 1e-300
    pred_energy = np.sum(w * channel_pred.sum(axis=1) ** 2) + 1e-300
    cross_pred = max(
        abs(np.sum(w * channel_pred[:, a] * channel_pred[:, b]))
        for a, b in [(0, 1), (0, 2), (1, 2)]
    ) / pred_energy
    checks = {
        "completeness": float(np.abs(net.sum(axis=1) - (y_hat - model.y_mean_)).max()),
        "ion_additivity": float(np.abs(ion_total_energy - total).max() / scale_e),
        "prediction_cross_energy": float(cross_pred),
    }

    R2 = None
    R2_channels = None
    if y is not None:
        y = np.asarray(y, dtype=np.float64).ravel()
        ss_tot = float(np.sum(w * (y - model.y_mean_) ** 2))
        if ss_tot > 0:
            R2 = 1.0 - float(np.sum(w * (y - y_hat) ** 2)) / ss_tot
            R2_channels = np.array(
                [np.sum(w * channel_pred[:, k] ** 2) for k in range(3)]
            ) / ss_tot
            checks["r2_split"] = float(abs(R2 - R2_channels.sum()))

    mem = triplets.memberships or {}
    nan = np.full(n, np.nan)
    return OrthogonalAttribution(
        contributions=contributions,
        channel_predictions=channel_pred,
        intercept=float(model.y_mean_),
        R2=R2,
        R2_channels=R2_channels,
        ion_energy=ion_energy,
        pi_T=pi[:, 0],
        pi_I=pi[:, 1],
        pi_F=pi[:, 2],
        sample_G=sample_G,
        tau=mem.get("tau", nan),
        iota=mem.get("iota", nan),
        phi=mem.get("phi", nan),
        sample_weights=w,
        checks=checks,
        feature_names=feature_names,
    )


def orthogonal_nsr(attribution: OrthogonalAttribution, gamma: float = 0.7, epsilon: float = 1e-10):
    """
    NSR / pi_G from the exact attribution (drop-in for ``compute_nsr``).

    pi_G(j) is the exact share of ion j's predictive energy carried by the
    baseline subspace; NSR = E_T / (E_I + E_F).
    """
    from neutrohydro.attribution import NSRResult

    E_T = attribution.ion_energy[:, 0]
    E_P = attribution.ion_energy[:, 1] + attribution.ion_energy[:, 2]
    pi_G = np.clip(attribution.pi_T, 0.0, 1.0)
    classification = np.full(len(pi_G), "mixed", dtype="U12")
    classification[pi_G >= gamma] = "baseline"
    classification[pi_G <= 1 - gamma] = "perturbation"
    return NSRResult(
        NSR=(E_T + epsilon) / (E_P + epsilon),
        pi_G=pi_G,
        pi_A=1.0 - pi_G,
        E_T=E_T,
        E_P=E_P,
        classification=classification,
        gamma=gamma,
    )


def orthogonal_sample_attribution(attribution: OrthogonalAttribution):
    """SampleAttributionResult from the exact attribution (drop-in)."""
    from neutrohydro.attribution import SampleAttributionResult

    c = attribution.contributions.sum(axis=2)
    G = np.clip(attribution.sample_G, 0.0, 1.0)
    return SampleAttributionResult(G=G, A=1.0 - G, w=np.abs(c), c=c)


# ----------------------------------------------------------------------
# Sensitivity, scenarios and process alignment
# ----------------------------------------------------------------------

def channel_sensitivity(encoder: OrthogonalNDGEncoder, model, s_y: float = 1.0) -> dict:
    """
    Gradient of the prediction w.r.t. standardized ions, split by channel.

    ``grad = b_T + b_I + b_F`` with ``b_c`` in the channel-c subspace, so
    the vectors are mutually orthogonal and ``|grad|^2 = sum |b_c|^2``.
    Multiply by ``s_y`` (the target scale) for original target units.
    """
    p = encoder.basis_.center.shape[0]
    beta = model.components_.beta
    rho = {"T": 1.0, "I": np.sqrt(model.rho_I), "F": np.sqrt(model.rho_F)}
    P = encoder.projectors
    b = {c: s_y * rho[c] * (P[c] @ beta[k * p:(k + 1) * p]) for k, c in enumerate(CHANNELS)}
    b["total"] = b["T"] + b["I"] + b["F"]
    return b


def scenario_response(
    encoder: OrthogonalNDGEncoder,
    model,
    delta_std: NDArray[np.floating],
    s_y: float = 1.0,
) -> dict:
    """
    Predicted response to a hypothetical change of ion composition.

    Parameters
    ----------
    delta_std : (p,) or (K, p)
        Change(s) in standardized ion units (see
        :func:`directions_to_standardized` to convert e.g. meq/L).
    s_y : float
        Target scale (``preprocessor.params_.s_y``) for original units.

    Returns
    -------
    dict with ``response`` (K,) total predicted change, ``by_channel``
    (K, 3) exact split, and ``direction_split`` (K, 3) fraction of each
    change vector lying in the T, I and F subspaces.
    """
    D = np.atleast_2d(np.asarray(delta_std, dtype=np.float64))
    b = channel_sensitivity(encoder, model, s_y)
    by_channel = np.column_stack([D @ b[c] for c in CHANNELS])
    return {
        "response": by_channel.sum(axis=1),
        "by_channel": by_channel,
        "direction_split": subspace_alignment(encoder, D),
    }


def directions_to_standardized(
    directions: NDArray[np.floating],
    preprocessor,
    reference: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    Map composition changes (original units) to standardized ion space.

    Uses the Jacobian of the preprocessing map (chain rule):
    linear scaling gives ``dx_std = dc / s``; with a log transform the map
    is ``dx_std = dc / ((c_ref + delta_x) * s)`` at a reference composition
    (default: the training medians).

    Parameters
    ----------
    directions : (K, p)
        Rows are composition changes in the units of the model's X, with
        columns in the model's feature order.
    preprocessor : Preprocessor
        Fitted preprocessor.
    reference : (p,), optional
        Reference composition for the log-transform Jacobian.
    """
    prm = preprocessor.params_
    D = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    scale = prm.s + prm.delta_s
    if prm.log_transform:
        if reference is None:
            reference = np.exp(prm.mu) - prm.delta_x
        scale = scale * (np.asarray(reference, dtype=np.float64) + prm.delta_x)
    return D / scale


def subspace_alignment(
    encoder: OrthogonalNDGEncoder,
    directions: Union[NDArray[np.floating], dict],
    names: Optional[Sequence[str]] = None,
):
    """
    Exact T/I/F split of process directions.

    For a direction a (e.g. a mineral's stoichiometric vector or an
    end-member such as seawater, in standardized units), returns
    ``|P_c a|^2 / |a|^2`` for c in (T, I, F); the three numbers sum to 1.
    A geogenic weathering vector lying mostly in T supports a geogenic
    baseline; an anthropogenic vector (e.g. nitrate fertiliser) lying in T
    warns that the "baseline" itself is contaminated.

    Because the subspaces have different dimensions, a *random* direction
    already puts r_c / p of its energy in subspace c. The ``enrich_*``
    columns divide each share by that chance level (> 1: the process is
    preferentially aligned with the subspace), and ``dominant`` is the
    channel with the largest enrichment.

    Returns
    -------
    pandas.DataFrame with columns ``process``, ``T``, ``I``, ``F``,
    ``enrich_T``, ``enrich_I``, ``enrich_F``, ``dominant``.
    """
    import pandas as pd

    if isinstance(directions, dict):
        names = list(directions.keys())
        D = np.array([np.asarray(v, dtype=np.float64) for v in directions.values()])
    else:
        D = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    if names is None:
        names = [f"direction_{k}" for k in range(D.shape[0])]

    P = encoder.projectors
    norms = np.sum(D ** 2, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    shares = np.column_stack([np.sum((D @ P[c]) ** 2, axis=1) / safe for c in CHANNELS])
    shares[norms <= 0] = np.nan
    p = D.shape[1]
    dims = np.array([encoder.basis_.basis(c).shape[1] for c in CHANNELS], dtype=float)
    chance = dims / p
    with np.errstate(divide="ignore", invalid="ignore"):
        enrich = np.where(chance > 0, shares / np.where(chance > 0, chance, 1.0), np.nan)
    df = pd.DataFrame(shares, columns=list(CHANNELS))
    df.insert(0, "process", list(names))
    for k, c in enumerate(CHANNELS):
        df[f"enrich_{c}"] = enrich[:, k]
    df["dominant"] = [
        CHANNELS[int(np.nanargmax(r))] if np.any(np.isfinite(r)) else "undefined"
        for r in enrich
    ]
    return df


# ----------------------------------------------------------------------
# Diagnostic for any encoder: how entangled are the channels?
# ----------------------------------------------------------------------

def jacobian_entanglement(
    encoder,
    X_std: NDArray[np.floating],
    model=None,
    n_samples: int = 20,
    eps: float = 1e-4,
    random_state: Optional[int] = 0,
) -> dict:
    """
    Finite-difference Jacobian entanglement of an encoder's channels.

    For sampled rows i, computes J_c = d channel_c(x_i) / d x_i (the row is
    perturbed inside the full batch, so batch-dependent encoders are
    measured as they are used) and, for each channel pair,

        E_ab = ||J_a J_b^T||_F / (||J_a||_F ||J_b||_F)   in [0, 1],

    0 meaning the channels see orthogonal input directions (disentangled)
    and 1 meaning they see the same directions. A channel whose Jacobian
    vanishes cannot carry sample information (e.g. a median baseline).

    Parameters
    ----------
    encoder : NDGEncoder or OrthogonalNDGEncoder (fitted)
    X_std : (n, p) standardized data
    model : PNPLS, optional
        If given, the channels are the model's effective (precision-weighted)
        design blocks instead of the raw encoder outputs.
    n_samples : int
        Rows to evaluate.
    eps : float
        Finite-difference step.

    Returns
    -------
    dict with mean entanglement per pair (``"T-I"``, ``"T-F"``, ``"I-F"``),
    the mean Jacobian Frobenius norm per channel (``"norms"``) and the
    per-sample values (``"per_sample"``).
    """
    X = np.asarray(X_std, dtype=np.float64)
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    rows = rng.choice(n, size=min(n_samples, n), replace=False)

    def channels(Xb: NDArray[np.floating]) -> dict:
        trip = encoder.transform(Xb)
        if model is not None:
            D = model._weighted_design(trip)
            return {c: D[:, k * p:(k + 1) * p] for k, c in enumerate(CHANNELS)}
        return {"T": trip.T, "I": trip.I, "F": trip.F}

    pairs = [("T", "I"), ("T", "F"), ("I", "F")]
    per_sample = {f"{a}-{b}": [] for a, b in pairs}
    norms = {c: [] for c in CHANNELS}
    for i in rows:
        J = {c: np.zeros((p, p)) for c in CHANNELS}
        for j in range(p):
            Xp = X.copy()
            Xm = X.copy()
            Xp[i, j] += eps
            Xm[i, j] -= eps
            cp, cm = channels(Xp), channels(Xm)
            for c in CHANNELS:
                J[c][:, j] = (cp[c][i] - cm[c][i]) / (2 * eps)
        nrm = {c: np.linalg.norm(J[c]) for c in CHANNELS}
        for c in CHANNELS:
            norms[c].append(nrm[c])
        for a, b in pairs:
            denom = nrm[a] * nrm[b]
            per_sample[f"{a}-{b}"].append(
                np.linalg.norm(J[a] @ J[b].T) / denom if denom > 1e-12 else np.nan
            )

    out = {}
    for key, vals in per_sample.items():
        vals = np.asarray(vals, dtype=float)
        out[key] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
    out["norms"] = {c: float(np.mean(v)) for c, v in norms.items()}
    out["per_sample"] = {k: np.asarray(v, dtype=float) for k, v in per_sample.items()}
    return out
