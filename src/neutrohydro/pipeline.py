"""
Interactive pipeline for NeutroHydro.

Provides a unified workflow from raw data to:
- Model fitting
- NVIP variable importance
- NSR/pi_G attribution
- Sample-level G_i
- Optional mineral inference
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from neutrohydro.preprocessing import Preprocessor, PreprocessorParams
from neutrohydro.encoder import NDGEncoder, TripletData, BaselineType, FalsityMap
from neutrohydro.model import PNPLS, PLSComponents
from neutrohydro.nvip import compute_nvip, NVIPResult
from neutrohydro.attribution import (
    compute_nsr,
    compute_sample_baseline_fraction,
    NSRResult,
    SampleAttributionResult,
)
from neutrohydro.minerals import MineralInverter, MineralInversionResult
from neutrohydro.orthogonal import (
    OrthogonalNDGEncoder,
    OrthogonalAttribution,
    orthogonal_attribution,
    orthogonal_nsr,
    orthogonal_sample_attribution,
    verify_jacobian_orthogonality,
)


@dataclass
class PipelineConfig:
    """Configuration for NeutroHydro pipeline."""

    # Preprocessing
    log_transform: bool = False
    delta_x: float = 1e-12
    delta_s: float = 1e-10

    # NDG Encoder
    # baseline_type: "robust_pca", "low_rank", "median", "hydrofacies", or
    # "orthogonal" (Jacobian-orthogonal encoding with exact attribution; see
    # neutrohydro.orthogonal). For "orthogonal", baseline_rank is the
    # baseline rank (None = select by bootstrap parallel analysis).
    baseline_type: str = "robust_pca"
    baseline_rank: Optional[int] = 2
    falsity_map: str = "exponential"
    falsity_params: Optional[dict] = None
    indeterminacy: str = "knn"            # legacy encoder: "knn", "rolling", "none"
    n_neighbors: int = 5                  # legacy encoder neighbourhood size
    indeterminacy_rank: Optional[int] = None  # orthogonal: I-band size (None = auto)
    orthogonal_robust: bool = True        # orthogonal: falsity-reweighted covariance
    random_state: Optional[int] = 0       # orthogonal: seed for rank selection

    # PNPLS Model
    n_components: int = 5
    rho_I: float = 1.0
    rho_F: float = 1.0
    lambda_F: float = 1.0

    # Attribution
    epsilon: float = 1e-10
    gamma: float = 0.7

    # Mineral inference
    run_mineral_inference: bool = False
    mineral_eta: float = 1.0
    mineral_tau_s: float = 0.01
    mineral_tau_r: float = 1.0

    # Thermodynamic validation
    run_thermodynamic_validation: bool = False
    si_threshold: float = 0.5


@dataclass
class PipelineResults:
    """Container for all pipeline results."""

    # Fitted objects
    preprocessor: Preprocessor
    encoder: NDGEncoder
    model: PNPLS
    triplets: TripletData

    # Results
    nvip: NVIPResult
    nsr: NSRResult
    sample_attribution: SampleAttributionResult

    # Predictions
    y_pred: NDArray[np.floating]
    y_pred_original: NDArray[np.floating]
    r2_train: float
    y_train: Optional[NDArray[np.floating]] = None # Original target values

    # Optional mineral results
    mineral_result: Optional[MineralInversionResult] = None

    # Exact attribution (orthogonal encoding only)
    orthogonal: Optional[OrthogonalAttribution] = None

    # Metadata
    feature_names: Optional[list[str]] = None
    config: Optional[PipelineConfig] = None


class NeutroHydroPipeline:
    """
    Unified pipeline for neutrosophic groundwater chemometrics.

    Orchestrates the full workflow:
    1. Preprocessing (robust centering/scaling)
    2. NDG encoding (T, I, F triplets)
    3. PNPLS regression
    4. NVIP variable importance
    5. NSR/pi_G attribution
    6. Sample-level G_i
    7. Optional mineral inference

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. If None, uses defaults.

    Examples
    --------
    >>> pipeline = NeutroHydroPipeline()
    >>> results = pipeline.fit(X, y, feature_names=ion_names)
    >>> print(f"R2: {results.r2_train:.3f}")
    >>> print(results.nvip.VIP_agg)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.results_: Optional[PipelineResults] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        feature_names: Optional[list[str]] = None,
        groups: Optional[NDArray[np.integer]] = None,
        c_meq: Optional[NDArray[np.floating]] = None,
        pH: Optional[NDArray[np.floating]] = None,
        Eh: Optional[NDArray[np.floating]] = None,
        temp: float = 25.0,
    ) -> PipelineResults:
        """
        Fit the complete pipeline.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix with ion concentrations.
        y : ndarray of shape (n_samples,)
            Target vector (e.g., log TDS, log EC).
        feature_names : list of str, optional
            Names of ion features.
        groups : ndarray of shape (n_samples,), optional
            Group labels for hydrofacies-conditioned baseline.
        c_meq : ndarray of shape (n_samples, n_ions), optional
            Ion concentrations in meq/L for mineral inference.
            If None and mineral inference is requested, uses X directly.
        pH : ndarray of shape (n_samples,), optional
            pH values for thermodynamic validation.
        Eh : ndarray of shape (n_samples,), optional
            Redox (Eh) values in mV for thermodynamic validation.
        temp : float, default=25.0
            Temperature in Celsius for thermodynamic validation.

        Returns
        -------
        results : PipelineResults
            Container with all fitted objects and results.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        if feature_names is None:
            feature_names = [f"Ion_{j}" for j in range(p)]

        # 1. Preprocessing
        preprocessor = Preprocessor(
            log_transform=self.config.log_transform,
            delta_x=self.config.delta_x,
            delta_s=self.config.delta_s,
        )
        X_std, y_std = preprocessor.fit_transform(X, y, feature_names)

        orthogonal_mode = self.config.baseline_type == "orthogonal"

        # 2. NDG Encoding
        if orthogonal_mode:
            encoder = OrthogonalNDGEncoder(
                rank_T=self.config.baseline_rank,
                rank_I=self.config.indeterminacy_rank,
                robust=self.config.orthogonal_robust,
                lambda_F=self.config.lambda_F,
                falsity_map=self.config.falsity_map,
                falsity_params=self.config.falsity_params,
                random_state=self.config.random_state,
            )
            triplets = encoder.fit_transform(X_std, groups)
        else:
            encoder = NDGEncoder(
                baseline_type=self.config.baseline_type,
                baseline_rank=self.config.baseline_rank,
                falsity_map=self.config.falsity_map,
                falsity_params=self.config.falsity_params,
                indeterminacy=self.config.indeterminacy,
                n_neighbors=self.config.n_neighbors,
            )
            triplets = encoder.fit_transform(X_std, groups)

        # 3. PNPLS Model
        model = PNPLS(
            n_components=self.config.n_components,
            rho_I=self.config.rho_I,
            rho_F=self.config.rho_F,
            lambda_F=self.config.lambda_F,
        )
        model.fit(triplets, y_std)

        # 4. NVIP computation
        nvip_result = compute_nvip(model, feature_names=feature_names)

        ortho = None
        if orthogonal_mode:
            # 5-6. Exact attribution replaces the VIP-energy heuristic
            ortho = orthogonal_attribution(model, triplets, y_std, feature_names)
            ortho.checks.update(verify_jacobian_orthogonality(encoder, X_std))
            nsr = orthogonal_nsr(ortho, self.config.gamma, self.config.epsilon)
            sample_attribution = orthogonal_sample_attribution(ortho)
        else:
            # 5. NSR / pi_G
            nsr = compute_nsr(nvip_result, self.config.epsilon, self.config.gamma)

            # 6. Sample-level G_i
            sample_attribution = compute_sample_baseline_fraction(model, triplets, nsr)

        # Predictions
        y_pred_std = model.predict(triplets)
        y_pred_original = preprocessor.inverse_transform_y(y_pred_std)

        # R2
        r2_train = model.score(triplets, y_std)

        # 7. Optional mineral inference
        mineral_result = None
        if self.config.run_mineral_inference:
            if c_meq is None:
                c_meq = X  # Assume X is already in meq/L

            inverter = MineralInverter(
                ion_order=feature_names,
                eta=self.config.mineral_eta,
                tau_s=self.config.mineral_tau_s,
                tau_r=self.config.mineral_tau_r,
            )
            # Use all samples if dimensions match
            if c_meq.shape[1] == len(inverter.ion_names):
                mineral_result = inverter.invert(
                    c_meq, 
                    nsr.pi_G,
                    use_thermodynamics=self.config.run_thermodynamic_validation,
                    pH=pH,
                    Eh=Eh,
                    temp=temp,
                    si_threshold=self.config.si_threshold
                )

        self.results_ = PipelineResults(
            preprocessor=preprocessor,
            encoder=encoder,
            model=model,
            triplets=triplets,
            nvip=nvip_result,
            nsr=nsr,
            sample_attribution=sample_attribution,
            y_pred=y_pred_std,
            y_pred_original=y_pred_original,
            y_train=y, # Save original target
            r2_train=r2_train,
            mineral_result=mineral_result,
            orthogonal=ortho,
            feature_names=feature_names,
            config=self.config,
        )
        self.is_fitted_ = True

        return self.results_

    def predict(
        self,
        X: NDArray[np.floating],
        groups: Optional[NDArray[np.integer]] = None,
    ) -> NDArray[np.floating]:
        """
        Predict target values for new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New predictor matrix.
        groups : ndarray, optional
            Group labels for hydrofacies baseline.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values in original scale.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before predict")

        X = np.asarray(X, dtype=np.float64)

        # Preprocess
        X_std, _ = self.results_.preprocessor.transform(X)

        # Encode
        triplets = self.results_.encoder.transform(X_std, groups)

        # Predict
        y_pred_std = self.results_.model.predict(triplets)

        # Inverse transform
        return self.results_.preprocessor.inverse_transform_y(y_pred_std)

    def get_summary(self) -> dict:
        """
        Get summary of pipeline results.

        Returns
        -------
        summary : dict
            Dictionary with key metrics and results.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted first")

        from neutrohydro.attribution import attribution_summary

        attr_summary = attribution_summary(
            self.results_.nsr,
            self.results_.sample_attribution,
            self.results_.feature_names,
        )

        summary = {
            "config": asdict(self.results_.config) if self.results_.config else {},
            "model": {
                "n_components": self.results_.model.components_.n_components,
                "r2_train": self.results_.r2_train,
                "explained_variance": self.results_.model.components_.explained_variance.tolist(),
            },
            "nvip": {
                "VIP_agg": self.results_.nvip.VIP_agg.tolist(),
                "VIP_T": self.results_.nvip.VIP_T.tolist(),
                "VIP_I": self.results_.nvip.VIP_I.tolist(),
                "VIP_F": self.results_.nvip.VIP_F.tolist(),
            },
            "attribution": attr_summary,
            "feature_names": self.results_.feature_names,
        }

        ortho = self.results_.orthogonal
        if ortho is not None:
            basis = self.results_.encoder.basis_
            summary["orthogonal"] = {
                "rank_T": basis.rank_T,
                "rank_I": basis.rank_I,
                "variance_share": basis.explained_variance_ratio,
                "R2_weighted": ortho.R2,
                "R2_T": None if ortho.R2_channels is None else float(ortho.R2_channels[0]),
                "R2_I": None if ortho.R2_channels is None else float(ortho.R2_channels[1]),
                "R2_F": None if ortho.R2_channels is None else float(ortho.R2_channels[2]),
                "checks": {k: float(v) for k, v in ortho.checks.items()},
            }

        return summary

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """
        Export results as pandas DataFrames.

        Returns
        -------
        dataframes : dict
            Dictionary with keys:
            - 'nvip': NVIP results per ion
            - 'nsr': NSR and pi_G per ion
            - 'samples': Sample-level G_i
            - 'predictions': Predictions vs actual (if available)
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted first")

        from neutrohydro.nvip import nvip_to_dataframe
        from neutrohydro.attribution import nsr_to_dataframe, sample_attribution_to_dataframe

        dataframes = {
            'nvip': nvip_to_dataframe(self.results_.nvip, self.results_.feature_names),
            'nsr': nsr_to_dataframe(self.results_.nsr, self.results_.feature_names),
            'samples': sample_attribution_to_dataframe(self.results_.sample_attribution),
        }
        
        # Predictions
        if self.results_.y_train is not None:
            dataframes['predictions'] = pd.DataFrame({
                'Actual': self.results_.y_train,
                'Predicted': self.results_.y_pred_original,
                'Residual': self.results_.y_train - self.results_.y_pred_original
            })

        if self.results_.orthogonal is not None:
            dataframes['orthogonal_ions'] = self.results_.orthogonal.ions_dataframe()
            dataframes['orthogonal_samples'] = self.results_.orthogonal.samples_dataframe()

        # Include mineral results if available
        if self.results_.mineral_result:
            # We use a default inverter to call the helper method
            inv = MineralInverter() 
            dataframes['minerals'] = inv.results_to_dataframe(self.results_.mineral_result)

        return dataframes

    def save_results(self, output_dir: str | Path) -> None:
        """
        Save results to CSV files.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save results.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted first")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dfs = self.to_dataframes()

        for name, df in dfs.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)

        # Save summary as JSON
        import json
        summary = self.get_summary()
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def run_pipeline(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    feature_names: Optional[list[str]] = None,
    **config_kwargs,
) -> PipelineResults:
    """
    Convenience function to run the full pipeline.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Predictor matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    feature_names : list of str, optional
        Names of features.
    **config_kwargs
        Keyword arguments passed to PipelineConfig.

    Returns
    -------
    results : PipelineResults
        Pipeline results.

    Examples
    --------
    >>> results = run_pipeline(X, y, n_components=3, log_transform=True)
    >>> print(f"R2: {results.r2_train:.3f}")
    """
    config = PipelineConfig(**config_kwargs)
    pipeline = NeutroHydroPipeline(config)
    return pipeline.fit(X, y, feature_names)
