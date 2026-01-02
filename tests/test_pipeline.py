"""
Tests for the pipeline module.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from neutrohydro.pipeline import (
    NeutroHydroPipeline,
    PipelineConfig,
    PipelineResults,
    run_pipeline,
)


class TestPipelineConfig:
    """Tests for PipelineConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.log_transform is False
        assert config.baseline_type == "robust_pca"
        assert config.n_components == 5
        assert config.rho_I == 1.0
        assert config.rho_F == 1.0
        assert config.gamma == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            log_transform=True,
            n_components=3,
            baseline_type="low_rank",
            baseline_rank=5,
        )

        assert config.log_transform is True
        assert config.n_components == 3
        assert config.baseline_type == "low_rank"
        assert config.baseline_rank == 5


class TestNeutroHydroPipeline:
    """Tests for NeutroHydroPipeline class."""

    def test_fit_basic(self, sample_data, ion_names):
        """Test basic pipeline fitting."""
        X, y = sample_data
        pipeline = NeutroHydroPipeline()
        results = pipeline.fit(X, y, feature_names=ion_names)

        assert isinstance(results, PipelineResults)
        assert pipeline.is_fitted_

    def test_results_structure(self, sample_data, ion_names):
        """Test that results have expected structure."""
        X, y = sample_data
        pipeline = NeutroHydroPipeline()
        results = pipeline.fit(X, y, feature_names=ion_names)

        # Check fitted objects
        assert results.preprocessor is not None
        assert results.encoder is not None
        assert results.model is not None
        assert results.triplets is not None

        # Check results
        assert results.nvip is not None
        assert results.nsr is not None
        assert results.sample_attribution is not None

        # Check predictions
        assert len(results.y_pred) == len(y)
        assert len(results.y_pred_original) == len(y)

        # Check metrics
        assert 0.0 < results.r2_train <= 1.0

    def test_predict(self, sample_data, ion_names):
        """Test prediction on new data."""
        X, y = sample_data
        pipeline = NeutroHydroPipeline()
        pipeline.fit(X, y, feature_names=ion_names)

        # Predict on same data
        y_pred = pipeline.predict(X)

        assert len(y_pred) == len(y)

    def test_predict_without_fit_raises(self, sample_data):
        """Test that predict before fit raises error."""
        X, y = sample_data
        pipeline = NeutroHydroPipeline()

        with pytest.raises(RuntimeError):
            pipeline.predict(X)

    def test_get_summary(self, sample_data, ion_names):
        """Test get_summary method."""
        X, y = sample_data
        pipeline = NeutroHydroPipeline()
        pipeline.fit(X, y, feature_names=ion_names)

        summary = pipeline.get_summary()

        assert 'model' in summary
        assert 'nvip' in summary
        assert 'attribution' in summary
        assert 'feature_names' in summary

    def test_to_dataframes(self, sample_data, ion_names):
        """Test to_dataframes method."""
        X, y = sample_data
        pipeline = NeutroHydroPipeline()
        pipeline.fit(X, y, feature_names=ion_names)

        dfs = pipeline.to_dataframes()

        assert 'nvip' in dfs
        assert 'nsr' in dfs
        assert 'samples' in dfs

    def test_save_results(self, sample_data, ion_names):
        """Test save_results method."""
        X, y = sample_data
        pipeline = NeutroHydroPipeline()
        pipeline.fit(X, y, feature_names=ion_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            pipeline.save_results(output_dir)

            # Check files exist
            assert (output_dir / "nvip.csv").exists()
            assert (output_dir / "nsr.csv").exists()
            assert (output_dir / "samples.csv").exists()
            assert (output_dir / "summary.json").exists()

    def test_log_transform_config(self, sample_data, ion_names):
        """Test pipeline with log transform."""
        X, y = sample_data
        config = PipelineConfig(log_transform=True)
        pipeline = NeutroHydroPipeline(config)
        results = pipeline.fit(X, y, feature_names=ion_names)

        assert results.preprocessor.params_.log_transform is True

    def test_low_rank_baseline(self, sample_data, ion_names):
        """Test pipeline with low-rank baseline."""
        X, y = sample_data
        config = PipelineConfig(baseline_type="low_rank", baseline_rank=3)
        pipeline = NeutroHydroPipeline(config)
        results = pipeline.fit(X, y, feature_names=ion_names)

        assert results.encoder.baseline_type.value == "low_rank"

    def test_different_components(self, sample_data, ion_names):
        """Test pipeline with different number of components."""
        X, y = sample_data

        config1 = PipelineConfig(n_components=2)
        pipeline1 = NeutroHydroPipeline(config1)
        results1 = pipeline1.fit(X, y, feature_names=ion_names)

        config2 = PipelineConfig(n_components=5)
        pipeline2 = NeutroHydroPipeline(config2)
        results2 = pipeline2.fit(X, y, feature_names=ion_names)

        # More components should generally give better fit (or equal)
        assert results2.r2_train >= results1.r2_train - 0.01  # Allow small tolerance


class TestRunPipeline:
    """Tests for run_pipeline convenience function."""

    def test_run_pipeline_basic(self, sample_data, ion_names):
        """Test run_pipeline function."""
        X, y = sample_data
        results = run_pipeline(X, y, feature_names=ion_names)

        assert isinstance(results, PipelineResults)

    def test_run_pipeline_with_config(self, sample_data, ion_names):
        """Test run_pipeline with configuration kwargs."""
        X, y = sample_data
        results = run_pipeline(
            X, y,
            feature_names=ion_names,
            n_components=3,
            log_transform=True,
        )

        assert results.config.n_components == 3
        assert results.config.log_transform is True


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self, sample_data, ion_names):
        """Test complete workflow from data to results."""
        X, y = sample_data
        n, p = X.shape

        # Create and fit pipeline
        config = PipelineConfig(
            n_components=5,
            log_transform=False,
            baseline_type="median",
        )
        pipeline = NeutroHydroPipeline(config)
        results = pipeline.fit(X, y, feature_names=ion_names)

        # Verify NVIP L2 decomposition
        from neutrohydro.nvip import verify_l2_decomposition
        assert verify_l2_decomposition(results.nvip)

        # Verify pi_G + pi_A = 1
        np.testing.assert_array_almost_equal(
            results.nsr.pi_G + results.nsr.pi_A, 1.0
        )

        # Verify G + A = 1
        np.testing.assert_array_almost_equal(
            results.sample_attribution.G + results.sample_attribution.A, 1.0
        )

        # Verify predictions are reasonable
        assert not np.any(np.isnan(results.y_pred_original))

        # Verify feature names preserved
        assert results.feature_names == ion_names

    def test_reproducibility(self, sample_data, ion_names):
        """Test that pipeline is reproducible."""
        X, y = sample_data

        pipeline1 = NeutroHydroPipeline()
        results1 = pipeline1.fit(X, y, feature_names=ion_names)

        pipeline2 = NeutroHydroPipeline()
        results2 = pipeline2.fit(X, y, feature_names=ion_names)

        np.testing.assert_array_almost_equal(
            results1.nvip.VIP_agg,
            results2.nvip.VIP_agg
        )
        np.testing.assert_array_almost_equal(
            results1.nsr.pi_G,
            results2.nsr.pi_G
        )
