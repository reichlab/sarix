"""
Generate regression test reference outputs with tight priors.

This script creates deterministic model outputs using tight priors to minimize
MCMC sampling variability. These references are used for regression testing
to catch bugs when modifying the model implementation.
"""
import numpy as np
from sarixfourier.sarix_fourier import SARIX
import json
import sys


def generate_simple_ar1_reference():
    """Generate reference output for simple AR(1) model with tight priors"""
    print("Generating simple AR(1) reference...")

    # Fixed seed for reproducibility
    np.random.seed(42)

    # Simple synthetic data: 3 locations, 50 timepoints, 2 features
    xy = np.random.randn(3, 50, 2) + 10

    # Run model with tight priors and fixed seed
    np.random.seed(999)
    model = SARIX(
        xy,
        p=1,
        d=0,
        D=0,
        P=0,
        season_period=1,
        transform='none',
        theta_pooling='shared',
        sigma_pooling='shared',
        forecast_horizon=3,
        num_warmup=20,
        num_samples=30,
        num_chains=1,
        # Tight priors for minimal MCMC variability
        sigma_prior_scale=0.05,
        theta_sd_prior_scale=0.05
    )

    # Save predictions and key parameters
    reference = {
        'predictions_mean': np.mean(model.predictions, axis=0).tolist(),
        'predictions_std': np.std(model.predictions, axis=0).tolist(),
        'theta_mean': np.mean(model.samples['theta'], axis=0).tolist(),
        'sigma_mean': np.mean(model.samples['sigma'], axis=0).tolist(),
        'predictions_shape': list(model.predictions.shape),
        'model_config': {
            'p': model.p,
            'd': model.d,
            'D': model.D,
            'P': model.P,
            'season_period': model.season_period,
            'transform': model.transform,
            'forecast_horizon': model.forecast_horizon,
            'sigma_prior_scale': model.sigma_prior_scale,
            'theta_sd_prior_scale': model.theta_sd_prior_scale,
        }
    }

    with open('test/fixtures/regression_simple_ar1.json', 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"  Predictions shape: {model.predictions.shape}")
    print(f"  Mean prediction: {np.mean(model.predictions):.6f}")
    print(f"  Prediction std dev: {np.std(model.predictions):.6f}")
    print("  Saved to test/fixtures/regression_simple_ar1.json")


def generate_fourier_seasonal_reference():
    """Generate reference output for Fourier seasonal model with tight priors"""
    print("\nGenerating Fourier seasonal reference...")

    # Fixed seed for data generation
    np.random.seed(123)

    # Synthetic seasonal data: 5 locations, 60 weeks, 2 features
    n_locations = 5
    n_weeks = 60
    n_features = 2

    day_of_year = np.array([7*i % 365 for i in range(n_weeks)])

    xy = np.zeros((n_locations, n_weeks, n_features))
    for i in range(n_locations):
        for j in range(n_weeks):
            # Simple seasonal pattern
            seasonal = 2 * np.sin(2 * np.pi * day_of_year[j] / 365)
            xy[i, j, :] = 10 + seasonal + i * 0.5 + np.random.randn(n_features) * 0.5

    xy = np.abs(xy) + 1

    # Run model with tight priors and fixed seed
    np.random.seed(777)
    model = SARIX(
        xy,
        p=1,
        d=0,
        day_of_year=day_of_year,
        fourier_K=2,
        theta_pooling='shared',
        sigma_pooling='shared',
        transform='none',
        num_warmup=20,
        num_samples=30,
        num_chains=1,
        forecast_horizon=4,
        # Tight priors for minimal MCMC variability
        sigma_prior_scale=0.05,
        theta_sd_prior_scale=0.05,
        fourier_beta_sd_prior_scale=0.05
    )

    # Save predictions and key parameters
    reference = {
        'predictions_mean': np.mean(model.predictions, axis=0).tolist(),
        'predictions_std': np.std(model.predictions, axis=0).tolist(),
        'theta_mean': np.mean(model.samples['theta'], axis=0).tolist(),
        'sigma_mean': np.mean(model.samples['sigma'], axis=0).tolist(),
        'fourier_beta_mean': np.mean(model.samples['fourier_beta'], axis=0).tolist(),
        'predictions_shape': list(model.predictions.shape),
        'model_config': {
            'p': model.p,
            'd': model.d,
            'fourier_K': model.fourier_K,
            'transform': model.transform,
            'forecast_horizon': model.forecast_horizon,
            'sigma_prior_scale': model.sigma_prior_scale,
            'theta_sd_prior_scale': model.theta_sd_prior_scale,
            'fourier_beta_sd_prior_scale': model.fourier_beta_sd_prior_scale,
        }
    }

    with open('test/fixtures/regression_fourier_seasonal.json', 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"  Predictions shape: {model.predictions.shape}")
    print(f"  Mean prediction: {np.mean(model.predictions):.6f}")
    print(f"  Prediction std dev: {np.std(model.predictions):.6f}")
    print("  Saved to test/fixtures/regression_fourier_seasonal.json")


def generate_cross_platform_simple_reference():
    """Generate reference output for cross-platform simple model"""
    print("\nGenerating cross-platform simple reference...")

    # Fixed seed for data generation
    np.random.seed(42)
    xy = np.random.randn(3, 30, 2) + 10

    # Run model with fixed seed
    np.random.seed(999)
    model = SARIX(
        xy,
        p=1,
        d=0,
        D=0,
        P=0,
        season_period=1,
        transform='none',
        theta_pooling='shared',
        sigma_pooling='shared',
        forecast_horizon=3,
        num_warmup=20,
        num_samples=30,
        num_chains=1
    )

    # Save predictions and key parameters
    reference = {
        'predictions_mean': np.mean(model.predictions, axis=0).tolist(),
        'predictions_std': np.std(model.predictions, axis=0).tolist(),
        'theta_mean': np.mean(model.samples['theta'], axis=0).tolist(),
        'sigma_mean': np.mean(model.samples['sigma'], axis=0).tolist(),
        'predictions_shape': list(model.predictions.shape),
        'model_config': {
            'p': model.p,
            'd': model.d,
            'D': model.D,
            'P': model.P,
            'season_period': model.season_period,
            'transform': model.transform,
            'forecast_horizon': model.forecast_horizon,
        }
    }

    with open('test/fixtures/reference_simple.json', 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"  Predictions shape: {model.predictions.shape}")
    print(f"  Mean prediction: {np.mean(model.predictions):.6f}")
    print(f"  Prediction std dev: {np.std(model.predictions):.6f}")
    print("  Saved to test/fixtures/reference_simple.json")


def generate_cross_platform_fourier_reference():
    """Generate reference output for cross-platform Fourier model"""
    print("\nGenerating cross-platform Fourier reference...")

    # Fixed seed for data generation
    np.random.seed(123)
    n_locations = 5
    n_weeks = 60
    n_features = 2

    day_of_year = np.array([7*i % 365 for i in range(n_weeks)])

    xy = np.zeros((n_locations, n_weeks, n_features))
    for i in range(n_locations):
        for j in range(n_weeks):
            seasonal = 2 * np.sin(2 * np.pi * day_of_year[j] / 365)
            xy[i, j, :] = 10 + seasonal + i * 0.5 + np.random.randn(n_features) * 0.5

    xy = np.abs(xy) + 1

    # Run model with fixed seed
    np.random.seed(777)
    model = SARIX(
        xy,
        p=2,
        d=0,
        day_of_year=day_of_year,
        fourier_K=2,
        theta_pooling='shared',
        sigma_pooling='shared',
        transform='sqrt',
        num_warmup=20,
        num_samples=30,
        num_chains=1,
        forecast_horizon=4
    )

    # Save predictions and key parameters
    reference = {
        'predictions_mean': np.mean(model.predictions, axis=0).tolist(),
        'predictions_std': np.std(model.predictions, axis=0).tolist(),
        'theta_mean': np.mean(model.samples['theta'], axis=0).tolist(),
        'sigma_mean': np.mean(model.samples['sigma'], axis=0).tolist(),
        'fourier_beta_mean': np.mean(model.samples['fourier_beta'], axis=0).tolist(),
        'predictions_shape': list(model.predictions.shape),
        'model_config': {
            'p': model.p,
            'd': model.d,
            'fourier_K': model.fourier_K,
            'transform': model.transform,
            'forecast_horizon': model.forecast_horizon,
        }
    }

    with open('test/fixtures/reference_fourier.json', 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"  Predictions shape: {model.predictions.shape}")
    print(f"  Mean prediction: {np.mean(model.predictions):.6f}")
    print(f"  Prediction std dev: {np.std(model.predictions):.6f}")
    print("  Saved to test/fixtures/reference_fourier.json")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating regression test references")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print("=" * 60)

    # Regression tests with tight priors
    generate_simple_ar1_reference()
    generate_fourier_seasonal_reference()

    # Cross-platform reproducibility references
    generate_cross_platform_simple_reference()
    generate_cross_platform_fourier_reference()

    print("\n" + "=" * 60)
    print("Reference generation complete!")
    print("=" * 60)
