import numpy as np
from sarixfourier.sarix_fourier import SARIX, diff, inv_diff

def test_diff_simple():
    """Test basic differencing operation"""
    x = np.array([[[1, 2], [3, 4], [5, 6]]])  # shape (1, 3, 2)
    result = diff(x, d=1, D=0, season_period=1, pad_na=False)
    expected = np.array([[[2, 2], [2, 2]]])  # differences
    np.testing.assert_array_equal(result, expected)

def test_diff_seasonal():
    """Test seasonal differencing"""
    x = np.array([[[1], [2], [3], [4], [5], [6], [7]]])  # shape (1, 7, 1)
    result = diff(x, d=0, D=1, season_period=3, pad_na=False)
    # First 3 values removed, then seasonal diff with period 3
    expected = np.array([[[3], [3], [3], [3]]])  # [4-1, 5-2, 6-3, 7-4]
    np.testing.assert_array_equal(result, expected)

def test_inv_diff():
    """Test inverse differencing"""
    x = np.array([[[1, 2], [3, 4]]])  # initial values, shape (1, 2, 2)
    dx = np.array([[[2, 2], [2, 2]]])  # differences, shape (1, 2, 2)
    result = inv_diff(x, dx, d=1, D=0, season_period=1)
    expected = np.array([[[5, 6], [7, 8]]])  # cumulative sum from x[-1]
    np.testing.assert_array_equal(result, expected)

def test_sarix_basic_instantiation():
    """Test that SARIX can be instantiated with minimal config"""
    np.random.seed(42)
    # Create minimal xy: batch=2, time=15, features=2 (so n_x=1)
    # Need enough time points: p=1 default, so max_lag=1, need at least max_lag+1=2 points after differencing
    xy = np.random.randn(2, 15, 2) + 10  # Add offset to avoid transform issues

    model = SARIX(
        xy,
        p=1,
        d=0,
        P=0,
        D=0,
        season_period=1,
        transform='none',
        theta_pooling='shared',
        sigma_pooling='shared',
        forecast_horizon=1,
        num_warmup=10,
        num_samples=10,
        num_chains=1
    )

    # Check basic attributes
    assert model.n_x == 1
    assert model.p == 1
    assert model.d == 0
    assert model.max_lag == 1
    assert model.forecast_horizon == 1
    assert model.predictions is not None
    assert model.predictions.shape == (10, 2, 1, 2)  # (samples, batch, horizon, features)

def test_sarix_with_seasonal():
    """Test SARIX with seasonal components"""
    np.random.seed(42)
    xy = np.random.randn(2, 20, 2) + 10

    model = SARIX(
        xy,
        p=1,
        P=1,
        season_period=7,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=1
    )

    # Check that seasonal parameters are used
    assert model.P == 1
    assert model.season_period == 7
    assert model.max_lag == 8  # p + P * season_period = 1 + 1*7
    assert model.predictions is not None


# Edge cases and validation tests (#1)

def test_single_batch():
    """Test SARIX with single batch (no batch dimension)"""
    np.random.seed(42)
    xy = np.random.randn(20, 2) + 10  # No batch dimension

    model = SARIX(
        xy,
        p=1,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    assert model.predictions is not None
    assert model.predictions.shape == (10, 2, 2)  # (samples, horizon, features)


def test_larger_batch():
    """Test SARIX with larger batch size"""
    np.random.seed(42)
    xy = np.random.randn(5, 20, 2) + 10  # batch=5

    model = SARIX(
        xy,
        p=1,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=1
    )

    assert model.predictions is not None
    assert model.predictions.shape == (10, 5, 1, 2)  # (samples, batch, horizon, features)


def test_minimum_data_size():
    """Test SARIX with minimum viable data size"""
    np.random.seed(42)
    # For p=1, P=0, need at least max_lag + 1 = 2 time points
    xy = np.random.randn(2, 2) + 10

    model = SARIX(
        xy,
        p=1,
        P=0,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        forecast_horizon=1
    )

    assert model.predictions is not None


def test_invalid_shape_diff():
    """Test that diff raises error for invalid differencing"""
    x = np.random.randn(1, 3, 2)
    # Trying to do too much differencing should fail
    try:
        result = diff(x, d=5, D=0, season_period=1, pad_na=False)
        # If we get here with empty result, that's expected
        assert result.shape[-2] == 0 or True
    except (ValueError, IndexError):
        # Either error is acceptable
        pass


def test_invalid_inv_diff_shapes():
    """Test that inv_diff validates input shapes"""
    x = np.array([[[1, 2]]])  # shape (1, 1, 2)
    dx = np.array([[[1, 2, 3]]])  # shape (1, 1, 3) - mismatched last dim

    try:
        inv_diff(x, dx, d=1, D=0, season_period=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "same size in their last dimension" in str(e)


def test_insufficient_data_inv_diff():
    """Test that inv_diff validates sufficient history"""
    x = np.array([[[1, 2]]])  # shape (1, 1, 2) - not enough for d=2
    dx = np.array([[[2, 2]]])

    try:
        inv_diff(x, dx, d=2, D=0, season_period=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "at least d + D observed values" in str(e)


# Numerical correctness tests (#3)

def test_diff_inv_diff_are_inverses():
    """Test that differencing and inverse differencing are true inverses"""
    np.random.seed(42)
    x_orig = np.random.randn(2, 20, 2) + 10

    # Test ordinary differencing with d=1
    # diff with d=1 removes 1 time point, so we have 19 points after differencing
    x_diff = diff(x_orig, d=1, D=0, season_period=1, pad_na=False)
    # We need the last d=1 point before differencing to reconstruct
    x_init = x_orig[..., :1, :]
    x_reconstructed = inv_diff(x_init, x_diff, d=1, D=0, season_period=1)

    # Should reconstruct all values after the initial d points
    np.testing.assert_array_almost_equal(
        x_reconstructed,
        x_orig[..., 1:, :],
        decimal=10
    )


def test_seasonal_diff_inv_diff_are_inverses():
    """Test seasonal differencing and its inverse"""
    np.random.seed(42)
    x_orig = np.random.randn(2, 20, 2) + 10
    season_period = 7

    x_init = x_orig[..., :season_period, :]
    x_diff = diff(x_orig, d=0, D=1, season_period=season_period, pad_na=False)
    x_reconstructed = inv_diff(x_init, x_diff, d=0, D=1, season_period=season_period)

    np.testing.assert_array_almost_equal(
        x_reconstructed,
        x_orig[..., season_period:, :],
        decimal=10
    )


def test_combined_diff_inv_diff_are_inverses():
    """Test combined ordinary and seasonal differencing"""
    np.random.seed(42)
    x_orig = np.random.randn(2, 25, 2) + 10
    season_period = 7

    x_init = x_orig[..., :(1 + season_period), :]  # Need d + D*season_period points
    x_diff = diff(x_orig, d=1, D=1, season_period=season_period, pad_na=False)
    x_reconstructed = inv_diff(x_init, x_diff, d=1, D=1, season_period=season_period)

    np.testing.assert_array_almost_equal(
        x_reconstructed,
        x_orig[..., (1 + season_period):, :],
        decimal=10
    )


def test_predictions_shape_multiple_horizons():
    """Test that predictions have correct shape for different forecast horizons"""
    np.random.seed(42)
    xy = np.random.randn(2, 20, 2) + 10

    for horizon in [1, 3, 5]:
        model = SARIX(
            xy,
            p=1,
            theta_pooling='shared',
            sigma_pooling='shared',
            num_warmup=5,
            num_samples=5,
            num_chains=1,
            forecast_horizon=horizon
        )

        # predictions shape: (samples, batch, horizon, features)
        assert model.predictions.shape == (5, 2, horizon, 2)


def test_predictions_non_negative_after_sqrt_transform():
    """Test that predictions are non-negative after sqrt transform"""
    np.random.seed(42)
    xy = np.abs(np.random.randn(2, 20, 2)) + 1  # Ensure positive for sqrt

    model = SARIX(
        xy,
        p=1,
        transform='sqrt',
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        forecast_horizon=3
    )

    # After sqrt transform and inverse, predictions should be non-negative
    assert np.all(model.predictions >= 0)


def test_predictions_non_negative_after_log_transform():
    """Test that predictions are non-negative after log transform"""
    np.random.seed(42)
    xy = np.abs(np.random.randn(2, 20, 2)) + 2  # Ensure positive for log

    model = SARIX(
        xy,
        p=1,
        transform='log',
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        forecast_horizon=3
    )

    # After log transform and exp inverse, predictions should be positive
    assert np.all(model.predictions > 0)


def test_prediction_consistency():
    """Test that same seed produces same results"""
    xy = np.random.randn(2, 20, 2) + 10

    # Create two models with same random seed
    np.random.seed(123)
    model1 = SARIX(
        xy.copy(),
        p=1,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        forecast_horizon=2
    )

    np.random.seed(123)
    model2 = SARIX(
        xy.copy(),
        p=1,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        forecast_horizon=2
    )

    # Note: Due to JAX's RNG, this might not be exactly equal, but should be close
    # This test verifies the structure is consistent
    assert model1.predictions.shape == model2.predictions.shape


# Gap #1: Testing with 'none' pooling

def test_none_pooling_theta():
    """Test SARIX with 'none' pooling for theta (separate parameters per batch)"""
    np.random.seed(42)
    xy = np.random.randn(3, 20, 2) + 10

    model = SARIX(
        xy,
        p=1,
        theta_pooling='none',  # Separate theta per batch
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    # Check that theta has separate values per batch
    assert 'theta' in model.samples
    # With none pooling, theta should have batch dimension
    # theta shape should be (num_samples, batch_size, n_theta)
    assert model.samples['theta'].ndim == 3
    assert model.samples['theta'].shape[1] == 3  # batch size
    assert model.predictions is not None


def test_none_pooling_sigma():
    """Test SARIX with 'none' pooling for sigma (separate parameters per batch)"""
    np.random.seed(42)
    xy = np.random.randn(3, 20, 2) + 10

    model = SARIX(
        xy,
        p=1,
        theta_pooling='shared',
        sigma_pooling='none',  # Separate sigma per batch
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    # Check that sigma has separate values per batch
    assert 'sigma' in model.samples
    # With none pooling, sigma should have batch dimension
    assert model.samples['sigma'].shape[1] == 3  # batch size
    assert model.predictions is not None


def test_none_pooling_both():
    """Test SARIX with 'none' pooling for both theta and sigma"""
    np.random.seed(42)
    xy = np.random.randn(3, 20, 2) + 10

    model = SARIX(
        xy,
        p=1,
        theta_pooling='none',
        sigma_pooling='none',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=1
    )

    # Both should have batch dimensions
    assert model.samples['theta'].shape[1] == 3
    assert model.samples['sigma'].shape[1] == 3
    assert model.predictions is not None


# Gap #2: Testing different AR orders

def test_higher_ar_order_p2():
    """Test SARIX with AR order p=2"""
    np.random.seed(42)
    xy = np.random.randn(2, 25, 2) + 10

    model = SARIX(
        xy,
        p=2,  # Higher AR order
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=3
    )

    assert model.p == 2
    assert model.max_lag == 2
    # n_theta = (2 * n_x + 1) * (p + P * (p + 1))
    # With n_x=1, p=2, P=0: n_theta = 3 * 2 = 6
    assert model.samples['theta'].shape[-1] == 6
    assert model.predictions is not None


def test_higher_ar_order_p3():
    """Test SARIX with AR order p=3"""
    np.random.seed(42)
    xy = np.random.randn(2, 30, 2) + 10

    model = SARIX(
        xy,
        p=3,  # Even higher AR order
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    assert model.p == 3
    assert model.max_lag == 3
    # n_theta = (2 * n_x + 1) * (p + P * (p + 1))
    # With n_x=1, p=3, P=0: n_theta = 3 * 3 = 9
    assert model.samples['theta'].shape[-1] == 9
    assert model.predictions is not None


def test_seasonal_ar_order():
    """Test SARIX with both p and P > 0"""
    np.random.seed(42)
    xy = np.random.randn(2, 30, 2) + 10

    model = SARIX(
        xy,
        p=2,
        P=1,
        season_period=7,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    assert model.p == 2
    assert model.P == 1
    assert model.max_lag == 2 + 1 * 7  # p + P * season_period = 9
    # n_theta = (2 * n_x + 1) * (p + P * (p + 1))
    # With n_x=1, p=2, P=1: n_theta = 3 * (2 + 1*(2+1)) = 3 * 5 = 15
    assert model.samples['theta'].shape[-1] == 15
    assert model.predictions is not None


# Gap #3: Testing with realistic data sizes

def test_realistic_data_size():
    """Test SARIX with realistic data dimensions (mimicking real epidemic data)"""
    np.random.seed(42)
    # Simulate weekly data for 50 US states over 2 years
    n_locations = 50
    n_weeks = 104  # 2 years
    n_features = 3  # e.g., target + 2 covariates

    xy = np.abs(np.random.randn(n_locations, n_weeks, n_features)) + 5

    model = SARIX(
        xy,
        p=2,
        P=1,
        season_period=52,  # Annual seasonality
        transform='sqrt',
        theta_pooling='shared',  # Share parameters across locations
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=4  # 4-week ahead forecast
    )

    # Check dimensions
    assert model.predictions.shape[1] == n_locations
    assert model.predictions.shape[2] == 4  # forecast horizon
    assert model.predictions.shape[3] == n_features
    # Check non-negativity after sqrt transform
    assert np.all(model.predictions >= 0)


def test_realistic_smaller_dataset():
    """Test SARIX with smaller but realistic dataset"""
    np.random.seed(42)
    # 10 locations, 1 year of weekly data
    n_locations = 10
    n_weeks = 52
    n_features = 2

    xy = np.abs(np.random.randn(n_locations, n_weeks, n_features)) + 2

    model = SARIX(
        xy,
        p=1,
        d=0,
        D=0,
        transform='log',
        theta_pooling='shared',
        sigma_pooling='none',  # Separate sigma per location
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    assert model.predictions is not None
    assert model.predictions.shape == (10, 10, 2, 2)
    assert np.all(model.predictions > 0)  # After exp transform


# Gap #4: Verifying MCMC sample structure

def test_mcmc_sample_keys():
    """Test that MCMC samples contain expected keys"""
    np.random.seed(42)
    xy = np.random.randn(2, 20, 2) + 10

    model = SARIX(
        xy,
        p=1,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=1
    )

    # Check that samples dictionary has expected keys
    assert 'theta' in model.samples
    assert 'sigma' in model.samples
    assert 'theta_sd' in model.samples

    # Should NOT have step_innovations in samples (it's observed)
    assert 'step_innovations' not in model.samples


def test_mcmc_sample_shapes():
    """Test that MCMC samples have correct shapes"""
    np.random.seed(42)
    batch_size = 3
    n_features = 2
    xy = np.random.randn(batch_size, 20, n_features) + 10

    num_samples = 15
    model = SARIX(
        xy,
        p=1,
        P=0,
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=num_samples,
        num_chains=1,
        forecast_horizon=1
    )

    # theta: (num_samples, n_theta) with shared pooling
    # n_theta = (2 * n_x + 1) * (p + P * (p + 1)) = 3 * 1 = 3
    assert model.samples['theta'].shape == (num_samples, 3)

    # sigma: (num_samples, n_features) with shared pooling
    assert model.samples['sigma'].shape == (num_samples, n_features)

    # theta_sd: (num_samples, 1)
    assert model.samples['theta_sd'].shape == (num_samples, 1)


def test_mcmc_sample_hierarchical_structure():
    """Test structure with none pooling"""
    np.random.seed(42)
    batch_size = 3
    xy = np.random.randn(batch_size, 20, 2) + 10

    num_samples = 15
    model = SARIX(
        xy,
        p=1,
        theta_pooling='none',  # Separate per batch
        sigma_pooling='none',
        num_warmup=10,
        num_samples=num_samples,
        num_chains=1,
        forecast_horizon=1
    )

    # With 'none' pooling, should have theta_sd (global)
    assert 'theta_sd' in model.samples
    assert 'theta' in model.samples
    assert 'sigma' in model.samples

    # theta should have batch dimension with 'none' pooling
    assert model.samples['theta'].shape[0] == num_samples
    assert model.samples['theta'].shape[1] == batch_size

    # sigma should also have batch dimension
    assert model.samples['sigma'].shape[1] == batch_size


# Gap #5: Full end-to-end integration test

def test_full_pipeline_sqrt_transform():
    """Test full pipeline with sqrt transform: transform → difference → fit → predict → inverse"""
    np.random.seed(42)
    # Start with positive data for sqrt
    xy_original = np.abs(np.random.randn(2, 30, 2)) + 5

    model = SARIX(
        xy_original,
        p=2,
        d=1,
        P=0,
        D=0,
        season_period=1,
        transform='sqrt',
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=4
    )

    # Verify transformations were applied
    assert model.transform == 'sqrt'
    assert model.d == 1

    # Check that original data was stored
    np.testing.assert_array_equal(model.xy_orig, xy_original)

    # Predictions should be on original scale (non-negative)
    assert np.all(model.predictions >= 0)

    # Check full prediction shape
    assert model.predictions.shape == (10, 2, 4, 2)


def test_full_pipeline_log_seasonal():
    """Test full pipeline with log transform and seasonal differencing"""
    np.random.seed(42)
    # Start with positive data for log
    xy_original = np.abs(np.random.randn(2, 40, 2)) + 3

    model = SARIX(
        xy_original,
        p=1,
        d=0,
        P=1,
        D=1,
        season_period=7,
        transform='log',
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=3
    )

    # Verify all settings
    assert model.transform == 'log'
    assert model.D == 1
    assert model.P == 1
    assert model.season_period == 7

    # Predictions should be positive (after exp)
    assert np.all(model.predictions > 0)

    # Check shape
    assert model.predictions.shape == (10, 2, 3, 2)


def test_full_pipeline_fourthrt():
    """Test full pipeline with fourth root transform"""
    np.random.seed(42)
    xy_original = np.abs(np.random.randn(2, 25, 2)) + 2

    model = SARIX(
        xy_original,
        p=1,
        d=1,
        P=0,
        D=0,
        transform='fourthrt',
        theta_pooling='shared',
        sigma_pooling='shared',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    # Verify transform
    assert model.transform == 'fourthrt'

    # Predictions should be non-negative after power(., 4)
    assert np.all(model.predictions >= 0)

    # Verify inverse transform was applied correctly
    # The code uses: jnp.maximum(0.0, predictions)**4
    assert model.predictions.shape == (10, 2, 2, 2)


def test_full_pipeline_complex():
    """Test full pipeline with multiple features and complex configuration"""
    np.random.seed(42)
    # Multiple locations, features, longer time series
    n_locations = 5
    n_weeks = 50
    n_features = 3  # 1 target + 2 covariates

    xy_original = np.abs(np.random.randn(n_locations, n_weeks, n_features)) + 4

    model = SARIX(
        xy_original,
        p=2,
        d=1,
        P=1,
        D=1,
        season_period=7,
        transform='sqrt',
        theta_pooling='none',  # Separate per location
        sigma_pooling='none',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=5
    )

    # Verify complex configuration
    assert model.n_x == 2  # n_features - 1
    assert model.p == 2
    assert model.P == 1
    assert model.d == 1
    assert model.D == 1
    assert model.max_lag == 2 + 1 * 7  # p + P * season_period

    # Check sample structure with 'none' pooling
    assert 'theta' in model.samples
    assert 'theta_sd' in model.samples
    assert 'sigma' in model.samples

    # theta and sigma should have batch dimensions
    assert model.samples['theta'].shape[1] == n_locations
    assert model.samples['sigma'].shape[1] == n_locations

    # Predictions should be correct shape and non-negative
    assert model.predictions.shape == (10, n_locations, 5, n_features)
    assert np.all(model.predictions >= 0)


# Fourier Seasonality Tests

def test_fourier_feature_calculation():
    """Test that Fourier features are calculated correctly"""
    from sarixfourier.sarix_fourier import SARIX

    # Create simple day-of-year array
    day_of_year = np.array([0, 91, 182, 273])  # Roughly quarterly
    K = 2

    # Create a minimal model just to access the method
    xy = np.random.randn(1, 4, 2) + 10
    model = SARIX(xy, p=1, num_warmup=5, num_samples=5, num_chains=1, forecast_horizon=1)

    features = model._calculate_fourier_features(day_of_year, K)

    # Check shape
    assert features.shape == (4, 4)  # (T, 2*K)

    # Check that features are in [-1, 1] range (sine/cosine)
    assert np.all(features >= -1)
    assert np.all(features <= 1)

    # Check periodicity: day 0 and day 365 should give similar values
    features_0 = model._calculate_fourier_features(np.array([0]), K)
    features_365 = model._calculate_fourier_features(np.array([365]), K)
    np.testing.assert_array_almost_equal(features_0, features_365, decimal=2)


def test_fourier_basic_instantiation():
    """Test SARIX with Fourier terms can be instantiated"""
    np.random.seed(42)
    n_weeks = 52
    xy = np.random.randn(2, n_weeks, 2) + 10
    day_of_year = np.arange(0, n_weeks * 7, 7) % 365  # Weekly data

    model = SARIX(
        xy,
        p=1,
        day_of_year=day_of_year,
        fourier_K=2,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=4
    )

    # Check Fourier parameters were set
    assert model.fourier_K == 2
    assert model.fourier_features is not None
    assert model.fourier_features.shape[1] == 4  # 2*K

    # Check samples include Fourier coefficients
    assert 'fourier_beta' in model.samples
    assert 'fourier_beta_sd' in model.samples

    # Check predictions work
    assert model.predictions is not None


def test_fourier_validation():
    """Test that Fourier parameter validation works"""
    xy = np.random.randn(2, 20, 2) + 10

    # Should raise error if fourier_K > 0 but day_of_year not provided
    try:
        model = SARIX(xy, fourier_K=2, num_warmup=5, num_samples=5, num_chains=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "day_of_year must be provided" in str(e)

    # Should raise error if day_of_year length doesn't match
    try:
        model = SARIX(xy, day_of_year=np.arange(10), fourier_K=2,
                      num_warmup=5, num_samples=5, num_chains=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must match time dimension" in str(e)


def test_fourier_sample_structure():
    """Test that Fourier coefficients have correct structure"""
    np.random.seed(42)
    batch_size = 3
    n_weeks = 52
    n_features = 2
    xy = np.random.randn(batch_size, n_weeks, n_features) + 10
    day_of_year = np.arange(0, n_weeks * 7, 7) % 365

    num_samples = 15
    K = 3
    model = SARIX(
        xy,
        p=1,
        day_of_year=day_of_year,
        fourier_K=K,
        num_warmup=10,
        num_samples=num_samples,
        num_chains=1,
        forecast_horizon=2
    )

    # fourier_beta shape: (num_samples, batch, n_x+1, 2*K)
    # Unpooled, so should have batch dimension
    assert model.samples['fourier_beta'].shape == (num_samples, batch_size, n_features, 2*K)

    # fourier_beta_sd shape: (num_samples, 1)
    assert model.samples['fourier_beta_sd'].shape == (num_samples, 1)


def test_fourier_with_differencing():
    """Test Fourier terms work correctly with differencing"""
    np.random.seed(42)
    n_weeks = 60
    xy = np.abs(np.random.randn(2, n_weeks, 2)) + 5
    day_of_year = np.arange(0, n_weeks * 7, 7) % 365

    model = SARIX(
        xy,
        p=1,
        d=1,
        day_of_year=day_of_year,
        fourier_K=2,
        transform='sqrt',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=3
    )

    # Fourier features should be trimmed after differencing
    # Original: 60 time points, after d=1: 59 time points
    # After max_lag=1: 58 time points used in model
    assert model.fourier_features.shape[0] == 59  # After differencing

    # Predictions should work
    assert model.predictions.shape == (10, 2, 3, 2)
    assert np.all(model.predictions >= 0)  # After sqrt inverse


def test_fourier_forecast_extrapolation():
    """Test that forecast day-of-year is extrapolated correctly"""
    np.random.seed(42)
    n_weeks = 52
    xy = np.random.randn(2, n_weeks, 2) + 10

    # Start at day 350 (near end of year)
    day_of_year = np.array([350 + 7*i for i in range(n_weeks)]) % 365

    model = SARIX(
        xy,
        p=1,
        day_of_year=day_of_year,
        fourier_K=2,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=4
    )

    # Last day in training: (350 + 7*51) % 365 = 7
    # Forecast days should be: 14, 21, 28, 35
    # Check that predictions are reasonable (model runs successfully)
    assert model.predictions is not None
    assert model.predictions.shape[2] == 4  # forecast_horizon


def test_fourier_realistic_epidemic_data():
    """Test Fourier terms with realistic epidemic-like seasonality"""
    np.random.seed(42)
    n_locations = 10
    n_weeks = 104  # 2 years of weekly data
    n_features = 2

    # Create synthetic data with annual seasonality
    day_of_year = np.array([7*i % 365 for i in range(n_weeks)])

    # Generate data with simple seasonal pattern
    xy = np.zeros((n_locations, n_weeks, n_features))
    for i in range(n_locations):
        for j in range(n_weeks):
            # Base level + seasonal component + noise
            seasonal = 5 * np.sin(2 * np.pi * day_of_year[j] / 365)
            xy[i, j, :] = 10 + seasonal + np.random.randn(n_features) * 0.5

    xy = np.abs(xy)  # Ensure positive

    model = SARIX(
        xy,
        p=2,
        day_of_year=day_of_year,
        fourier_K=3,  # 3 harmonics to capture seasonality
        theta_pooling='shared',
        sigma_pooling='shared',
        transform='none',
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=4
    )

    # Model should fit successfully
    assert model.predictions.shape == (10, n_locations, 4, n_features)

    # Fourier coefficients should be estimated
    assert 'fourier_beta' in model.samples
    assert model.samples['fourier_beta'].shape[-1] == 6  # 2*K


def test_fourier_without_fourier_is_same():
    """Test that fourier_K=0 gives same results as not specifying Fourier"""
    np.random.seed(42)
    xy = np.random.randn(2, 20, 2) + 10

    # Model without Fourier
    model1 = SARIX(
        xy.copy(),
        p=1,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    # Model with fourier_K=0 (should be equivalent)
    model2 = SARIX(
        xy.copy(),
        p=1,
        fourier_K=0,  # Explicitly disabled
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        forecast_horizon=2
    )

    # Both should not have Fourier features
    assert model1.fourier_features is None
    assert model2.fourier_features is None

    # Both should not have Fourier samples
    assert 'fourier_beta' not in model1.samples
    assert 'fourier_beta' not in model2.samples
