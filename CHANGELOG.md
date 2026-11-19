# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Calendar effects**: new `calendar_labels` parameter (integer array over the data and forecast time
  points, laid out like `xy` with shape `(..., T + forecast_horizon, n_x + 1)` and broadcast over batches
  and variables; 0 = no effect, 1..K = a multiplicative effect, e.g. holiday weeks). A trailing axis of 1
  gives every variable effects; 0 in a variable's column gives it none
  - Each variable with effects is adjusted on its modeled scale before differencing, the likelihood
    includes the Jacobian of the adjustment, and forecasts at labelled time points are mapped back
  - Effects are separate for each variable (and each batch with `'unshared'` pooling) and share one
    standard deviation, like the other coefficients; the indices of the variables with effects are stored
    in the `calendar_variables` attribute, and `calendar_effect` has shape
    `(num_samples,) [+ batch_shape] + (len(calendar_variables), K)`
  - New `calendar_pooling` parameter (`'unshared'`, `'shared'` or `'partial'`), required when
    `calendar_labels` is given and must be `None` otherwise
  - `calendar_pooling='partial'` gives each batch its own effects, shrunk toward a learned common effect
    `calendar_effect_mu` for each variable and label (prior `Normal(0, calendar_effect_mu_prior_scale)`).
    It uses a non-centred parameterization: `calendar_effect = calendar_effect_mu + calendar_effect_sd *
    calendar_effect_decentered`, with `calendar_effect_decentered ~ Normal(0, 1)`
  - New `calendar_effect_mu_prior_scale` parameter (default 1.0; a scalar or one value per variable)
  - New `calendar_effect_sd_prior_scale` parameter for the HalfCauchy prior on the effects' standard
    deviation; effects are stored in the samples as `calendar_effect` (and `calendar_effect_sd`)
  - Warns when an effect has no modeled observations and so is drawn from the prior: a label in 1..K
    that never occurs for a variable with effects, or one that occurs only in the forecast horizon or the
    conditioned-on first time points (checked per variable, and per batch with `'unshared'` pooling)
- **Input scale description**: new `input_power` and `input_center` parameters (only with
  `transform='none'`) declare that `xy` was power-transformed (power 0 = log) and centred before being
  passed to SARIX; calendar effects use them to act as multipliers on the original scale. `input_power`
  is a scalar or one value per variable (shape `(n_x + 1,)`); `input_center` has one value per batch and
  variable (broadcast to `batch_shape + (n_x + 1,)`), must be finite for variables with calendar effects
  and a non-zero power, and may be NaN elsewhere
- **Divergence count**: the number of divergent MCMC transitions is stored in the new `num_divergences` attribute
- **Models with no autoregressive terms**: SARIX now supports `p=0` and `P=0`, where the predictive mean
  comes only from the intercept and/or Fourier terms. `theta` and `theta_sd` are not sampled in this case.
- **Optional intercept**: New `include_intercept` parameter adds an intercept for each variable
  - New `intercept_pooling` parameter (`'unshared'`, `'shared'` or `'partial'`), required when `include_intercept=True`
    and must be `None` otherwise
  - New `intercept_sd_prior_scale` parameter for the HalfCauchy prior on the intercept standard deviation
  - `intercept_pooling='partial'` gives each batch its own intercept with a hierarchical prior
    `intercept ~ Normal(intercept_mu, intercept_sd)`, where `intercept_mu` is a learned mean per variable
    with prior `Normal(0, intercept_mu_prior_scale)`. Intercepts are then shrunk toward a shared, possibly
    non-zero mean rather than toward 0
  - New `intercept_mu_prior_scale` parameter. By default (`None`) it is set from the data: for each
    variable, 2.5 times the root mean square of its values on the modeled scale. A scalar or a
    per-variable array can be passed instead
  - The intercept is applied on the modeled scale (after transformation and differencing), so it is a
    level with `d=0` and a drift with `d=1`

### Changed
- **Breaking**: All `SARIX` constructor arguments after `xy` are now keyword-only, and they are
  reordered into groups: data (`day_of_year`), model structure, pooling, prior scales, and
  forecasting/MCMC settings. Code that passes these arguments by name is unaffected; code that
  passes them by position now raises a `TypeError`.
- An unknown `transform` value now raises a `ValueError` instead of being treated as `'none'`.

### Deprecated
- The `'none'` option for `theta_pooling`, `sigma_pooling`, `fourier_pooling` and `intercept_pooling`
  is renamed to `'unshared'`, to avoid confusion with Python's `None` (which `fourier_pooling` and
  `intercept_pooling` use to mean "not applicable"). `'none'` is still accepted for now as an alias for
  `'unshared'` but emits a `FutureWarning`, and will be removed in a future release. The defaults of
  `theta_pooling` and `sigma_pooling` are now `'unshared'` (same behavior as before).

## [0.2.0] - 2025-10-15

### Added
- **Fourier seasonality support**: Added flexible Fourier-based seasonal patterns to the SARIX model
  - New `fourier_K` parameter to control the number of harmonic pairs
  - New `day_of_year` parameter for specifying temporal information
  - Fourier features are calculated before differencing to capture seasonal patterns in levels
  - Automatic extrapolation of day-of-year values for forecast periods
  - Location-specific (unpooled) Fourier coefficients with hierarchical priors
- Comprehensive test suite with regression and analytical validation tests (src/sarix/sarix.py:1-702)
- Cross-platform reference fixtures for reproducible testing
- GitHub Actions CI workflow for automated testing on pull requests
- Prior scale parameters for better model customization:
  - `sigma_prior_scale`: Controls innovation variance prior
  - `theta_sd_prior_scale`: Controls AR coefficient variance prior
  - `fourier_beta_sd_prior_scale`: Controls Fourier coefficient variance prior

### Changed
- **Breaking**: Renamed package from `sarixfourier` to `sarix`
- **Breaking**: Renamed main module from `sarix_fourier.py` to `sarix.py`
- Migrated from `setup.py` to modern `pyproject.toml` configuration
- Updated package version from 0.0.1 to 0.2.0
- Updated package description to reflect Fourier seasonality support
- Minimum Python version now 3.11+

### Improved
- Cleaned up repository structure
- Added UV package manager support via `uv.lock`
- Enhanced model flexibility with configurable prior scales
- Better documentation of Fourier seasonality features in docstrings

### Removed
- Old `sarixfourier` module structure
- Legacy `setup.py` configuration file
- Dependency on covidcast package (removed in earlier version)

## [0.0.1] - 2024-01-01 (approximate)

### Added
- Initial implementation of SARIX (Seasonal AutoRegressive Integrated with eXogenous variables) model
- Support for autoregressive (AR) terms with parameter `p`
- Support for ordinary differencing with parameter `d`
- Support for seasonal AR terms with parameter `P`
- Support for seasonal differencing with parameter `D`
- Data transformations: sqrt, fourthrt, log
- Parameter pooling options (none, shared) for both AR coefficients and innovation variances
- MCMC inference using NumPyro and JAX
- Batch processing for multiple time series
- Multi-step ahead forecasting
- Utility functions `diff()` and `inv_diff()` for differencing operations

[0.2.0]: https://github.com/reichlab/sarix/compare/v0.0.1...v0.2.0
[0.0.1]: https://github.com/reichlab/sarix/releases/tag/v0.0.1
