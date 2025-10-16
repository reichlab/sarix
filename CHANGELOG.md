# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
