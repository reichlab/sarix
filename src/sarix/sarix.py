"""
Local Trend
================
"""

import os
import time

from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.transforms import AffineTransform, LowerCholeskyAffine
from numpyro.infer import MCMC, NUTS
from numpyro.distributions.util import is_prng_key, validate_sample


def diff(x, d=0, D=0, season_period=7, pad_na=False):
    """
    Apply differencing and seasonal differencing to all variables in a
    batch of matrices with observations in rows and variables in columns.

    Parameters
    ----------
    x: a numpy array to difference, with shape `batch_shape + (T, n_vars)`
    d: number of ordinary differences to compute
    D: number of seasonal differences to compute
    seasonal_period: number of time points per seasonal period
    pad_na: boolean; if True, result has shape `batch_shape + (T, n_vars)` and
        the leading D + d rows in axis -2 have values `np.nan`. Otherwise, the
        result has shape `batch_shape + (T - D - d, n_vars)`.

    Returns
    -------
    a copy of x after taking differences
    """
    # non-seasonal differencing
    x = onp.diff(x, n = d, axis = -2)

    # seasonal differencing
    for i in range(D):
        x = x[..., season_period:, :] - x[..., :-season_period, :]

    if pad_na:
        batch_shape = x.shape[:-2]
        n_vars = x.shape[-1]
        leading_nans = onp.full(batch_shape + (D * season_period + d, n_vars), onp.nan)
        x = onp.concatenate([leading_nans, x], axis = -2)

    return x


def inv_diff(x, dx, d=0, D=0, season_period=7):
    '''
    Invert ordinary and seasonal differencing (go from seasonally differenced
    time series to original time series).

    Inputs
    ------
    dx a (batch of) first-order and/or seasonally differenced time series
        with shape `batch_shape_dx + (T_dx, n_vars)`. For example, if d=0, D=1,
        `dx` has values like `x_{t} - x_{t - season_period}`.
    x a (batch of) time series with shape `batch_shape_x + (T_x, n_vars)`.
    d order of first differencing
    D order of seasonal differencing
    seasonal_period: number of time points per seasonal period

    Returns
    -------
    an array with the same shape as `dx` containing reconstructed values of
    the original time series `x` in the time points `T_x, ..., T_x + T_dx - 1`
    (with zero-based indexing so that x covers the time points `0, ..., T_x - 1`)

    Notes
    -----
    It is assumed that dx "starts" one time index after x "ends": that is, if
        d = 0 and D = 1 then if we had observed x[..., T_x, :] we could calculate
        dx[..., 0, :] = x[..., T_x, :] - x[..., T - ts_frequency, :]
    '''
    # record information about shapes
    batch_shape_x = x.shape[:-2]
    T_x = x.shape[-2]
    n_vars = x.shape[-1]
    batch_shape_dx = dx.shape[:-2]
    T_dx = dx.shape[-2]

    # validate shapes
    if dx.shape[-1] != n_vars:
        raise ValueError("x and dx must have the same size in their last dimension")

    try:
        broadcast_batch_shape = jnp.broadcast_shapes(batch_shape_x, batch_shape_dx)
        if broadcast_batch_shape != batch_shape_dx:
            raise ValueError()
    except ValueError:
        raise ValueError("The batch shapes of x and dx must be broadcastable to the batch shape of dx")

    if T_x < d + D:
        raise ValueError("There must be at least d + D observed values in x to invert differencing")

    # invert ordinary differencing
    for i in range(1, d + 1):
        x_dm1 = diff(x, d=d-i, D=D, season_period=season_period, pad_na=True)

        x_dm1 = onp.broadcast_to(x_dm1, batch_shape_dx + x_dm1.shape[-2:])
        dx_full = onp.concatenate([x_dm1, dx], axis=-2)
        for t in range(T_dx):
            dx_full[..., T_x + t, :] = dx_full[..., T_x + t - 1, :] + dx_full[..., T_x + t, :]

        dx = dx_full[..., -T_dx:, :]

    # invert seasonal differencing
    for i in range(1, D + 1):
        x_dm1 = diff(x, d=0, D=D-i, season_period=season_period, pad_na=True)

        x_dm1 = onp.broadcast_to(x_dm1, batch_shape_dx + x_dm1.shape[-2:])
        dx_full = onp.concatenate([x_dm1, dx], axis=-2)
        for t in range(T_dx):
            dx_full[..., T_x + t, :] = dx_full[..., T_x + t - season_period, :] + dx_full[..., T_x + t, :]

        dx = dx_full[..., -T_dx:, :]

    return dx



class SARIX():
    """
    Seasonal AutoRegressive Integrated model with eXogenous variables (SARIX).

    This model combines autoregressive dynamics with optional seasonal patterns,
    differencing, transformations, and Fourier-based seasonal terms.

    Parameters
    ----------
    xy : array_like
        Input data with shape (batch, time, features) or (time, features).
        The last feature dimension is treated as the target variable (y),
        and all other features are exogenous variables (x).

    p : int, default=1
        Order of autoregressive terms (number of lags).

    d : int, default=0
        Order of differencing (non-seasonal).

    P : int, default=0
        Order of seasonal autoregressive terms.

    D : int, default=0
        Order of seasonal differencing.

    season_period : int, default=1
        Number of time points per seasonal period (e.g., 52 for weekly data
        with annual seasonality).

    transform : {'none', 'sqrt', 'fourthrt', 'log'}, default='none'
        Transformation to apply to data before modeling.
        - 'none': No transformation
        - 'sqrt': Square root transformation
        - 'fourthrt': Fourth root transformation
        - 'log': Natural log transformation

    theta_pooling : {'none', 'shared'}, default='none'
        How to share AR coefficients across batches.
        - 'none': Separate parameters per batch/location
        - 'shared': Single set of parameters shared across all batches

    sigma_pooling : {'none', 'shared'}, default='none'
        How to share innovation standard deviations across batches.
        - 'none': Separate parameters per batch/location
        - 'shared': Single set of parameters shared across all batches

    forecast_horizon : int, default=1
        Number of time steps to forecast into the future.

    num_warmup : int, default=1000
        Number of MCMC warmup iterations.

    num_samples : int, default=1000
        Number of MCMC samples to draw after warmup.

    num_chains : int, default=1
        Number of MCMC chains to run.

    day_of_year : array_like, optional
        Day-of-year values (0-365) for each time point. Required if fourier_K > 0.
        Used to calculate Fourier seasonal features.

    fourier_K : int, default=0
        Number of Fourier harmonic pairs to include for seasonal patterns.
        If 0, no Fourier terms are included. Each harmonic pair includes
        a sine and cosine term, so fourier_K=3 adds 6 features total.

        Design note: Fourier terms are applied to the data BEFORE differencing
        to capture seasonal patterns in levels rather than changes.

    fourier_pooling : {'none', 'shared'}, required if fourier_K > 0
        How to share Fourier regression coefficients across batches.
        - 'none': Separate coefficients per batch/location
        - 'shared': Single set of coefficients shared across all batches
        Required when fourier_K > 0, must be None when fourier_K=0.

    Notes
    -----
    **Fourier Seasonality:**
    When fourier_K > 0, the model adds seasonal terms of the form:
        Σ_{k=1}^{K} [β_{sin,k} * sin(2πk*day/365.25) + β_{cos,k} * cos(2πk*day/365.25)]

    These terms are separate for each location (unpooled) and each variable.

    **Day-of-year format:**
    The day_of_year parameter expects integer values 0-365 (or 1-366).
    For weekly data, extract this from your dates using:
        day_of_year = pd.to_datetime(dates).dayofyear.values

    **Forecast extrapolation:**
    When making forecasts, day-of-year values are automatically extrapolated
    assuming 7-day spacing (weekly data), wrapping around year boundaries.

    Examples
    --------
    Basic SARIX model without Fourier terms:

    >>> xy = np.random.randn(10, 52, 2)  # 10 locations, 52 weeks, 2 features
    >>> model = SARIX(xy, p=2, forecast_horizon=4)

    SARIX with Fourier seasonality:

    >>> dates = pd.date_range('2020-01-01', periods=52, freq='W')
    >>> day_of_year = dates.dayofyear.values
    >>> model = SARIX(xy, p=2, day_of_year=day_of_year, fourier_K=3,
    ...               fourier_pooling='shared', forecast_horizon=4)
    """
    def __init__(self,
                 xy,
                 p=1,
                 d=0,
                 P=0,
                 D=0,
                 season_period=1,
                 transform='none',
                 theta_pooling='none',
                 sigma_pooling='none',
                 forecast_horizon=1,
                 num_warmup=1000, num_samples=1000, num_chains=1,
                 day_of_year=None,
                 fourier_K=0,
                 fourier_pooling=None,
                 sigma_prior_scale=1.0,
                 theta_sd_prior_scale=1.0,
                 fourier_beta_sd_prior_scale=1.0):
        self.n_x = xy.shape[-1] - 1
        self.xy = xy.copy()
        self.p = p
        self.d = d
        self.P = P
        self.D = D
        self.max_lag = p + P * season_period
        self.transform = transform
        self.theta_pooling = theta_pooling
        self.sigma_pooling = sigma_pooling
        self.season_period = season_period
        self.forecast_horizon = forecast_horizon
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.fourier_K = fourier_K
        self.fourier_pooling = fourier_pooling
        self.day_of_year = day_of_year
        self.sigma_prior_scale = sigma_prior_scale
        self.theta_sd_prior_scale = theta_sd_prior_scale
        self.fourier_beta_sd_prior_scale = fourier_beta_sd_prior_scale

        # Validate Fourier parameters
        if fourier_K > 0 and day_of_year is None:
            raise ValueError("day_of_year must be provided when fourier_K > 0")
        if fourier_K > 0 and len(day_of_year) != xy.shape[-2]:
            raise ValueError(f"day_of_year length ({len(day_of_year)}) must match time dimension ({xy.shape[-2]})")
        if fourier_K > 0 and fourier_pooling is None:
            raise ValueError("fourier_pooling must be specified when fourier_K > 0")
        if fourier_K == 0 and fourier_pooling is not None:
            raise ValueError("fourier_pooling should not be specified when fourier_K=0")
        if fourier_pooling is not None and fourier_pooling not in ['none', 'shared']:
            raise ValueError("fourier_pooling must be 'none' or 'shared'")

        # set up batch shapes for parameter pooling
        # xy has shape batch_shape + (T, n_x + 1)
        batch_shape = xy.shape[:-2]

        if theta_pooling == 'none':
            # separate parameters per batch
            self.theta_batch_shape = batch_shape
        elif theta_pooling == 'shared':
            # no batches for theta; will broadcast to share across all batches
            self.theta_batch_shape = ()
        else:
            raise ValueError("theta_pooling must be 'none' or 'shared'")

        if sigma_pooling == 'none':
            # separate parameters per batch
            self.sigma_batch_shape = batch_shape
        elif sigma_pooling == 'shared':
            # no batches for sigma; will broadcast to share across all batches
            self.sigma_batch_shape = ()
        else:
            raise ValueError("sigma_pooling must be 'none' or 'shared'")

        if fourier_K > 0:
            if fourier_pooling == 'none':
                # separate parameters per batch
                self.fourier_batch_shape = batch_shape
            elif fourier_pooling == 'shared':
                # no batches for fourier; will broadcast to share across all batches
                self.fourier_batch_shape = ()
        else:
            # Not using Fourier terms, but set for consistency
            self.fourier_batch_shape = batch_shape

        # do transformation
        self.xy_orig = xy.copy()
        if transform == "sqrt":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.sqrt(self.xy)
        elif transform == "fourthrt":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.power(self.xy, 0.25)
        elif transform == "log":
            self.xy[self.xy <= 0] = 1.0
            self.xy = onp.log(self.xy)

        # Calculate Fourier features BEFORE differencing
        # Design decision: Fourier terms capture seasonal patterns in levels,
        # not changes, so we compute them on the original (transformed) scale
        if self.fourier_K > 0:
            self.fourier_features = self._calculate_fourier_features(
                self.day_of_year, self.fourier_K
            )
        else:
            self.fourier_features = None

        # do differencing; save xy before differencing for later use when
        # inverting differencing
        transformed_xy = self.xy
        self.xy = diff(self.xy, self.d, self.D, self.season_period, pad_na=False)

        # Trim Fourier features to match differenced data length
        # After differencing, we lose d + D*season_period time points
        if self.fourier_features is not None:
            n_diff = self.d + self.D * self.season_period
            self.fourier_features = self.fourier_features[n_diff:, :]

        # pre-calculate state update matrix
        self.update_X = self.state_update_X(self.xy[..., :self.max_lag, :],
                                            self.xy[..., self.max_lag:, :])

        # do inference
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        self.run_inference(rng_key)

        # generate predictions
        self.predictions_modeled_scale = self.predict(rng_key_predict)

        # undo differencing
        self.predictions = inv_diff(transformed_xy,
                                    self.predictions_modeled_scale,
                                    self.d, self.D, self.season_period)

        # undo transformation to get predictions on original scale
        if transform == "log":
            self.predictions = onp.exp(self.predictions)
        elif transform == "fourthrt":
            self.predictions = jnp.maximum(0.0, self.predictions)**4
        elif transform == "sqrt":
            self.predictions = jnp.maximum(0.0, self.predictions)**2


    def _calculate_fourier_features(self, day_of_year, K):
        """
        Calculate Fourier seasonal features from day-of-year values.

        Parameters
        ----------
        day_of_year : array_like
            Day-of-year values (0-365) for each time point, shape (T,)
        K : int
            Number of Fourier harmonic pairs

        Returns
        -------
        fourier_features : ndarray
            Array of shape (T, 2*K) containing sine and cosine features.
            For each harmonic k=1..K, includes:
            - sin(2π * k * day / 365.25)
            - cos(2π * k * day / 365.25)

        Notes
        -----
        We use 365.25 to account for leap years in the annual cycle.
        The features are ordered as: [sin_1, cos_1, sin_2, cos_2, ..., sin_K, cos_K]
        """
        day_of_year = onp.asarray(day_of_year)
        T = len(day_of_year)
        features = []

        # Normalize to [0, 1) range for annual cycle
        # Using 365.25 to account for leap years
        t_normalized = day_of_year / 365.25

        for k in range(1, K + 1):
            # Add sine and cosine for each harmonic
            features.append(onp.sin(2 * onp.pi * k * t_normalized))
            features.append(onp.cos(2 * onp.pi * k * t_normalized))

        # Stack into (T, 2*K) array
        return onp.stack(features, axis=-1)

    def run_inference(self, rng_key):
        '''
        helper function for doing hmc inference
        '''
        start = time.time()
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=self.num_chains,
                    progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
        mcmc.run(rng_key, self.xy)
        mcmc.print_summary()
        print('\nMCMC elapsed time:', time.time() - start)
        self.samples = mcmc.get_samples()


    def make_state_transition_matrix(self, theta):
        batch_shape = theta.shape[:-1]
        n_ar_coef = self.p + self.P * (self.p + 1)

        A_x_cols = [
            jnp.concatenate(
                    [
                        jnp.zeros(batch_shape + (i * n_ar_coef, 1)),
                        jnp.expand_dims(theta[..., (i * n_ar_coef):((i + 1) * n_ar_coef)], -1),
                        jnp.zeros(batch_shape + ((self.n_x - i) * n_ar_coef, 1))
                    ],
                    axis = -2) \
                for i in range(self.n_x)
        ]
        A_y_col = [ jnp.expand_dims(theta[..., (self.n_x * n_ar_coef):], -1) ]

        A = jnp.concatenate(A_x_cols + A_y_col, axis = -1)

        return A


    def state_update_X(self, init_stoch_state, stoch_state):
        stoch_state = jnp.concatenate([init_stoch_state, stoch_state], axis=-2)

        # lagged values of x
        lagged_x = [
            self.build_lagged_var(stoch_state[..., :, i:(i+1)]) \
                for i in range(self.n_x)
        ]

        # lagged values of y
        lagged_y = self.build_lagged_var(stoch_state[..., :, self.n_x:(self.n_x + 1)])

        # concatenate
        return jnp.concatenate(lagged_x + [lagged_y], axis = -1)


    def build_lagged_var(self, x):
        # lagged state, highest degree term
        lagged_state = [
            self.lagged_vals_one_seasonal_lag(x=x,
                                              seasonal_lag=P_ind*self.season_period,
                                              p=self.p) for P_ind in range(self.P+1)]
        lagged_state = [x for x in lagged_state if x is not None]
        lagged_state = jnp.concatenate(lagged_state, axis = -1)

        # return entries in rows starting at the last row of init_stoch_shape,
        # going up to second-to-last column. These are the entries used to determine
        # means for stoch_state
        return lagged_state[..., (self.max_lag - 1):(-1), :]


    def lagged_vals_one_seasonal_lag(self, x, seasonal_lag, p):
        if seasonal_lag == 0:
            # no seasonal lag, just terms up to p
            to_concat = [self.lagged_col(x, l) for l in range(p)]
        else:
            # lags from seasonal_lag to (seasonal_lag + p)
            to_concat = [self.lagged_col(x, seasonal_lag - 1 + l) for l in range(p+1)]

        if to_concat == []:
            return None

        result = jnp.concatenate(to_concat, axis=-1)
        return result


    def lagged_col(self, x, lag):
        batch_shape = x.shape[:-2]
        T = x.shape[-2]
        return jnp.concatenate(
            [jnp.full(batch_shape + (lag, 1), jnp.nan), x[..., :(T-lag), 0:1]],
            axis=-2)


    def model(self, xy):
        # Vector of innovation standard deviations for the n_x + 1 variables
        sigma = numpyro.sample(
            "sigma",
            dist.HalfCauchy(self.sigma_prior_scale * jnp.ones(self.sigma_batch_shape + (self.n_x + 1,))))

        # Lower cholesky factor of the covariance matrix has
        # standard deviations on the diagonal
        # The first line below creates (potentially batched) diagonal matrices
        # with shape self.sigma_batch_shape + (n_x + 1, n_x + 1)
        # If xy has batch dimensions, we then insert another dimension at
        # postion -3 for appropriate broadcasting with the time dimension of
        # observed values
        Sigma_chol = jnp.expand_dims(sigma, -2) * jnp.eye(sigma.shape[-1])
        if len(xy.shape) > 2:
            Sigma_chol = jnp.expand_dims(Sigma_chol, -3)

        # state transition matrix parameters
        n_theta = (2 * self.n_x + 1) * (self.p + self.P * (self.p + 1))
        theta_sd = numpyro.sample(
            "theta_sd",
            dist.HalfCauchy(self.theta_sd_prior_scale * jnp.ones(1))
        )
        theta = numpyro.sample(
            "theta",
            dist.Normal(loc=jnp.zeros(self.theta_batch_shape + (n_theta,)),
                        scale=jnp.full(self.theta_batch_shape + (n_theta,),
                                       theta_sd))
        )

        # assemble state transition matrix A
        A = self.make_state_transition_matrix(theta)

        # predictive means based on AR structure
        step_means_ar = jnp.matmul(self.update_X, A)

        # Add Fourier seasonal terms if present
        if self.fourier_K > 0:
            # Sample Fourier coefficients: separate for each location and variable
            # Shape: (fourier_batch_shape, n_x+1, 2*K)
            fourier_beta_sd = numpyro.sample(
                "fourier_beta_sd",
                dist.HalfCauchy(self.fourier_beta_sd_prior_scale * jnp.ones(1))
            )
            fourier_beta = numpyro.sample(
                "fourier_beta",
                dist.Normal(
                    loc=jnp.zeros(self.fourier_batch_shape + (self.n_x+1, 2*self.fourier_K)),
                    scale=fourier_beta_sd
                )
            )

            # Fourier features for the time period after max_lag
            # Shape: (T - max_lag, 2*K)
            F = jnp.array(self.fourier_features[self.max_lag:, :])

            # Compute Fourier contribution to means
            # einsum: (batch, n_x+1, 2K) x (T-max_lag, 2K) -> (batch, T-max_lag, n_x+1)
            step_means_fourier = jnp.einsum('...ik,tk->...ti', fourier_beta, F)

            # Total predictive mean = AR terms + Fourier terms
            step_means = step_means_ar + step_means_fourier
        else:
            step_means = step_means_ar

        # step innovations are (state - step_means),
        # with shape (batch_shape, T - self.max_lag, n_x + 1)
        step_innovations = xy[..., self.max_lag:, :] - step_means

        # sample innovations
        numpyro.sample(
            "step_innovations",
            dist.MultivariateNormal(
                loc=jnp.zeros((self.n_x + 1,)), scale_tril=Sigma_chol),
            obs=step_innovations
        )


    def predict(self, rng_key):
        '''
        Predict future values of all signals based on a single sample of
        parameter values from the posterior distribution.
        '''
        # load in parameter estimates and update to target batch size
        theta = self.samples['theta']
        sigma = self.samples['sigma']
        xy_batch_shape = self.xy.shape[:-2]
        theta_batch_shape = theta.shape[:-1]
        sigma_batch_shape = sigma.shape[:-1]

        if self.theta_pooling == 'shared':
            # goal is shape theta_batch_shape + xy_batch_shape + theta.shape[-1]
            # first insert 1's corresponding to xy_batch_shape, then broadcast
            ones = (1,) * len(xy_batch_shape)
            theta = theta.reshape(theta_batch_shape + ones + (theta.shape[-1],))
            target = theta_batch_shape + xy_batch_shape + (theta.shape[-1],)
            theta = jnp.broadcast_to(theta, target)

        if self.sigma_pooling == 'shared':
            # goal is shape sigma_batch_shape + xy_batch_shape + sigma.shape[-1]
            # first insert 1's corresponding to xy_batch_shape, then broadcast
            ones = (1,) * len(xy_batch_shape)
            sigma = sigma.reshape(sigma_batch_shape + ones + (sigma.shape[-1],))
            target = sigma_batch_shape + xy_batch_shape + (sigma.shape[-1],)
            sigma = jnp.broadcast_to(sigma, target)

        batch_shape = theta.shape[:-1]

        # state transition matrix
        A = self.make_state_transition_matrix(theta)

        # convert sigma to a batch of covariance matrix Cholesky factors
        Sigma_chol = jnp.expand_dims(sigma, -2) * jnp.eye(sigma.shape[-1])

        # Load Fourier coefficients if present
        if self.fourier_K > 0:
            fourier_beta = self.samples['fourier_beta']
            # fourier_beta has shape (num_samples, fourier_batch_shape..., n_x+1, 2*K)
            # Need to match batch_shape
            fourier_batch_shape = fourier_beta.shape[:-2]

            if self.fourier_pooling == 'shared':
                # goal is shape fourier_batch_shape + xy_batch_shape + (n_x+1, 2*K)
                # first insert 1's corresponding to xy_batch_shape, then broadcast
                ones = (1,) * len(xy_batch_shape)
                fourier_beta = fourier_beta.reshape(fourier_batch_shape + ones + fourier_beta.shape[-2:])
                target = fourier_batch_shape + xy_batch_shape + fourier_beta.shape[-2:]
                fourier_beta = jnp.broadcast_to(fourier_beta, target)

            # Generate forecast day-of-year values
            # Extrapolate from last observed day, assuming 7-day spacing (weekly data)
            last_day = self.day_of_year[-1]
            forecast_days = onp.array([(last_day + 7*(h+1)) % 365 for h in range(self.forecast_horizon)])

            # Calculate Fourier features for forecast period
            F_forecast = self._calculate_fourier_features(forecast_days, self.fourier_K)
            F_forecast = jnp.array(F_forecast)  # Shape: (forecast_horizon, 2*K)
        else:
            fourier_beta = None
            F_forecast = None

        # generate innovations
        # note that the use of sample_shape = forecast_horizon means that
        # innovations.shape = (forecast_horizon,) + batch_shape + (n_x + 1)
        # we would really like shape batch_shape + (forecast_horizon, n_x + 1)
        # we deal with this in the loop below when adding to the mean for each
        # forecast horizon by dropping the leading dimension when indexing, then
        # inserting an extra dimension at position -2 before adding.
        innovations = dist.MultivariateNormal(
                loc=jnp.zeros((self.n_x + 1,)),
                scale_tril=Sigma_chol) \
            .sample(rng_key, sample_shape=(self.forecast_horizon, ))

        # generate step-ahead forecasts iteratively
        y_pred = []
        recent_lags = jnp.broadcast_to(self.xy[..., -self.max_lag:, :],
                                       batch_shape + (self.max_lag, self.xy.shape[-1]))
        dummy_values = jnp.zeros(batch_shape + (1, self.xy.shape[-1]))
        for h in range(self.forecast_horizon):
            update_X = self.state_update_X(recent_lags, dummy_values)
            # AR contribution
            ar_mean = jnp.matmul(update_X, A)

            # Add Fourier contribution if present
            if self.fourier_K > 0:
                # Compute Fourier mean for this forecast step
                # einsum: (..., n_x+1, 2K) x (2K,) -> (..., n_x+1)
                fourier_mean = jnp.einsum('...ik,k->...i', fourier_beta, F_forecast[h, :])
                # Expand to (..., 1, n_x+1) for addition
                fourier_mean = jnp.expand_dims(fourier_mean, -2)
            else:
                fourier_mean = 0

            new_y_pred = ar_mean + fourier_mean + jnp.expand_dims(innovations[h, ...], -2)
            y_pred.append(new_y_pred)
            recent_lags = jnp.concatenate([recent_lags[..., 1:, :], new_y_pred],
                                          axis=-2)

        y_pred = jnp.concatenate(y_pred, axis=-2)
        return onp.asarray(y_pred)


    def plot(self, save_path = None):
        t = onp.arange(self.y_nbhd.shape[0])
        t_pred = onp.arange(self.y_nbhd.shape[0] + self.forecast_horizon)
        n_betas = self.samples['betas'].shape[1]

        percentile_levels = [2.5, 97.5]
        median_prediction = onp.median(self.predictions, axis=0)
        percentiles = onp.percentile(self.predictions, percentile_levels, axis=0)
        median_prediction_orig = onp.median(self.predictions_orig, axis=0)
        percentiles_orig = onp.percentile(self.predictions_orig, percentile_levels, axis=0)

        fig, ax = plt.subplots(n_betas + 1, 1, figsize=(10,3 * (n_betas + 1)))

        ax[0].fill_between(t_pred, percentiles_orig[0, :], percentiles_orig[1, :], color='lightblue')
        ax[0].plot(t_pred, median_prediction_orig, 'blue', ls='solid', lw=2.0)
        ax[0].plot(t, self.y_orig, 'black', ls='solid')
        ax[0].set(xlabel="t", ylabel="y", title="Mean predictions with 95% CI")

        # plot 95% confidence level of predictions
        ax[1].fill_between(t_pred, percentiles[0, :], percentiles[1, :], color='lightblue')
        ax[1].plot(t_pred, median_prediction, 'blue', ls='solid', lw=2.0)
        ax[1].plot(t, self.y, 'black', ls='solid')
        ax[1].set(xlabel="t", ylabel="y (" + self.transform + " scale)", title="Mean predictions with 95% CI")

        for i in range(1, n_betas):
            beta_median = onp.median(self.samples['betas'][:, i, :], axis=0)
            beta_percentiles = onp.percentile(self.samples['betas'][:, i, :], percentile_levels, axis=0)
            ax[i + 1].fill_between(t_pred, beta_percentiles[0, :], beta_percentiles[1, :], color='lightblue')
            ax[i + 1].plot(t_pred, beta_median, 'blue', ls='solid', lw=2.0)
            ax[i + 1].set(xlabel="t", ylabel="incidence deriv " + str(i))

        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()
