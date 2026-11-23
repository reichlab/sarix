"""
Local Trend
================
"""

import os
import time
import warnings

from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp

import jax
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
    # accept array-likes such as lists and data frames; jax arrays (including
    # traced ones inside the model) are left as they are
    if not isinstance(x, jax.Array):
        x = onp.asarray(x)

    # non-seasonal differencing, by slicing so that this also works on traced
    # jax arrays inside the model
    for _ in range(d):
        x = x[..., 1:, :] - x[..., :-1, :]

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



POOLING_OPTIONS = ('unshared', 'shared')
PARTIAL_POOLING_OPTIONS = POOLING_OPTIONS + ('partial',)

# power of the power transform each `transform` option applies (0 means log)
TRANSFORM_POWERS = {'none': 1.0, 'sqrt': 0.5, 'fourthrt': 0.25, 'log': 0.0}


def _per_variable(value, name, n_vars):
    """
    Return value as a float array of shape () or (n_vars,), raising a
    ValueError for any other shape.
    """
    value = onp.asarray(value, dtype=float)
    if value.shape not in ((), (n_vars,)):
        raise ValueError(f"{name} must be a scalar or have shape (n_x + 1,) = ({n_vars},), "
                         f"but got shape {value.shape}")
    return value


def _all_positive(value):
    """
    Whether value (a scalar or array) is positive and finite everywhere. None
    and non-numeric values give False.
    """
    try:
        value = onp.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(onp.all(onp.isfinite(value) & (value > 0)))


def _log_df_excess_moments(df_prior_scale):
    """
    Mean and standard deviation of log(df - 2) when df has the
    Gamma(2, 1 / df_prior_scale) distribution truncated to df > 2, computed by
    numerical integration over u = log(df - 2).
    """
    u = onp.linspace(-30.0, onp.log(100.0 * df_prior_scale), 200001)
    df = 2.0 + onp.exp(u)
    # Gamma(2, rate) density is proportional to df * exp(-rate * df); times d df / du = exp(u)
    log_dens = onp.log(df) - df / df_prior_scale + u
    w = onp.exp(log_dens - log_dens.max())
    w /= w.sum()
    mean = onp.sum(w * u)
    return mean, onp.sqrt(onp.sum(w * (u - mean) ** 2))


def _standardized_t_scale_tril(scale_tril, df_excess):
    """
    Scale factor of a multivariate t with df = 2 + df_excess degrees of freedom
    whose covariance is scale_tril @ scale_tril.T:
    scale_tril * sqrt((df - 2) / df). Taking df_excess = df - 2 rather than df
    avoids rounding df - 2 to 0 in float32 when df is close to 2.
    df_excess broadcasts against the batch shape of scale_tril.
    """
    factor = jnp.sqrt(df_excess / (2.0 + df_excess))
    return scale_tril * factor[..., None, None]


def _format_options(options):
    """Format pooling options for an error message, e.g. "'a', 'b' or 'c'"."""
    quoted = [f"'{option}'" for option in options]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def _validate_pooling(value, name, options=POOLING_OPTIONS, allow_none_alias=True):
    """
    Validate a pooling option and return it as a plain string.

    If allow_none_alias, the deprecated value 'none' is accepted as an alias
    for 'unshared' with a FutureWarning.
    """
    if allow_none_alias and value == 'none':
        warnings.warn(
            f"{name}='none' is deprecated and will be removed in a future "
            f"release; use {name}='unshared' instead.",
            FutureWarning,
            stacklevel=3,
        )
        return 'unshared'
    if value not in options:
        raise ValueError(f"{name} must be {_format_options(options)}")
    # return the canonical plain string (value may be a str enum member)
    return options[options.index(value)]


class SARIX():
    """
    Seasonal AutoRegressive Integrated model with eXogenous variables (SARIX).

    This model combines autoregressive dynamics with optional seasonal patterns,
    differencing, transformations, and Fourier-based seasonal terms.

    Parameters
    ----------
    All parameters after `xy` are keyword-only.

    **Data**

    xy : array_like
        Input data with shape batch_shape + (T, n_x + 1): T time points and
        n_x + 1 variables, with optional leading batch dimensions batch_shape
        (e.g. one per location; empty for shape (T, n_x + 1)).
        The last variable is treated as the target variable (y),
        and the other n_x are exogenous variables (x).

    day_of_year : array_like, optional
        Day-of-year values (0-365) for each time point. Required if fourier_K > 0.
        Used to calculate Fourier seasonal features.

    calendar_labels : array_like of int, optional
        Calendar label (for example, for holiday effects) for each time point
        and variable, including the forecast horizon, laid out like xy, with
        shape `(..., T + forecast_horizon, n_x + 1)`. A 0 in any position
        means no effect. With K the largest label, labels 1, ..., K each get a
        separate multiplicative effect (see Notes), so leave no gaps. For
        example, the user might place a 1 in each index of a Christmas week
        and 2 in each index of a New Year's week. Each variable with any
        non-zero label gets its own effects (see `calendar_variables` under
        Attributes); to give effects to the target only, put 0 at all time
        points in the exogenous variables' columns.
        The shape is broadcast to `batch_shape + (T + forecast_horizon, n_x + 1)`
        and must include the full time dimension. The leading batch
        dimensions may be omitted to share the labels across batches. A
        trailing dimension of length 1 (shape `(..., T + forecast_horizon, 1)`)
        applies the same labels to all variables, so every exogenous variable
        is also adjusted, with its own effects.

    **Model structure**

    p : int, default=1
        Order of autoregressive terms (number of lags). If p=0 and P=0, the
        model has no autoregressive terms, and the predictive mean comes only
        from the intercept and/or Fourier terms (if enabled).

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

    input_power : float, optional
        Only with transform='none'. Declares that xy was already transformed
        before being passed to SARIX, so that each variable is
        `a * u**input_power - input_center` for some positive quantity `u` on the
        original scale and some `a > 0`. `input_power=0` means `a * log(u) -
        input_center`. A scalar, or shape (n_x + 1,) for one power per
        variable. Default 1 (xy is on the original scale). Calendar effects
        multiply u, so they need to know this scale. Used only for calendar
        effects: xy is modeled as given, and predictions are returned on its
        scale. For example, for xy = sqrt(u) - m, pass input_power=0.5 and
        input_center=m.

    input_center : float or array_like, optional
        Only with transform='none'. The centering constant in input_power's
        description, applied *after* the power transformation.
        One value per batch and variable: anything that broadcasts to
        batch_shape + (n_x + 1,), e.g. a scalar, shape (n_x + 1,) for one
        value per variable shared by all batches, or the full shape. Default
        0. Only used for variables with calendar effects and a non-zero
        power, where it must be finite; other values may be NaN (e.g. when a
        variable's scale is unknown).

    fourier_K : int, default=0
        Number of Fourier harmonic pairs to include for seasonal patterns.
        If 0, no Fourier terms are included. Each harmonic pair includes
        a sine and cosine term, so fourier_K=3 adds 6 features total.

        Design note: Fourier terms are applied to the data BEFORE differencing
        to capture seasonal patterns in levels rather than changes.

    include_intercept : bool, default=False
        Whether to include an intercept term for each variable. The intercept
        is applied on the modeled scale, i.e. after transformation and
        differencing: with d=0 it is a level, and with d=1 it is a drift.

    **Pooling**

    For all pooling arguments except innovation_df_pooling, the deprecated
    value 'none' is accepted as an alias for 'unshared' and emits a
    FutureWarning.

    theta_pooling : {'unshared', 'shared'}, default='unshared'
        How to share AR coefficients across batches.
        - 'unshared': Separate parameters per batch/location
        - 'shared': Single set of parameters shared across all batches

    sigma_pooling : {'unshared', 'shared'}, default='unshared'
        How to share innovation standard deviations across batches.
        - 'unshared': Separate parameters per batch/location
        - 'shared': Single set of parameters shared across all batches

    fourier_pooling : {'unshared', 'shared'}, required if fourier_K > 0
        How to share Fourier regression coefficients across batches.
        - 'unshared': Separate coefficients per batch/location
        - 'shared': Single set of coefficients shared across all batches
        Required when fourier_K > 0, must be None when fourier_K=0.

    intercept_pooling : {'unshared', 'shared', 'partial'}, required if include_intercept
        How to share intercepts across batches.
        - 'unshared': Separate intercepts per batch/location, each with prior
          Normal(0, intercept_sd), so they are all shrunk toward 0
        - 'shared': Single set of intercepts shared across all batches
        - 'partial': Separate intercepts per batch/location, each with prior
          Normal(intercept_mu, intercept_sd), where intercept_mu is a learned
          mean for each variable. Intercepts are shrunk toward that shared
          mean rather than toward 0, which suits series whose level on the
          modeled scale is far from 0. This hierarchical prior can produce
          some divergent transitions during MCMC, particularly when the
          batch levels are tightly clustered (so intercept_sd is near 0) and
          the model has autoregressive terms; check the divergence count
          reported by MCMC and consider more warmup iterations if it is high.
        Required when include_intercept is True, must be None otherwise.

    calendar_pooling : {'unshared', 'shared', 'partial'}, required if calendar_labels given
        How to share calendar effects across batches.
        - 'unshared': Separate effects per batch/location, each with prior
          Normal(0, calendar_effect_sd), so they are all shrunk toward 0
        - 'shared': Single set of effects shared across all batches
        - 'partial': Separate effects per batch/location, each with prior
          Normal(calendar_effect_mu, calendar_effect_sd), where
          calendar_effect_mu is a learned mean for each variable and label.
          Effects are shrunk toward that common effect rather than toward 0.
          They are sampled non-centred, as calendar_effect_mu +
          calendar_effect_sd * calendar_effect_decentered with
          calendar_effect_decentered ~ Normal(0, 1), which reduces divergent
          transitions when the effects are tightly clustered
          (calendar_effect_sd near 0); check `num_divergences` after fitting.
        Required when calendar_labels is given, must be None otherwise.

    **Prior scales**

    sigma_prior_scale : float, default=1.0
        Scale of the HalfCauchy prior on the innovation standard deviations.

    theta_sd_prior_scale : float, default=1.0
        Scale of the HalfCauchy prior on the standard deviation of the AR
        coefficients.

    fourier_beta_sd_prior_scale : float, default=1.0
        Scale of the HalfCauchy prior on the standard deviation of the
        Fourier coefficients.

    intercept_sd_prior_scale : float, default=1.0
        Scale of the HalfCauchy prior on the standard deviation of the
        intercepts.

    intercept_mu_prior_scale : float or array_like, optional
        Scale of the Normal(0, intercept_mu_prior_scale) prior on the
        per-variable intercept means. Only used when
        intercept_pooling='partial'. May be a scalar or have shape (n_x + 1,)
        for a separate scale per variable. If None (the default), it is set
        from the data: for each variable, 2.5 times the root mean square of
        its values on the modeled scale (after transformation and
        differencing), across all batches and time points. Because the root
        mean square grows with the level of the series as well as its spread,
        this keeps the prior wide relative to the series' level. The resolved
        value is stored in the `intercept_mu_prior_scale` attribute.

    calendar_effect_sd_prior_scale : float, default=1.0
        Scale of the HalfCauchy prior on the standard deviation of the
        calendar effects.

    calendar_effect_mu_prior_scale : float or array_like, default=1.0
        Scale of the Normal(0, calendar_effect_mu_prior_scale) prior on the
        common calendar effects. Only used when calendar_pooling='partial'.
        May be a scalar or have shape (n_x + 1,) for a separate scale per
        variable. The effects are log multipliers on the original scale, so
        unlike intercept_mu_prior_scale the default does not depend on the
        data. The exception is input_power=0 with data a * log(u) - c where
        a != 1 (for example, logged and then standardized data): the effects
        are then a times the log multipliers, so the prior scale should be
        chosen on the scale of xy (see Notes).

    **Innovation distribution**

    innovation_dist : {'normal', 't'}, default='normal'
        Distribution of the innovations.
        - 'normal': multivariate normal with standard deviations sigma.
        - 't': multivariate Student's t with innovation_df > 2 degrees of
          freedom, standardized so that sigma is still the standard deviation:
          its scale is sigma * sqrt((innovation_df - 2) / innovation_df). The
          tails are heavier than the normal's, with the same variance.

    innovation_df_pooling : {'unshared', 'shared', 'partial'}, required if innovation_dist='t'
        How to share the degrees of freedom of t innovations across batches.
        - 'unshared': Separate df per batch/location, each with the prior below
        - 'shared': Single df shared across all batches, with the prior below
        - 'partial': Separate df per batch/location, shrunk toward a common
          value: log(innovation_df - 2) = innovation_df_mu + innovation_df_sd *
          innovation_df_decentered with innovation_df_decentered ~ Normal(0, 1)
          (non-centred). innovation_df_mu has a Normal prior with the mean and
          standard deviation of log(df - 2) under the prior below, and
          innovation_df_sd ~ HalfCauchy(innovation_df_sd_prior_scale).
        Required when innovation_dist='t', must be None otherwise. The deprecated
        alias 'none' is not accepted. The samples hold innovation_df and
        innovation_df_excess = innovation_df - 2 (used for computation, as it
        keeps its precision when innovation_df is close to 2).

    innovation_df_prior_scale : float, default=10.0
        The prior on the degrees of freedom is Gamma(2, 1 / innovation_df_prior_scale)
        (shape 2, scale innovation_df_prior_scale; Juarez and Steel, 2010)
        truncated to df > 2, which keeps the innovation variance finite. With the
        default, its median is about 17 and its mean about 20.

    innovation_df_sd_prior_scale : float, default=1.0
        Scale of the HalfCauchy prior on innovation_df_sd, the spread of
        log(innovation_df - 2) across batches. Only used when
        innovation_df_pooling='partial'.

    **Forecasting and MCMC**

    forecast_horizon : int, default=1
        Number of time steps to forecast into the future.

    num_warmup : int, default=1000
        Number of MCMC warmup iterations.

    num_samples : int, default=1000
        Number of MCMC samples to draw after warmup.

    num_chains : int, default=1
        Number of MCMC chains to run.

    Attributes
    ----------
    samples : dict
        Posterior samples from MCMC, keyed by sample site.

    predictions : ndarray
        Forecasts on the scale of xy (with `transform` undone, and with
        calendar effects included), with shape
        (num_samples,) + batch_shape + (forecast_horizon, n_x + 1).

    predictions_modeled_scale : ndarray
        Forecasts on the modeled scale (transformed and differenced), with
        the same shape as predictions.

    num_divergences : int
        Number of divergent MCMC transitions.

    calendar_variables : ndarray of int
        Only set when calendar_labels is given. Indices of the variables
        (columns of xy) with at least one non-zero label, which get calendar
        effects.

    Notes
    -----
    **Fourier Seasonality:**
    When fourier_K > 0, the model adds seasonal terms of the form:
        Σ_{k=1}^{K} [β_{sin,k} * sin(2πk*day/365.25) + β_{cos,k} * cos(2πk*day/365.25)]

    These terms are separate for each location (unpooled) and each variable.

    **Calendar effects:**
    A label k > 0 for variable j at a time point is taken to multiply that
    variable's original-scale value u by exp(eta_jk); the exception is data
    given as a * log(u) - c with input_power=0, where a is unknown and the
    multiplier is exp(eta_jk / a). The AR model is fitted to
    the series with these effects divided out (the likelihood includes the
    Jacobian of this adjustment), and forecasts at labelled time points are
    multiplied back. The adjustment is made on the modeled scale using the
    power and centre given by `transform` or by input_power and input_center.
    The priors on the effects are described under
    calendar_pooling and the calendar prior scales.

    With n_adjusted = len(calendar_variables), the samples hold 'calendar_effect'
    with shape (num_samples,) + batch_shape + (n_adjusted, K) ('unshared'
    or 'partial') or (num_samples, n_adjusted, K) ('shared'), where K is the
    largest label. With 'partial', 'calendar_effect_mu' and
    'calendar_effect_decentered' are stored as well.
    `predictions_modeled_scale` holds forecasts of the transformed,
    differenced series with the calendar effects removed; `predictions`
    includes them.

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

    SARIX with a holiday effect on the target only (0 for the exogenous
    variable), with labels shared across locations:

    >>> labels = np.zeros((52 + 4, 2), dtype=int)  # (T + forecast_horizon, n_x + 1)
    >>> labels[[50, 51], 1] = 1  # holiday weeks
    >>> model = SARIX(xy, p=2, calendar_labels=labels, calendar_pooling='shared',
    ...               forecast_horizon=4)
    """
    def __init__(self,
                 xy,
                 *,
                 # data
                 day_of_year=None,
                 calendar_labels=None,
                 # model structure
                 p=1,
                 d=0,
                 P=0,
                 D=0,
                 season_period=1,
                 transform='none',
                 input_power=None,
                 input_center=None,
                 fourier_K=0,
                 include_intercept=False,
                 # pooling
                 theta_pooling='unshared',
                 sigma_pooling='unshared',
                 fourier_pooling=None,
                 intercept_pooling=None,
                 calendar_pooling=None,
                 # prior scales
                 sigma_prior_scale=1.0,
                 theta_sd_prior_scale=1.0,
                 fourier_beta_sd_prior_scale=1.0,
                 intercept_sd_prior_scale=1.0,
                 intercept_mu_prior_scale=None,
                 calendar_effect_sd_prior_scale=1.0,
                 calendar_effect_mu_prior_scale=1.0,
                 # innovation distribution
                 innovation_dist='normal',
                 innovation_df_pooling=None,
                 innovation_df_prior_scale=10.0,
                 innovation_df_sd_prior_scale=1.0,
                 # forecasting and MCMC
                 forecast_horizon=1,
                 num_warmup=1000,
                 num_samples=1000,
                 num_chains=1):
        theta_pooling = _validate_pooling(theta_pooling, 'theta_pooling')
        sigma_pooling = _validate_pooling(sigma_pooling, 'sigma_pooling')
        if fourier_pooling is not None:
            fourier_pooling = _validate_pooling(fourier_pooling, 'fourier_pooling')
        if intercept_pooling is not None:
            intercept_pooling = _validate_pooling(intercept_pooling, 'intercept_pooling',
                                                  PARTIAL_POOLING_OPTIONS)
        if calendar_pooling is not None:
            calendar_pooling = _validate_pooling(calendar_pooling, 'calendar_pooling', PARTIAL_POOLING_OPTIONS)

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
        self.include_intercept = include_intercept
        self.intercept_pooling = intercept_pooling
        self.intercept_sd_prior_scale = intercept_sd_prior_scale
        self.intercept_mu_prior_scale = intercept_mu_prior_scale
        self.calendar_pooling = calendar_pooling
        self.calendar_effect_sd_prior_scale = calendar_effect_sd_prior_scale
        mu_scale = _per_variable(calendar_effect_mu_prior_scale, 'calendar_effect_mu_prior_scale', xy.shape[-1])
        if onp.any(mu_scale <= 0):
            raise ValueError("calendar_effect_mu_prior_scale must be positive")
        self.calendar_effect_mu_prior_scale = onp.broadcast_to(mu_scale, (xy.shape[-1],))

        # Validate innovation distribution parameters
        if innovation_dist not in ('normal', 't'):
            raise ValueError("innovation_dist must be 'normal' or 't'")
        if innovation_dist == 't' and innovation_df_pooling is None:
            raise ValueError("innovation_df_pooling must be specified when innovation_dist='t'")
        if innovation_dist == 'normal' and innovation_df_pooling is not None:
            raise ValueError("innovation_df_pooling should not be specified when innovation_dist='normal'")
        if innovation_df_pooling is not None:
            # a new parameter: the deprecated 'none' alias is not accepted
            innovation_df_pooling = _validate_pooling(innovation_df_pooling, 'innovation_df_pooling',
                                                      PARTIAL_POOLING_OPTIONS, allow_none_alias=False)
        # the prior scales are only checked when they are used; NaN, inf and None are rejected
        if innovation_dist == 't' and not _all_positive(innovation_df_prior_scale):
            raise ValueError("innovation_df_prior_scale must be positive and finite")
        if innovation_df_pooling == 'partial' and not _all_positive(innovation_df_sd_prior_scale):
            raise ValueError("innovation_df_sd_prior_scale must be positive and finite")
        self.innovation_dist = innovation_dist
        self.innovation_df_pooling = innovation_df_pooling
        self.innovation_df_prior_scale = innovation_df_prior_scale
        self.innovation_df_sd_prior_scale = innovation_df_sd_prior_scale
        if innovation_df_pooling == 'partial':
            # prior on the common mean of log(df - 2): the mean and sd of log(df - 2) under the
            # (truncated Gamma) prior used for df with 'unshared' and 'shared' pooling
            self.innovation_df_mu_prior = _log_df_excess_moments(innovation_df_prior_scale)

        # Validate Fourier parameters
        if fourier_K > 0 and day_of_year is None:
            raise ValueError("day_of_year must be provided when fourier_K > 0")
        if fourier_K > 0 and len(day_of_year) != xy.shape[-2]:
            raise ValueError(f"day_of_year length ({len(day_of_year)}) must match time dimension ({xy.shape[-2]})")
        if fourier_K > 0 and fourier_pooling is None:
            raise ValueError("fourier_pooling must be specified when fourier_K > 0")
        if fourier_K == 0 and fourier_pooling is not None:
            raise ValueError("fourier_pooling should not be specified when fourier_K=0")

        # Validate intercept parameters
        if include_intercept and intercept_pooling is None:
            raise ValueError("intercept_pooling must be specified when include_intercept is True")
        if not include_intercept and intercept_pooling is not None:
            raise ValueError("intercept_pooling should not be specified when include_intercept is False")
        if intercept_mu_prior_scale is not None:
            self.intercept_mu_prior_scale = _per_variable(intercept_mu_prior_scale, 'intercept_mu_prior_scale',
                                                          xy.shape[-1])

        # set up batch shapes for parameter pooling
        # xy has shape batch_shape + (T, n_x + 1)
        batch_shape = xy.shape[:-2]

        if theta_pooling == 'unshared':
            # separate parameters per batch
            self.theta_batch_shape = batch_shape
        elif theta_pooling == 'shared':
            # no batches for theta; will broadcast to share across all batches
            self.theta_batch_shape = ()
        else:
            raise ValueError("theta_pooling must be 'unshared' or 'shared'")

        if sigma_pooling == 'unshared':
            # separate parameters per batch
            self.sigma_batch_shape = batch_shape
        elif sigma_pooling == 'shared':
            # no batches for sigma; will broadcast to share across all batches
            self.sigma_batch_shape = ()
        else:
            raise ValueError("sigma_pooling must be 'unshared' or 'shared'")

        if fourier_K > 0:
            if fourier_pooling == 'unshared':
                # separate parameters per batch
                self.fourier_batch_shape = batch_shape
            elif fourier_pooling == 'shared':
                # no batches for fourier; will broadcast to share across all batches
                self.fourier_batch_shape = ()
        else:
            # Not using Fourier terms, but set for consistency
            self.fourier_batch_shape = batch_shape

        # one df for all batches ('shared'), or one per batch ('unshared' and 'partial')
        self.innovation_df_batch_shape = () if innovation_df_pooling == 'shared' else batch_shape

        if intercept_pooling == 'shared':
            # no batches for intercept; will broadcast to share across all batches
            self.intercept_batch_shape = ()
        else:
            # separate parameters per batch, for both 'unshared' and 'partial'
            # (also set for consistency when not using an intercept)
            self.intercept_batch_shape = batch_shape

        # scale of each variable as modeled: power and centering constant
        n_vars = self.n_x + 1
        if transform not in TRANSFORM_POWERS:
            raise ValueError(f"transform must be one of {tuple(TRANSFORM_POWERS)}")
        if transform != 'none' and (input_power is not None or input_center is not None):
            raise ValueError("input_power and input_center describe data transformed before being "
                             "passed to SARIX and can only be used with transform='none'")
        power = _per_variable(TRANSFORM_POWERS[transform] if input_power is None else input_power, 'input_power',
                              n_vars)
        if onp.any(power < 0):
            raise ValueError("input_power must be non-negative")
        self.scale_power = onp.broadcast_to(power, (n_vars,))
        center_shape = batch_shape + (n_vars,)
        try:
            self.input_center = onp.broadcast_to(
                onp.asarray(0.0 if input_center is None else input_center, dtype=float), center_shape)
        except ValueError:
            raise ValueError(f"input_center must broadcast to shape {center_shape} (batch_shape + (n_x + 1,))")

        # calendar labels: integer array laid out like xy, batch_shape + (T + forecast_horizon, n_x + 1)
        if calendar_labels is None:
            if calendar_pooling is not None:
                raise ValueError("calendar_pooling should not be specified when calendar_labels is None")
            self.calendar_labels = None
            self.n_calendar = 0
        else:
            if calendar_pooling is None:
                raise ValueError("calendar_pooling must be specified when calendar_labels is given")
            labels = onp.asarray(calendar_labels)

            if not onp.issubdtype(labels.dtype, onp.integer):
                raise ValueError("calendar_labels must be integers")
            if labels.min() < 0:
                raise ValueError("calendar_labels must be non-negative")

            n_timepoints_total = xy.shape[-2] + forecast_horizon
            if labels.ndim < 2 or labels.shape[-2] != n_timepoints_total or labels.shape[-1] not in (1, n_vars):
                raise ValueError(
                    f"calendar_labels must have shape (..., T + forecast_horizon, 1) or "
                    f"(..., T + forecast_horizon, n_x + 1) = (..., {n_timepoints_total}, 1) or (..., {n_timepoints_total}, {n_vars}), "
                    f"but got shape {labels.shape}")

            try:
                labels = onp.broadcast_to(labels, batch_shape + (n_timepoints_total, n_vars))
            except ValueError:
                raise ValueError(f"calendar_labels must broadcast to shape {batch_shape + (n_timepoints_total, n_vars)}")

            self.n_calendar = int(labels.max())
            if self.n_calendar == 0:
                raise ValueError("calendar_labels has no non-zero labels")

            # variables with any non-zero label get effects; the others are left unchanged
            self.calendar_variables = onp.flatnonzero(labels.reshape(-1, n_vars).any(axis=0))
            adjusted = self.calendar_variables
            uses_center = adjusted[self.scale_power[adjusted] > 0]
            if not onp.all(onp.isfinite(self.input_center[..., uses_center])):
                raise ValueError("input_center must be finite for variables with calendar effects "
                                 f"and a non-zero input_power (variables {uses_center.tolist()})")
            # power and center of the adjusted variables, as used by _calendar_adjust. Centers
            # are only used where the power is > 0; set them to 0 elsewhere so that a NaN center
            # cannot reach the gradient through the unused branch of jnp.where
            self._adjusted_power = self.scale_power[adjusted]  # (n_adjusted,)
            self._adjusted_center = onp.where(self._adjusted_power > 0,
                                              self.input_center[..., adjusted], 0.0)  # batch_shape + (n_adjusted,)
            self.calendar_labels = labels
            self._warn_unobserved_calendar_labels(labels, xy.shape[-2], calendar_pooling)
        # separate effects per batch for 'unshared' and 'partial'
        self.calendar_batch_shape = () if calendar_pooling == 'shared' else batch_shape

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

        # with power > 0, each variable with calendar effects is a * u**power - center
        # for a positive u, so xy + center should not be negative
        if self.calendar_labels is not None:
            pos = self.calendar_variables[self._adjusted_power > 0]
            neg = (self.xy[..., pos] + self.input_center[..., None, pos] < 0).any(
                axis=tuple(range(self.xy.ndim - 1)))
            if neg.any():
                warnings.warn(
                    f"variables {pos[neg].tolist()} have calendar effects but take values below "
                    "-input_center, which cannot come from a positive quantity on the original scale; "
                    "check transform, input_power and input_center, as the effects may not be multipliers")

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
        # inverting differencing (and, with calendar effects, for adjusting
        # the series inside the model before differencing)
        self.xy_transformed = self.xy
        self.xy = diff(self.xy, self.d, self.D, self.season_period, pad_na=False)

        # set a data-based scale for the intercept_mu prior if not provided;
        # this uses the modeled scale, where the intercept is applied
        if self.intercept_pooling == 'partial' and self.intercept_mu_prior_scale is None:
            self.intercept_mu_prior_scale = self._default_intercept_mu_prior_scale(self.xy)

        # Trim Fourier features to match differenced data length
        # After differencing, we lose d + D*season_period time points
        if self.fourier_features is not None:
            n_diff = self.d + self.D * self.season_period
            self.fourier_features = self.fourier_features[n_diff:, :]

        # pre-calculate state update matrix; there is none when the model has
        # no autoregressive terms (p=0 and P=0). With calendar effects the
        # lagged values depend on the effects and are built inside the model.
        if self.max_lag > 0 and self.calendar_labels is None:
            self.update_X = self.state_update_X(self.xy[..., :self.max_lag, :],
                                                self.xy[..., self.max_lag:, :])
        else:
            self.update_X = None

        # do inference
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        self.run_inference(rng_key)

        # generate predictions
        self.predictions_modeled_scale = self.predict(rng_key_predict)

        # undo differencing (with calendar effects, of the adjusted series,
        # whose recent values depend on the sampled effects), then undo the
        # calendar adjustment at forecast time points with labels
        if self.calendar_labels is None:
            self.predictions = inv_diff(self.xy_transformed, self.predictions_modeled_scale,
                                        self.d, self.D, self.season_period)
        else:
            _, base, future_offsets = self._calendar_forecast_start()
            self.predictions = inv_diff(base, self.predictions_modeled_scale,
                                        self.d, self.D, self.season_period)
            self.predictions = onp.asarray(self._calendar_adjust(
                self.predictions, future_offsets, inverse=True))

        # undo transformation to get predictions on original scale
        if transform == "log":
            self.predictions = onp.exp(self.predictions)
        elif transform == "fourthrt":
            self.predictions = onp.maximum(0.0, self.predictions)**4
        elif transform == "sqrt":
            self.predictions = onp.maximum(0.0, self.predictions)**2


    @staticmethod
    def _default_intercept_mu_prior_scale(xy):
        """
        Data-based scale for the prior on intercept_mu: for each variable,
        2.5 times the root mean square of xy (on the modeled scale) across all
        batches and time points.

        Parameters
        ----------
        xy : ndarray
            Data on the modeled scale, with shape batch_shape + (T, n_x + 1)

        Returns
        -------
        ndarray of shape (n_x + 1,). Variables whose values are all zero get a
        scale of 1.0, since a scale of 0 is not a valid prior.
        """
        n_vars = xy.shape[-1]
        rms = onp.sqrt(onp.nanmean(onp.square(xy.reshape(-1, n_vars)), axis=0))
        return onp.where(rms > 0, 2.5 * rms, 1.0)


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
        self.num_divergences = int(mcmc.get_extra_fields()['diverging'].sum())


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


    def _warn_unobserved_calendar_labels(self, labels, T, pooling):
        """
        Warn about calendar effects that no data time point informs, so that
        they are drawn from the prior. There is one effect per variable with
        calendar effects and label 1, ..., K. An effect is uninformed if its
        label does not occur for that variable in time points 0, ..., T - 1:
        it never occurs, or occurs only in the forecast horizon. (Every data
        time point enters the likelihood, including the first
        d + D * season_period + max_lag, through the differences and lags of
        the first modeled step.) With 'unshared' pooling, each batch has its
        own effects, so this is checked per batch.
        labels has shape batch_shape + (T + forecast_horizon, n_x + 1).
        """
        K = self.n_calendar
        ks = onp.arange(1, K + 1)
        # whether each label occurs in the data time points, per batch: (n_batches, n_x + 1, K)
        lab = labels[..., :T, :].reshape((-1, T, labels.shape[-1]))
        observed = (lab[..., None] == ks).any(axis=1)
        # with 'shared' pooling, the batches share the effects, and with 'partial' pooling, each batch's effects
        # are informed through the common mean by any batch, so one batch with the label is enough
        if pooling in ('shared', 'partial'):
            observed = observed.any(axis=0, keepdims=True)
        n_batches = observed.shape[0]
        n_missing = (~observed).sum(axis=0)  # number of batches without the label: (n_x + 1, K)

        unobserved = []
        for j in self.calendar_variables:
            parts = [f"label {k}" + (f" (missing in {n_missing[j, k - 1]} of {n_batches} batches)"
                                     if pooling == 'unshared' else "")
                     for k in ks if n_missing[j, k - 1] > 0]
            if parts:
                unobserved.append(f"variable {j}: " + ", ".join(parts))
        if unobserved:
            warnings.warn(f"some calendar effects have no observations and are drawn from the prior (labels run "
                          f"from 1 to {K}; these never occur, or occur only in the forecast horizon): "
                          f"{'; '.join(unobserved)}")


    def _calendar_offsets(self, effect, n_sample_dims):
        """
        Map sampled calendar effects to time points.

        effect has shape sample_shape + self.calendar_batch_shape + (n_adjusted, K),
        with n_sample_dims leading sample dimensions, where
        n_adjusted = len(self.calendar_variables). Returns eta at every time
        point for each variable with calendar effects (0 where the label is 0),
        with shape sample_shape + batch_shape + (T + forecast_horizon, n_adjusted).
        """
        sample_shape = effect.shape[:n_sample_dims]
        labels = self.calendar_labels[..., self.calendar_variables]  # batch_shape + (T + h, n_adjusted)
        batch_shape = labels.shape[:-2]
        if self.calendar_pooling == 'shared':
            # no batch dimensions in effect; insert size-1 ones so it broadcasts over batches
            effect = effect.reshape(sample_shape + (1,) * len(batch_shape) + effect.shape[-2:])
        # full is a lookup table from label to effect: full[..., j, k] is the effect of label k
        # for adjusted variable j, with a 0 prepended in the first position of the last axis
        # for label 0 (no effect), so the labels can be used directly as indices.
        # Shape sample_shape + batch_shape + (n_adjusted, K + 1), with size-1 batch
        # dimensions under 'shared' pooling.
        full = jnp.concatenate([jnp.zeros(effect.shape[:-1] + (1,)), effect], axis=-1)
        # take_along_axis needs the same number of dimensions as full; it broadcasts the rest
        # after swapaxes + reshape, dimensions are broadcastable to sample_shape + batch_shape + (n_adjusted, T + h)
        labels = jnp.swapaxes(labels, -1, -2)  # batch_shape + (n_adjusted, T + h), to index full's last axis
        labels = labels.reshape((1,) * len(sample_shape) + labels.shape)
        # look up each time point's effect in full: eta[..., j, t] = full[..., j, labels[..., j, t]],
        # then swap back to time before variable: sample_shape + batch_shape + (T + h, n_adjusted)
        return jnp.swapaxes(jnp.take_along_axis(full, labels, axis=-1), -1, -2)


    def _calendar_adjust(self, xy, offsets, inverse=False):
        """
        Divide (or, with inverse=True, multiply) the original-scale quantity of
        each variable with calendar effects by exp(offsets) (exp(offsets / a)
        for power 0; see below), working on the modeled scale. xy has shape
        (..., T, n_x + 1) and offsets shape (..., T, n_adjusted); their leading
        dimensions broadcast. Variables without calendar effects are unchanged.
        """
        idx = self.calendar_variables
        y = xy[..., idx]
        sign = 1.0 if inverse else -1.0
        lam = jnp.asarray(self._adjusted_power)
        c = jnp.asarray(self._adjusted_center)[..., None, :]  # batch_shape + (1, n_adjusted), broadcasts over time

        # Write the modeled scale as y = a * u**lam - c (lam > 0) or y = a * log(u) - c (lam == 0),
        # where u is the positive original-scale quantity. We want z, the same transform of
        # u multiplied by the calendar effect (exp(sign * eta) with eta = offsets for lam > 0;
        # see below for lam == 0).
        # With sign = -1 (inverse=False) this removes the effects: y is observed, z is adjusted.
        # With sign = +1 (inverse=True) it restores them: the input is adjusted and the output is
        # on the observed scale. Applying one after the other gives back the input.
        # - lam > 0: y = a * u**lam - c, so
        #   z = a * [u * exp(sign * eta)]**lam - c
        #     = a * u**lam * exp(sign * eta * lam) - c
        #     = exp(sign * lam * eta) * (y + c) - c.
        #   The center c is needed but a is not.
        # - lam == 0: y = a * log(u) - c, so
        #   z = a * log(u * exp(sign * \tilde{eta})) - c
        #     = a * log(u) + a * sign * \tilde{eta} - c
        #     = y + a * sign * \tilde{eta}
        #   a is unknown, so eta is defined on this scale (eta = a * \tilde{eta}; see the class Notes):
        #       z = y + sign * eta,
        #   which needs neither a nor c.
        # The lam > 0 formula does not reduce to this one as lam -> 0 (it would give z = y),
        # so the two cases are computed separately.
        z = jnp.where(lam == 0, y + sign * offsets, jnp.exp(sign * lam * offsets) * (y + c) - c)
        out_shape = jnp.broadcast_shapes(xy.shape, z.shape[:-1] + xy.shape[-1:])
        return jnp.broadcast_to(xy, out_shape).at[..., idx].set(jnp.broadcast_to(z, out_shape[:-1] + (len(idx),)))


    def _calendar_forecast_start(self):
        """
        With calendar effects, the values the forecasts start from depend on
        the sampled effects. Only the last d + D * season_period + max_lag
        time points are needed. Returns, for each posterior sample:
        - lag_source: the adjusted, differenced series at the last max_lag
          time points, which the AR forecasts start from;
        - base: the adjusted (undifferenced, transformed-scale) series at the
          last d + D * season_period time points, for inverting differencing;
        - future_offsets: the effects at the forecast time points, with shape
          (num_samples,) + batch_shape + (forecast_horizon, n_adjusted).
        """
        offsets = self._calendar_offsets(self.samples['calendar_effect'], 1)
        T = self.xy_transformed.shape[-2]
        n_diff = self.d + self.D * self.season_period
        n_tail = n_diff + self.max_lag
        z_tail = self._calendar_adjust(jnp.asarray(self.xy_transformed[..., T - n_tail:, :]),
                                       offsets[..., T - n_tail:T, :])
        lag_source = diff(z_tail, self.d, self.D, self.season_period)
        base = onp.asarray(z_tail[..., n_tail - n_diff:, :])
        return lag_source, base, offsets[..., T:, :]


    def _calendar_model_data(self):
        """
        Sample the calendar effects and return the adjusted, differenced
        series and its lagged-value matrix (None if there are no AR terms).
        Adds the log Jacobian of the adjustment for the modeled time points.
        """
        effect_sd = numpyro.sample(
            "calendar_effect_sd",
            dist.HalfCauchy(self.calendar_effect_sd_prior_scale * jnp.ones(1))
        )
        effect_shape = self.calendar_batch_shape + (len(self.calendar_variables), self.n_calendar)
        if self.calendar_pooling == 'partial':
            # learned mean for each variable and label, shared across batches; each batch's effects are
            # shrunk toward it. Non-centred: effect = mu + sd * decentered with decentered ~ Normal(0, 1),
            # which avoids the funnel between the effects and effect_sd when effect_sd is near 0
            mu_scale = self.calendar_effect_mu_prior_scale[self.calendar_variables][:, None]
            effect_mu = numpyro.sample(
                "calendar_effect_mu",
                dist.Normal(loc=jnp.zeros(effect_shape[-2:]), scale=mu_scale)
            )
            decentered = numpyro.sample("calendar_effect_decentered", dist.Normal(jnp.zeros(effect_shape), 1.0))
            effect = numpyro.deterministic("calendar_effect", effect_mu + effect_sd * decentered)
        else:
            effect = numpyro.sample(
                "calendar_effect",
                dist.Normal(loc=jnp.zeros(effect_shape), scale=effect_sd)
            )
        T = self.xy_transformed.shape[-2]
        offsets = self._calendar_offsets(effect, 0)[..., :T, :]
        z = self._calendar_adjust(jnp.asarray(self.xy_transformed), offsets)

        # Jacobian of the calendar adjustment. The AR likelihood below is written for z, the
        # adjusted series, but the data are y, and z = z(y; eta) depends on the parameters eta.
        # The density of the data is therefore
        #     p(y | theta, eta) = p_AR(z(y; eta) | theta, eta) * |dz/dy|,
        # the usual change-of-variables formula. Without the |dz/dy| factor, the model could raise
        # its likelihood by choosing eta to shrink z towards -c, rather than by fitting the data.
        # The adjustment acts on each time point and variable separately, so the Jacobian is
        # diagonal and log |dz/dy| is a sum over elements of
        #     log(dz/dy) = log(exp(-lam_j * eta)) = -lam_j * eta    (lam_j > 0),
        #     log(dz/dy) = log(1) = 0                              (lam_j == 0, where z = y - eta),
        # and the second case is covered by the same expression because lam_j = 0.
        # Only the time points the likelihood scores count: the first d + D * season_period +
        # max_lag are conditioned on (they start the differencing and the AR lags), not modeled.
        # Differencing itself has Jacobian 1, so it adds nothing.
        # numpyro.factor adds an arbitrary term to the model's log density (here, log |dz/dy|),
        # alongside the log-likelihood and log-prior terms from numpyro.sample sites.
        n_skip = self.d + self.D * self.season_period + self.max_lag
        lam = jnp.asarray(self._adjusted_power)
        numpyro.factor("calendar_log_jacobian", -jnp.sum(lam * offsets[..., n_skip:, :]))

        xy = diff(z, self.d, self.D, self.season_period)
        if self.max_lag > 0:
            update_X = self.state_update_X(xy[..., :self.max_lag, :], xy[..., self.max_lag:, :])
        else:
            update_X = None
        return xy, update_X


    def model(self, xy):
        # with calendar effects, the modeled series depends on the effects
        if self.calendar_labels is not None:
            xy, update_X = self._calendar_model_data()
        else:
            update_X = self.update_X

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

        if self.max_lag > 0:
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
            step_means = jnp.matmul(update_X, A)
        else:
            # no autoregressive terms (p=0 and P=0); the predictive mean comes
            # only from the intercept and Fourier terms below, if present
            step_means = 0.0

        # Add intercepts if present
        if self.include_intercept:
            intercept_sd = numpyro.sample(
                "intercept_sd",
                dist.HalfCauchy(self.intercept_sd_prior_scale * jnp.ones(1))
            )
            intercept_loc = jnp.zeros(self.intercept_batch_shape + (self.n_x+1,))
            if self.intercept_pooling == 'partial':
                # learned mean for each variable, shared across batches;
                # intercepts are shrunk toward it rather than toward 0
                intercept_mu = numpyro.sample(
                    "intercept_mu",
                    dist.Normal(
                        loc=jnp.zeros((self.n_x+1,)),
                        scale=self.intercept_mu_prior_scale
                    )
                )
                intercept_loc = intercept_loc + intercept_mu
            # Shape: (intercept_batch_shape, n_x+1)
            intercept = numpyro.sample(
                "intercept",
                dist.Normal(loc=intercept_loc, scale=intercept_sd)
            )

            # Insert a time dimension so the intercept broadcasts over time
            step_means = step_means + jnp.expand_dims(intercept, -2)

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

            # Total predictive mean = AR terms + intercept + Fourier terms
            step_means = step_means + step_means_fourier

        # step innovations are (state - step_means),
        # with shape (batch_shape, T - self.max_lag, n_x + 1)
        step_innovations = xy[..., self.max_lag:, :] - step_means

        # sample innovations with chosen distribution
        if self.innovation_dist == 'normal':
            numpyro.sample(
                "step_innovations",
                dist.MultivariateNormal(
                    loc=jnp.zeros((self.n_x + 1,)), scale_tril=Sigma_chol),
                obs=step_innovations
            )
        else:
            # Student's t, standardized so that sigma is the innovation standard deviation.
            # df - 2 has shape innovation_df_batch_shape; with batch dimensions, insert a
            # dimension for time, as for Sigma_chol
            df_excess = self._sample_innovation_df()
            if len(xy.shape) > 2 and self.innovation_df_pooling != 'shared':
                df_excess = jnp.expand_dims(df_excess, -1)
            numpyro.sample(
                "step_innovations",
                dist.MultivariateStudentT(
                    df=2.0 + df_excess,
                    loc=jnp.zeros((self.n_x + 1,)),
                    scale_tril=_standardized_t_scale_tril(Sigma_chol, df_excess)),
                obs=step_innovations
            )


    def _sample_innovation_df(self):
        """
        Sample the degrees of freedom of t innovations and return
        innovation_df_excess = innovation_df - 2, with shape
        innovation_df_batch_shape. Both are recorded in the samples. Working
        with df - 2 avoids rounding it to 0 in float32 when df is close to 2.

        'unshared' and 'shared': df has the Gamma(2, 1 / innovation_df_prior_scale)
        prior truncated to df > 2 (Juarez and Steel, 2010), written as a
        positive parameter df - 2 plus the Gamma log density of df; the
        truncation's normalizing constant does not depend on parameters.
        'partial': log(df - 2) = mu + sd * decentered for each batch, with
        decentered ~ Normal(0, 1) (non-centred), mu ~ Normal with the mean and
        sd of log(df - 2) under that truncated Gamma prior, and
        sd ~ HalfCauchy(innovation_df_sd_prior_scale).
        """
        shape = self.innovation_df_batch_shape
        if self.innovation_df_pooling == 'partial':
            mu_loc, mu_scale = self.innovation_df_mu_prior
            mu = numpyro.sample("innovation_df_mu", dist.Normal(mu_loc, mu_scale))
            sd = numpyro.sample("innovation_df_sd", dist.HalfCauchy(self.innovation_df_sd_prior_scale))
            decentered = numpyro.sample("innovation_df_decentered", dist.Normal(jnp.zeros(shape), 1.0))
            df_excess = numpyro.deterministic("innovation_df_excess", jnp.exp(mu + sd * decentered))
        else:
            df_excess = numpyro.sample("innovation_df_excess",
                                       dist.ImproperUniform(constraints.positive, shape, ()))
            numpyro.factor("innovation_df_prior",
                           dist.Gamma(2.0, 1.0 / self.innovation_df_prior_scale).log_prob(2.0 + df_excess).sum())
        numpyro.deterministic("innovation_df", 2.0 + df_excess)
        return df_excess


    def predict(self, rng_key):
        '''
        Predict future values of all signals based on a single sample of
        parameter values from the posterior distribution.
        '''
        # load in parameter estimates and update to target batch size
        sigma = self.samples['sigma']
        xy_batch_shape = self.xy.shape[:-2]
        sigma_sample_batch_shape = sigma.shape[:-1] # `(num_samples,) + self.sigma_batch_shape`, dropping last dimension which is per variable (self.n_x + 1,)

        if self.max_lag > 0:
            theta = self.samples['theta']
            theta_sample_batch_shape = theta.shape[:-1] # `(num_samples,) + self.theta_batch_shape`, dropping last dimension which is per coefficient (n_theta,)

            if self.theta_pooling == 'shared':
                # goal is shape theta_sample_batch_shape + xy_batch_shape + theta.shape[-1]
                # first insert 1's corresponding to xy_batch_shape, then broadcast
                ones = (1,) * len(xy_batch_shape)
                theta = theta.reshape(theta_sample_batch_shape + ones + (theta.shape[-1],))
                target = theta_sample_batch_shape + xy_batch_shape + (theta.shape[-1],)
                theta = jnp.broadcast_to(theta, target)

            # state transition matrix
            A = self.make_state_transition_matrix(theta)

        if self.sigma_pooling == 'shared':
            # goal is shape sigma_sample_batch_shape + xy_batch_shape + sigma.shape[-1]
            # first insert 1's corresponding to xy_batch_shape, then broadcast
            ones = (1,) * len(xy_batch_shape)
            sigma = sigma.reshape(sigma_sample_batch_shape + ones + (sigma.shape[-1],))
            target = sigma_sample_batch_shape + xy_batch_shape + (sigma.shape[-1],)
            sigma = jnp.broadcast_to(sigma, target)

        # batch_shape is (num_samples,) + xy_batch_shape. It is taken from sigma
        # rather than theta because theta is not sampled when there are no
        # autoregressive terms (p=0 and P=0), while sigma always is. After the
        # pooling broadcasts above, both have this same leading shape.
        batch_shape = sigma.shape[:-1]

        # convert sigma to a batch of covariance matrix Cholesky factors
        Sigma_chol = jnp.expand_dims(sigma, -2) * jnp.eye(sigma.shape[-1])

        # with calendar effects, the recent (adjusted, differenced) values the
        # forecasts start from depend on the sampled effects
        if self.calendar_labels is not None:
            lag_source, _, _ = self._calendar_forecast_start()
        else:
            lag_source = self.xy

        # Load intercepts if present
        if self.include_intercept:
            intercept = self.samples['intercept']
            # intercept has shape (num_samples,) + self.intercept_batch_shape + (n_x+1,)
            intercept_sample_batch_shape = intercept.shape[:-1] # `(num_samples,) + self.intercept_batch_shape`, dropping last dimension which is per variable (self.n_x + 1,)

            # With 'unshared' and 'partial' pooling, there is already one
            # intercept per batch. For 'partial', the learned intercept_mu
            # enters the forecasts through these per-batch intercepts (it would
            # only be used directly to forecast a new batch, which predict()
            # does not do).

            if self.intercept_pooling == 'shared':
                # goal is shape intercept_sample_batch_shape + xy_batch_shape + (n_x+1,)
                # first insert 1's corresponding to xy_batch_shape, then broadcast
                ones = (1,) * len(xy_batch_shape)
                intercept = intercept.reshape(intercept_sample_batch_shape + ones + (intercept.shape[-1],))
                target = intercept_sample_batch_shape + xy_batch_shape + (intercept.shape[-1],)
                intercept = jnp.broadcast_to(intercept, target)

            # Expand to (..., 1, n_x+1) for addition; same for every horizon
            intercept_mean = jnp.expand_dims(intercept, -2)
        else:
            intercept_mean = 0

        # Load Fourier coefficients if present
        if self.fourier_K > 0:
            fourier_beta = self.samples['fourier_beta']
            # fourier_beta has shape (num_samples,) + self.fourier_batch_shape + (n_x+1, 2*K)
            # Need to match batch_shape
            fourier_sample_batch_shape = fourier_beta.shape[:-2] # `(num_samples,) + self.fourier_batch_shape`, dropping last dimensions which are per variable and coefficient

            if self.fourier_pooling == 'shared':
                # goal is shape fourier_sample_batch_shape + xy_batch_shape + (n_x+1, 2*K)
                # first insert 1's corresponding to xy_batch_shape, then broadcast
                ones = (1,) * len(xy_batch_shape)
                fourier_beta = fourier_beta.reshape(fourier_sample_batch_shape + ones + fourier_beta.shape[-2:])
                target = fourier_sample_batch_shape + xy_batch_shape + fourier_beta.shape[-2:]
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
        if self.innovation_dist == 'normal':
            innovations = dist.MultivariateNormal(
                    loc=jnp.zeros((self.n_x + 1,)),
                    scale_tril=Sigma_chol) \
                .sample(rng_key, sample_shape=(self.forecast_horizon, ))
        else:
            # df - 2 has shape (num_samples,) + innovation_df_batch_shape; broadcast to batch_shape
            df_excess = self.samples['innovation_df_excess']
            if self.innovation_df_pooling == 'shared':
                df_excess = df_excess.reshape(df_excess.shape + (1,) * len(xy_batch_shape))
            df_excess = jnp.broadcast_to(df_excess, batch_shape)
            innovations = dist.MultivariateStudentT(
                    df=2.0 + df_excess,
                    loc=jnp.zeros((self.n_x + 1,)),
                    scale_tril=_standardized_t_scale_tril(Sigma_chol, df_excess)) \
                .sample(rng_key, sample_shape=(self.forecast_horizon, ))

        # generate step-ahead forecasts iteratively
        y_pred = []
        dummy_values = jnp.zeros(batch_shape + (1, self.xy.shape[-1]))
        if self.max_lag > 0:
            recent_lags = jnp.broadcast_to(lag_source[..., -self.max_lag:, :],
                                           batch_shape + (self.max_lag, self.xy.shape[-1]))
        for h in range(self.forecast_horizon):
            # AR contribution
            if self.max_lag > 0:
                update_X = self.state_update_X(recent_lags, dummy_values)
                ar_mean = jnp.matmul(update_X, A)
            else:
                ar_mean = dummy_values

            # Add Fourier contribution if present
            if self.fourier_K > 0:
                # Compute Fourier mean for this forecast step
                # einsum: (..., n_x+1, 2K) x (2K,) -> (..., n_x+1)
                fourier_mean = jnp.einsum('...ik,k->...i', fourier_beta, F_forecast[h, :])
                # Expand to (..., 1, n_x+1) for addition
                fourier_mean = jnp.expand_dims(fourier_mean, -2)
            else:
                fourier_mean = 0

            new_y_pred = ar_mean + intercept_mean + fourier_mean + jnp.expand_dims(innovations[h, ...], -2)
            y_pred.append(new_y_pred)
            if self.max_lag > 0:
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
