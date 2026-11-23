"""Tests for the level-dependent innovation scale (sigma_level=True)."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from sarix.sarix import SARIX

FAST = dict(num_warmup=20, num_samples=20, num_chains=1)


def ar1_data(B=2, T=60, phi=0.7, sd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((B, T))
    for t in range(1, T):
        z[:, t] = phi * z[:, t - 1] + sd * rng.standard_normal(B)
    return z[..., None] + 5.0


def _trace(m, values):
    return handlers.trace(handlers.seed(handlers.substitute(m.model, data=values), 0)).get_trace(m.xy)


def _step_log_prob(tr):
    site = tr["step_innovations"]
    return site["fn"].log_prob(site["value"]).sum()


OFFSET = 0.7  # sigma_level_offset used in the tests
OFFSET_VALUES = {"sigma_level_offset": jnp.array([OFFSET])}  # trace value, one per variable


def _factor(coef, x):
    """The level factor g(x) = softplus(OFFSET + coef * x) / softplus(OFFSET)."""
    return np.logaddexp(0.0, OFFSET + coef * x) / np.logaddexp(0.0, OFFSET)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(sigma_level=True), "sigma_level_pooling must be specified"),
        (dict(sigma_level_pooling='shared'), "sigma_level_pooling should not be specified"),
        (dict(sigma_level=True, sigma_level_pooling='bogus'), "must be 'unshared', 'shared' or 'partial'"),
        (dict(sigma_level=True, sigma_level_pooling='none'), "must be 'unshared', 'shared' or 'partial'"),
        (dict(sigma_level=True, sigma_level_pooling='shared', d=1), "requires d=0 and D=0"),
        (dict(sigma_level=True, sigma_level_pooling='shared', p=0), "at least one autoregressive lag"),
        (dict(sigma_level=True, sigma_level_pooling='shared', sigma_level_prior_scale=0.0), "must be positive"),
        (dict(sigma_level=True, sigma_level_pooling='shared', sigma_level_prior_scale=np.nan), "must be positive"),
        (dict(sigma_level=True, sigma_level_pooling='shared', sigma_level_prior_scale=np.inf), "must be positive"),
        (dict(sigma_level=True, sigma_level_pooling='partial', sigma_level_sd_prior_scale=None),
         "sigma_level_sd_prior_scale must be positive"),
        (dict(sigma_level=True, sigma_level_pooling='partial', sigma_level_sd_prior_scale=np.nan),
         "sigma_level_sd_prior_scale must be positive"),
        (dict(sigma_level=True, sigma_level_pooling='shared', sigma_level_prior_scale=[1.0, 2.0]), "must be a scalar"),
        (dict(sigma_level=True, sigma_level_pooling='shared', sigma_level_offset_prior_scale=0.0),
         "sigma_level_offset_prior_scale must be positive"),
    ],
)
def test_validation(kwargs, match):
    args = dict(p=1) | kwargs
    with pytest.raises(ValueError, match=match):
        SARIX(ar1_data(T=30), **args, **FAST)


def test_unused_prior_scales_are_not_validated():
    """The sigma_level prior scales are not checked when they are not used, so None is accepted."""
    unused = dict(sigma_level_prior_scale=np.nan, sigma_level_sd_prior_scale=None,
                  sigma_level_offset_prior_scale=None)
    SARIX(ar1_data(T=30), p=1, **unused, **FAST)
    SARIX(ar1_data(T=30), p=1, sigma_level=True, sigma_level_pooling='shared',
          sigma_level_sd_prior_scale=None, **FAST)


def test_factor_is_finite_for_very_negative_offset():
    """For very negative offsets, softplus underflows in float32; the factor approaches exp(coef * x)."""
    for offset in [-20.0, -90.0, -120.0]:
        factor = SARIX._sigma_level_factor(1.0, 0.5, 0.0, jnp.float32(offset))
        np.testing.assert_allclose(factor, np.exp(0.5), rtol=1e-5)
    grad = jax.grad(lambda a: SARIX._sigma_level_factor(1.0, 0.5, 0.0, a))(-120.0)
    assert np.isfinite(grad)


def test_center_is_mean_of_series():
    xy = ar1_data(B=3, T=30)
    m = SARIX(xy, p=1, sigma_level=True, sigma_level_pooling='shared', **FAST)
    np.testing.assert_allclose(m.sigma_level_center, xy.mean(axis=-2))


def test_default_prior_scale_is_inverse_sd_of_level():
    """By default the prior scale is 1 / sd of the level about its center, per variable."""
    xy = ar1_data(B=3, T=30) * np.array([1.0, 100.0])  # two variables on very different scales
    m = SARIX(xy, p=1, sigma_level=True, sigma_level_pooling='shared', **FAST)
    expected = 1 / np.sqrt(np.mean((xy - xy.mean(axis=-2, keepdims=True)) ** 2, axis=(0, 1)))
    np.testing.assert_allclose(m.sigma_level_prior_scale, expected, rtol=1e-6)


@pytest.mark.parametrize("scale", [0.5, [0.5, 2.0]])
def test_given_prior_scale_is_used_as_is(scale):
    xy = ar1_data(B=2, T=30) * np.array([1.0, 100.0])
    m = SARIX(xy, p=1, sigma_level=True, sigma_level_pooling='partial', sigma_level_prior_scale=scale, **FAST)
    np.testing.assert_allclose(m.sigma_level_prior_scale, scale)


@pytest.mark.parametrize("pooling", ["unshared", "shared", "partial"])
def test_default_prior_scale_fits_large_modeled_scale(pooling):
    """Counts in the hundreds with transform='none' fit: the chain moves and forecasts are near the data."""
    xy = 300.0 + 100.0 * (ar1_data(B=2, T=60, sd=1.0) - 5.0)
    n = 50
    m = SARIX(xy, p=1, sigma_level=True, sigma_level_pooling=pooling, forecast_horizon=1, num_warmup=100,
              num_samples=n, num_chains=1)
    for site in ["sigma", "theta", "sigma_level_coef"]:
        values = np.asarray(m.samples[site]).reshape(n, -1)
        assert np.isfinite(values).all()
        assert len(np.unique(values, axis=0)) > n // 2, f"{site} does not move"
    median_h1 = np.median(m.predictions[:, :, 0, 0], axis=0)
    np.testing.assert_array_less(np.abs(median_h1 - xy[:, -1, 0]), 2 * xy.std())


def test_predict_warns_on_non_finite_paths():
    xy = ar1_data(B=2, T=30)
    xy[1, -1, 0] = xy[1, :, 0].mean() + 0.5  # second batch ends above its mean
    m = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='unshared', sigma_level=True,
              sigma_level_pooling='unshared', forecast_horizon=3, **FAST)
    m.samples = _fixed_samples(10, 2, 1.0)
    # above the center, an infinite coefficient gives an infinite innovation scale on one path
    m.samples["sigma_level_coef"] = m.samples["sigma_level_coef"].at[3, 1, 0].set(np.inf)
    with pytest.warns(UserWarning, match="1 of 20 forecast paths"):
        m.predict(jax.random.PRNGKey(0))


def test_fit_warns_on_paths_non_finite_only_on_original_scale(monkeypatch):
    """With transform='log', a path finite on the modeled scale can overflow in exp; fitting warns about it."""
    pred = np.zeros((10, 2, 3, 1))
    pred[3, 1, 2, 0] = 1000.0  # finite, but exp overflows
    monkeypatch.setattr(SARIX, "predict", lambda self, rng_key: pred)
    with pytest.warns(UserWarning, match="1 of 20 forecast paths .* finite on the modeled scale"):
        m = SARIX(ar1_data(B=2, T=30), p=1, transform='log', sigma_level=True, sigma_level_pooling='shared',
                  forecast_horizon=3, **FAST)
    assert np.isinf(m.predictions[3, 1, 2, 0])


@pytest.mark.parametrize("innovation_dist", ["normal", "t"])
def test_zero_coef_matches_constant_sigma(innovation_dist):
    """With sigma_level_coef = 0 the likelihood is that of the model without sigma_level."""
    xy = ar1_data(B=3, T=30)
    t_args = dict(innovation_dist='t', innovation_df_pooling='shared') if innovation_dist == 't' else {}
    base = dict(p=1, theta_pooling='shared', sigma_pooling='unshared', **t_args, **FAST)
    values = {"theta_sd": jnp.ones(1), "theta": jnp.array([0.6]), "sigma": jnp.array([[0.1], [0.2], [0.3]]),
              "innovation_df_excess": jnp.array(3.0)}
    m0 = SARIX(xy, **base)
    m1 = SARIX(xy, sigma_level=True, sigma_level_pooling='unshared', **base)
    lp0 = _step_log_prob(_trace(m0, values))
    lp1 = _step_log_prob(_trace(m1, values | {"sigma_level_coef": jnp.zeros((3, 1))}))
    np.testing.assert_allclose(lp1, lp0, rtol=1e-6)


@pytest.mark.parametrize("pooling", ["unshared", "shared"])
@pytest.mark.parametrize("innovation_dist", ["normal", "t"])
def test_likelihood_uses_level_dependent_sd(pooling, innovation_dist):
    """Each innovation has sd sigma * g(y[t-1] - center)."""
    xy = ar1_data(B=3, T=30)
    t_args = dict(innovation_dist='t', innovation_df_pooling='shared') if innovation_dist == 't' else {}
    m = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='unshared', sigma_level=True,
              sigma_level_pooling=pooling, **t_args, **FAST)
    theta, sigma, df = 0.6, jnp.array([[0.1], [0.2], [0.3]]), 4.0
    coef = jnp.array([[0.5], [1.0], [-2.0]]) if pooling == 'unshared' else jnp.array([1.5])
    tr = _trace(m, {"theta_sd": jnp.ones(1), "theta": jnp.array([theta]), "sigma": sigma,
                    "sigma_level_coef": coef, "innovation_df_excess": jnp.array(df - 2)} | OFFSET_VALUES)

    y = xy[..., 0]
    innov = y[:, 1:] - theta * y[:, :-1]
    coef_b = jnp.broadcast_to(coef, (3, 1))
    sd = sigma * _factor(np.asarray(coef_b), y[:, :-1] - y.mean(axis=1, keepdims=True))
    if innovation_dist == 'normal':
        expected = dist.Normal(0.0, sd).log_prob(innov).sum()
    else:
        expected = dist.StudentT(df, 0.0, sd * jnp.sqrt((df - 2) / df)).log_prob(innov).sum()
    np.testing.assert_allclose(_step_log_prob(tr), expected, rtol=1e-5)


def test_level_uses_calendar_adjusted_series():
    """With calendar effects, the level is the series with the effects divided out."""
    B, T, h = 2, 30, 2
    xy = ar1_data(B=B, T=T)
    lab = np.zeros((T + h, 1), dtype=int)
    lab[[5, 12, 19], 0] = 1
    m = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='shared', calendar_labels=lab,
              calendar_pooling='shared', input_power=0.0, sigma_level=True, sigma_level_pooling='shared',
              forecast_horizon=h, **FAST)
    effect, coef, theta, sigma = 0.3, 2.0, 0.6, 0.2
    tr = _trace(m, {"theta_sd": jnp.ones(1), "theta": jnp.array([theta]), "sigma": jnp.array([sigma]),
                    "sigma_level_coef": jnp.array([coef]), "calendar_effect_sd": jnp.ones(1),
                    "calendar_effect": jnp.array([[effect]])} | OFFSET_VALUES)
    # power 0: the adjusted series is y - effect at labelled time points
    z = xy[..., 0] - effect * (lab[:T, 0] == 1)
    innov = z[:, 1:] - theta * z[:, :-1]
    sd = sigma * _factor(coef, z[:, :-1] - xy[..., 0].mean(axis=1, keepdims=True))
    np.testing.assert_allclose(_step_log_prob(tr), dist.Normal(0.0, sd).log_prob(innov).sum(), rtol=1e-5)


def _fixed_samples(n, B, coef):
    s = {"theta_sd": jnp.ones((n, 1)), "theta": jnp.ones((n, 1)), "sigma": jnp.ones((n, B, 1))}
    if coef is not None:
        s["sigma_level_coef"] = jnp.full((n, B, 1), coef)
        s["sigma_level_offset"] = jnp.full((n, 1), OFFSET)
    return s


def test_predict_zero_coef_matches_constant_sigma():
    xy = ar1_data(B=2, T=30)
    base = dict(p=1, theta_pooling='shared', sigma_pooling='unshared', forecast_horizon=3, **FAST)
    m0 = SARIX(xy, **base)
    m1 = SARIX(xy, sigma_level=True, sigma_level_pooling='unshared', **base)
    m0.samples = _fixed_samples(500, 2, None)
    m1.samples = _fixed_samples(500, 2, 0.0)
    key = jax.random.PRNGKey(1)
    np.testing.assert_allclose(m1.predict(key), m0.predict(key), rtol=1e-6)


@pytest.mark.parametrize("pooling", ["unshared", "shared", "partial"])
def test_predict_scales_innovations_by_level(pooling):
    """One step ahead, the forecast sd is sigma * g(y[T-1] - center); later steps use simulated values."""
    xy = ar1_data(B=2, T=30)
    xy[1, -1, 0] = xy[1, :, 0].mean() + 0.5  # second batch ends above its mean
    m = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='unshared', sigma_level=True,
              sigma_level_pooling=pooling, forecast_horizon=2, **FAST)
    n, coef = 40000, 1.0
    m.samples = _fixed_samples(n, 2, coef)
    if pooling == 'shared':
        m.samples["sigma_level_coef"] = jnp.full((n, 1), coef)
    pred = m.predict(jax.random.PRNGKey(0))[..., 0]  # (n, B, h); theta = 1, so a random walk
    expected_sd = _factor(coef, xy[:, -1, 0] - m.sigma_level_center[:, 0])
    np.testing.assert_allclose(pred[:, :, 0].std(axis=0), expected_sd, rtol=0.03)
    # second step: the innovation scale follows the first step's simulated value
    step2 = pred[:, :, 1] - pred[:, :, 0]
    expected2 = _factor(coef, pred[:, :, 0] - m.sigma_level_center[:, 0])
    np.testing.assert_allclose((step2 / expected2).std(axis=0), 1.0, rtol=0.03)


@pytest.mark.parametrize("pooling", ["unshared", "shared"])
def test_coef_is_scaled_standard_normal(pooling):
    """sigma_level_coef = sigma_level_prior_scale * sigma_level_coef_z, with sigma_level_coef_z ~ Normal(0, 1)."""
    m = SARIX(ar1_data(B=3, T=30), p=1, theta_pooling='shared', sigma_level=True, sigma_level_pooling=pooling,
              sigma_level_prior_scale=0.3, **FAST)
    z = jnp.array([-1.0, 0.5, 2.0])[:, None] if pooling == 'unshared' else jnp.array([2.0])
    tr = _trace(m, {"sigma_level_coef_z": z})
    np.testing.assert_allclose(tr["sigma_level_coef"]["value"], 0.3 * z, rtol=1e-6)
    np.testing.assert_allclose(tr["sigma_level_coef_z"]["fn"].scale, 1.0)
    np.testing.assert_allclose(np.asarray(m.samples["sigma_level_coef"]),
                               0.3 * np.asarray(m.samples["sigma_level_coef_z"]), rtol=1e-6)


def test_partial_pooling_coef():
    scale, sd_scale = 0.3, 0.7
    m = SARIX(ar1_data(B=3, T=30), p=1, theta_pooling='shared', sigma_level=True, sigma_level_pooling='partial',
              sigma_level_prior_scale=scale, sigma_level_sd_prior_scale=sd_scale, **FAST)
    dec = jnp.array([[-1.0], [0.0], [2.0]])
    tr = _trace(m, {"sigma_level_mu_z": jnp.array([0.4]), "sigma_level_sd_z": jnp.array([0.5]),
                    "sigma_level_decentered": dec})
    np.testing.assert_allclose(tr["sigma_level_coef"]["value"], scale * (0.4 + 0.5 * dec), rtol=1e-6)
    np.testing.assert_allclose(tr["sigma_level_mu"]["value"], [scale * 0.4], rtol=1e-6)
    np.testing.assert_allclose(tr["sigma_level_sd"]["value"], [scale * 0.5], rtol=1e-6)
    # the spread's prior is in units of the prior scale
    np.testing.assert_allclose(tr["sigma_level_mu_z"]["fn"].scale, 1.0)
    np.testing.assert_allclose(tr["sigma_level_sd_z"]["fn"].scale, sd_scale)
    assert m.samples["sigma_level_coef"].shape == (FAST["num_samples"], 3, 1)
    assert m.samples["sigma_level_mu"].shape == (FAST["num_samples"], 1)
    assert m.samples["sigma_level_sd"].shape == (FAST["num_samples"], 1)


@pytest.mark.parametrize("pooling", ["unshared", "shared", "partial"])
def test_fit_with_fourier_calendar_and_t(pooling):
    B, T, h = 3, 60, 3
    lab = np.zeros((T + h, 1), dtype=int)
    lab[np.arange(5, T + h, 13), 0] = 1
    # 20 warmup draws can leave the chain near its start (df near 2), where forecast paths can overflow
    m = SARIX(ar1_data(B=B, T=T), p=2, theta_pooling='shared', sigma_pooling='unshared', fourier_K=2,
              fourier_pooling='shared', day_of_year=(np.arange(T) * 7) % 365 + 1, calendar_labels=lab,
              calendar_pooling='shared', innovation_dist='t', innovation_df_pooling='shared', sigma_level=True,
              sigma_level_pooling=pooling, forecast_horizon=h, **(FAST | dict(num_warmup=100)))
    coef_shape = (1,) if pooling == 'shared' else (B, 1)
    assert m.samples["sigma_level_coef"].shape == (FAST["num_samples"],) + coef_shape
    assert m.predictions.shape == (FAST["num_samples"], B, h, 1)
    assert np.isfinite(m.predictions).all()


def test_unbatched_fit():
    m = SARIX(ar1_data(B=1, T=40)[0], p=1, sigma_level=True, sigma_level_pooling='shared', forecast_horizon=2,
              **FAST)
    assert m.samples["sigma_level_coef"].shape == (FAST["num_samples"], 1)
    assert m.predictions.shape == (FAST["num_samples"], 2, 1)


def test_recovers_softplus_coefficient():
    """Data whose innovation sd grows linearly with the level give a clearly positive softplus coefficient."""
    rng = np.random.default_rng(0)
    B, T, phi, a, coef = 4, 200, 0.8, 1.0, 3.0
    y = np.zeros((B, T))
    for t in range(1, T):
        y[:, t] = phi * y[:, t - 1] + 0.1 * np.logaddexp(0.0, a + coef * y[:, t - 1]) * rng.standard_normal(B)
    assert np.abs(y).max() < 3
    m = SARIX(y[..., None], p=1, theta_pooling='shared', sigma_pooling='shared', sigma_level=True,
              sigma_level_pooling='shared', num_warmup=300, num_samples=300, num_chains=1)
    est = np.asarray(m.samples["sigma_level_coef"])[:, 0]
    assert np.quantile(est, 0.05) > 0.5
    assert m.samples["sigma_level_offset"].shape == (300, 1)


def test_forecasts_stay_finite_with_large_coef():
    """With a large coefficient, g grows linearly in the level, so forecasts stay finite over a long horizon."""
    xy = ar1_data(B=2, T=30)
    xy[:, -1, 0] += 0.5  # start above the center
    m = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='unshared', sigma_level=True,
              sigma_level_pooling='unshared', forecast_horizon=12, **FAST)
    m.samples = _fixed_samples(2000, 2, 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pred = m.predict(jax.random.PRNGKey(0))
    assert np.isfinite(pred).all()
