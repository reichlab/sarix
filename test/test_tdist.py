"""Tests for t-distributed innovations (innovation_dist='t')."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from sarix.sarix import SARIX, _log_df_excess_moments

FAST = dict(num_warmup=20, num_samples=20, num_chains=1)


def ar1_data(B=2, T=60, phi=0.7, sd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((B, T))
    for t in range(1, T):
        z[:, t] = phi * z[:, t - 1] + sd * rng.standard_t(3, size=B)
    return z[..., None] + 5.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(innovation_dist='cauchy'), "innovation_dist must be 'normal' or 't'"),
        (dict(innovation_dist='t'), "innovation_df_pooling must be specified"),
        (dict(innovation_df_pooling='shared'), "innovation_df_pooling should not be specified"),
        (dict(innovation_dist='t', innovation_df_pooling='bogus'), "must be 'unshared', 'shared' or 'partial'"),
        (dict(innovation_dist='t', innovation_df_pooling='shared', innovation_df_prior_scale=0.0), "must be positive"),
        (dict(innovation_dist='t', innovation_df_pooling='shared', innovation_df_prior_scale=float('nan')),
         "innovation_df_prior_scale must be positive and finite"),
        (dict(innovation_dist='t', innovation_df_pooling='shared', innovation_df_prior_scale=None),
         "innovation_df_prior_scale must be positive and finite"),
        (dict(innovation_dist='t', innovation_df_pooling='partial', innovation_df_sd_prior_scale=float('inf')),
         "innovation_df_sd_prior_scale must be positive and finite"),
    ],
)
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SARIX(ar1_data(T=30), p=1, **FAST, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(innovation_df_prior_scale=None, innovation_df_sd_prior_scale=None),  # normal innovations: unused
        dict(innovation_dist='t', innovation_df_pooling='shared', innovation_df_sd_prior_scale=float('nan')),
    ],
)
def test_unused_prior_scales_are_not_validated(kwargs):
    SARIX(ar1_data(T=30), p=1, **FAST, **kwargs)


@pytest.mark.parametrize(
    "innovation_dist, match",
    [("t", "must be 'unshared', 'shared' or 'partial'"), ("normal", "should not be specified")],
)
def test_df_pooling_rejects_deprecated_none_alias(innovation_dist, match):
    """innovation_df_pooling is new, so 'none' is an error, not a deprecated alias with a FutureWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with pytest.raises(ValueError, match=match):
            SARIX(ar1_data(T=30), p=1, innovation_dist=innovation_dist, innovation_df_pooling='none', **FAST)


def _trace(m, values):
    return handlers.trace(handlers.seed(handlers.substitute(m.model, data=values), 0)).get_trace(m.xy)


@pytest.mark.parametrize("pooling", ["unshared", "shared"])
def test_likelihood_is_standardized_student_t(pooling):
    """The innovations' log density is a sum of t log densities with sd sigma and each batch's df."""
    xy = ar1_data(B=3, T=30)
    m = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='unshared', innovation_dist='t',
              innovation_df_pooling=pooling, **FAST)
    theta, sigma = 0.6, jnp.array([[0.1], [0.2], [0.3]])
    df = jnp.array([2.5, 5.0, 30.0]) if pooling == 'unshared' else jnp.array(4.0)
    tr = _trace(m, {"theta_sd": jnp.ones(1), "theta": jnp.array([theta]), "sigma": sigma,
                    "innovation_df_excess": df - 2})
    np.testing.assert_allclose(tr["innovation_df"]["value"], df, rtol=1e-6)
    site = tr["step_innovations"]
    actual = site["fn"].log_prob(site["value"]).sum()

    innov = xy[:, 1:, 0] - theta * xy[:, :-1, 0]
    df_b = jnp.broadcast_to(df, (3,))[:, None]
    expected = dist.StudentT(df_b, 0.0, sigma * jnp.sqrt((df_b - 2) / df_b)).log_prob(innov).sum()
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # prior on df: Gamma(2, 1 / innovation_df_prior_scale) density (truncation constant omitted)
    np.testing.assert_allclose(tr["innovation_df_prior"]["fn"].log_prob(tr["innovation_df_prior"]["value"]).sum(),
                               dist.Gamma(2.0, 0.1).log_prob(df).sum(), rtol=1e-6)


@pytest.mark.parametrize("pooling", ["unshared", "partial"])
def test_likelihood_finite_when_df_rounds_to_2(pooling):
    """df - 2 = exp(-20) is lost when df is formed in float32, but the likelihood stays finite."""
    m = SARIX(ar1_data(B=2, T=30), p=1, theta_pooling='shared', innovation_dist='t',
              innovation_df_pooling=pooling, **FAST)
    excess = jnp.full((2,), jnp.exp(-20.0))
    values = {"theta_sd": jnp.ones(1), "theta": jnp.array([0.6]), "sigma": jnp.full((2, 1), 0.1)}
    if pooling == 'partial':
        values.update(innovation_df_mu=-20.0, innovation_df_sd=0.5, innovation_df_decentered=jnp.zeros(2))
    else:
        values["innovation_df_excess"] = excess
    tr = _trace(m, values)
    assert (tr["innovation_df"]["value"] == 2.0).all()
    np.testing.assert_allclose(tr["innovation_df_excess"]["value"], excess, rtol=1e-5)
    site = tr["step_innovations"]
    assert np.isfinite(site["fn"].log_prob(site["value"]).sum())


def test_partial_pooling_df():
    m = SARIX(ar1_data(B=3, T=30), p=1, theta_pooling='shared', innovation_dist='t',
              innovation_df_pooling='partial', **FAST)
    mu_loc, mu_scale = m.innovation_df_mu_prior
    tr = _trace(m, {"innovation_df_mu": 1.0, "innovation_df_sd": 0.5,
                    "innovation_df_decentered": jnp.array([-1.0, 0.0, 2.0])})
    np.testing.assert_allclose(tr["innovation_df"]["value"], 2.0 + np.exp(1.0 + 0.5 * np.array([-1.0, 0.0, 2.0])),
                               rtol=1e-6)
    np.testing.assert_allclose(tr["innovation_df_excess"]["value"], np.exp(1.0 + 0.5 * np.array([-1.0, 0.0, 2.0])),
                               rtol=1e-6)
    assert tr["innovation_df_mu"]["fn"].loc == mu_loc and tr["innovation_df_mu"]["fn"].scale == mu_scale
    assert m.samples["innovation_df"].shape == (FAST["num_samples"], 3)
    assert (m.samples["innovation_df"] > 2).all()


def test_log_df_excess_moments():
    """Moments of log(df - 2) under the truncated Gamma prior match Monte Carlo."""
    rng = np.random.default_rng(0)
    df = rng.gamma(2.0, 10.0, 2_000_000)
    u = np.log(df[df > 2] - 2)
    np.testing.assert_allclose(_log_df_excess_moments(10.0), (u.mean(), u.std()), atol=0.005)


@pytest.mark.parametrize("pooling", ["unshared", "shared", "partial"])
def test_predict_uses_each_batch_df_and_sigma_is_sd(pooling):
    """Forecast innovations have sd sigma, with heavier tails for a batch with small df."""
    m = SARIX(ar1_data(B=2, T=30), p=1, theta_pooling='shared', sigma_pooling='unshared', innovation_dist='t',
              innovation_df_pooling=pooling, forecast_horizon=1, **FAST)
    n = 20000
    df = jnp.tile(jnp.array([2.5, 1e6]), (n, 1)) if pooling != 'shared' else jnp.full((n,), 3.0)
    m.samples = {"theta_sd": jnp.ones((n, 1)), "theta": jnp.zeros((n, 1)), "sigma": jnp.ones((n, 2, 1)),
                 "innovation_df_excess": df - 2}
    pred = m.predict(jax.random.PRNGKey(0))[..., 0, 0]
    assert pred.shape == (n, 2)
    # standardized: the interquartile range is narrower than the normal's 1.349 when df is small
    iqr = np.subtract(*np.percentile(pred, [75, 25], axis=0))
    if pooling == 'shared':
        np.testing.assert_allclose(iqr[0], iqr[1], rtol=0.05)
        assert iqr[0] < 1.2
    else:
        assert iqr[0] < 0.9
        np.testing.assert_allclose(iqr[1], 1.349, rtol=0.05)


@pytest.mark.parametrize("pooling", ["unshared", "shared", "partial"])
def test_fit_with_fourier_and_calendar_effects(pooling):
    B, T, h = 3, 60, 3
    lab = np.zeros((T + h, 1), dtype=int)
    lab[np.arange(5, T + h, 13), 0] = 1
    m = SARIX(ar1_data(B=B, T=T), p=2, theta_pooling='shared', sigma_pooling='unshared', innovation_dist='t',
              innovation_df_pooling=pooling, fourier_K=2, fourier_pooling='shared',
              day_of_year=(np.arange(T) * 7) % 365 + 1, calendar_labels=lab, calendar_pooling='shared',
              forecast_horizon=h, **FAST)
    df_shape = () if pooling == 'shared' else (B,)
    assert m.samples["innovation_df"].shape == (FAST["num_samples"],) + df_shape
    np.testing.assert_allclose(m.samples["innovation_df"], 2.0 + m.samples["innovation_df_excess"], rtol=1e-6)
    assert m.predictions.shape == (FAST["num_samples"], B, h, 1)
    assert np.isfinite(m.predictions).all()


@pytest.mark.parametrize("pooling", ["unshared", "shared", "partial"])
def test_unbatched_fit(pooling):
    m = SARIX(ar1_data(B=1, T=40)[0], p=1, innovation_dist='t', innovation_df_pooling=pooling, forecast_horizon=2,
              **FAST)
    assert m.samples["innovation_df"].shape == (FAST["num_samples"],)
    assert m.predictions.shape == (FAST["num_samples"], 2, 1)
