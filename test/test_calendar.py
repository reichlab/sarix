"""Tests for calendar (e.g. holiday) effects and the input_power / input_center scale description."""

import warnings

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density

from sarix.sarix import SARIX

FAST = dict(num_warmup=20, num_samples=20, num_chains=1)


def ar1_data(B=2, T=60, phi=0.7, sd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((B, T))
    for t in range(1, T):
        z[:, t] = phi * z[:, t - 1] + sd * rng.normal(size=B)
    return z


def labels(T, h, period=13):
    lab = np.zeros(T + h, dtype=int)
    lab[np.arange(5, T + h, period)] = 1
    lab[np.arange(6, T + h, period)] = 2
    return lab


def cal(lab, n_vars=1):
    """Labels laid out like xy: the target's labels in the last column, 0 for exogenous variables."""
    out = np.zeros(np.shape(lab) + (n_vars,), dtype=np.asarray(lab).dtype)
    out[..., -1] = lab
    return out


# ---------------------------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(calendar_pooling='shared'), "calendar_pooling should not be specified"),
        (dict(calendar_labels=cal(labels(30, 1))), "calendar_pooling must be specified"),
        (dict(calendar_labels=cal(labels(30, 2)), calendar_pooling='shared'), r"must have shape \(\.\.\., T"),
        (dict(calendar_labels=labels(30, 1), calendar_pooling='shared'), r"must have shape \(\.\.\., T"),
        (dict(calendar_labels=cal(labels(30, 1), 3), calendar_pooling='shared'), r"must have shape \(\.\.\., T"),
        (dict(calendar_labels=-cal(labels(30, 1)), calendar_pooling='shared'), "non-negative"),
        (dict(calendar_labels=cal(labels(30, 1)).astype(float), calendar_pooling='shared'), "integers"),
        (dict(calendar_labels=np.zeros((31, 1), dtype=int), calendar_pooling='shared'), "no non-zero"),
        (dict(calendar_labels=np.zeros((3, 31, 1), dtype=int), calendar_pooling='shared'), "must broadcast"),
        (dict(calendar_labels=cal(labels(30, 1)), calendar_pooling='bogus'),
         "must be 'unshared', 'shared' or 'partial'"),
        (dict(calendar_effect_mu_prior_scale=np.ones(3)), r"calendar_effect_mu_prior_scale must be a scalar"),
        (dict(calendar_effect_mu_prior_scale=0.0), "calendar_effect_mu_prior_scale must be positive"),
        (dict(transform='sqrt', input_power=0.5), "only be used with transform='none'"),
        (dict(transform='log', input_center=1.0), "only be used with transform='none'"),
        (dict(input_power=-1.0), "non-negative"),
        (dict(input_power=np.ones(2)), r"input_power must be a scalar or have shape \(n_x \+ 1,\)"),
        (dict(input_center=np.ones(3)), "input_center must broadcast"),
        (dict(input_center=np.array([[1.0], [np.nan]]), calendar_labels=cal(labels(30, 1)),
              calendar_pooling='shared'), r"input_center must be finite .* \(variables \[0\]\)"),
        (dict(transform='cuberoot'), "transform must be one of"),
    ],
)
def test_validation(kwargs, match):
    xy = ar1_data(B=2, T=30)[..., None] + 5.0
    with pytest.raises(ValueError, match=match):
        SARIX(xy, p=1, forecast_horizon=1, **FAST, **kwargs)


def _fit_with_labels(lab, pooling='shared', T=30, h=2):
    xy = ar1_data(B=2, T=T)[..., None] + 5.0
    return SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='shared', calendar_labels=cal(lab),
                 calendar_pooling=pooling, forecast_horizon=h, **FAST)


def test_warns_on_label_gaps():
    lab = np.zeros(32, dtype=int)
    lab[[5, 10, 15]] = [1, 2, 4]
    with pytest.warns(UserWarning, match=r"no observations .*labels run from 1 to 4.*: variable 0: label 3$"):
        m = _fit_with_labels(lab)
    assert m.samples['calendar_effect'].shape == (20, 1, 4)


@pytest.mark.parametrize("pooling", ["shared", "partial"])
def test_warns_on_labels_only_in_forecast_horizon(pooling):
    lab = np.zeros(32, dtype=int)
    lab[[5, 30]] = [1, 2]
    # no per-batch counts unless the batches have separate effects ('unshared')
    with pytest.warns(UserWarning, match=r"only in the forecast horizon\): variable 0: label 2$"):
        _fit_with_labels(lab, pooling=pooling)


@pytest.mark.parametrize("pooling", ["shared", "partial"])
def test_no_warning_for_labels_in_first_time_points(pooling):
    """The first d + D * season_period + max_lag time points enter the likelihood through the differences
    and lags of the first modeled step, so their effects are informed by the data."""
    lab = np.zeros(32, dtype=int)
    lab[[5, 0]] = [1, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _fit_with_labels(lab, pooling=pooling)


@pytest.mark.parametrize("batch_1_where", [31, None])  # forecast horizon; absent
def test_unobserved_labels_checked_per_batch_when_unshared(batch_1_where):
    lab = np.zeros((2, 32), dtype=int)
    lab[:, 5] = 1
    lab[0, 10] = 2  # label 2 observed for batch 0 only
    if batch_1_where is not None:
        lab[1, batch_1_where] = 2
    with pytest.warns(UserWarning, match=r"variable 0: label 2 \(missing in 1 of 2 batches\)$"):
        _fit_with_labels(lab, pooling='unshared')
    for pooling in ('shared', 'partial'):  # informed by batch 0 (directly, or through the common mean)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _fit_with_labels(lab, pooling=pooling)


def xy2(B=2, T=30):
    """Two variables (one exogenous, then the target), positive."""
    return np.stack([ar1_data(B=B, T=T, seed=1), ar1_data(B=B, T=T, seed=2)], -1) + 5.0


def test_trailing_axis_of_one_gives_effects_to_all_variables():
    T, h = 30, 2
    m = SARIX(xy2(T=T), p=1, theta_pooling='shared', sigma_pooling='shared', calendar_labels=cal(labels(T, h)),
              calendar_pooling='shared', forecast_horizon=h, **FAST)
    assert m.calendar_variables.tolist() == [0, 1]
    assert m.samples['calendar_effect'].shape == (20, 2, 2)


def test_only_variables_with_labels_get_effects():
    T, h = 30, 2
    m = SARIX(xy2(T=T), p=1, theta_pooling='shared', sigma_pooling='shared',
              calendar_labels=cal(labels(T, h), 2), calendar_pooling='unshared', forecast_horizon=h, **FAST)
    assert m.calendar_variables.tolist() == [1]
    assert m.samples['calendar_effect'].shape == (20, 2, 1, 2)


def test_label_gaps_checked_per_variable():
    T, h = 30, 2
    lab = np.zeros((T + h, 2), dtype=int)
    lab[5, :] = 1
    lab[10, 1] = 2  # label 2 only for the target
    with pytest.warns(UserWarning, match=r"no observations .*: variable 0: label 2$") as rec:
        SARIX(xy2(T=T), p=1, theta_pooling='shared', sigma_pooling='shared', calendar_labels=lab,
              calendar_pooling='shared', forecast_horizon=h, **FAST)
    assert not any("variable 1" in str(w.message) for w in rec)


def test_label_shapes_equivalent():
    """(T + h, 1), batch_shape + (T + h, 1) and, with one variable, the same thing."""
    T, h = 30, 2
    lab = labels(T, h)
    a = _fit_with_labels(lab, T=T, h=h)
    xy = ar1_data(B=2, T=T)[..., None] + 5.0
    b = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='shared', calendar_labels=np.stack([cal(lab)] * 2),
              calendar_pooling='shared', forecast_horizon=h, **FAST)
    np.testing.assert_array_equal(a.calendar_labels, b.calendar_labels)
    assert b.calendar_labels.shape == (2, T + h, 1)


def test_no_calendar_no_new_sample_sites():
    xy = ar1_data(B=2, T=30)[..., None] + 5.0
    m = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='shared', forecast_horizon=1, **FAST)
    assert set(m.samples) == {'sigma', 'theta_sd', 'theta'}


# ---------------------------------------------------------------------------------------------
# likelihood: adjusted series, differencing, Jacobian
# ---------------------------------------------------------------------------------------------


def manual_log_density(x, lab, params, lam, c, d, p=1):
    """log p(x) written out directly: AR(1) (shared theta) on the differenced adjusted target."""
    T = x.shape[-1]
    eta = np.concatenate([[0.0], np.asarray(params['calendar_effect']).ravel()])[lab[:T]]  # (T,) shared
    if lam == 0:
        z = x - eta
    else:
        z = np.exp(-lam * eta) * (x + c[:, None]) - c[:, None]
    w = np.diff(z, n=d, axis=-1)
    theta = float(params['theta'][0])
    innov = w[:, p:] - theta * w[:, :-p]
    sigma = params['sigma']  # (B, 1) unshared
    lp = dist.Normal(0.0, sigma).log_prob(innov).sum()
    lp += dist.HalfCauchy(1.0).log_prob(sigma).sum()
    lp += dist.HalfCauchy(1.0).log_prob(params['theta_sd']).sum()
    lp += dist.Normal(0.0, params['theta_sd']).log_prob(params['theta']).sum()
    lp += dist.HalfCauchy(1.0).log_prob(params['calendar_effect_sd']).sum()
    lp += dist.Normal(0.0, params['calendar_effect_sd']).log_prob(params['calendar_effect']).sum()
    n_skip = d + p
    lp += -lam * eta[n_skip:].sum() * x.shape[0]  # Jacobian, per batch
    return float(lp)


@pytest.mark.parametrize("transform, input_power, center, d", [
    ('none', 0.25, np.array([1.0, 2.0]), 0),
    ('none', 0.25, np.array([1.0, 2.0]), 1),
    ('none', None, np.array([0.0, 0.0]), 1),
    ('log', None, np.array([0.0, 0.0]), 1),
])
def test_log_density_matches_manual(transform, input_power, center, d):
    T, h = 40, 2
    x = ar1_data(B=2, T=T) + (3.0 if transform == 'log' else 0.5)
    lab = labels(T, h)
    kw = dict(input_power=input_power, input_center=center[:, None]) if transform == 'none' else {}
    m = SARIX(x[..., None], p=1, d=d, transform=transform, theta_pooling='shared', sigma_pooling='unshared',
              calendar_labels=cal(lab), calendar_pooling='shared', forecast_horizon=h, **FAST, **kw)
    params = {'sigma': jnp.array([[0.2], [0.3]]), 'theta_sd': jnp.array([0.8]), 'theta': jnp.array([0.5]),
              'calendar_effect_sd': jnp.array([0.6]), 'calendar_effect': jnp.array([[0.3, -0.2]])}
    got, _ = log_density(m.model, (m.xy,), {}, params)
    xm = np.log(x) if transform == 'log' else x
    lam = 0.0 if transform == 'log' else (1.0 if input_power is None else input_power)
    np.testing.assert_allclose(float(got), manual_log_density(xm, lab, params, lam, center, d), rtol=1e-5)


def two_variable_setup(T, h):
    """Exogenous variable on the log scale (power 0, centre unknown) and target on a fourth-root, centred
    scale, with different labels for the two variables."""
    x = xy2(T=T)
    lab = np.stack([labels(T, h, period=13), labels(T, h, period=7)], -1)  # (T + h, 2)
    power, center = np.array([0.0, 0.25]), np.array([[np.nan, 1.0], [np.nan, 2.0]])
    return x, lab, power, center


def adjust_two_variables(x, eta, lab, power, center, inverse=False):
    """Manual adjustment; eta has shape (..., B, 2, K) and lab (n, 2) for the n time points of x."""
    sign = 1.0 if inverse else -1.0
    full = np.concatenate([np.zeros(eta.shape[:-1] + (1,)), eta], -1)
    e0, e1 = full[..., 0, lab[:, 0]], full[..., 1, lab[:, 1]]  # (..., B, n)
    c = center[:, 1:]
    z1 = np.exp(sign * power[1] * e1) * (x[..., 1] + c) - c
    return np.stack(np.broadcast_arrays(x[..., 0] + sign * e0, z1), -1), e1


def test_log_density_two_variables():
    """With effects on both variables: log density = that of a model without calendar effects fitted to the
    manually adjusted series + the effects' prior + the Jacobian (from the target only, as power 0 has none)."""
    T, h, d = 40, 2, 1
    x, lab, power, center = two_variable_setup(T, h)
    common = dict(p=1, d=d, theta_pooling='shared', sigma_pooling='unshared', forecast_horizon=h, **FAST)
    m = SARIX(x, input_power=power, input_center=center, calendar_labels=lab, calendar_pooling='unshared',
              **common)
    assert m.calendar_variables.tolist() == [0, 1]
    eta = np.array([[[0.3, -0.2], [0.5, 0.1]], [[-0.1, 0.2], [0.4, -0.3]]])  # (B, 2 variables, K)
    z, e1 = adjust_two_variables(x, eta, lab[:T], power, center)
    plain = SARIX(z, **common)
    params = {k: v[0] for k, v in plain.samples.items()}
    expect, _ = log_density(plain.model, (plain.xy,), {}, params)
    params.update(calendar_effect_sd=jnp.array([0.6]), calendar_effect=jnp.asarray(eta))
    got, _ = log_density(m.model, (m.xy,), {}, params)
    expect = float(expect) + float(dist.HalfCauchy(1.0).log_prob(0.6))
    expect += float(dist.Normal(0.0, 0.6).log_prob(eta).sum()) - power[1] * e1[:, d + 1:].sum()
    np.testing.assert_allclose(float(got), expect, rtol=1e-5)


def test_log_density_partial_pooling():
    """Non-centred partial pooling: effect = mu + sd * decentered. Log density = that of a model without
    calendar effects on the adjusted series + priors on sd, mu (per-variable scale) and decentered + Jacobian."""
    T, h, d = 40, 2, 1
    x, lab, power, center = two_variable_setup(T, h)
    mu_scale = np.array([0.5, 2.0])
    common = dict(p=1, d=d, theta_pooling='shared', sigma_pooling='unshared', forecast_horizon=h, **FAST)
    m = SARIX(x, input_power=power, input_center=center, calendar_labels=lab, calendar_pooling='partial',
              calendar_effect_mu_prior_scale=mu_scale, **common)
    assert {'calendar_effect_mu', 'calendar_effect_decentered', 'calendar_effect'} <= set(m.samples)
    assert m.samples['calendar_effect'].shape == (20, 2, 2, 2)
    assert m.samples['calendar_effect_mu'].shape == (20, 2, 2)
    sd = 0.6
    mu = np.array([[0.3, -0.2], [0.5, 0.1]])  # (variable, label)
    dec = np.array([[[0.5, -1.0], [0.2, 0.3]], [[-0.4, 1.2], [0.0, -0.7]]])  # (B, variable, label)
    eta = mu + sd * dec
    z, e1 = adjust_two_variables(x, eta, lab[:T], power, center)
    plain = SARIX(z, **common)
    params = {k: v[0] for k, v in plain.samples.items()}
    expect, _ = log_density(plain.model, (plain.xy,), {}, params)
    params.update(calendar_effect_sd=jnp.array([sd]), calendar_effect_mu=jnp.asarray(mu),
                  calendar_effect_decentered=jnp.asarray(dec))
    got, _ = log_density(m.model, (m.xy,), {}, params)
    expect = float(expect) + float(dist.HalfCauchy(1.0).log_prob(sd))
    expect += float(dist.Normal(0.0, mu_scale[:, None]).log_prob(mu).sum())
    expect += float(dist.Normal(0.0, 1.0).log_prob(dec).sum()) - power[1] * e1[:, d + 1:].sum()
    np.testing.assert_allclose(float(got), expect, rtol=1e-5)


# ---------------------------------------------------------------------------------------------
# forecasts
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0, 1])
def test_forecasts_mapped_back(d):
    """predictions = inverse adjustment of (differencing inverted from the adjusted series)."""
    T, h = 40, 6
    lam, c = 0.25, np.array([1.0, 2.0])
    x = ar1_data(B=2, T=T)
    lab = labels(T, h, period=7)
    assert lab[T:].any() and (lab[T:] == 0).any()
    m = SARIX(x[..., None], p=1, d=d, theta_pooling='shared', sigma_pooling='shared', input_power=lam,
              input_center=c[:, None], calendar_labels=cal(lab), calendar_pooling='unshared', forecast_horizon=h, **FAST)
    eta = np.asarray(m.samples['calendar_effect'])[..., 0, :]  # (S, B, K): the target, the only variable
    assert eta.shape == (20, 2, 2)
    full = np.concatenate([np.zeros(eta.shape[:-1] + (1,)), eta], -1)  # (S, B, K + 1)
    e_hist = full[..., lab[:T]]  # (S, B, T)
    e_fut = full[..., lab[T:]]  # (S, B, h)
    z = np.exp(-lam * e_hist) * (x + c[:, None]) - c[:, None]
    zp = m.predictions_modeled_scale[..., 0]  # (S, B, h)
    if d == 1:
        zp = z[..., -1:] + np.cumsum(zp, -1)
    expect = np.exp(lam * e_fut) * (zp + c[:, None]) - c[:, None]
    np.testing.assert_allclose(m.predictions[..., 0], expect, rtol=1e-5, atol=1e-6)


def test_forecasts_mapped_back_two_variables():
    T, h, d = 40, 6, 1
    x, lab, power, center = two_variable_setup(T, h)
    assert lab[T:].any(axis=0).all()
    m = SARIX(x, p=1, d=d, theta_pooling='shared', sigma_pooling='shared', input_power=power, input_center=center,
              calendar_labels=lab, calendar_pooling='shared', forecast_horizon=h, **FAST)
    eta = np.asarray(m.samples['calendar_effect'])[:, None]  # (S, 1, 2, K): shared across batches
    z, _ = adjust_two_variables(x, eta, lab[:T], power, center)
    zp = z[..., -1:, :] + np.cumsum(m.predictions_modeled_scale, -2)
    expect, _ = adjust_two_variables(zp, eta, lab[T:], power, center, inverse=True)
    np.testing.assert_allclose(m.predictions, expect, rtol=1e-5, atol=1e-6)


def test_equivalent_to_sarix_fourthrt_transform():
    """Calendar effects on data fourth-rooted by SARIX == on data fourth-rooted beforehand with input_power."""
    T, h = 40, 3
    y = (ar1_data(B=2, T=T) + 2.0) ** 4
    lab = labels(T, h)
    common = dict(p=1, theta_pooling='shared', sigma_pooling='shared', calendar_labels=cal(lab),
                  calendar_pooling='shared', forecast_horizon=h, **FAST)  # fmt: skip
    a = SARIX(y[..., None], transform='fourthrt', **common)
    b = SARIX(y[..., None] ** 0.25, transform='none', input_power=0.25, **common)
    np.testing.assert_allclose(a.samples['calendar_effect'], b.samples['calendar_effect'], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(a.predictions, np.maximum(b.predictions, 0.0) ** 4, rtol=1e-3)


def test_unshared_and_no_batch_shapes():
    T, h = 40, 2
    x = ar1_data(B=1, T=T)[0]  # (T,): no batch dimension
    m = SARIX(x[:, None] + 1.0, p=1, theta_pooling='shared', sigma_pooling='shared',
              calendar_labels=cal(labels(T, h)), calendar_pooling='shared', forecast_horizon=h, **FAST)
    assert m.samples['calendar_effect'].shape == (20, 1, 2)
    assert m.predictions.shape == (20, h, 1)


def test_input_center_per_variable():
    """Only the target's centre is used; an exogenous variable's centre may be NaN (unknown)."""
    T, h = 40, 2
    x = np.stack([ar1_data(B=2, T=T, seed=1), ar1_data(B=2, T=T, seed=2)], -1) + 1.0  # (B, T, 2)
    common = dict(p=1, theta_pooling='shared', sigma_pooling='shared', input_power=0.25,
                  calendar_labels=cal(labels(T, h), 2), calendar_pooling='shared', forecast_horizon=h, **FAST)  # fmt: skip
    m = SARIX(x, input_center=np.array([[np.nan, 1.0], [np.nan, 2.0]]), **common)
    assert m.input_center.shape == (2, 2)
    z = m._calendar_adjust(jnp.asarray(x), jnp.ones((2, T, 1)))
    c = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(z[..., 1], np.exp(-0.25) * (x[..., 1] + c) - c, rtol=1e-6)
    np.testing.assert_allclose(z[..., 0], x[..., 0])
    # shape (n_x + 1,): one centre per variable, shared by all batches
    m2 = SARIX(x, input_center=np.array([5.0, 1.0]), **common)
    np.testing.assert_allclose(m2.input_center, [[5.0, 1.0], [5.0, 1.0]])


def test_exogenous_variables_not_adjusted():
    T, h = 40, 2
    x = np.stack([ar1_data(B=2, T=T, seed=1), ar1_data(B=2, T=T, seed=2)], -1) + 1.0  # (B, T, 2)
    m = SARIX(x, p=1, theta_pooling='shared', sigma_pooling='shared', calendar_labels=cal(labels(T, h), 2),
              calendar_pooling='shared', forecast_horizon=h, **FAST)
    z = m._calendar_adjust(jnp.asarray(x), jnp.ones((2, T, 1)))
    np.testing.assert_allclose(z[..., 0], x[..., 0])
    np.testing.assert_allclose(z[..., 1], np.exp(-1.0) * x[..., 1], rtol=1e-6)


# ---------------------------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------------------------


def test_recovers_effects():
    """Fourth-root, centred scale with known multipliers exp(0.4), exp(0.8)."""
    B, T, h = 3, 300, 4
    lam, c = 0.25, np.array([1.0, 1.5, 2.0])
    z = ar1_data(B=B, T=T, phi=0.8, seed=3)
    lab = labels(T, h)
    eta = np.array([0.0, 0.4, 0.8])
    x = np.exp(lam * eta[lab[:T]]) * (z + c[:, None]) - c[:, None]
    m = SARIX(x[..., None], p=1, theta_pooling='shared', sigma_pooling='shared', input_power=lam, input_center=c[:, None],
              calendar_labels=cal(lab), calendar_pooling='shared', forecast_horizon=h, num_warmup=200, num_samples=200)
    est = np.asarray(m.samples['calendar_effect'])[:, 0].mean(0)
    sd = np.asarray(m.samples['calendar_effect'])[:, 0].std(0)
    assert np.all(np.abs(est - eta[1:]) < 4 * sd + 0.02), (est, sd)


def test_recovers_separate_effects_per_variable():
    """Labels shared by both variables (trailing axis of 1), different known effects and powers."""
    B, T, h = 3, 300, 4
    lam = np.array([1.0, 0.25])
    c = np.array([[2.0, 1.0], [2.0, 1.5], [2.0, 2.0]])
    z = np.stack([ar1_data(B=B, T=T, phi=0.8, seed=4), ar1_data(B=B, T=T, phi=0.8, seed=3)], -1)
    lab = labels(T, h)
    eta = np.array([[0.0, -0.3, 0.2], [0.0, 0.4, 0.8]])  # (variable, label)
    e = np.stack([eta[0, lab[:T]], eta[1, lab[:T]]], -1)  # (T, 2)
    x = np.exp(lam * e) * (z + c[:, None, :]) - c[:, None, :]
    m = SARIX(x, p=1, theta_pooling='shared', sigma_pooling='shared', input_power=lam, input_center=c,
              calendar_labels=cal(lab), calendar_pooling='shared', forecast_horizon=h, num_warmup=200,
              num_samples=200)
    s = np.asarray(m.samples['calendar_effect'])  # (S, 2, K)
    est, sd = s.mean(0), s.std(0)
    assert np.all(np.abs(est - eta[:, 1:]) < 4 * sd + 0.02), (est, sd)


@pytest.mark.parametrize("effect_sd", [0.0, 0.15])  # identical effects (the funnel case); spread-out effects
def test_partial_pooling_recovers_effects_without_divergences(effect_sd):
    """Each batch's effects = common effects + effect_sd * noise, on a fourth-root, centred scale."""
    B, T, h = 10, 156, 4
    lam = 0.25
    rng = np.random.default_rng(5)
    c = rng.uniform(1.0, 2.0, size=B)
    z = ar1_data(B=B, T=T, phi=0.8, seed=6)
    lab = labels(T, h)
    mu = np.array([0.4, 0.8])
    eta = mu + effect_sd * rng.normal(size=(B, 2))  # (B, K)
    e = np.concatenate([np.zeros((B, 1)), eta], -1)[:, lab[:T]]  # (B, T)
    x = np.exp(lam * e) * (z + c[:, None]) - c[:, None]
    m = SARIX(x[..., None], p=1, theta_pooling='shared', sigma_pooling='shared', input_power=lam,
              input_center=c[:, None], calendar_labels=cal(lab), calendar_pooling='partial', forecast_horizon=h,
              num_warmup=500, num_samples=500)
    assert m.num_divergences <= 5, m.num_divergences  # at most 1% of draws
    mu_s = np.asarray(m.samples['calendar_effect_mu'])[:, 0]  # (S, K)
    assert np.all(np.abs(mu_s.mean(0) - mu) < 4 * mu_s.std(0) + 0.02), (mu_s.mean(0), mu_s.std(0))
    eff = np.asarray(m.samples['calendar_effect'])[:, :, 0]  # (S, B, K)
    lo, hi = np.quantile(eff, [0.025, 0.975], axis=0)
    assert np.mean((lo <= eta) & (eta <= hi)) >= 0.85  # 95% intervals for the 20 batch-level effects
    sd_s = np.asarray(m.samples['calendar_effect_sd'])[:, 0]
    assert np.quantile(sd_s, 0.025) <= effect_sd + 0.02 and effect_sd <= np.quantile(sd_s, 0.975) + 0.02


# ---------------------------------------------------------------------------------------------
# input checks and output types
# ---------------------------------------------------------------------------------------------


def test_diff_accepts_array_likes():
    import pandas as pd

    from sarix.sarix import diff

    x = np.arange(20.0).reshape(10, 2) ** 2
    expect = np.diff(x, n=2, axis=0)
    np.testing.assert_allclose(diff(x.tolist(), d=2), expect)
    np.testing.assert_allclose(diff(pd.DataFrame(x), d=2), expect)


def test_warns_when_values_below_negative_center():
    """With power > 0, xy + input_center < 0 cannot come from a positive original-scale quantity."""
    xy = ar1_data(B=2, T=30)[..., None]  # centred around 0
    with pytest.warns(UserWarning, match=r"variables \[0\] have calendar effects but take values below"):
        _ = SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='shared', calendar_labels=cal(labels(30, 1)),
                  calendar_pooling='shared', forecast_horizon=1, **FAST)
    with warnings.catch_warnings():  # the same data with a centre that makes them positive
        warnings.simplefilter("error")
        SARIX(xy, p=1, theta_pooling='shared', sigma_pooling='shared', input_center=5.0,
              calendar_labels=cal(labels(30, 1)), calendar_pooling='shared', forecast_horizon=1, **FAST)


@pytest.mark.parametrize("transform", ["sqrt", "fourthrt"])
def test_predictions_are_numpy_arrays(transform):
    xy = ar1_data(B=2, T=30)[..., None] + 5.0
    m = SARIX(xy, p=1, transform=transform, theta_pooling='shared', sigma_pooling='shared', calendar_labels=cal(labels(30, 1)),
              calendar_pooling='shared', forecast_horizon=1, **FAST)
    assert type(m.predictions) is np.ndarray
