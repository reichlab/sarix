import os
import time
import jax.numpy as jnp
import numpy as onp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def diff(x, d=0, D=0, season_period=7, pad_na=False):
    x = onp.diff(x, n = d, axis = -2)
    for i in range(D):
        x = x[..., season_period:, :] - x[..., :-season_period, :]
    if pad_na:
        batch_shape = x.shape[:-2]
        n_vars = x.shape[-1]
        leading_nans = onp.full(batch_shape + (D * season_period + d, n_vars), onp.nan)
        x = onp.concatenate([leading_nans, x], axis = -2)
    return x

def inv_diff(x, dx, d=0, D=0, season_period=7):
    batch_shape_x = x.shape[:-2]
    T_x = x.shape[-2]
    n_vars = x.shape[-1]
    batch_shape_dx = dx.shape[:-2]
    T_dx = dx.shape[-2]
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
    for i in range(1, d + 1):
        x_dm1 = diff(x, d=d-i, D=D, season_period=season_period, pad_na=True)
        x_dm1 = onp.broadcast_to(x_dm1, batch_shape_dx + x_dm1.shape[-2:])
        dx_full = onp.concatenate([x_dm1, dx], axis=-2)
        for t in range(T_dx):
            dx_full[..., T_x + t, :] = dx_full[..., T_x + t - 1, :] + dx_full[..., T_x + t, :]
        dx = dx_full[..., -T_dx:, :]
    for i in range(1, D + 1):
        x_dm1 = diff(x, d=0, D=D-i, season_period=season_period, pad_na=True)
        x_dm1 = onp.broadcast_to(x_dm1, batch_shape_dx + x_dm1.shape[-2:])
        dx_full = onp.concatenate([x_dm1, dx], axis=-2)
        for t in range(T_dx):
            dx_full[..., T_x + t, :] = dx_full[..., T_x + t - season_period, :] + dx_full[..., T_x + t, :]
        dx = dx_full[..., -T_dx:, :]
    return dx

class SARIX:
    def __init__(self,
                 xy,
                 p=1,
                 d=0,
                 P=0,
                 D=0,
                 season_period=1,
                 transform='none',
                 theta_pooling='none',   # 'none' = one per location, 'shared' = shared across locations
                 sigma_pooling='none',   # unchanged from your version
                 forecast_horizon=1,
                 num_warmup=1000,
                 num_samples=1000,
                 num_chains=1,
                 fourier_K=0,            # number of harmonics for Fourier terms
                 fourier_pooling='none'  # <-- new option: 'none' or 'shared'
                 ):
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
        self.fourier_K = fourier_K
        self.fourier_pooling = fourier_pooling
        self.season_period = season_period
        self.forecast_horizon = forecast_horizon
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains

        batch_shape = xy.shape[:-2]
        self.batch_shape = batch_shape
        self.theta_batch_shape = batch_shape if theta_pooling == 'none' else ()
        self.sigma_batch_shape = batch_shape if sigma_pooling == 'none' else ()
        self.fourier_batch_shape = batch_shape if fourier_pooling == 'none' else ()

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

        transformed_xy = self.xy
        self.xy = diff(self.xy, self.d, self.D, self.season_period, pad_na=False)

        T = self.xy.shape[-2]
        if self.fourier_K > 0:
            self.fourier_features = self._fourier_terms(T, self.season_period, self.fourier_K)
        else:
            self.fourier_features = None

        self.update_X = self.state_update_X(self.xy[..., :self.max_lag, :],
                                            self.xy[..., self.max_lag:, :])
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        self.run_inference(rng_key)
        self.predictions_modeled_scale = self.predict(rng_key_predict)
        self.predictions = inv_diff(transformed_xy,
                                    self.predictions_modeled_scale,
                                    self.d, self.D, self.season_period)
        if transform == "log":
            self.predictions = onp.exp(self.predictions)
        elif transform == "fourthrt":
            self.predictions = jnp.maximum(0.0, self.predictions)**4
        elif transform == "sqrt":
            self.predictions = jnp.maximum(0.0, self.predictions)**2

    def _fourier_terms(self, T, season_period, K):
        t = onp.arange(T)
        terms = []
        for k in range(1, K+1):
            terms.append(onp.sin(2 * onp.pi * k * t / season_period))
            terms.append(onp.cos(2 * onp.pi * k * t / season_period))
        return onp.stack(terms, axis=-1)  # shape (T, 2*K)

    def make_state_transition_matrix(self, theta):
        batch_shape = theta.shape[:-1]
        n_ar_coef = (self.n_x + 1) * self.p
        A = jnp.reshape(theta, batch_shape + (self.n_x+1, self.p*(self.n_x+1)))
        A = jnp.swapaxes(A, -2, -1)  # (...batch, p*(n_x+1), n_x+1)
        return A

    def state_update_X(self, init_stoch_state, stoch_state):
        stoch_state = jnp.concatenate([init_stoch_state, stoch_state], axis=-2)
        lagged_rows = []
        for lag in range(1, self.p + 1):
            lagged_rows.append(stoch_state[..., self.max_lag - lag:-(lag), :])
        result = jnp.concatenate(lagged_rows, axis=-1)
        return result

    def model(self, xy):
        batch_size = xy.shape[0]
        T = xy.shape[1]

        # Partial pooling for AR coefficients
        n_theta = self.p * (self.n_x + 1)
        if self.theta_pooling == 'none':
            theta_group_sd = numpyro.sample("theta_group_sd", dist.HalfCauchy(jnp.ones(1)))
            theta_mu = numpyro.sample("theta_mu", dist.Normal(jnp.zeros((n_theta,)), theta_group_sd))
            theta_sd = numpyro.sample("theta_sd", dist.HalfCauchy(jnp.ones(1)))
            theta = numpyro.sample(
                "theta",
                dist.Normal(loc=theta_mu[None, :], scale=theta_sd)
            )  # shape (batch, n_theta)
        else:  # shared pooling
            theta_sd = numpyro.sample("theta_sd", dist.HalfCauchy(jnp.ones(1)))
            theta = numpyro.sample(
                "theta",
                dist.Normal(loc=jnp.zeros((n_theta,)), scale=theta_sd)
            )   # shape (n_theta,)
            theta = jnp.broadcast_to(theta, (batch_size,) + theta.shape)

        sigma = numpyro.sample(
            "sigma", dist.HalfCauchy(jnp.ones((batch_size, self.n_x+1)))
        )
        Sigma_chol = jnp.expand_dims(sigma, -2) * jnp.eye(sigma.shape[-1])

        # Fourier terms
        if self.fourier_K > 0:
            fourier_coef_shape = (self.n_x+1, 2*self.fourier_K)
            if self.fourier_pooling == 'none':
                # Partial pooling for locations
                beta_group_sd = numpyro.sample("beta_group_sd", dist.HalfCauchy(jnp.ones(1)))
                beta_mu = numpyro.sample("beta_mu", dist.Normal(jnp.zeros(fourier_coef_shape), beta_group_sd))
                beta_sd = numpyro.sample("beta_sd", dist.HalfCauchy(jnp.ones(1)))
                fourier_B = numpyro.sample("fourier_B", dist.Normal(loc=beta_mu[None,:,:], scale=beta_sd))  # shape (batch, n_x+1, 2K)
            else:  # 'shared'
                beta_sd = numpyro.sample("beta_sd", dist.HalfCauchy(jnp.ones(1)))
                fourier_B = numpyro.sample("fourier_B", dist.Normal(loc=jnp.zeros(fourier_coef_shape), scale=beta_sd))
                fourier_B = jnp.broadcast_to(fourier_B, (batch_size,) + fourier_B.shape)
        else:
            fourier_B = None

        A = self.make_state_transition_matrix(theta)
        update_X = self.update_X
        step_means_ar = jnp.sum(update_X * A, axis=-1)  # shape (batch, time, n_x+1)

        # Fourier means
        if self.fourier_K > 0:
            F = jnp.array(self.fourier_features)  # (time, 2K)
            step_means_fourier = jnp.einsum('bic,tc->bit', fourier_B, F)
        else:
            step_means_fourier = 0

        step_means = step_means_ar + step_means_fourier
        step_innovations = xy[..., self.max_lag:, :] - step_means
        numpyro.sample(
            "step_innovations",
            dist.MultivariateNormal(
                loc=jnp.zeros((self.n_x+1,)), scale_tril=Sigma_chol),
            obs=step_innovations
        )

    def run_inference(self, rng_key):
        start = time.time()
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples,
                    num_chains=self.num_chains, progress_bar=True)
        mcmc.run(rng_key, self.xy)
        self.samples = mcmc.get_samples()
        print('\nMCMC elapsed time:', time.time() - start)

    def predict(self, rng_key):
        theta = self.samples['theta']
        sigma = self.samples['sigma']
        batch_size = theta.shape[0]
        n_xp1 = self.n_x + 1

        if self.fourier_K > 0:
            fourier_B = self.samples['fourier_B']
            T = self.xy.shape[-2]
            T_pred = T + self.forecast_horizon
            F_pred = self._fourier_terms(T_pred, self.season_period, self.fourier_K)
            F_future = F_pred[-self.forecast_horizon:, :]  # (horizon, 2K)
        Sigma_chol = jnp.expand_dims(sigma, -2) * jnp.eye(sigma.shape[-1])
        innovations = dist.MultivariateNormal(
                loc=jnp.zeros((n_xp1,)), scale_tril=Sigma_chol) \
            .sample(rng_key, sample_shape=(self.forecast_horizon, batch_size))

        y_pred = []
        recent_lags = jnp.broadcast_to(self.xy[..., -self.max_lag:, :],
                                       (batch_size, self.max_lag, self.xy.shape[-1]))
        dummy_values = jnp.zeros((batch_size, 1, self.xy.shape[-1]))

        for h in range(self.forecast_horizon):
            update_X = self.state_update_X(recent_lags, dummy_values)
            step_means_ar = jnp.sum(update_X * self.make_state_transition_matrix(theta), axis=-1)
            if self.fourier_K > 0:
                if self.fourier_pooling == 'none':
                    step_means_fourier = jnp.einsum('bic,c->bi', fourier_B, F_future[h])
                else:
                    step_means_fourier = jnp.einsum('ic,c->bi', fourier_B, F_future[h])
            else:
                step_means_fourier = 0
            new_y_pred = step_means_ar + step_means_fourier + innovations[h, ...][..., None, :]
            y_pred.append(new_y_pred)
            recent_lags = jnp.concatenate([recent_lags[..., 1:, :], new_y_pred], axis=-2)
        y_pred = jnp.concatenate(y_pred, axis=-2)
        return onp.asarray(y_pred)