import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from hbmep.config import Config
from src.bsp.recruitment import functional as functional, smooth_functional as smooth_functional
from hbmep.model import GammaModel
from hbmep.model.utils import Site as site


def get_name(self):
    return (
        f'{self.NAME}_'
        f'{"_".join(self.response)}_'
        f'{"_".join(self.features)}_'
        f'{self.intensity}_'
        f'mix{"T" if self.use_mixture else "F"}'
        f'_v{"A" if self.visit is None else self.visit}'
        f'{"_smooth" if self.smooth else ""}'
        f'_{self.size_type}'
        f'{"_s50" if self.s50parameterization else ""}'
        f'{"_s" if self.small else ""}'
    )


def set_defaults(self):
    return {
        "use_mixture": True,
        "small": False,
        "visit": 0,
        "smooth": False,
        "size_type": 'auc',
        "s50parameterization": False,
        "run_kwargs": {
            "max_tree_depth": (20, 20),
            "target_accept_prob": 0.95,
            "extra_fields": [
                "potential_energy",
                "num_steps",
                "accept_prob",
            ],
        },
        "mcmc_params": {
            "num_warmup": 4000,
            "num_samples": 4000,
            "thinning": 4,
            "num_chains": 4,
        },
    }


class RectifiedLogistic(GammaModel):
    NAME = "RectifiedLogistic"

    def __init__(self, config: Config):
        super(RectifiedLogistic, self).__init__(config=config)
        self.__dict__.update(set_defaults(self))

    @property
    def subname(self):
        return get_name(self)

    def _model(self, intensity, features, response_obs=None):
        # Pick F based on whether we want smooth or not
        F = smooth_functional if self.smooth else functional
        # Decide which rectified logistic function to call
        logistic_fn = F.rectified_logistic_s50 if self.s50parameterization else F.rectified_logistic

        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        if response_obs is None:
            mask_obs = np.ones_like(response_obs, dtype=bool)
        else:
            mask_obs = np.invert(np.isnan(response_obs))

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 100., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(10.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(1.))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(25.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(25.))
        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(10.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(1.))

        if self.use_mixture:
            # Outlier distribution
            q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.025))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                    ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                    c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                    c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                    c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        mu = numpyro.deterministic(
            site.mu,
            logistic_fn(
                x=intensity,
                a=a[feature0, feature1],
                b=b[feature0, feature1],
                L=L[feature0, feature1],
                ell=ell[feature0, feature1],
                H=H[feature0, feature1]
            )
        )
        beta = numpyro.deterministic(
            site.beta,
            self.rate(
                mu,
                c_1[feature0, feature1],
                c_2[feature0, feature1]
            )
        )
        alpha = numpyro.deterministic(
            site.alpha,
            self.concentration(mu, beta)
        )

        if self.use_mixture:
            # Mixture
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions = [
                dist.Gamma(concentration=alpha, rate=beta),
                dist.HalfNormal(scale=25.0)
            ]
            Mixture = dist.MixtureGeneral(
                mixing_distribution=mixing_distribution,
                component_distributions=component_distributions
            )

        with numpyro.handlers.mask(mask=mask_obs):
            numpyro.sample(
                site.obs,
                (
                    Mixture if self.use_mixture
                    else dist.Gamma(concentration=alpha, rate=beta)
                ),
                obs=response_obs
            )


class RectifiedLogisticCo_a(GammaModel):
    NAME = "RectifiedLogisticCo_a"

    def __init__(self, config: Config):
        super(RectifiedLogisticCo_a, self).__init__(config=config)
        self.__dict__.update(set_defaults(self))

    @property
    def subname(self):
        return get_name(self)

    def _model(self, intensity, features, response_obs=None):
        # Pick F based on whether we want smooth or not
        F = smooth_functional if self.smooth else functional
        # Decide which rectified logistic function to call
        logistic_fn = F.rectified_logistic_s50 if self.s50parameterization else F.rectified_logistic

        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        if response_obs is None:
            mask_obs = np.ones_like(response_obs, dtype=bool)
        else:
            mask_obs = np.invert(np.isnan(response_obs))

        # Hyper Priors
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.))
        b_scale = numpyro.sample("b_scale", dist.HalfNormal(10.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(1.))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(25.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(25.))
        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(10.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(1.))

        if self.use_mixture:
            # Outlier distribution
            q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.025))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                a_loc = numpyro.sample(
                    "a_loc", dist.TruncatedNormal(50.0, 100.0, low=0)
                )
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )
                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                    ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                    c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                    c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                    c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        mu = numpyro.deterministic(
            site.mu,
            logistic_fn(
                x=intensity,
                a=a[feature0, feature1],
                b=b[feature0, feature1],
                L=L[feature0, feature1],
                ell=ell[feature0, feature1],
                H=H[feature0, feature1]
            )
        )
        beta = numpyro.deterministic(
            site.beta,
            self.rate(
                mu,
                c_1[feature0, feature1],
                c_2[feature0, feature1]
            )
        )
        alpha = numpyro.deterministic(
            site.alpha,
            self.concentration(mu, beta)
        )

        if self.use_mixture:
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions = [
                dist.Gamma(concentration=alpha, rate=beta),
                dist.HalfNormal(scale=25.0)
            ]
            Mixture = dist.MixtureGeneral(
                mixing_distribution=mixing_distribution,
                component_distributions=component_distributions
            )

        with numpyro.handlers.mask(mask=mask_obs):
            numpyro.sample(
                site.obs,
                (
                    Mixture if self.use_mixture
                    else dist.Gamma(concentration=alpha, rate=beta)
                ),
                obs=response_obs
            )


class RectifiedLogisticELL(GammaModel):
    NAME = "RectifiedLogisticELL"  # the B is for biased

    def __init__(self, config: Config):
        super(RectifiedLogisticELL, self).__init__(config=config)
        self.__dict__.update(set_defaults(self))

    @property
    def subname(self):
        return get_name(self)

    def _model(self, intensity, features, response_obs=None):
        # Pick F based on whether we want smooth or not
        F = smooth_functional if self.smooth else functional
        # Decide which rectified logistic function to call
        logistic_fn = F.rectified_logistic_s50 if self.s50parameterization else F.rectified_logistic

        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        if response_obs is None:
            mask_obs = np.ones_like(response_obs, dtype=bool)
        else:
            mask_obs = np.invert(np.isnan(response_obs))

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 100., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(10.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(1.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(25.))
        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(10.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(1.))

        ell_px = numpyro.sample('ell_px', dist.HalfNormal(10.))
        ell_p1 = numpyro.deterministic('ell_p1', ell_px + 1.0)  # constant addition >1 biases towards relu onset.

        if self.use_mixture:
            # Outlier distribution
            q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.025))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    ell = numpyro.sample(site.ell, dist.Beta(ell_p1, 1.))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                    c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                    c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                    c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        mu = numpyro.deterministic(
            site.mu,
            logistic_fn(
                x=intensity,
                a=a[feature0, feature1],
                b=b[feature0, feature1],
                L=L[feature0, feature1],
                ell=H[feature0, feature1]*ell[feature0, feature1],
                H=H[feature0, feature1]
            )
        )
        beta = numpyro.deterministic(
            site.beta,
            self.rate(
                mu,
                c_1[feature0, feature1],
                c_2[feature0, feature1]
            )
        )
        alpha = numpyro.deterministic(
            site.alpha,
            self.concentration(mu, beta)
        )

        if self.use_mixture:
            # Mixture
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions = [
                dist.Gamma(concentration=alpha, rate=beta),
                dist.HalfNormal(scale=25.0)
            ]
            Mixture = dist.MixtureGeneral(
                mixing_distribution=mixing_distribution,
                component_distributions=component_distributions
            )

        with numpyro.handlers.mask(mask=mask_obs):
            numpyro.sample(
                site.obs,
                (
                    Mixture if self.use_mixture
                    else dist.Gamma(concentration=alpha, rate=beta)
                ),
                obs=response_obs
            )


class RectifiedLogisticELLCo_a(GammaModel):
    NAME = "RectifiedLogisticELLCo_a"  # the B is for biased

    def __init__(self, config: Config):
        super(RectifiedLogisticELLCo_a, self).__init__(config=config)
        self.__dict__.update(set_defaults(self))

    @property
    def subname(self):
        return get_name(self)

    def _model(self, intensity, features, response_obs=None):
        # Pick F based on whether we want smooth or not
        F = smooth_functional if self.smooth else functional
        # Decide which rectified logistic function to call
        logistic_fn = F.rectified_logistic_s50 if self.s50parameterization else F.rectified_logistic

        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        if response_obs is None:
            mask_obs = np.ones_like(response_obs, dtype=bool)
        else:
            mask_obs = np.invert(np.isnan(response_obs))

        # Hyper Priors
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(10.))
        L_scale = numpyro.sample("L_scale", dist.HalfNormal(1.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(25.))
        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(10.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(1.))

        ell_px = numpyro.sample('ell_px', dist.HalfNormal(10.))
        ell_p1 = numpyro.deterministic('ell_p1', ell_px + 1.0)  # constant addition >1 biases towards relu onset.

        if self.use_mixture:
            # Outlier distribution
            q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.025))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                a_loc = numpyro.sample(
                    "a_loc", dist.TruncatedNormal(50.0, 100.0, low=0)
                )
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    ell = numpyro.sample(site.ell, dist.Beta(ell_p1, 1.))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                    c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                    c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                    c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        mu = numpyro.deterministic(
            site.mu,
            logistic_fn(
                x=intensity,
                a=a[feature0, feature1],
                b=b[feature0, feature1],
                L=L[feature0, feature1],
                ell=H[feature0, feature1]*ell[feature0, feature1],
                H=H[feature0, feature1]
            )
        )
        beta = numpyro.deterministic(
            site.beta,
            self.rate(
                mu,
                c_1[feature0, feature1],
                c_2[feature0, feature1]
            )
        )
        alpha = numpyro.deterministic(
            site.alpha,
            self.concentration(mu, beta)
        )

        if self.use_mixture:
            # Mixture
            mixing_distribution = dist.Categorical(
                probs=jnp.stack([1 - q, q], axis=-1)
            )
            component_distributions = [
                dist.Gamma(concentration=alpha, rate=beta),
                dist.HalfNormal(scale=25.0)
            ]
            Mixture = dist.MixtureGeneral(
                mixing_distribution=mixing_distribution,
                component_distributions=component_distributions
            )

        with numpyro.handlers.mask(mask=mask_obs):
            numpyro.sample(
                site.obs,
                (
                    Mixture if self.use_mixture
                    else dist.Gamma(concentration=alpha, rate=beta)
                ),
                obs=response_obs
            )

