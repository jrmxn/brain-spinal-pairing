import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LKJ, MultivariateNormal
import jax.numpy as jnp
from src.bsp.core.utils import make_parameter_mask, make_obs_mask


def g(x, mean, sd, bs, peakiness):
    z = jnp.exp(-0.5 * jnp.power((x - mean) / sd, 2 * peakiness))
    return bs * z


def model00(cpi, time, response_obs, run_index, visit_index, participant_index, descriptor_index, intensity_index,
            average_count, model_options):
    #
    #
    # Compute the number of unique participants, runs, visits, and cxscs
    num_participants = np.max(participant_index) + 1
    num_runs = np.max(run_index) + 1
    num_visits = np.max(visit_index) + 1
    num_intensity = np.max(intensity_index) + 1
    num_descriptor = np.max(descriptor_index) + 1
    assert num_visits == 1, 'Only handling a single visit currently.'

    rescale_sd = False if np.all(average_count == 1) else True
    if rescale_sd:
        scale_renorm = 1 / jnp.sqrt(average_count)

    # Handle NaN values in cpi
    is_only = np.isnan(cpi)
    cpi = np.where(is_only, 100.0, cpi)

    # Extract model options
    scale_c_prior = model_options.get('scale_c_prior', 0.5)
    c_limits = model_options.get('c_limits', 2)

    # Determine the number of muscles
    if isinstance(response_obs, int):
        num_muscles = response_obs
        response_obs = None
    else:
        num_muscles = response_obs.shape[1]

    # Create parameter and observation masks
    mask_run, mask_visit, mask_participant, mask_descriptor, mask_intensity, _ = make_parameter_mask(
        num_muscles, num_runs, num_visits, num_participants, num_descriptor, num_intensity,
        response_obs, run_index, visit_index, participant_index, descriptor_index, intensity_index,
        model_options.get('mask', None))
    mask_obs = make_obs_mask(response_obs, model_options.get('mask', None))

    # Sample additional hyperparameters if 'use_b' is True
    scale_scale = numpyro.sample('scale_scale', dist.HalfNormal(10.0))
    loc_loc_a = numpyro.sample('loc_loc_a', dist.Normal(0.0, 10.0))
    scale_loc_a = numpyro.sample('scale_loc_a', dist.HalfNormal(10.0))
    scale_a_run = numpyro.sample('scale_a_run', dist.HalfNormal(10.0))
    use_b = model_options.get('use_b', False)
    if use_b:
        scale_loc_b = numpyro.sample('scale_loc_b', dist.HalfNormal(10.0))
        scale_b_run = numpyro.sample('scale_b_run', dist.HalfNormal(10.0))

    # Sample hyperparameters
    with numpyro.plate('muscles', num_muscles):
        scale_s = numpyro.sample('scale_s', dist.HalfNormal(10.0))

    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            # with numpyro.plate('descriptor', num_descriptor):
            if num_visits > 1:
                scale_sd = numpyro.sample('scale_sd', dist.HalfNormal(10.0))

    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.plate('participants', num_participants):
                    with numpyro.plate('visits', num_visits):
                        scale = numpyro.sample('scale', dist.HalfNormal(scale_scale))

    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            with numpyro.plate('descriptor', 1):  # to make it same dim as model00co
            # with numpyro.plate('descriptor', num_descriptor):
                #     with numpyro.handlers.mask(mask=mask_descriptor):
                loc_s = numpyro.sample('loc_s', dist.Normal(0, 10))

    # determine c
    if scale_c_prior > 0:
        scale_c = numpyro.sample('scale_c', dist.HalfNormal(scale_c_prior))
        with numpyro.plate('muscles', num_muscles):
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.plate('participants', num_participants):
                    c = numpyro.sample('c', dist.TruncatedNormal(loc=0, scale=scale_c, low=-c_limits, high=+c_limits))
    else:
        c = numpyro.deterministic('c', jnp.full((num_participants, num_descriptor, num_muscles), 0))

    # determine w
    with numpyro.plate('muscles', num_muscles):
        scale_w = numpyro.sample('scale_w', dist.HalfNormal(2.5))
        with numpyro.plate('descriptor', num_descriptor):
            with numpyro.plate('participants', num_participants):
                w = numpyro.sample('w', dist.TruncatedNormal(loc=0.5, scale=scale_w, low=0.5, high=2 * 2.5))

    # Sample 's' parameter
    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.plate('participants', num_participants):
                    with numpyro.handlers.mask(mask=mask_participant):
                        s_raw = numpyro.sample('s_raw', dist.Normal(0, 1))
                        s0 = numpyro.deterministic('s0', loc_s + s_raw * scale_s)
                        if num_visits > 1:
                            with numpyro.plate('visits_diff', num_visits - 1):
                                with numpyro.handlers.mask(mask=mask_visit[1:, ...]):
                                    sd = numpyro.sample('sd', dist.Normal(0, scale_sd))
                            s_rest = s0[None, ...] + sd
                            s = numpyro.deterministic('s', jnp.concatenate([s0[None, ...], s_rest], axis=0))
                        else:
                            s = numpyro.deterministic('s', s0[None, ...])

    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.plate('participants', num_participants):
                    with numpyro.plate('visits', num_visits):
                        with numpyro.handlers.mask(mask=mask_visit):
                            loc_a_raw = numpyro.sample('loc_a_raw', dist.Normal(0, 1))
                            loc_a = numpyro.deterministic('loc_a', loc_loc_a + scale_loc_a * loc_a_raw)
                            if model_options['use_b']:
                                loc_b_raw = numpyro.sample('loc_b_raw', dist.Normal(0, 1))
                                loc_b = numpyro.deterministic('loc_b', scale_loc_b * loc_b_raw)

                            with numpyro.plate('run', num_runs):
                                with numpyro.handlers.mask(mask=mask_run):
                                    a_raw = numpyro.sample('a_run', dist.Laplace(0, 1))
                                    a = numpyro.deterministic('a', loc_a + jnp.multiply(a_raw, scale_a_run))
                                    if model_options['use_b']:
                                        b_raw = numpyro.sample('b_run',
                                                               dist.Laplace(0, 1))  # <- made this laplace
                                        b = numpyro.deterministic('b', loc_b + jnp.multiply(b_raw, scale_b_run))
                                    else:
                                        b = jnp.zeros_like(a)

    indices = (run_index, visit_index, participant_index, descriptor_index, intensity_index)
    baseline = a[indices] + b[indices] * time

    exp_poly_bell_contribution = jnp.where(
        is_only,
        jnp.zeros_like(baseline),
        g(cpi,
          c[participant_index, descriptor_index, :],
          w[participant_index, descriptor_index, :],
          s[visit_index, participant_index, descriptor_index, intensity_index, :], 1)
    )

    f = baseline + exp_poly_bell_contribution
    scale_flat = scale[visit_index, participant_index, descriptor_index, intensity_index, :]
    if rescale_sd:
        scale_flat = scale_flat * scale_renorm

    with numpyro.handlers.mask(mask=mask_obs):
        numpyro.sample('y', dist.Normal(f, scale_flat), obs=response_obs)


def model00co(cpi, time, response_obs, run_index, visit_index, participant_index, descriptor_index, intensity_index,
    average_count, model_options):
    #
    #
    # Compute the number of unique participants, runs, visits, and cxscs
    num_participants = np.max(participant_index) + 1
    num_runs = np.max(run_index) + 1
    num_visits = np.max(visit_index) + 1
    num_intensity = np.max(intensity_index) + 1
    num_descriptor = np.max(descriptor_index) + 1
    assert num_visits == 1, 'Only handling a single visit currently.'

    rescale_sd = False if np.all(average_count == 1) else True
    if rescale_sd:
        scale_renorm = 1 / jnp.sqrt(average_count)

    # Handle NaN values in cpi
    is_only = np.isnan(cpi)
    cpi = np.where(is_only, 100.0, cpi)

    # Extract model options
    scale_c_prior = model_options.get('scale_c_prior', 0.5)
    c_limits = model_options.get('c_limits', 2)

    # Determine the number of muscles
    if isinstance(response_obs, int):
        num_muscles = response_obs
        response_obs = None
    else:
        num_muscles = response_obs.shape[1]

    # Create parameter and observation masks
    mask_run, mask_visit, mask_participant, mask_descriptor, mask_intensity, _ = make_parameter_mask(
        num_muscles, num_runs, num_visits, num_participants, num_descriptor, num_intensity,
        response_obs, run_index, visit_index, participant_index, descriptor_index, intensity_index,
        model_options.get('mask', None))
    mask_obs = make_obs_mask(response_obs, model_options.get('mask', None))

    # Sample additional hyperparameters if 'use_b' is True
    scale_scale = numpyro.sample('scale_scale', dist.HalfNormal(10.0))
    loc_loc_a = numpyro.sample('loc_loc_a', dist.Normal(0.0, 10.0))
    scale_loc_a = numpyro.sample('scale_loc_a', dist.HalfNormal(10.0))
    scale_a_run = numpyro.sample('scale_a_run', dist.HalfNormal(10.0))
    use_b = model_options.get('use_b', False)
    if use_b:
        scale_loc_b = numpyro.sample('scale_loc_b', dist.HalfNormal(10.0))
        scale_b_run = numpyro.sample('scale_b_run', dist.HalfNormal(10.0))

    # Sample hyperparameters
    with numpyro.plate('muscles', num_muscles):
        scale_s = numpyro.sample('scale_s', dist.HalfNormal(10.0))

    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            with numpyro.plate('descriptor', num_descriptor):
                if num_visits > 1:
                    scale_sd = numpyro.sample('scale_sd', dist.HalfNormal(10.0))

    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.plate('participants', num_participants):
                    with numpyro.plate('visits', num_visits):
                        scale = numpyro.sample('scale', dist.HalfNormal(scale_scale))

    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            # with numpyro.plate('descriptor', 1):  # to make it same dim as model00co
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.handlers.mask(mask=mask_descriptor):
                    loc_s = numpyro.sample('loc_s', dist.Normal(0, 10))

    # determine c
    if scale_c_prior > 0:
        scale_c = numpyro.sample('scale_c', dist.HalfNormal(scale_c_prior))
        with numpyro.plate('muscles', num_muscles):
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.plate('participants', num_participants):
                    c = numpyro.sample('c', dist.TruncatedNormal(loc=0, scale=scale_c, low=-c_limits, high=+c_limits))
    else:
        c = numpyro.deterministic('c', jnp.full((num_participants, num_descriptor, num_muscles), 0))

    # determine w
    with numpyro.plate('muscles', num_muscles):
        scale_w = numpyro.sample('scale_w', dist.HalfNormal(2.5))
        with numpyro.plate('descriptor', num_descriptor):
            with numpyro.plate('participants', num_participants):
                w = numpyro.sample('w', dist.TruncatedNormal(loc=0.5, scale=scale_w, low=0.5, high=2 * 2.5))

    # Sample 's' parameter
    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.plate('participants', num_participants):
                    with numpyro.handlers.mask(mask=mask_participant):
                        s_raw = numpyro.sample('s_raw', dist.Normal(0, 1))
                        s0 = numpyro.deterministic('s0', loc_s + s_raw * scale_s)
                        if num_visits > 1:
                            with numpyro.plate('visits_diff', num_visits - 1):
                                with numpyro.handlers.mask(mask=mask_visit[1:, ...]):
                                    sd = numpyro.sample('sd', dist.Normal(0, scale_sd))
                            s_rest = s0[None, ...] + sd
                            s = numpyro.deterministic('s', jnp.concatenate([s0[None, ...], s_rest], axis=0))
                        else:
                            s = numpyro.deterministic('s', s0[None, ...])

    with numpyro.plate('muscles', num_muscles):
        with numpyro.plate('intensity', num_intensity):
            with numpyro.plate('descriptor', num_descriptor):
                with numpyro.plate('participants', num_participants):
                    with numpyro.plate('visits', num_visits):
                        with numpyro.handlers.mask(mask=mask_visit):
                            loc_a_raw = numpyro.sample('loc_a_raw', dist.Normal(0, 1))
                            loc_a = numpyro.deterministic('loc_a', loc_loc_a + scale_loc_a * loc_a_raw)
                            if model_options['use_b']:
                                loc_b_raw = numpyro.sample('loc_b_raw', dist.Normal(0, 1))
                                loc_b = numpyro.deterministic('loc_b', scale_loc_b * loc_b_raw)

                            with numpyro.plate('run', num_runs):
                                with numpyro.handlers.mask(mask=mask_run):
                                    a_raw = numpyro.sample('a_run', dist.Laplace(0, 1))
                                    a = numpyro.deterministic('a', loc_a + jnp.multiply(a_raw, scale_a_run))
                                    if model_options['use_b']:
                                        b_raw = numpyro.sample('b_run',
                                                               dist.Laplace(0, 1))  # <- made this laplace
                                        b = numpyro.deterministic('b', loc_b + jnp.multiply(b_raw, scale_b_run))
                                    else:
                                        b = jnp.zeros_like(a)

    indices = (run_index, visit_index, participant_index, descriptor_index, intensity_index)
    baseline = a[indices] + b[indices] * time

    exp_poly_bell_contribution = jnp.where(
        is_only,
        jnp.zeros_like(baseline),
        g(cpi,
          c[participant_index, descriptor_index, :],
          w[participant_index, descriptor_index, :],
          s[visit_index, participant_index, descriptor_index, intensity_index, :], 1)
    )

    f = baseline + exp_poly_bell_contribution
    scale_flat = scale[visit_index, participant_index, descriptor_index, intensity_index, :]
    if rescale_sd:
        scale_flat = scale_flat * scale_renorm

    with numpyro.handlers.mask(mask=mask_obs):
        numpyro.sample('y', dist.Normal(f, scale_flat), obs=response_obs)

