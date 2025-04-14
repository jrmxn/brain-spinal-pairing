import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import gammainc
from hbmep.nn.functional import rectified_logistic
from hbmep.nn.functional import solve_rectified_logistic


def rate(mu, c_1, c_2):
    """
    Matches:
    (1 / c_1) + (1 / (c_2 * mu))
    """
    return (1.0 / c_1) + jnp.true_divide(1.0, (c_2 * mu))


def concentration(mu, beta):
    """
    Matches:
    mu * beta
    """
    return mu * beta


def gamma_cdf(x, alpha, beta):
    """
    CDF of Gamma(alpha, beta) at 'x' in the 'rate' parameterization.
    alpha>0, beta>0, x>=0
    """
    return gammainc(alpha, beta * x)


def median_objective(x, y, a, b, L, ell, H, c1, c2):
    mu = rectified_logistic(x, a, b, L, ell, H)
    beta_val = rate(mu, c1, c2)
    alpha_val = concentration(mu, beta_val)
    return gamma_cdf(y, alpha_val, beta_val) - 0.5


@jax.jit
def solve_rectified_logistic_median1(
        y, a, b, L, ell, H, c1, c2,
        x_lower=0.0, x_upper=150.0,
        max_steps=100,
        bracket_tol=1e-6,
        func_tol=1e-6
):
    """
    """

    @jax.jit
    def f(x_candidate):
        return median_objective(x_candidate, y, a, b, L, ell, H, c1, c2)

    # Initialize bracket
    xL = jnp.ones_like(a) * x_lower
    xU = jnp.ones_like(a) * x_upper

    def cond_fun(state):
        iteration, xL_curr, xU_curr = state
        still_iterating = iteration < max_steps

        mid = 0.5 * (xL_curr + xU_curr)
        bracket_size_ok = jnp.abs(xU_curr - xL_curr) < bracket_tol
        func_val_ok = jnp.abs(f(mid)) < func_tol

        # converged if bracket & function are small.
        all_converged = jnp.all(jnp.logical_and(bracket_size_ok, func_val_ok))

        return jnp.logical_and(still_iterating, jnp.logical_not(all_converged))

    def body_fun(state):
        iteration, xL_curr, xU_curr = state
        mid = 0.5 * (xL_curr + xU_curr)
        fL = f(xL_curr)
        fM = f(mid)
        same_sign = (fL * fM) > 0

        xL_next = jnp.where(same_sign, mid, xL_curr)
        xU_next = jnp.where(same_sign, xU_curr, mid)

        return (iteration + 1, xL_next, xU_next)

    state0 = (0, xL, xU)
    iteration_final, xL_final, xU_final = jax.lax.while_loop(cond_fun, body_fun, state0)

    return 0.5 * (xL_final + xU_final)


@jax.jit
def solve_rectified_logistic_median2(
        y, a, b, L, ell, H, c1, c2,
        max_steps=100,
        tol=1e-7
    ):
    """
    This one steps from mean guess
    """

    x0 = solve_rectified_logistic(y, a, b, L, ell, H)

    def f(x):
        return median_objective(x, y, a, b, L, ell, H, c1, c2)

    f_prime = jax.grad(f)

    def cond_fun(state):
        iteration, x_curr = state
        still_iter = iteration < max_steps
        residual = jnp.abs(f(x_curr))
        all_converged = jnp.all(residual < tol)
        return jnp.logical_and(still_iter, jnp.logical_not(all_converged))

    def body_fun(state):
        iteration, x_curr = state
        f_val = f(x_curr)
        fp_val = f_prime(x_curr)
        step = jnp.where(fp_val != 0, f_val / fp_val, 0.0)
        x_next = x_curr - step
        return (iteration + 1, x_next)

    init_state = (0, x0)
    _, x_final = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return x_final