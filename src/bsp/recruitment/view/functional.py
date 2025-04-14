import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import gammaincinv

# -----------------------
# Rectified logistic and distribution utilities
# -----------------------
def _linear_transform(x, a, b):
    return jnp.multiply(b, x - a)

def _logistic_transform(x, a, b, ell, H):
    z = _linear_transform(x, a, b) - jnp.log(H) + jnp.log(ell)
    z = jax.nn.sigmoid(z)
    z = (H + ell) * z
    z = -ell + z
    return z

def rectified_logistic(x, a, b, L, ell, H):
    """
    Rectified-logistic function in threshold parameterization
    mu(x) = L + relu(logistic_transform(...))
    """
    z = _logistic_transform(x, a, b, ell, H)
    z = jax.nn.relu(z)
    return L + z

def rate(mu, c1, c2):
    """
    beta = (1/c1) + 1/(c2 * mu)
    """
    return (1.0 / c1) + 1.0 / (c2 * mu)

def concentration(mu, beta):
    """
    alpha = mu * beta
    """
    return mu * beta

def gamma_ppf(p, alpha, beta):
    # Gamma(alpha, rate=beta) quantile
    return gammaincinv(alpha, p) / beta

def get_gamma_statistics(x, a, b, L, ell, H, c1, c2):
    """
    Returns (2.5%, mean, 97.5%) for the Gamma distribution across x.
    """
    mu = rectified_logistic(x, a, b, L, ell, H)
    mu = jnp.clip(mu, 1e-10, None)  # avoid zero division
    beta_val = rate(mu, c1, c2)
    alpha_val = concentration(mu, beta_val)

    # 2.5% and 97.5% quantiles + mean
    q025 = gamma_ppf(0.025, alpha_val, beta_val)
    mean_val = mu
    q975 = gamma_ppf(0.975, alpha_val, beta_val)
    return q025, mean_val, q975


# -----------------------
# Main interactive code
# -----------------------
def main():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.38)
    ax.set_ylim([0, 25])  # Limit from 0 to 25
    ax.set_xlim([-1, 100])  # Limit from 0 to 25

    # X domain
    x = jnp.linspace(-1, 100, 300)
    x_np = np.array(x)

    # Initial parameter values
    a_init = 0.0
    b_init = 50.0
    L_init = 0.1
    ell_init = 1.0  # = 10^0
    H_init = 10.0
    c1_init = 1.0
    c2_init = 1.0

    # Compute initial lines (mean, then 2.5%/97.5% quantiles)
    mu_init = rectified_logistic(x, a_init, b_init, L_init, ell_init, H_init)
    line_mu, = ax.plot(x_np, np.array(mu_init), lw=2, label="Mean (rect. logistic)")

    line_q025, = ax.plot(x_np, np.zeros_like(x_np), '--', label="2.5%")
    line_q975, = ax.plot(x_np, np.zeros_like(x_np), '--', label="97.5%")

    q025_init, qmean_init, q975_init = get_gamma_statistics(
        x, a_init, b_init, L_init, ell_init, H_init, c1_init, c2_init
    )
    line_q025.set_ydata(np.array(q025_init))
    line_q975.set_ydata(np.array(q975_init))

    # Labels/titles
    ax.set_xlabel("x")
    ax.set_ylabel("Gamma statistic")
    ax.set_title("Gamma dist. summary (Mean, 2.5%, 97.5%)")

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    slider_height = 0.03
    ax_a     = plt.axes([0.25, 0.30, 0.65, slider_height], facecolor=axcolor)
    ax_b     = plt.axes([0.25, 0.25, 0.65, slider_height], facecolor=axcolor)
    ax_L     = plt.axes([0.25, 0.20, 0.65, slider_height], facecolor=axcolor)
    ax_logel = plt.axes([0.25, 0.15, 0.65, slider_height], facecolor=axcolor)
    ax_H     = plt.axes([0.25, 0.10, 0.65, slider_height], facecolor=axcolor)
    ax_c1    = plt.axes([0.25, 0.05, 0.65, slider_height], facecolor=axcolor)
    ax_c2    = plt.axes([0.25, 0.00, 0.65, slider_height], facecolor=axcolor)

    slider_a       = Slider(ax_a,     'a',   0.0, 100.0,  valinit=a_init)
    slider_b       = Slider(ax_b,     'b',    0.1,   5.0,  valinit=b_init)
    slider_L       = Slider(ax_L,     'L',   -5.0,   5.0,  valinit=L_init)
    slider_log_ell = Slider(ax_logel, 'ell', -4.0,   1.0,  valinit=np.log10(ell_init))
    slider_H       = Slider(ax_H,     'H',    0.1,  20.0,  valinit=H_init)
    slider_c1      = Slider(ax_c1,    'c1',   0.01,  10.,  valinit=c1_init)
    slider_c2      = Slider(ax_c2,    'c2',   0.01,  2.5,  valinit=c2_init)

    slider_log_ell.valtext.set_text(f"{10**slider_log_ell.val:.3g}")

    def update(_):
        a = slider_a.val
        b = slider_b.val
        L = slider_L.val
        ell = 10.0 ** slider_log_ell.val
        slider_log_ell.valtext.set_text(f"{ell:.3g}")
        H = slider_H.val
        c1 = slider_c1.val
        c2 = slider_c2.val

        # Always update the mean
        mu_new = rectified_logistic(x, a, b, L, ell, H)
        line_mu.set_ydata(np.array(mu_new))

        # Always update the 2.5% & 97.5% lines
        q025, qmean, q975 = get_gamma_statistics(x, a, b, L, ell, H, c1, c2)
        line_q025.set_ydata(np.array(q025))
        line_q975.set_ydata(np.array(q975))

        fig.canvas.draw_idle()

    # Bind slider updates
    slider_a.on_changed(update)
    slider_b.on_changed(update)
    slider_L.on_changed(update)
    slider_log_ell.on_changed(update)
    slider_H.on_changed(update)
    slider_c1.on_changed(update)
    slider_c2.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()
