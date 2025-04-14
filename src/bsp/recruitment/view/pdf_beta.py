import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox
import numpyro.distributions as dist


def main():
    # Create a figure with a decent size and adjust layout so there's space below the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95)

    # Position the main plot higher so we have space below for sliders and text boxes
    ax.set_position([0.1, 0.45, 0.8, 0.45])  # [left, bottom, width, height]

    # Initial x range and Beta parameters
    x_min_init = 0.0
    x_max_init = 1.0
    x = jnp.linspace(x_min_init, x_max_init, 1000)
    alpha_init = 2.0
    beta_init = 2.0

    beta_dist = dist.Beta(concentration1=alpha_init, concentration0=beta_init)
    y = jnp.exp(beta_dist.log_prob(x))
    line, = ax.plot(np.array(x), np.array(y), lw=2)
    ax.set_xlabel("x")
    ax.set_ylabel("pdf")
    ax.set_title("Numpyro Beta Distribution PDF")
    ax.set_xlim(x_min_init, x_max_init)

    # Add grid to the plot
    ax.grid(True)

    # Create axes for sliders and text boxes below the main plot
    axcolor = 'lightgoldenrodyellow'

    # --- Alpha Slider & Range Boxes ---
    ax_alpha = fig.add_axes([0.1, 0.35, 0.55, 0.03], facecolor=axcolor)
    slider_alpha = Slider(ax_alpha, 'Alpha', 0.1, 50.0, valinit=alpha_init)

    ax_alpha_min = fig.add_axes([0.68, 0.355, 0.07, 0.03])
    ax_alpha_max = fig.add_axes([0.77, 0.355, 0.07, 0.03])
    text_box_alpha_min = TextBox(ax_alpha_min, 'Min', initial=str(slider_alpha.valmin))
    text_box_alpha_max = TextBox(ax_alpha_max, 'Max', initial=str(slider_alpha.valmax))

    # --- Beta Slider & Range Boxes ---
    ax_beta = fig.add_axes([0.1, 0.30, 0.55, 0.03], facecolor=axcolor)
    slider_beta = Slider(ax_beta, 'Beta', 0.1, 50.0, valinit=beta_init)

    ax_beta_min = fig.add_axes([0.68, 0.305, 0.07, 0.03])
    ax_beta_max = fig.add_axes([0.77, 0.305, 0.07, 0.03])
    text_box_beta_min = TextBox(ax_beta_min, 'Min', initial=str(slider_beta.valmin))
    text_box_beta_max = TextBox(ax_beta_max, 'Max', initial=str(slider_beta.valmax))

    # --- x Range Text Boxes ---
    ax_xmin = fig.add_axes([0.1, 0.2, 0.35, 0.03])
    ax_xmax = fig.add_axes([0.55, 0.2, 0.35, 0.03])
    text_box_xmin = TextBox(ax_xmin, 'x min', initial=str(x_min_init))
    text_box_xmax = TextBox(ax_xmax, 'x max', initial=str(x_max_init))

    def update(_):
        # Update the Beta pdf when sliders move
        alpha = slider_alpha.val
        beta = slider_beta.val
        current_dist = dist.Beta(concentration1=alpha, concentration0=beta)
        y_new = jnp.exp(current_dist.log_prob(x))
        line.set_ydata(np.array(y_new))
        fig.canvas.draw_idle()

    def update_x_range(_):
        # Update x range from the text boxes
        nonlocal x
        try:
            new_xmin = float(text_box_xmin.text)
            new_xmax = float(text_box_xmax.text)
            if new_xmin >= new_xmax:
                print("x min must be less than x max")
                return
            x = jnp.linspace(new_xmin, new_xmax, 1000)
            # Recompute PDF with new x
            alpha = slider_alpha.val
            beta = slider_beta.val
            current_dist = dist.Beta(concentration1=alpha, concentration0=beta)
            y_new = jnp.exp(current_dist.log_prob(x))
            line.set_data(np.array(x), np.array(y_new))
            ax.set_xlim(new_xmin, new_xmax)
            fig.canvas.draw_idle()
        except ValueError:
            print("Invalid input for x range")

    def update_alpha_range(_):
        # Update alpha slider range
        try:
            new_min = float(text_box_alpha_min.text)
            new_max = float(text_box_alpha_max.text)
            if new_min >= new_max:
                print("Alpha min must be less than max")
                return
            slider_alpha.valmin = new_min
            slider_alpha.valmax = new_max
            slider_alpha.ax.set_xlim(new_min, new_max)
            # Clamp current value if out of new bounds
            if slider_alpha.val < new_min:
                slider_alpha.set_val(new_min)
            elif slider_alpha.val > new_max:
                slider_alpha.set_val(new_max)
            fig.canvas.draw_idle()
        except ValueError:
            print("Invalid input for alpha slider range")

    def update_beta_range(_):
        # Update beta slider range
        try:
            new_min = float(text_box_beta_min.text)
            new_max = float(text_box_beta_max.text)
            if new_min >= new_max:
                print("Beta min must be less than max")
                return
            slider_beta.valmin = new_min
            slider_beta.valmax = new_max
            slider_beta.ax.set_xlim(new_min, new_max)
            # Clamp current value if out of new bounds
            if slider_beta.val < new_min:
                slider_beta.set_val(new_min)
            elif slider_beta.val > new_max:
                slider_beta.set_val(new_max)
            fig.canvas.draw_idle()
        except ValueError:
            print("Invalid input for beta slider range")

    # Register callbacks
    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)
    text_box_xmin.on_submit(update_x_range)
    text_box_xmax.on_submit(update_x_range)
    text_box_alpha_min.on_submit(update_alpha_range)
    text_box_alpha_max.on_submit(update_alpha_range)
    text_box_beta_min.on_submit(update_beta_range)
    text_box_beta_max.on_submit(update_beta_range)

    plt.show()


if __name__ == "__main__":
    main()
