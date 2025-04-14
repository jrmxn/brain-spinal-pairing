import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox
import numpyro.distributions as dist


def main():
    # Create a figure with a decent size and adjust layout so there's space below the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95)

    # Position the main plot a bit higher so we have space below for sliders and text boxes
    ax.set_position([0.1, 0.45, 0.8, 0.45])  # [left, bottom, width, height]

    # Initial x range and Gamma parameters
    x_min_init = 1e-3
    x_max_init = 5
    x = jnp.linspace(x_min_init, x_max_init, 1000)
    conc_init = 2.0
    rate_init = 1.0

    gamma_dist = dist.Gamma(concentration=conc_init, rate=rate_init)
    y = jnp.exp(gamma_dist.log_prob(x))
    line, = ax.plot(np.array(x), np.array(y), lw=2)
    ax.set_xlabel("x")
    ax.set_ylabel("pdf")
    ax.set_title("Numpyro Gamma Distribution PDF")
    ax.set_xlim(x_min_init, x_max_init)

    # Add grid to the plot
    ax.grid(True)

    # Create axes for sliders and text boxes below the main plot
    axcolor = 'lightgoldenrodyellow'

    # --- Concentration Slider & Range Boxes ---
    ax_conc = fig.add_axes([0.1, 0.35, 0.55, 0.03], facecolor=axcolor)
    slider_conc = Slider(ax_conc, 'Concentration', 0.1, 500.0, valinit=conc_init)

    ax_conc_min = fig.add_axes([0.68, 0.355, 0.07, 0.03])
    ax_conc_max = fig.add_axes([0.77, 0.355, 0.07, 0.03])
    text_box_conc_min = TextBox(ax_conc_min, 'Min', initial=str(slider_conc.valmin))
    text_box_conc_max = TextBox(ax_conc_max, 'Max', initial=str(slider_conc.valmax))

    # --- Rate Slider & Range Boxes ---
    ax_rate = fig.add_axes([0.1, 0.30, 0.55, 0.03], facecolor=axcolor)
    slider_rate = Slider(ax_rate, 'Rate', 0.1, 500.0, valinit=rate_init)

    ax_rate_min = fig.add_axes([0.68, 0.305, 0.07, 0.03])
    ax_rate_max = fig.add_axes([0.77, 0.305, 0.07, 0.03])
    text_box_rate_min = TextBox(ax_rate_min, 'Min', initial=str(slider_rate.valmin))
    text_box_rate_max = TextBox(ax_rate_max, 'Max', initial=str(slider_rate.valmax))

    # --- x Range Text Boxes ---
    # Placed a bit lower so they don't collide with sliders
    ax_xmin = fig.add_axes([0.1, 0.2, 0.35, 0.03])
    ax_xmax = fig.add_axes([0.55, 0.2, 0.35, 0.03])
    text_box_xmin = TextBox(ax_xmin, 'x min', initial=str(x_min_init))
    text_box_xmax = TextBox(ax_xmax, 'x max', initial=str(x_max_init))

    def update(val):
        # Update the gamma pdf when sliders move
        conc = slider_conc.val
        rate = slider_rate.val
        gamma_dist = dist.Gamma(concentration=conc, rate=rate)
        y_new = jnp.exp(gamma_dist.log_prob(x))
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
            conc = slider_conc.val
            rate = slider_rate.val
            gamma_dist = dist.Gamma(concentration=conc, rate=rate)
            y_new = jnp.exp(gamma_dist.log_prob(x))
            line.set_data(np.array(x), np.array(y_new))
            ax.set_xlim(new_xmin, new_xmax)
            fig.canvas.draw_idle()
        except ValueError:
            print("Invalid input for x range")

    def update_conc_range(_):
        # Update concentration slider range
        try:
            new_min = float(text_box_conc_min.text)
            new_max = float(text_box_conc_max.text)
            if new_min >= new_max:
                print("Concentration min must be less than max")
                return
            slider_conc.valmin = new_min
            slider_conc.valmax = new_max
            slider_conc.ax.set_xlim(new_min, new_max)
            # Clamp current value if out of new bounds
            if slider_conc.val < new_min:
                slider_conc.set_val(new_min)
            elif slider_conc.val > new_max:
                slider_conc.set_val(new_max)
            fig.canvas.draw_idle()
        except ValueError:
            print("Invalid input for concentration slider range")

    def update_rate_range(_):
        # Update rate slider range
        try:
            new_min = float(text_box_rate_min.text)
            new_max = float(text_box_rate_max.text)
            if new_min >= new_max:
                print("Rate min must be less than max")
                return
            slider_rate.valmin = new_min
            slider_rate.valmax = new_max
            slider_rate.ax.set_xlim(new_min, new_max)
            # Clamp current value if out of new bounds
            if slider_rate.val < new_min:
                slider_rate.set_val(new_min)
            elif slider_rate.val > new_max:
                slider_rate.set_val(new_max)
            fig.canvas.draw_idle()
        except ValueError:
            print("Invalid input for rate slider range")

    # Register callbacks
    slider_conc.on_changed(update)
    slider_rate.on_changed(update)
    text_box_xmin.on_submit(update_x_range)
    text_box_xmax.on_submit(update_x_range)
    text_box_conc_min.on_submit(update_conc_range)
    text_box_conc_max.on_submit(update_conc_range)
    text_box_rate_min.on_submit(update_rate_range)
    text_box_rate_max.on_submit(update_rate_range)

    plt.show()


if __name__ == "__main__":
    main()
