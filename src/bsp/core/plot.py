import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import pandas as pd
from src.bsp.core.utils import configure_figure
cmti = configure_figure()
from scipy.signal import savgol_filter


#  matplotlib.use('Agg')  # or 'TkAgg', 'Qt5Agg', etc.
#  %matplotlib qt

def adjust_brightness(rgb_in, val):
    hsv = rgb_to_hsv(rgb_in)
    hsv[2] = val  # Adjust the brightness (value component in HSV)
    rgb_out = hsv_to_rgb(hsv)
    return rgb_out

color_segment = {'C4': 'red', 'C5': 'blue', 'C6': 'green', 'C7': 'purple', 'C8': 'orange', 'T1': 'brown', 'NI': 'red'}

def get_cmap_muscles_alt():
    vec_muscle = np.array(["Trapezius", "Deltoid", "Biceps", "Triceps", "ECR", "FCR", "APB", "ADM", "TA", "EDB", "AH", "FDI", "auc_target"])

    cmap_mus_dark = np.array([
        adjust_brightness(np.array([0.6350, 0.0780, 0.1840]), 0.5),  # trapz
        np.array([1, 133, 113]) / 255,  # delt
        np.array([166, 97, 26]) / 255,  # biceps
        np.array([44, 123, 182]) / 255,  # triceps
        np.array([52, 0, 102]) / 255,  # ecr
        adjust_brightness(np.array([0.5, 0.5, 0.5]), 0.3),  # fcr
        np.array([208, 28, 139]) / 255,  # apb
        np.array([77, 172, 38]) / 255,  # adm
        np.array([215, 25, 28]) / 255,  # ta
        np.array([123, 50, 148]) / 255,  # edb
        adjust_brightness(np.array([153, 79, 0]) / 256, 0.4),  # ah
        np.array([231, 226, 61]) / 255,  # fdi
        np.array([255, 100, 0]) / 255,  # auc_target
    ])

    cmap_mus_light = np.array([
        adjust_brightness(np.array([0.6350, 0.0780, 0.1840]), 0.8),  # trapz
        np.array([128, 205, 193]) / 255,  # delt
        np.array([223, 194, 125]) / 255,  # biceps
        np.array([171, 217, 233]) / 255,  # triceps
        adjust_brightness(np.array([200, 40, 0]) / 255, 0.6),  # ecr
        adjust_brightness(np.array([0.5, 0.5, 0.5]), 0.6),  # fcr
        np.array([241, 182, 218]) / 255,  # apb
        np.array([184, 225, 134]) / 255,  # adm
        np.array([253, 174, 97]) / 255,  # ta
        np.array([194, 165, 207]) / 255,  # edb
        adjust_brightness(np.array([153, 79, 0]) / 256, 0.6),  # ah
        adjust_brightness(np.array([23, 54, 124]) / 256, 0.6),  # fdi
        np.array([255, 100, 0]) / 255,  # auc_target
    ])

    # Create a DataFrame to hold muscle names and corresponding colors
    T_color = pd.DataFrame({
        'muscle': vec_muscle,
        'cmap_mus_light': [tuple(c) for c in cmap_mus_light],
        'cmap_mus_dark': [tuple(c) for c in cmap_mus_dark],
    })

    # Convert RGB to hex
    T_color['cmap_mus_light_hex'] = T_color['cmap_mus_light'].apply(
        lambda x: '#%02x%02x%02x' % tuple([int(255 * v) for v in x]))
    T_color['cmap_mus_dark_hex'] = T_color['cmap_mus_dark'].apply(
        lambda x: '#%02x%02x%02x' % tuple([int(255 * v) for v in x]))

    return cmap_mus_dark, cmap_mus_light, vec_muscle, T_color


def plot_data_with_posterior_predictive(cfg, y, x, condition_index, participant_index, visit_index, cxsc_index, run_index, cxsc_none_or_val,
                                        pred_means_new, pred_intervals_new, x_new,
                                        mapping, xlabel, output_file, show=False):
    if cfg['DATA_OPTIONS']['response_transform'] == 'log10':
        base = 10
    elif cfg['DATA_OPTIONS']['response_transform'] == 'log2':
        base = 2
    else:
        raise Exception('base?')
    vec_participant_ix = np.sort(participant_index.unique())
    colors, _, vec_muscle_color, _ = get_cmap_muscles_alt()
    vec_visit_ix = np.sort(visit_index.unique())
    if np.isscalar(cxsc_none_or_val):
        vec_cxsc_ix = np.array([cxsc_none_or_val])
        print(f'Plotting {xlabel} predictions for intensity index {cxsc_none_or_val}...')
    elif cxsc_none_or_val is None:
        print(f'Plotting {xlabel} predictions...')
        vec_cxsc_ix = np.sort(np.unique(cxsc_none_or_val))
    else:
        raise Exception('?')

    vec_run_ix = np.sort(run_index.unique())
    num_muscles = y.shape[1]
    x_new = x_new.reshape(-1)
    colors_run = plt.colormaps['tab20'](np.linspace(0, 1, len(vec_run_ix)))
    type_run = ["o", "d", "s", "X", "p", "h", "H"]
    valid_axes = []
    for ix_i, _ in enumerate(vec_cxsc_ix):
        total_plots = np.unique(np.stack((participant_index.values[cxsc_index.values == vec_cxsc_ix[ix_i]],
                                          visit_index.values[cxsc_index.values == vec_cxsc_ix[ix_i]])).transpose(),
                                axis=0).shape[0]
        if total_plots < 1:
            continue
        fig, axes = plt.subplots(total_plots, num_muscles, figsize=(3.5 * num_muscles * cmti, 3.5 * total_plots * cmti), sharex=True,
                                 squeeze=False)
        gg = []
        plot_index = 0
        str_i = mapping.get('intensity', vec_cxsc_ix[ix_i], '?')
        for ix_p in vec_participant_ix:

            for ix_v in range(len(vec_visit_ix)):
                # Check if there is data for the current participant and visit
                if np.any((participant_index == ix_p) & (visit_index == vec_visit_ix[ix_v]) & (
                        cxsc_index == vec_cxsc_ix[ix_i])):
                    for ix_m in range(num_muscles):
                        str_condition = mapping.get('participant_condition', ix_p)
                        str_p = mapping.get('participant', ix_p, 'UNK')
                        str_v = mapping.get('visit', ix_v, '?')
                        str_m = mapping.get('muscle', ix_m, '?')
                        ix_condition = mapping.get_inverse('condition', str_condition)
                        y_local = y[(participant_index == ix_p) & (visit_index == vec_visit_ix[ix_v]) & (
                                    cxsc_index == vec_cxsc_ix[ix_i]), ix_m]
                        if np.all(np.isnan(y_local)):
                            ylim_ = np.array([-10, +10]).astype(float)
                        else:
                            ylim_ = np.array([np.nanmin(y_local), np.nanmax(y_local)])

                        y_new_local = np.array(pred_means_new[ix_condition, ix_p, ix_v, ix_i, :, ix_m])
                        y_new_local_i = np.array(pred_intervals_new[:, ix_condition, ix_p, ix_v, ix_i, :, ix_m])
                        y_new_local_0 = y_new_local_i[0, ...]
                        y_new_local_1 = y_new_local_i[1, ...]
                        smooth = False
                        if smooth:  # REMOVE THIS IN FUTURE
                            window_length, polyorder = 13, 3
                            y_new_local = savgol_filter(y_new_local, window_length=window_length, polyorder=polyorder)
                            y_new_local_0 = savgol_filter(y_new_local_0, window_length=window_length, polyorder=polyorder)
                            y_new_local_1 = savgol_filter(y_new_local_1, window_length=window_length, polyorder=polyorder)


                        ax = axes[plot_index, ix_m]
                        c = colors[vec_muscle_color == str_m, :]
                        ax.fill_between(x_new, y_new_local_0, y_new_local_1, color=c, alpha=0.1, linewidth=0, edgecolor=None)
                        for ix_r in range(len(vec_run_ix)):
                            if ix_r == 0:
                                label = f'{str_p}, (V{str_v}, {str_i})'
                            else:
                                label = None
                            x_local = x[(participant_index == ix_p) & (visit_index == vec_visit_ix[ix_v]) & (
                                        cxsc_index == vec_cxsc_ix[ix_i]) & (run_index == vec_run_ix[ix_r])].reshape(-1)
                            y_local = y[(participant_index == ix_p) & (visit_index == vec_visit_ix[ix_v]) & (
                                        cxsc_index == vec_cxsc_ix[ix_i]) & (run_index == vec_run_ix[ix_r]), ix_m]
                            ax.scatter(x_local, y_local, label=label, marker=type_run[ix_r], color=np.ones((1, 3))*0.3,
                                       linewidths=1, edgecolors='white', alpha=0.2, s=8)  #  color=colors_run[ix_r])

                            # datax = pd.DataFrame({'x': x_local, 'y': y_local})
                            # grouped_data = datax.groupby('x').mean().reset_index()
                            # x_local_new = grouped_data['x'].values
                            # y_local_new = grouped_data['y'].values
                            # ax.scatter(x_local_new, y_local_new, label=label, marker=type_run[ix_r],
                            #            color=np.ones((1, 3)) * 0.2,
                            #            linewidths=1, edgecolors='white', alpha=0.8, s=8)

                        ax.plot(x_new, y_new_local, color=c, linewidth=1)
                        if plot_index == total_plots - 1:
                            ax.set_xlabel(xlabel)
                        if ix_m == 0:
                            ax.set_ylabel(rf'MEP size ($\log_{{{base}}}$)')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                        p = 0.2
                        if not np.isnan(ylim_).any():
                            ylim_delta = (ylim_[1] - ylim_[0])
                            ylim_ += ylim_delta * np.array([-p, p])
                            ylim_[1] = ylim_[1] + 1e-4

                            ax.set_ylim(ylim_)
                        if (str_p == 'SCA05') and (str_m == "APB") and (str_i == 'supra-sub'):
                            ax.set_ylim([-3.5, 2.75])
                            z1, z2 = -0.8, -0.5
                            ax.axhline(z1, color='k', linestyle='--', linewidth=0.5)
                            ax.axhline(z2, color='k', linestyle='--', linewidth=0.5)

                            # Copy the ax into its own figure
                            fig_single, ax_single = plt.subplots(1, 1, figsize=((1 + 3.5 * 1) * cmti, (0.5 + 3.5 * 1) * cmti))
                            ax_single.plot(x_new, y_new_local, color=c, linewidth=1)
                            ax_single.set_xlabel(xlabel)
                            ax_single.set_ylabel(rf'MEP size ($\log_{{{base}}}$)')
                            ax_single.spines['top'].set_visible(False)
                            ax_single.spines['right'].set_visible(False)
                            ax_single.set_xlim([np.min(x_new), np.max(x_new)])
                            ax_single.set_ylim([z1, z2])
                            fig_single.tight_layout()
                            separate_plot_file = output_file.with_stem(output_file.stem + '_SCA05_APB')
                            fig_single.savefig(separate_plot_file, bbox_inches='tight')
                            # fig_single.show()
                            plt.close(fig_single)

                        ax.set_xlim([np.min(x_new), np.max(x_new)])
                        # ax.legend()
                        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                        valid_axes.append(ax)

                    plot_index += 1

        # fig.tight_layout()
        fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.2)
        for ax in axes.flat:
            ax.legend()
        op = Path(str(output_file).replace(output_file.stem, output_file.stem + '_' + str_i))
        if output_file is not None:
            fig.savefig(op, bbox_inches='tight')
        if show:
            fig.show()
        plt.close(fig)
        print(f'Saved plot to {op}')


def plot_posteriors(cfg, samples, variables, mapping, map_str, output_file, show=False, skip=False):
    if skip:
        return False
    colors, _, vec_muscle_color, _ = get_cmap_muscles_alt()
    if map_str is None:
        num_plots = 1
    else:
        num_plots = len(mapping.get(map_str))
    fig, axes = plt.subplots(len(variables), num_plots,
                             figsize=(5 * num_plots, 5 * len(variables)), squeeze=False)

    for i, var in enumerate(variables):
        if var not in samples:
            continue  # Skip the variable if it's not in samples

        samples_ = samples[var]
        if var == 'bell1_size_muscle':
            vec_cxsc = [i for i in
                        [mapping.get_inverse('intensity', 'supra-sub'), mapping.get_inverse('intensity', 'sub-sub')] if
                        i is not None]
            samples_ = samples_[:, vec_cxsc, :]
        samples_ = samples_.squeeze()
        if samples_.ndim == 1:  # Scalar variable
            sns.histplot(samples_, kde=True, ax=axes[i, 0])
            p = np.mean(samples_ > 0)
            axes[i, 0].set_title(f'{var}, p>0={p:0.2f}')
            axes[i, 0].set_xlabel(var)
            axes[i, 0].set_ylabel('Density')
            # for j in range(1, num_participants):
            #     axes[j, i].axis('off')
        else:  # Array variable
            s = len(samples_.shape)
            if (s == 2) or (s == 3):  # will have to add more code to handle the other dimensions
                for j in range(num_plots):
                    str_muscle = mapping.get(map_str, j)
                    if map_str == 'muscle':
                        c = colors[vec_muscle_color == str_muscle, :]
                    else:
                        c = np.array(np.ones((1, 3)) * 0.5)
                    # sns.histplot(samples_[..., j], kde=True, color=c, ax=axes[i, j])
                    sns.kdeplot(samples_[..., j], color=c, ax=axes[i, j], fill=False)
                    sns.kdeplot(samples_[..., j], color=c, ax=axes[i, j], fill=True, clip=(0, np.max(samples_[..., j])), linewidth=0)
                    p = np.mean(samples_[..., j] > 0, axis=0)
                    p = p.reshape(-1)
                    p_str = ", ".join([f"{val:0.2f}" for val in p])
                    axes[i, j].set_title(f'{str_muscle}, p>0=[{p_str}]')
                    axes[i, j].set_xlabel(var)
                    axes[i, j].set_ylabel(f'{var} Density')
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file)
    if show:
        plt.show()
    plt.close()
