import hashlib
import inspect
import dill as pickle
import tempfile
import toml
from pathlib import Path
import uuid
import re
import numpy as np
import jax.numpy as jnp
import arviz as az
import config
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import pandas as pd


def configure_figure():
    plt.rcParams.update({
        'font.size': 8,  # Base font size for axes labels
        'axes.titlesize': 8,  # Font size for axes titles
        'axes.labelsize': 8,  # Font size for axes labels
        'xtick.labelsize': 6,  # Font size for x-tick labels
        'ytick.labelsize': 6,  # Font size for y-tick labels
        'legend.fontsize': 6,  # Font size for the legend
    })
    # Ensure text remains editable and fonts are embedded
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    cmti = 1 / 2.54
    return cmti


def get_model_hash(model_function):
    source_code = inspect.getsource(model_function)
    return hashlib.sha256(source_code.encode('utf-8')).hexdigest()


def get_data_hash(data):
    data_string = data.to_csv(index=False)
    return hashlib.sha256(data_string.encode('utf-8')).hexdigest()


def get_toml_hash(toml_path):
    with open(toml_path, 'r') as file:
        file_content = file.read()

    return hashlib.sha256(file_content.encode('utf-8')).hexdigest()

def get_hash(cfg_file_path, model_function, data):
    model_hash = get_model_hash(model_function)
    data_hash = get_data_hash(data)
    # cfg_hash = hash_module(config)
    cfg_hash = get_toml_hash(cfg_file_path)
    combined_hash = model_hash + data_hash + cfg_hash
    return hashlib.sha256(combined_hash.encode('utf-8')).hexdigest()


def hash_module(module):
    """
    """
    # Find the file path of the module
    file_path = inspect.getfile(module)

    # Compute the hash of the file contents
    hasher = hashlib.sha256()

    with open(file_path, 'rb') as f:
        content = f.read()
        hasher.update(content)

    return hasher.hexdigest()


# Compute the hash of the config module
config_hash = hash_module(config)


def ungroup_samples(samples, num_chains):
    # this should probably have been called group samples...
    ungrouped_samples = {}
    for param, values in samples.items():
        # Reshape from (num_chains, num_samples, ...) to (num_chains * num_samples, ...)
        num_samples = values.shape[1]
        new_shape = (num_chains * num_samples,) + values.shape[2:]
        ungrouped_samples[param] = values.reshape(new_shape)
    return ungrouped_samples


def save_model(mcmc, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(mcmc, f)
        # pickle.dump((mcmc, mapping), f)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class bidir_dict:
    def __init__(self):
        self.data = {}

    def add_mapping(self, name, forward_dict=None):
        if forward_dict is None:
            forward_dict = {}
        self.data[name] = {
            'forward': forward_dict,
            'backward': {v: k for k, v in forward_dict.items()}
        }

    def get(self, category, key=None, default=None):
        if category in self.data:
            if key is None:
                return self.data[category]['forward']  # Return all forward mappings
            return self.data[category]['forward'].get(key, default)
        return default

    def get_inverse(self, category, key=None, default=None):
        if category in self.data:
            if key is None:
                return self.data[category]['backward']  # Return all backward mappings
            return self.data[category]['backward'].get(key, default)
        return default

    def __setitem__(self, category_key, value):
        category, key = category_key
        if category not in self.data:
            self.add_mapping(category)
        if key in self.data[category]['forward']:
            # Remove the old backward mapping.
            del self.data[category]['backward'][self.data[category]['forward'][key]]
        if value in self.data[category]['backward']:
            # Remove the old forward mapping.
            del self.data[category]['forward'][self.data[category]['backward'][value]]
        self.data[category]['forward'][key] = value
        self.data[category]['backward'][value] = key

    def __delitem__(self, category_key):
        category, key = category_key
        value = self.data[category]['forward'].pop(key)
        del self.data[category]['backward'][value]

    def __repr__(self):
        # Build a human-readable string for the mapping.
        result = "bidir_dict:\n"
        for category, mapping in self.data.items():
            result += f"  {category}:\n"
            result += "    Forward:\n"
            for key, value in mapping['forward'].items():
                result += f"      {key}: {value}\n"
            result += "    Backward:\n"
            for key, value in mapping['backward'].items():
                result += f"      {key}: {value}\n"
        return result


def map_run_indices(new_time, data):
    """
    Maps new_time to corresponding ix_run values from data based on the closest time match.
    """
    # Convert the DataFrame columns to numpy arrays for efficient computation
    time_array = data['time'].to_numpy()
    ix_run_array = data['run_index'].to_numpy()

    # Create an empty array for the run indices
    run_index_new = np.zeros(new_time.shape).reshape(-1)

    # Iterate through each element in new_time
    for i, t in enumerate(new_time.reshape(-1)):
        # Find the index of the closest time in the time_array
        closest_index = np.argmin(np.abs(time_array - t))
        # Assign the corresponding ix_run value to run_index_new
        run_index_new[i] = ix_run_array[closest_index]

    # Reshape run_index_new to the same shape as new_time
    run_index_new = run_index_new.reshape(new_time.shape)
    run_index_new = run_index_new.astype(int).reshape(-1)
    return run_index_new


def generate_predictions(cfg, new_SPI_target, new_time, condition_index, participant_index, visit_index, cxsc_index_or_val, data, predictive, rng_key, y, num_samples, average_count):
    """
    Generate predictions for new SPI targets and extract posterior predictive means and credible intervals.
    """

    vec_visit = np.sort(np.unique(visit_index))
    if np.isscalar(cxsc_index_or_val):
        vec_cxsc = np.array([cxsc_index_or_val])
        print(f'Generating predictions for intensity index {cxsc_index_or_val}...')
    else:
        print(f'Generating predictions...')
        vec_cxsc = np.sort(np.unique(cxsc_index_or_val))

    num_visit = len(vec_visit)
    num_x = new_time.shape[0]
    num_muscles = y.shape[1]
    num_cxsc = len(vec_cxsc)
    num_participants = len(np.unique(participant_index))
    num_conditions = len(np.unique(condition_index))

    new_predictions = np.empty((num_conditions, num_participants, num_visit, num_cxsc, num_samples, num_x, num_muscles))
    for condition_ix, condition in enumerate(np.sort(np.unique(condition_index))):
        condition_index_new = np.full(new_SPI_target.shape, condition)[:, 0]
        for p_idx, participant in enumerate(np.sort(np.unique(participant_index))):
            participant_index_new = np.full(new_SPI_target.shape, participant)[:, 0]
            for v_idx, visit in enumerate(vec_visit):
                visit_index_new = np.full(participant_index_new.shape, visit)
                for c_idx, cxsc in enumerate(vec_cxsc):
                    cxsc_index_new = np.full(participant_index_new.shape, cxsc)

                    case_local = (data['participant_index'] == participant) & (data['visit_index'] == visit)
                    if np.sum(case_local) > 0:
                        data_local = data[case_local].copy()
                        new_run_index = map_run_indices(new_time, data_local)
                        pred = predictive(rng_key, new_SPI_target, new_time, int(y.shape[-1]), new_run_index.astype(int), visit_index_new, participant_index_new,
                                      condition_index_new,  cxsc_index_new, average_count, cfg['MODEL_OPTIONS'])
                        new_predictions[condition_ix, p_idx, v_idx, c_idx, :, :, :] = pred['y']

                    else:
                        new_predictions[condition_ix, p_idx, v_idx, c_idx, :, :, :] = np.nan

    ix_chain = 4  # chain is in
    pred_means_new = jnp.mean(new_predictions, axis=ix_chain)
    new_predictions = np.moveaxis(new_predictions, ix_chain, 0)
    new_predictions = np.expand_dims(new_predictions, axis=0)
    hdi = compute_hdi(new_predictions)
    pred_intervals_new = np.moveaxis(hdi, -1, 0)

    return pred_means_new, pred_intervals_new


def make_parameter_mask(num_muscles, num_runs, num_visits, num_participants, num_descriptors, num_intensities, response_obs, run_index, visit_index, participant_index, condition_index, intensity_index, mask_types):
    mask_muscle = np.zeros((num_runs, num_visits, num_participants, num_descriptors, num_intensities, num_muscles), dtype=bool)
    for intensity in range(num_intensities):
        for descriptor in range(num_descriptors):
            for participant in range(num_participants):
                for visit in range(num_visits):
                    for run in range(num_runs):
                        x = (intensity_index == intensity) & (condition_index == descriptor) & (participant_index == participant) & (visit_index == visit) & (run_index == run)
                        if x.any() and response_obs is not None:
                            mask_muscle[run, visit, participant, descriptor, intensity, :] = np.any(np.isfinite(response_obs[x, :]), axis=0)
                        else:
                            mask_muscle[run, visit, participant, descriptor, intensity, :] = x.any()

    mask_run = np.any(mask_muscle, axis=-1)
    mask_run = np.expand_dims(mask_run, axis=-1)
    mask_run = np.broadcast_to(mask_run, mask_muscle.shape)

    mask_visit = np.any(mask_run, axis=0)
    mask_participant = np.any(mask_visit, axis=0)
    mask_descriptor = np.any(mask_participant, axis=0)
    mask_intensity = np.any(mask_descriptor, axis=0)

    if 'run' not in mask_types or response_obs is None:
        mask_run = np.ones_like(mask_run, dtype=bool)

    if 'visit' not in mask_types or response_obs is None:
        mask_visit = np.ones_like(mask_visit, dtype=bool)

    if 'participant' not in mask_types or response_obs is None:
        mask_participant = np.ones_like(mask_participant, dtype=bool)

    if 'descriptor' not in mask_types or response_obs is None:
        mask_descriptor = np.ones_like(mask_descriptor, dtype=bool)

    if 'intensity' not in mask_types or response_obs is None:
        mask_intensity = np.ones_like(mask_intensity, dtype=bool)

    return mask_run, mask_visit, mask_participant, mask_descriptor, mask_intensity, mask_muscle


def make_obs_mask(response_obs, mask_types):
    if 'obs' not in mask_types or response_obs is None:
        mask_obs = np.ones_like(response_obs, dtype=bool)
    else:
        mask_obs = np.invert(np.isnan(response_obs))
    return mask_obs


def filter_posterior_samples(posterior_samples, masks):
    # GPT generated, have not verified.
    filtered_samples = {}

    for var_name, samples in posterior_samples.items():
        sample_shape = samples.shape[2:]  # Exclude chain and draw dimensions

        for mask_name, mask in masks.items():
            mask_shape = mask.shape

            # If the mask can be broadcasted to the sample shape, apply the mask
            if np.array_equal(sample_shape[-len(mask_shape):], mask_shape):
                filtered_samples[var_name] = samples[:, :, mask]
                break
        else:
            # If no matching mask is found, keep the samples unchanged
            filtered_samples[var_name] = samples

    return filtered_samples


def extract_dims_coords(posterior_samples, mcmc):
    dims = {}
    coords = {}

    for var_name, samples in posterior_samples.items():
        sample_shape = samples.shape[2:]  # Exclude chain and draw dimensions
        dim_names = []
        for dim_index, size in enumerate(sample_shape):
            dim_names.append(f"{var_name}_dim_{dim_index}")

        dims[var_name] = dim_names

        for dim_name, size in zip(dim_names, sample_shape):
            if dim_name not in coords:
                coords[dim_name] = np.arange(size)

    return dims, coords


def write_cfg_to_toml(cfg):
    """
    Writes the content of the cfg dictionary to a TOML file in a temporary directory,
    skipping any keys with Path values. Returns the file path of the created TOML file.
    """

    # Edits to the config
    cfg['LOAD_SAVED_MODEL'] = True  # because when you load it's going to be true anyway
    if cfg['DATA_OPTIONS']['type'] == 'intraoperative':
        cfg['DATA_OPTIONS']['noninvasive'] = None
        cfg['NI_RECOMPUTE_RUNS'] = None
        cfg['NI_RENAME_VISITS'] = None
        cfg['NI_EXCLUDE'] = None
    elif cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        cfg['DATA_OPTIONS']['intraoperative'] = None
        cfg['IO_EXCLUDE'] = None

    # Identify the temporary folder
    temp_dir = Path(tempfile.gettempdir())

    # Filter out entries with Path objects
    filtered_cfg = {
        key: value for key, value in cfg.items()
        if not isinstance(value, (Path, dict))
    }

    # Filter nested dictionaries if they contain Path objects
    for key, value in cfg.items():
        if isinstance(value, dict):
            filtered_cfg[key] = {k: v for k, v in value.items() if not isinstance(v, Path)}

    # Generate a unique filename for the TOML file
    unique_filename = f"config_{uuid.uuid4().hex}.toml"
    toml_file_path = temp_dir / unique_filename

    # Write the filtered dictionary to the TOML file
    with open(toml_file_path, 'w') as toml_file:
        toml.dump(filtered_cfg, toml_file)

    # Return the path of the created TOML file
    return toml_file_path


def extract_target_matrix(data):
    # Find the maximum indices to define the shape of the matrix
    max_run = data['run_index'].max() + 1
    max_visit = data['visit_index'].max() + 1
    max_participant = data['participant_index'].max() + 1
    max_cxsc = data['cxsc_index'].max() + 1

    # Initialize the output matrix with empty strings
    op_muscle = np.full((max_run, max_visit, max_participant, max_cxsc), '', dtype=object)
    op_segment = np.full((max_run, max_visit, max_participant, max_cxsc), '', dtype=object)

    # Iterate through the DataFrame and populate the matrix
    for _, row in data.iterrows():
        ix_r = row['run_index']
        ix_v = row['visit_index']
        ix_p = row['participant_index']
        ix_i = row['cxsc_index']
        if 'target_muscle' in row:
            op_muscle[ix_r, ix_v, ix_p, ix_i] = row['target_muscle']
        elif 'muscle_targeted' in row:
            op_muscle[ix_r, ix_v, ix_p, ix_i] = row['main_targeted_side'] + row['muscle_targeted']
        if 'sc_level' in row:
            op_segment[ix_r, ix_v, ix_p, ix_i] = row['sc_level']
        else:
            op_segment[ix_r, ix_v, ix_p, ix_i] = ''
    return op_muscle, op_segment


def compute_hdi(samples, n_chains=None, hdi_prob=0.95):
    if n_chains is not None:
        # put the first dimension as the number of chains
        n_draws = samples.shape[0] // n_chains
        samples = samples.reshape((n_chains, n_draws) + samples.shape[1:])

    hdi = az.hdi(np.array(samples), hdi_prob=hdi_prob)

    return hdi


def parse_details_from_paths(example_paths):
    # Use the first path to extract the base model
    base_model_match = re.search(r'/(v[\d_]+noninvasive_model\d+_[A-Z_]+n\d+)_', example_paths[0])
    if not base_model_match:
        raise ValueError("Could not extract the base model from the path.")
    base_model = base_model_match.group(1)

    # Extract session_ids and unique_ids from all example paths
    session_ids = []
    unique_ids = []
    for path in example_paths:
        session_id_match = re.search(r'_s(\d+)_', path)
        unique_id_match = re.search(r'_s\d+_([a-f0-9]{64})/', path)
        if session_id_match and unique_id_match:
            session_ids.append(int(session_id_match.group(1)))
            unique_ids.append(unique_id_match.group(1))

    return base_model, session_ids, unique_ids


def generate_filenames(base_model, session_ids, unique_ids):
    filenames = []
    for session_id, unique_id in zip(session_ids, unique_ids):
        filename = f"{base_model}_s{session_id:05d}_{unique_id}/mcmc_model.pkl"
        filenames.append(filename)
    return filenames


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


def plot_data_with_posterior_predictive(cfg, y, x, condition_index, participant_index, visit_index, cxsc_index, run_index, average_count, cxsc_none_or_val,
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
                            plt.close(fig_single)

                        ax.set_xlim([np.min(x_new), np.max(x_new)])

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


if __name__ == "__main__":
    pass
