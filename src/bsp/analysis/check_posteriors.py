from pathlib import Path
import pandas as pd
import numpy as np
import toml
import re
from numpyro.infer import Predictive
import shutil
import os
import jax.random as random
from numpyro.contrib.render import render_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr, gaussian_kde, wilcoxon, ranksums
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import importlib
import math
from config import BASE_DIR, DATA_FOLDER
from config_analysis import CONFIG_ANALYSIS
from src.bsp.core.filter import filter_data as filter
from src.bsp.core.filter import filter_ni as filter_ni  # just for mep plots
from src.bsp.core.utils import ungroup_samples, load_model, generate_predictions, extract_target_matrix, compute_hdi
from src.bsp.core.utils import plot_data_with_posterior_predictive, plot_posteriors, get_cmap_muscles_alt, color_segment
from src.bsp.core.utils import make_parameter_mask, make_obs_mask, configure_figure
from src.bsp.core.model import g
import pickle
from copy import deepcopy
import warnings
from hbmep.model.utils import Site as site
from hbmep.nn.functional import solve_rectified_logistic
from src.bsp.recruitment.functional_median import solve_rectified_logistic_median2 as solve_rectified_logistic_median


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="The figure layout has changed to tight")


site.m50uv = "m50uv"


CMTI = configure_figure()


model_module = importlib.import_module(f"src.bsp.core.model")


def optimal_division(total, priority="rows"):
    cols = int(math.sqrt(total))

    while True:
        rows = math.ceil(total / cols)

        # Check if rows * cols is greater than or equal to total
        if rows * cols >= total:
            if priority == "rows":
                return rows, cols
            elif priority == "columns":
                return cols, rows
            else:
                raise Exception("?")

        cols -= 1


def compute_central(samples, n_chains=None, op="median"):
    if n_chains is not None:
        # Put the first dimension as the number of chains
        n_draws = samples.shape[0] // n_chains
        samples = samples.reshape((n_chains, n_draws) + samples.shape[1:])

    # Compute the mean across the chains and samples
    if op == "median":
        v = np.median(samples, axis=(0, 1))
    elif op == "mean":
        v = np.mean(samples, axis=(0, 1))
    else:
        raise Exception("?")

    return v


hdi_g = lambda x, m, w, s: compute_hdi(np.expand_dims(g(x, m, w, s, 1), axis=-1)).squeeze()


cen_g = lambda x, m, w, s: compute_central(np.expand_dims(g(x, m, w, s, 1), axis=-1)).squeeze()


pctchg_core = lambda x, base: ((base ** x) - 1) * 100


h_use_robust_linear = lambda: False


h_skip_slow_plots = lambda: True


h_alias = lambda x: x.replace('SCA', 'UI').replace('SCS', 'SCI')


def generate_curves(pi, num_cxscs, num_visits, n_plots, num_participants, target_muscle, muscles,
                    mask_visit, posterior_samples_grouped, mapping, mask_muscle, zero_pi=True):
    """
    """

    s = posterior_samples_grouped["s"][:, :, :, :, :, :, :]
    mask_local = np.any(mask_muscle[:, :, :, :, :, :], axis=0)
    s = np.where(mask_local[None, None, ...], s, np.nan)

    m = posterior_samples_grouped["c"]
    w = posterior_samples_grouped["w"]
    mask_local = np.any(mask_muscle, axis=(0, 1, 4))
    m = np.where(mask_local[None, None, ...], m, np.nan)
    w = np.where(mask_local[None, None, ...], w, np.nan)

    Z = np.zeros((len(pi), num_cxscs, num_visits, n_plots, num_participants)) * np.nan
    for ix_m in range(n_plots):
        for ix_p in range(num_participants):
            ix_c = mapping.get_inverse("condition", mapping.get("participant_condition", ix_p))
            X = np.zeros((len(pi), num_cxscs, num_visits)) * np.nan
            for ix_i in range(num_cxscs):
                for ix_v in range(num_visits):
                    if ix_m == n_plots - 1:
                        case_target = [x for x in np.unique(target_muscle[:, ix_v, ix_p, ix_i]) if x != ""]
                        if len(case_target) == 0:
                            continue
                        str_target = case_target[0][1:]
                        if muscles == ["auc_target"]:
                            ix_target = 0
                        else:
                            ix_target = np.where([muscle == str_target for muscle in muscles])[0]
                            if np.size(ix_target) == 0:
                                continue  # the target is not in the set
                            else:
                                ix_target = ix_target[0]
                    else:
                        ix_target = ix_m

                    if mask_visit[ix_v, ix_p, ix_c, ix_i, ix_target]:
                        w_local = w[:, :, ix_p, ix_c, ix_target]
                        s_local = s[:, :, ix_v, ix_p, ix_c, ix_i, ix_target]
                        if zero_pi:
                            m_local = 0
                        else:
                            m_local = m[:, :, ix_p, ix_c, ix_target]

                        y_cen = np.array([cen_g(x_, m_local, w_local, s_local) for x_ in pi])
                        X[:, ix_i, ix_v] = y_cen
            Z[:, :, :, ix_m, ix_p] = X

    return Z


def get_target(target_muscle, muscles, ix_v, ix_p, ix_i):
    case_target = [x for x in np.unique(target_muscle[:, ix_v, ix_p, ix_i]) if x != ""]
    if len(case_target) == 0:
        return None, None
    str_target = case_target[0][1:]
    if muscles == ["auc_target"]:
        ix_target = 0
    else:
        if np.sum([muscle == str_target for muscle in muscles]) == 0:
            ix_target = 0
            print("target muscle not in muscle set - defaulting to 0 - ignore target muscle set")
        else:
            ix_target = np.where([muscle == str_target for muscle in muscles])[0]
            if np.size(ix_target) == 0:
                ix_target = None  # the target is not in the set
            else:
                ix_target = ix_target[0]
    return str_target, ix_target


def linear_model(x, y, x_eval_min=None, x_eval_max=None, add_offset=True, use_robust=None, num=100):
    def get_stars(p):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = "⠀"
        return star
    if x_eval_min is None:
        x_eval_min = np.nanmin(x)
    if x_eval_max is None:
        x_eval_max = np.nanmax(y)
    if use_robust is None:
        use_robust = h_use_robust_linear()
    x_fit = np.linspace(x_eval_min, x_eval_max, num)
    if (x.shape[-1] < 3) or (y.shape[-1] < 3):
        y_fit = x_fit * np.nan
        corr_value, corr_p = np.nan, np.nan
        slope_value, slope_p = np.nan, np.nan
        r_squared = np.nan
    else:
        if add_offset:
            X = sm.add_constant(x)
            if X.shape[-1] == 1: # if X is already a constant then add_constant doesn't add a constant... so:
                X_fit = x_fit[:, np.newaxis]
            else:
                X_fit = sm.add_constant(x_fit)
        else:
            X = x[:, np.newaxis]
            X_fit = x_fit[:, np.newaxis]

        if use_robust:
            lm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
            corr_value, corr_p = spearmanr(x, y)
            r_squared = np.nan
        else:
            lm_model = sm.OLS(y, X).fit()
            corr_value, corr_p = pearsonr(x, y)
            r_squared = lm_model.rsquared

        slope_p = lm_model.pvalues[-1]
        slope_value = lm_model.params[-1]
        y_fit = lm_model.predict(X_fit)

    corr_star = get_stars(corr_p)
    slope_star = get_stars(slope_p)

    stats = {"corr_value": corr_value, "corr_p": corr_p, "corr_star": corr_star, "r2_value": r_squared,
             "slope_value": slope_value, "slope_p": slope_p, "slope_star": slope_star}
    return x_fit, y_fit, stats


def get_m50uv(ps, site, m50uv=50e-3, use_median=True):
    """
    Units are in mV so 50e-3 makes sense.
    Using the mean is in fact not quite right, because the curve is defined as the mean of the gamma distribution - not the median.
     You would need the median curve in order to get the equivalent of the 5/10. Then you can take the mean of those posteriors.
     One way to do this is to predict with the same intensity instead of unique intensities.
     Then take the median over the intensities while maintaining the draws.
     Then take the mean over the draws to get the point estimate.
     But it's extensive because it requires a very fine intensity resolution AND many repeats per intensity.
     Another way is to numerically solve for the median crossing of m50uv.
    """
      # because base
    if use_median:
        m50uv = solve_rectified_logistic_median(
            m50uv, ps[site.a], ps[site.b],
            ps[site.L], ps[site.ell], ps[site.H],
            ps[site.c_1], ps[site.c_2]
        )
    else:
        m50uv = solve_rectified_logistic(
            m50uv, ps[site.a], ps[site.b],
            ps[site.L], ps[site.ell], ps[site.H]
        )

    return m50uv


def load_and_process_hbmep(str_intensity, model_name, feature1, mapping, cfg, visit=0, mepsizetype="auc", short=False):
    muscles = list(mapping.get("muscle").values())
    str_short = "_s" if short else ""
    str_smooth = "_smooth"  # "" or "_smooth"
    str_mepsizetype = "_" + mepsizetype
    str_s50 = ""  # "" or "_s50"
    visit = f"v{visit}"
    vs = cfg["DATA_OPTIONS"]["es"]
    vs = vs.split("_")[0] + "_"
    f_hbmep = lambda x: f"hbmep_{vs}{model_name}_{'_'.join(muscles)}_participant_visit_{feature1}_{str_intensity}_mixT_{visit}{str_smooth}{str_mepsizetype}{str_s50}{x}"
    source_path = Path(BASE_DIR) / "hbmep" / f_hbmep(str_short) / "inference.pkl"
    if not source_path.exists():
        warnings.warn(f"hbmep file does not exist:\n{source_path}\n Trying to fall back on short version.")
        source_path = Path(BASE_DIR) / "hbmep" / f_hbmep("_s") / "inference.pkl"
    if not source_path.exists():
        raise Exception("hbmep file does not exist.")
    with open(source_path, "rb") as f:
        model, mcmc, ps, unpack, hbmep_mask, data = pickle.load(f)

    es = "_" + source_path.parts[-2].replace("dnc_hbmep_", "")

    # Compute s50 using the rectified logistic function - skip now since we just use a different model for s50
    # site.s50 = "s50"
    # mp = ps[site.L] + ps[site.H] / 2
    # ps[site.s50] = solve_rectified_logistic(
    #     mp, ps[site.a], ps[site.b],
    #     ps[site.L], ps[site.ell], ps[site.H]
    # )

    if mepsizetype == "pkpk":
        ps[site.m50uv] = get_m50uv(ps, site)

    # Apply mask for each site
    nd_hbmep = ps[site.a].shape[0]
    np_hbmep = ps[site.a].shape[1]
    site_mask = []
    for k, v in ps.items():
        v = np.asarray(v)
        if v.shape == (nd_hbmep,):
            pass
        elif v.shape[1] == np_hbmep:
            site_mask.append(k)
        else:
            pass
    for s in site_mask:
        mask_bc = np.broadcast_to(hbmep_mask, ps[s].shape)
        ps[s] = np.where(mask_bc, ps[s], np.nan)

    # make sure we can just use the muscles string list for both hbmep and bsp
    assert all(x == y for x, y in zip(model.response, muscles))

    hbmep_dict = {
        "model": model,
        "mcmc": mcmc,
        "ps": ps,
        "unpack": unpack,
        "data": data,
        "es": es
    }
    return hbmep_dict


def make_prediction(dhbmep, str_intensity, num_conditions, num_visits, num_participants, mapping, num_points=50, max_intensity=80,compute_full_pp=True):
    model_hbmep, hbmep_ps, hbmep_unpack = dhbmep["model"], dhbmep["ps"], dhbmep["unpack"]
    if model_hbmep.use_mixture:
        p_outlier = hbmep_ps[site.outlier_prob]
        hbmep_ps[site.outlier_prob] = hbmep_ps[site.outlier_prob] * 0

    df_template = pd.DataFrame({
        str_intensity: np.linspace(0, max_intensity, num_points),
        model_hbmep.features[0]: np.zeros(num_points, dtype=int),
        model_hbmep.features[1]: np.zeros(num_points, dtype=int)
    })
    hbmep_unpack_pv = np.array([[int(x) for x in hbmep_unpack["participant_visit"][y].split("_")] for y in range(len(hbmep_unpack["participant_visit"]))])

    pp = [[[None for _ in range(num_conditions)] for _ in range(num_visits)] for _ in range(num_participants)]
    for ix_f in range(len(hbmep_unpack_pv)):
        f = hbmep_unpack_pv[ix_f]
        ix_p, ix_v = f[0], f[1]  # get these indices from hbmep (the feature was built with ix_p and ix_v in the first place)
        ix_c = mapping.get_inverse("condition", mapping.get("participant_condition", ix_p))
        # get whether ix_p is SCI/U, then look up whether in the ps, where SCI/U is:
        case_ps_f2 = mapping.get(model_hbmep.features[1], ix_p) == hbmep_unpack[model_hbmep.features[1]]
        df_local = deepcopy(df_template)
        df_local[model_hbmep.features[0]] = ix_f
        df_local[model_hbmep.features[1]] = np.where(case_ps_f2)[0][0]
        if compute_full_pp:
            vec_participants = list(mapping.get("participant").values())
        else:
            vec_participants = ["SCA03", "SCA09", "SCS01"]  # TODO: just to speed things up for now, remove eventually
        if mapping.get("participant", ix_p) in vec_participants:
            # add a row with the highest indexed condition because otherwise things fail sometimes...
            row = pd.DataFrame([dhbmep["data"][df_local.columns].max().astype(int).to_dict()])
            df_local = pd.concat([df_local, row]).reset_index(drop=True).copy()

            pred_local = model_hbmep.predict(df=df_local, posterior_samples=hbmep_ps, return_sites=[site.mu, site.obs])

            # remove added row
            pred_local = {u:v[:, :-1, :] for u, v in pred_local.items()}
            pp[ix_p][ix_v][ix_c] = pred_local

    return pp, df_template


def make_prediction_average(dhbmep, str_intensity, num_conditions, num_visits, num_participants, mapping, num_points=50, max_intensity=80):
    model_hbmep, hbmep_ps, hbmep_unpack = dhbmep["model"], dhbmep["ps"], dhbmep["unpack"]
    if model_hbmep.use_mixture:
        p_outlier = hbmep_ps[site.outlier_prob]
        hbmep_ps[site.outlier_prob] = hbmep_ps[site.outlier_prob] * 0

    hbmep_ps_avg = {}

    nd_hbmep = hbmep_ps[site.a].shape[0]
    np_hbmep = hbmep_ps[site.a].shape[1]
    for k, v in hbmep_ps.items():
        v = np.asarray(v)  # Ensure it's an array
        if v.shape == (nd_hbmep,):  # Keep these as they are
            hbmep_ps_avg[k] = v
        elif v.shape[1] == np_hbmep:  # Check if second dim is 28 (participants)
            # Mask was applied above so should be OK
            hbmep_ps_avg[k] = np.nanmean(v, axis=1, keepdims=True)  # Average over participants
        else:
            hbmep_ps_avg[k] = v  # Leave other shapes unchanged

    df_template = pd.DataFrame({
        str_intensity: np.linspace(0, max_intensity, num_points),
        model_hbmep.features[0]: np.zeros(num_points, dtype=int),
        model_hbmep.features[1]: np.zeros(num_points, dtype=int)
    })

    pp = [[[None for _ in range(num_conditions)] for _ in range(num_visits)] for _ in range(1)]
    for ix_c in range(hbmep_ps[site.a].shape[2]):
        df_local = deepcopy(df_template)
        df_local[model_hbmep.features[0]] = 0
        df_local[model_hbmep.features[1]] = ix_c
        pred_local = model_hbmep.predict(df=df_local, posterior_samples=hbmep_ps_avg)
        pp[0][0][ix_c] = pred_local

    return pp, df_template


def extract_s_vs_threshold(posterior_samples_grouped, muscles, dhbmep, mapping, mask_muscle, data, hdidelta_a_cutoff=None, site_flag=None):
    model_hbmep, hbmep_ps, hbmep_unpack = dhbmep["model"], dhbmep["ps"], dhbmep["unpack"]
    site_flag = site.a if site_flag is None else site_flag
    if site_flag == "a":
        hdidelta_cutoff = hdidelta_a_cutoff
    else:
        hdidelta_cutoff = None
    num_participants = len(mapping.get("participant"))  # should match outside
    ix_v = 0
    ix_i = mapping.get_inverse("intensity", "supra-sub")
    s = posterior_samples_grouped["s"][:, :, ix_v, :, :, ix_i, :]
    mask_local = np.any(mask_muscle[:, ix_v, :, :, ix_i, :], axis=0)
    s = np.where(mask_local[None, None, :, :, :], s, np.nan)
    s_mea = np.mean(s, axis=(0, 1))

    hbmep_muscles = model_hbmep.response
    muscle_map = {muscle: idx for idx, muscle in enumerate(muscles)}  # TODO: confirm this muscle indexing
    hbmep_param = np.full((num_participants, len(muscles)), np.nan)  # thresholds
    hbmep_param_hdidelta = np.full((num_participants, len(muscles)), np.nan)  # thresholds
    ie_size = np.full((num_participants, len(muscles)), np.nan)  # ie size
    stim_intensity = np.full((num_participants, 2), np.nan)  # stim intensity
    vec_condition = np.full((num_participants, 1), np.nan)  # stim intensity
    hbmep_unpack_pv = np.array([[int(x) for x in hbmep_unpack["participant_visit"][y].split("_")] for y in
                                range(len(hbmep_unpack["participant_visit"]))])
    for ix_p in range(num_participants):
        ix_c = mapping.get_inverse("condition", mapping.get("participant_condition", ix_p))
        case_ps_f0 = (hbmep_unpack_pv[:, 0] == ix_p) & (hbmep_unpack_pv[:, 1] == ix_v)
        case_ps_f1 = mapping.get(model_hbmep.features[1], ix_p) == hbmep_unpack[model_hbmep.features[1]]

        s_mea_p = s_mea[ix_p, ix_c, :]  # participant x cond (SCI vs U) x muscles
        ie_size[ix_p, :] = s_mea_p  # Direct assignment as x matches muscles_1
        if np.any(case_ps_f0):
            ps_full = hbmep_ps[site_flag][:, case_ps_f0, case_ps_f1, :]
            ps_mea = np.mean(ps_full, axis=0)[0]  # 1 x n_muscles_2
            ps_hdi = compute_hdi(ps_full[np.newaxis, ...])
            ps_hdidelta = ps_hdi[..., -1] - ps_hdi[..., 0]
            for idx, muscle in enumerate(hbmep_muscles):
                if muscle in muscle_map:
                    x_idx = muscle_map[muscle]
                    hbmep_param[ix_p, x_idx] = ps_mea[idx]
                    hbmep_param_hdidelta[ix_p, x_idx] = ps_hdidelta[0, idx]
                else:
                    print("?")# Assign y[idx] to the correct column in y_array
        else:
            print(f"{mapping.get('participant', ix_p)} missing an RC for V{mapping.get('visit', ix_v)}")

        filtered_data = data.loc[
            (data["participant_index"] == ix_p) &
            (data["visit_index"] == ix_v) &
            (data["cxsc_index"] == ix_i),
            ["TMSInt", "TSCSInt"]
        ]

        unique_rows = filtered_data.drop_duplicates().values

        # Flatten to a single list of unique values
        unique_values = list(set(unique_rows.flatten()))
        # Ensure all TSCSInt values in the filtered data are the same
        if unique_rows.shape[0] == 1:
            unique_rows = unique_rows[0]
            stim_intensity[ix_p, :] = unique_rows
        else:
            print(f'{ix_p} missing perhaps from hbmep?')

    if hdidelta_cutoff:
        hbmep_param[hbmep_param_hdidelta > hdidelta_cutoff] = np.nan

    return hbmep_param, ie_size, stim_intensity


def get_participant_summary(df_in: pd.DataFrame, mapping, posterior_samples_grouped, dhbmep_tms, dhbmep_tss, mask_muscle, target_muscle, hdidelta_a_cutoff=None, ix_v: int=0, str_intensity="supra-sub"):
    '''
    '''
    df = deepcopy(df_in)

    df = df.groupby(["participant_index", "visit_index", "TMSIntPct"], as_index=False).first()
    df = df[df["visit_index"] == ix_v].copy()
    df = df.loc[df.groupby("participant_index")["TMSIntPct"].idxmax()].reset_index(drop=True)
    df = df.set_index("participant_index")

    num_participants = len(mapping.get("participant"))  # should match outside
    muscles = list(mapping.get("muscle").values())

    def get_uems(row):
        local_ = row["UEMS_L"] if row["target_muscle"][0].upper() == "L" else row["UEMS_R"]
        return local_

    def get_grassp(row):
        local_ = row["GRASSP_L"] if row["target_muscle"][0].upper() == "L" else row["GRASSP_R"]
        return local_

    df["UEMS"] = df.apply(get_uems, axis=1)
    df["GRASSP"] = df.apply(get_grassp, axis=1)

    df_sci = df[df['participant_condition'] == "SCI"].copy()
    mage = int(np.median(df_sci['age_binned']))
    bin_labels = [f"<{mage:02d}", f"≥{mage:02d}"]
    categories, bins = pd.qcut(df_sci['age_ranked'], q=2, labels=bin_labels, retbins=True)
    categories = pd.Series(categories, index=df_sci.index)
    df['age_sci_rebinned'] = "Uninjured"
    df.loc[df['participant_condition'] == "SCI", 'age_sci_rebinned'] = categories

    mage = int(np.median(df['age_binned']))
    bin_labels = [f"<{mage:02d}", f"≥{mage:02d}"]
    categories, bins = pd.qcut(df['age_ranked'], q=2, labels=bin_labels, retbins=True)
    categories = pd.Series(categories, index=df.index)
    df['age_rebinned'] = categories

    str_upper = 'C1-C3'
    str_mid = 'C4-C6'
    str_lower = 'C7-T1'
    motor_level_remap = {
        'C1': str_upper, 'C2': str_upper, 'C3': str_upper,
        'C4': str_mid, 'C5': str_mid, 'C6': str_mid,
        'C7': str_lower, 'C8': str_lower, 'T1': str_lower
    }
    df['MotorLevelRebinned'] = df['MotorLevel'].map(motor_level_remap)

    ix_i = mapping.get_inverse("intensity", str_intensity)

    # augment the cmct
    m = posterior_samples_grouped["c"]
    mask_local = np.any(mask_muscle, axis=(0, 1, 4))
    m = np.where(mask_local[None, None, ...], m, np.nan)
    for ix_p in range(num_participants):
        ix_c = mapping.get_inverse("condition", mapping.get("participant_condition", ix_p))
        str_target, ix_target = get_target(target_muscle, muscles, ix_v, ix_p, ix_i)
        cmct_bsp_adjustment = m[:, :, ix_p, ix_c, ix_target].mean()
        case_p = ix_p == df.index  # which is now "participant_index"
        df.loc[case_p, "cmct_bsp_adjustment"] = cmct_bsp_adjustment
    df["cmct_original_experimental"] = df["cmct"] - df["cmct_post_adjustment_manual"]
    df["cmct_model"] = df["cmct"] + df["cmct_bsp_adjustment"]

    # get the target muscle facilitation
    s = posterior_samples_grouped["s"][:, :, ix_v, :, :, ix_i, :]
    mask_local = np.any(mask_muscle[:, ix_v, :, :, ix_i, :], axis=0)
    s = np.where(mask_local[None, None, ...], s, np.nan)
    for ix_p in range(num_participants):
        ix_c = mapping.get_inverse("condition", mapping.get("participant_condition", ix_p))
        str_target, ix_target = get_target(target_muscle, muscles, ix_v, ix_p, ix_i)
        s_mea = s[:, :, ix_p, ix_c, ix_target].mean()
        case_p = ix_p == df.index  # which is now "participant_index"
        df.loc[case_p, "s_target"] = s_mea

    #
    df["target_muscle_unsided"] = df["target_muscle"].str[1:]  # drop the first character (L/R)

    # get thresholds
    for ix_mode in range(2):
        if ix_mode == 0:
            dhbmep_type = dhbmep_tms
            str_stim_type = "tms"
        else:
            dhbmep_type = dhbmep_tss
            str_stim_type = "tss"
        hbmep_unpack_pv = np.array([[int(x) for x in dhbmep_type["unpack"]["participant_visit"][y].split("_")] for y in
                                    range(len(dhbmep_type["unpack"]["participant_visit"]))])
        for ix_p in range(num_participants):
            case_p = ix_p == df.index  # which is now "participant_index"
            str_target, ix_target = get_target(target_muscle, muscles, ix_v, ix_p, ix_i)
            case_ps_f0 = (hbmep_unpack_pv[:, 0] == ix_p) & (hbmep_unpack_pv[:, 1] == ix_v)
            case_ps_f1 = mapping.get(dhbmep_type["model"].features[1], ix_p) == dhbmep_type["unpack"][dhbmep_type["model"].features[1]]
            for site_ in [site.a, site.ell, site.m50uv]:
                if site_ in dhbmep_type["ps"].keys():
                    px = dhbmep_type["ps"][site_][:, case_ps_f0, case_ps_f1, ix_target].ravel()
                    if site_ == site.m50uv:
                        # TODO: note, this is nan because in some RC instances 50uv crossing is not defined. This introduces a clear bias in the estimate.
                        op = np.nanmean
                    else:
                        op = np.mean
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        val = op(px)

                    if (site_ == site.a) and hdidelta_a_cutoff and px.size > 0:
                        hdi_local = np.diff(compute_hdi(px))
                        if (hdi_local > hdidelta_a_cutoff).squeeze():
                            val = np.nan

                    df.loc[case_p, f"{site_}_{str_stim_type}_target"] = val

    return df


def merge_pp(pp0, pp1, num_participants, num_visits, num_conditions):
    for ix_p in range(num_participants):
        for ix_v in range(num_visits):
            for ix_c in range(num_conditions):
                if pp0[ix_p][ix_v][ix_c] is None:
                    pp0[ix_p][ix_v][ix_c] = pp1[ix_p][ix_v][ix_c]
    return pp0


def plot_bland_altman(ax, x, y, title=None):
    """
    """
    mean_vals = (x + y) / 2
    diff_vals = y - x
    md = np.mean(diff_vals)  # mean difference
    sd = np.std(diff_vals, ddof=1)  # standard deviation of difference

    ax.scatter(mean_vals, diff_vals)
    ax.axhline(md, linestyle="--")
    ax.axhline(md + 1.96 * sd, linestyle="--")
    ax.axhline(md - 1.96 * sd, linestyle="--")
    ax.set_xlabel("Mean")
    ax.set_ylabel("Difference")
    if title:
        ax.set_title(f"{title}")


def plot_with_fit(ax, x, y, color="k", linestyle="-", title=None, xlabel=None, ylabel=None, add_offset=True, flip_pr_text_location=True, show_data=True, show_n=False):
    """
    Plot data with a linear fit on the provided axes.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]
    x_fit, y_fit, stats = linear_model(x_clean, y_clean, np.nanmin(x), np.nanmax(x), add_offset=add_offset)
    if show_data:
        ax.plot(x, y, "o",
                markerfacecolor=color,
                markeredgecolor="w",
                markersize=4,
                alpha=0.7)

    if len(x_clean) > 1:
        ax.plot(x_fit, y_fit, linestyle=linestyle, color=color)
        str_text0 = rf"$R^{2} = {stats['r2_value']:.2f}^{{{stats['corr_star']}}}$"
        str_text1 = rf"$r = {stats['corr_value']:.2f}^{{{stats['corr_star']}}}$"
        str_text2 = rf"$b = {stats['slope_value']:.2f}^{{{stats['slope_star']}}}$"
        yt = [0.075, 0.95]
        if flip_pr_text_location:
            yt = yt[::-1]

        ax.text(
            0.95, yt[0], rf"{str_text0} {str_text1} {str_text2}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none")
        )
        str_text1 = rf"$p_r = {stats['corr_p']:.2f}$"
        str_text2 = rf"$p_b = {stats['slope_p']:.2f}$"
        ax.text(
            0.95, yt[1], str_text1 + " " + str_text2,
            transform=ax.transAxes, ha="right", va="top", fontsize=5,
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none")
        )
        if show_n:
            str_text = rf"$n = {np.sum(mask):d}$"
            ax.text(
                0.95, 0.5, str_text,
                transform=ax.transAxes, ha="right", va="top", fontsize=5,
                bbox=dict(facecolor="white", alpha=0.0, edgecolor="none")
            )

    if title:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return stats, mask


def plot_density_on_ax(samples, ax, xlim=None, xlabel="Value", label=None):
    """
    Plots the density of posterior samples on a given axis.
    """
    flattened_samples = samples.reshape(-1)  # Flatten the samples
    sns.kdeplot(flattened_samples, fill=True, ax=ax, label=label)

    ax.set_ylabel("Density")
    ax.set_xlabel(xlabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if label is not None:
        ax.legend()


def write_figure(fig, d_analysis, show, extra_dir=None):
    title = getattr(fig, "figure_name", "No_Name_Set")
    print(f"Writing {title} figure.")
    if extra_dir is None: extra_dir = ""
    output_file = d_analysis / extra_dir / f"{title}.png"
    output_file.parent.mkdir(exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    fig.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")

    if show:
        fig.show()
    plt.close(fig)

def tally_participants(df_summary, participant_list):
    subset = df_summary[df_summary["participant"].isin(participant_list)]
    counts = subset["sex"].value_counts()
    total = len(subset)
    m_count = counts.get("M", 0)
    f_count = counts.get("F", 0)
    return f"(n = {total}; {m_count}M/{f_count}F)"


o_model = None
def main(o_model=None, rl_model="", overwrite=False):
    if o_model is None:
        ix_model = CONFIG_ANALYSIS["ix_model"]
        o_model = CONFIG_ANALYSIS["o_model"][ix_model]
        overwrite = CONFIG_ANALYSIS["overwrite"]
        show = CONFIG_ANALYSIS["show"]
        ix_rl_model = CONFIG_ANALYSIS["ix_rl_model"]
        rl_model = CONFIG_ANALYSIS["rl_model"][ix_rl_model]
        skip = CONFIG_ANALYSIS["skip"]
        compute_full_pp = CONFIG_ANALYSIS["compute_full_pp"]
    else:
        show = False
        skip = False
        compute_full_pp = True


    h_hdidelta_a_cutoff = lambda : 30  # 20
    str_ex = lambda : rl_model

    d_analysis = BASE_DIR / "analysis" / str_ex() / o_model
    print(d_analysis)

    f_model = "mcmc_model.pkl"
    p_model = None
    for path in (BASE_DIR / 'bsp').rglob(f"**/{o_model}/{f_model}"):
        p_model = path
        break

    if (d_analysis / "summary.csv").exists() and not overwrite:
        print("Already done, not overwriting.")
        return

    if p_model is None:
        print("Model badly specified? Skipping.")
        return

    if not p_model.with_stem("summary").with_suffix(".csv").exists():
        print("Model still sampling...")
        return

    d_analysis.mkdir(exist_ok=True, parents=True)

    cfg_file_path = p_model.with_stem("config").with_suffix(".toml")
    cfg = toml.load(cfg_file_path)
    cfg["DATA_FOLDER"] = DATA_FOLDER

    shutil.copy(cfg_file_path, d_analysis / "config.toml")
    shutil.copy(p_model.with_stem("summary").with_suffix(".csv"), d_analysis / "summary_hbmep.csv")

    match = re.search(r"(model[^_]+)", o_model)
    model_version = match.group(1)

    muscles_from_cfg = cfg["DATA_OPTIONS"]["response"]

    if cfg["DATA_OPTIONS"]["type"] == "intraoperative":
        es = ""
    else:
        es =  "_" + o_model
    data, mapping, mep, mep_ch = filter(cfg, overwrite=True, es=es)

    pi = data["SPI_target"].values.reshape(-1, 1)
    participant_index = data["participant_index"].values
    condition_index = data["condition_index"].values
    run_index = data["run_index"].values
    visit_index = data["visit_index"].values
    cxsc_index = data["cxsc_index"].values
    time = data["time"].values.reshape(-1, 1)
    average_count = data['average_count'].values.reshape(-1, 1)

    muscles = list(mapping.get("muscle").values())
    assert muscles == muscles_from_cfg, "?"

    num_participants = len(mapping.get("participant"))  #  len(vec_participants), following vec_participants = np.unique(participant_index)
    num_conditions = len(mapping.get("condition"))  # len(vec_conditions), following vec_conditions = np.unique(condition_index)
    num_intensity = len(mapping.get("intensity"))  # len(vec_cxsc), following vec_cxsc = np.unique(cxsc_index)
    num_visits = len(mapping.get("visit"))  # len(vec_visits), following vec_visits = np.unique(visit_index)
    num_runs = len(mapping.get("run"))  # len(vec_runs), following vec_runs = np.unique(run_index)
    num_muscles = len(muscles)

    response_obs = data[muscles].values.reshape(-1, num_muscles)

    model = getattr(model_module, model_version)
    mcmc = load_model(p_model)

    # Get the posterior samples
    group_by_chain = True
    posterior_samples_grouped = mcmc.get_samples(group_by_chain=group_by_chain)
    posterior_samples = ungroup_samples(posterior_samples_grouped, cfg["MCMC_OPTIONS"]["num_chains"])

    predictive = Predictive(model, posterior_samples)
    rng_key = random.PRNGKey(1)

    mask_run, mask_visit, mask_participant, mask_condition, mask_cxsc, mask_muscle = make_parameter_mask(num_muscles, num_runs, num_visits, num_participants,
                                                                            num_conditions, num_intensity, response_obs,
                                                                            run_index, visit_index, participant_index, condition_index, cxsc_index, cfg["MODEL_OPTIONS"]["mask"])
    mask_obs = make_obs_mask(response_obs, cfg["MODEL_OPTIONS"]["mask"])
    num_samples_pred = int(cfg["MCMC_OPTIONS"]["num_chains"] * cfg["MCMC_OPTIONS"]["num_samples"]/cfg["MCMC_OPTIONS"]["thinning"])


    # %%
    # Render the model structure
    f_plate_notation = "dnc_" + o_model + ".pdf"
    model_trace = render_model(
        model,
        model_args=(pi, time, response_obs, run_index, visit_index, participant_index, condition_index, cxsc_index, average_count, cfg["MODEL_OPTIONS"]),
        render_distributions=True,
        filename=f_plate_notation,
    )
    shutil.copy(f_plate_notation, d_analysis / "plates.pdf")
    os.remove(f_plate_notation)

    colors, _, vec_muscle_color, _ = get_cmap_muscles_alt()

    colors_condition = np.zeros((2, 3))
    colors_condition[mapping.get_inverse("condition", "SCI"), :] = np.array([166/255, 97/255, 26/255])
    colors_condition[mapping.get_inverse("condition", "Uninjured"), :] = np.array([0.1, 0.1, 0.1])

    colors_visit = np.zeros((2, 3))
    colors_visit[0, :] = np.array([0, 0, 1])
    colors_visit[1, :] = np.array([0, 1, 1])

    color_pairing = "#2E4053"
    color_threshold = "#3F6072"

    vec_cxsc = [i for i in [mapping.get_inverse("intensity", "supra-sub")]]
    if cfg["DATA_OPTIONS"]["response_transform"] == "log10":
        base = 10
    elif cfg["DATA_OPTIONS"]["response_transform"] == "log2":
        base = 2
    else:
        raise Exception("base?")

    target_muscle, target_segment = extract_target_matrix(data)
    pctchg = lambda x: pctchg_core(x, base)

    pi_candidate = np.linspace(-15, 15, 151)


    # %% Load hbMEP results
    if cfg["DATA_OPTIONS"]["type"] == "noninvasive":
        if "co" in str(p_model):
            feature1 = "participant_condition"  # load the condition hbmep - it's not really necessary to pair like this, but helps to keep things straight.
            model_name = f"RectifiedLogistic{str_ex()}Co_a"
        else:
            feature1 = "participant_filler"
            model_name = f"RectifiedLogistic{str_ex()}"
        hbmep_tms = load_and_process_hbmep("TMSInt", model_name, feature1, mapping, cfg, visit=0)
        hbmep_tss = load_and_process_hbmep("TSCSInt", model_name, feature1, mapping, cfg, visit=0)
        hbmep_tms_short = load_and_process_hbmep("TMSInt", model_name, feature1, mapping, cfg, visit=0, short=True)
        hbmep_tss_short = load_and_process_hbmep("TSCSInt", model_name, feature1, mapping, cfg, visit=0, short=True)


    # %%
    def plot_hbmep_distribution(
            ax,
            ps_sci,
            ps_sca,
            str_intensity,
            alpha=0.3,
            skip=False,
            rotate=False,
    ):
        """
        Plots the SCI vs. Uninjured distributions on ax, optionally rotating
        so that the density is along the x-axis (rotate=True) or y-axis (rotate=False).
        """
        if skip:
            return None

        # Decide intensity label
        if str_intensity == "TSCSInt":
            units = "mA"
            str_pre = "spinal"
        elif str_intensity == "TMSInt":
            units = "% MSO"
            str_pre = "cortical"
        else:
            raise ValueError("str_intensity must be 'TSCSInt' or 'TMSInt'.")

        # Compute difference probability
        ps_diff = ps_sci - ps_sca
        pr = np.mean(ps_diff > 0, axis=0)

        ix_sci_color = mapping.get_inverse("condition", "SCI")
        ix_sca_color = mapping.get_inverse("condition", "Uninjured")
        color_sci = colors_condition[ix_sci_color, :]
        color_sca = colors_condition[ix_sca_color, :]

        plot_dim = "y" if rotate else "x"
        density_dim = "x" if rotate else "y"

        # Plot SCI
        sns.kdeplot(**{plot_dim: ps_sci}, color=color_sci, linestyle="-", ax=ax, fill=False)
        sns.kdeplot(**{plot_dim: ps_sci}, color=color_sci, ax=ax,
                    fill=True, linewidth=0, alpha=alpha)

        # Plot Uninjured
        sns.kdeplot(**{plot_dim: ps_sca}, color=color_sca, linestyle="-", ax=ax, fill=False)
        sns.kdeplot(**{plot_dim: ps_sca}, color=color_sca, ax=ax,
                    fill=True, linewidth=0, alpha=alpha)

        ax.text(
            0.95, 0.95, rf"$Probability = {pr:0.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none")
        )

        if rotate:
            ax.set_ylabel(f"{str_pre.capitalize()} threshold ({units})")
            ax.set_xlabel("Density")
        else:
            ax.set_xlabel(f"{str_pre.capitalize()} threshold ({units})")
            ax.set_ylabel("Density")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    def plot_hbmep_difference(
            ax,
            ps_sci,
            ps_sca,
            str_intensity,
            color_diff,
            ls="-",
            compute_hdi_func=None
    ):
        """
        Plots the difference distribution (SCI - Uninjured) on ax.
        """

        if str_intensity == "TSCSInt":
            units = "mA"
        elif str_intensity == "TMSInt":
            units = "% MSO"
        else:
            units = "arbitrary"

        ps_diff = ps_sci - ps_sca
        ax.axvline(0, linestyle="--", color="k", linewidth=0.5)

        sns.kdeplot(x=ps_diff, color=color_diff, linestyle="-", ax=ax, fill=False)
        sns.kdeplot(
            x=ps_diff, color=color_diff, linestyle="-", ax=ax,
            fill=True, clip=(0, np.max(ps_diff)), linewidth=0
        )

        if compute_hdi_func is not None:
            e_diff = compute_hdi_func(ps_diff)
            y_max = ax.get_ylim()[1]
            ax.plot(e_diff, np.ones(2) * 1.00 * y_max, "-", color=color_diff)

        ax.set_xlabel(f"Stimulation intensity ({units})")
        ax.set_ylabel("Density")
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Probability that difference is > 0
        pr = np.mean(ps_diff > 0, axis=0)
        ax.text(
            0.95, 0.95, rf"$Probability = {pr:0.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none")
        )


    def plot_hbmep_conditions(dhbmep, str_intensity, skip):
        """
        """

        if skip:
            return

        model_hbmep, hbmep_ps, hbmep_unpack, df_data = dhbmep["model"], dhbmep["ps"], dhbmep["unpack"], dhbmep["data"]

        fig_width = (2.0 + 2.5 * num_muscles) * CMTI
        fig_height = 8 * 1 * CMTI

        keys_with_shape_2 = [
            key for key in hbmep_ps
            if hbmep_ps[key].ndim > 1 and hbmep_ps[key].shape[1] == 2
        ]

        for cond_param in keys_with_shape_2:
            fig, axs = plt.subplots(
                2,
                num_muscles,
                figsize=(fig_width, fig_height),
                squeeze=False,
                constrained_layout=True,
                dpi=300
            )
            fig.figure_name = f"co_{cond_param}_{str_intensity}"

            for ix_m in range(num_muscles):
                str_muscle = mapping.get("muscle", ix_m)
                c = colors[vec_muscle_color == str_muscle, :]

                ix_sci = hbmep_unpack["participant_condition"] == "SCI"
                ps_sci = hbmep_ps[cond_param][:, ix_sci, ix_m].ravel()

                ix_sca = hbmep_unpack["participant_condition"] == "Uninjured"
                ps_sca = hbmep_ps[cond_param][:, ix_sca, ix_m].ravel()

                ax_top = axs[0, ix_m]
                ax_bottom = axs[1, ix_m]

                plot_hbmep_distribution(
                    ax=ax_top,
                    ps_sci=ps_sci,
                    ps_sca=ps_sca,
                    str_intensity=str_intensity,
                )

                plot_hbmep_difference(
                    ax=ax_bottom,
                    ps_sci=ps_sci,
                    ps_sca=ps_sca,
                    str_intensity=str_intensity,
                    color_diff=c
                )

            write_figure(fig, d_analysis, show)


    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        if 'co' in str(p_model):
            plot_hbmep_conditions(hbmep_tss, 'TSCSInt', skip)
            plot_hbmep_conditions(hbmep_tms, 'TMSInt', skip)


  # %% prediction with hbmep
    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        max_intensity = 80
        pp_tms, df_template_tms = make_prediction(hbmep_tms_short, 'TMSInt', num_conditions, num_visits, num_participants, mapping, max_intensity=max_intensity, compute_full_pp=compute_full_pp)
        pp_tss, df_template_tss = make_prediction(hbmep_tss_short, 'TSCSInt', num_conditions, num_visits, num_participants, mapping, max_intensity=max_intensity, compute_full_pp=compute_full_pp)
        pp_tms_avg, _ = make_prediction_average(hbmep_tms_short, 'TMSInt', num_conditions, num_visits, num_participants, mapping, max_intensity=max_intensity)
        pp_tss_avg, _ = make_prediction_average(hbmep_tss_short, 'TSCSInt', num_conditions, num_visits, num_participants, mapping, max_intensity=max_intensity)


    # %%
    def plot_cmct_adjustments(df_in: pd.DataFrame, ix_v: int = 0, skip: bool = False):
        """
        """
        if skip:
            return None

        df = get_participant_summary(df_in, mapping, posterior_samples_grouped, hbmep_tms, hbmep_tss, mask_muscle, target_muscle, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), ix_v=ix_v)

        df_melt = df.melt(
            id_vars=["participant"],
            value_vars=["cmct_original_experimental", "cmct", "cmct_model"],
            var_name="Measurement",
            value_name="Value"
        )

        fig_width = 17.6 * CMTI
        fig_height = 8 * CMTI

        fig, axs = plt.subplots(
            1,
            2,
            figsize=(fig_width, fig_height),
            dpi=300,
            gridspec_kw={'width_ratios': [3, 1]},
            squeeze=False,
            constrained_layout=True
        )
        fig.figure_name = f"cmct_adjustments_target_muscle_V{ix_v}"

        ax0 = axs[0, 0]
        sns.barplot(ax=ax0, data=df_melt, x="participant", y="Value", hue="Measurement")
        ax0.set_title(f"CMCT (V{ix_v})")
        ax0.tick_params(axis='x', rotation=45)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)

        legend = ax0.get_legend()
        legend.set_title(None)
        legend.get_frame().set_alpha(0)

        ax1 = axs[0, 1]
        x_positions = [1, 2, 3]
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(['Original\nCMCT', 'Corrected\nCMCT', 'Model\nCMCT'])
        ax1.set_ylabel("CMCT Value")

        for _, row in df.iterrows():
            y1 = row["cmct_original_experimental"]
            y2 = row["cmct"]
            y3 = row["cmct_model"]
            ax1.plot([1, 2], [y1, y2], color='gray', alpha=0.25,
                     marker='o', markersize=4, linestyle='-')
            ax1.plot([2, 3], [y2, y3], color='gray', alpha=0.25,
                     marker='o', markersize=2, linestyle='-')

        ax1.set_ylim(bottom=0)
        ax1.set_ylim(top=ax0.get_ylim()[1])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        diff1 = df["cmct"] - df["cmct_original_experimental"]
        _, p1 = wilcoxon(diff1)
        diff2 = df["cmct_model"] - df["cmct"]
        _, p2 = wilcoxon(diff2)

        ax1.text(1.5, 0, fr"$p={p1:.3g}$", ha='center', va='bottom', fontsize=4)
        mea1 = np.mean(diff1)
        std1 = np.nanstd(diff1)
        n_valid1 = np.sum(~np.isnan(diff1))
        sem1 = std1 / np.sqrt(n_valid1)
        ax1.text(1.5, 1.0, f'{mea1:0.2f}±{sem1:0.2f} (SD={std1:0.2f})', ha='center', va='bottom', fontsize=4)

        ax1.text(2.5, 0, fr"$p={p2:.3g}$", ha='center', va='bottom', fontsize=4)
        mea2 = np.mean(diff2)
        std2 = np.nanstd(diff2)
        n_valid2 = np.sum(~np.isnan(diff2))
        sem2 = std2 / np.sqrt(n_valid2)
        ax1.text(2.5, 1.0, f'{mea2:0.2f}±{sem2:0.2f} (SD={std2:0.2f})', ha='center', va='bottom', fontsize=4)

        fig.tight_layout()
        write_figure(fig, d_analysis, show)


    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        plot_cmct_adjustments(data, skip=skip)


    # %%
    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive') and (num_muscles == 1) and ("co" not in str(p_model)):
        df_s = get_participant_summary(data, mapping, posterior_samples_grouped, hbmep_tms, hbmep_tss, mask_muscle, target_muscle, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), ix_v=0)
        p_summary_out = d_analysis / f'summary_data_cut{h_hdidelta_a_cutoff()}.csv'
        df_s.to_csv(p_summary_out, index=False)
        print(f'Saved summary to {p_summary_out}')


    # %%
    def plot_cmct_conditions(df_in: pd.DataFrame, cmct_type:str='cmct_model', ix_v:int=0, skip:bool=False):
        """
        """
        if skip:
            return None

        df = get_participant_summary(df_in, mapping, posterior_samples_grouped, hbmep_tms, hbmep_tss, mask_muscle, target_muscle, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), ix_v=ix_v)
        df = df.reset_index()  # This converts the index (participant_index) into a column for next line
        df = df.groupby(["participant_index", "visit_index"], as_index=False).first()

        fig_width = 4 * CMTI
        fig_height = 6 * CMTI

        fig, axs = plt.subplots(
            1, 1,
            figsize=(fig_width, fig_height),
            squeeze=False,
            constrained_layout=True,
            dpi=300
        )
        fig.figure_name = f"{cmct_type}_V{ix_v}"
        ax = axs[0, 0]

        palette = {
            condition: colors_condition[mapping.get_inverse('condition', condition), :]
            for condition in ['Uninjured', 'SCI']
        }

        sns.boxplot(ax=ax, data=df, x="participant_condition", y=cmct_type, palette=palette, flierprops={'marker': 'o', 'markersize': 3})
        ax.set_xlabel("")
        ax.set_ylabel(f"CMCT [{cmct_type}] (ms)")

        if cmct_type=='cmct_model':
            ax.set_ylim(bottom=0)
            ax.set_ylim(top=ax.get_ylim()[1]+5)

        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        data_uninjured = df.loc[df["participant_condition"] == "Uninjured", cmct_type]
        data_sci = df.loc[df["participant_condition"] == "SCI", cmct_type]

        stat, p_value = ranksums(data_uninjured, data_sci)
        ax.set_title(rf"$p={p_value:0.3f}$")

        if p_value < 0.05:
            y_max = df[cmct_type].max()
            y_min = df[cmct_type].min()
            y_range = y_max - y_min
            offset = 0.05 * y_range  # 5% of the data range
            x1, x2 = 0, 1
            y = y_max + offset
            h = offset
            col = 'k'
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            ax.text((x1 + x2) * 0.5, y + h, "*", ha='center', va='bottom', color=col)

        write_figure(fig, d_analysis, show)


    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        plot_cmct_conditions(data, skip=skip)
        plot_cmct_conditions(data, cmct_type='cmct_bsp_adjustment', skip=skip)


    # %%
    def plot_summary_columns(x_column, y_column, hbmep_tms, hbmep_tss, figure_name="", ix_v=0, lim=None, skip=False):
        if skip:
            return None

        ds = get_participant_summary(
            data, mapping, posterior_samples_grouped,
            hbmep_tms, hbmep_tss,
            mask_muscle, target_muscle,
            hdidelta_a_cutoff=h_hdidelta_a_cutoff(),
            ix_v=ix_v
        )

        if x_column not in ds.columns or y_column not in ds.columns:
            print(f"Column '{x_column}' or '{y_column}' not found in data.")
            return

        # Drop rows where either column is missing
        temp = ds[[x_column, y_column]].dropna()
        if temp.empty:
            print("No valid data available for plotting.")
            return

        x = temp[x_column].values
        y = temp[y_column].values

        fig_width = 6 * CMTI
        fig_height = 9 * CMTI

        fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=300, constrained_layout=True)
        fig.figure_name = f"{figure_name}"

        axs[0].plot(lim, lim, color='black', linestyle='--', linewidth=1)
        plot_with_fit(axs[0], x, y, title=None)
        axs[0].set_xlabel(f"{x_column}")
        axs[0].set_ylabel(f"{y_column}")

        if lim:
            axs[0].set_xlim(lim)
            axs[0].set_ylim(lim)
        axs[0].set_box_aspect(1)

        plot_bland_altman(axs[1], x, y, title=None)
        axs[1].set_box_aspect(1)
        axs[1].set_xlabel('Mean')
        axs[1].set_ylabel('Difference')

        write_figure(fig, d_analysis, show)


    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive') and (num_muscles == 1):
        plot_summary_columns("TSCS_RMT", "a_tss_target", hbmep_tms, hbmep_tss, figure_name="tss_RMT_vs_a", lim=[0, 100], skip=skip)
        plot_summary_columns("TMS_RMT", "a_tms_target", hbmep_tms, hbmep_tss, figure_name="tms_RMT_vs_a", lim=[0, 100], skip=skip)

    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive') and (num_muscles == 1):
        try:
            print('Trying PKPK m50uV plots.')
            hbmep_tss_pkpk = load_and_process_hbmep("TSCSInt", model_name, feature1, mapping, cfg, visit=0, mepsizetype='pkpk')
            hbmep_tms_pkpk = load_and_process_hbmep("TMSInt", model_name, feature1, mapping, cfg, visit=0, mepsizetype='pkpk')
            print('You could probably improve the following, by excluding conditions where m50uv has x% of nans in the posterior.')
            plot_summary_columns("TSCS_RMT", "m50uv_tss_target", hbmep_tms_pkpk, hbmep_tss_pkpk, figure_name="tss_RMT_vs_m50uv", lim=[0, 100], skip=skip)
            plot_summary_columns("TMS_RMT", "m50uv_tms_target", hbmep_tms_pkpk, hbmep_tss_pkpk, figure_name="tms_RMT_vs_m50uv", lim=[0, 100], skip=skip)
        except:
            print('Failed to generate PKPK m50uV plots.')


    # %%
    def plot_target_muscles(df_in: pd.DataFrame, ix_v: int = 0, skip: bool = False):
        if skip:
            return None

        palette = {
            condition: colors_condition[mapping.get_inverse('condition', condition), :]
            for condition in ['Uninjured', 'SCI']
        }

        df = get_participant_summary(df_in, mapping, posterior_samples_grouped, hbmep_tms, hbmep_tss, mask_muscle, target_muscle, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), ix_v=ix_v)

        fig_width = 6 * CMTI
        fig_height = 4 * CMTI

        fig, axs = plt.subplots(
            1,
            1,
            figsize=(fig_width, fig_height),
            squeeze=False,
            constrained_layout=True,
            dpi=300
        )
        fig.figure_name = f"target_muscle_V{ix_v}"
        ax = axs[0, 0]

        sns.countplot(ax=ax, data=df, x="target_muscle_unsided", hue="participant_condition", palette=palette)
        ax.set_box_aspect(1)
        ax.set_xlabel("target muscle")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        legend = ax.get_legend()
        legend.set_title(None)
        legend.get_frame().set_alpha(0)

        # Color each xtick label differently using a colormap.
        xticklabels = ax.get_xticklabels()
        num_labels = len(xticklabels)
        colors_local = [colors[xticklabels[i].get_text() == vec_muscle_color, :][0] for i in range(num_labels)]
        for tick, color in zip(xticklabels, colors_local):
            tick.set_color(color)

        write_figure(fig, d_analysis, show)


    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive'):
        plot_target_muscles(data, ix_v=0, skip=skip)


    # %% Basic recruitment curve
    def plot_rc(ax, ix_p, ix_v, ix_c, ix_i, ix_m, model_hbmep, str_intensity, df_template, df_data, pp, pa, color='b',
                alpha=0.2, alpha_line=0.95, show_extras=True, visit_in_text=False, mepsize_units=None, ylim=None, skip=False):
        # TODO: the indexing here is a big mess. It's partially inside, partially outside.
        if skip:
            return None
        """
        Plot RC for a given participant, visit and cxsc index.
        """
        if str_intensity == 'TSCSInt':
            units = 'mA'
            str_pre = 'spinal'
        elif str_intensity == 'TMSInt':
            units = '% MSO'
            str_pre = 'cortical'

        if mepsize_units is None:
            mepsize_units = 'µV⋅s'

        x = df_template[model_hbmep.intensity].values
        if ix_p is None:
            Y_list = []
            for ix_p_local in range(len(pp)):
                if pp[ix_p_local][ix_v][ix_c] is not None:
                    Y_local = pp[ix_p_local][ix_v][ix_c][site.mu][:, :, ix_m]
                    Y_list.append(Y_local)
            Y_stacked = np.stack(Y_list, axis=0)
            Y = np.mean(Y_stacked, axis=0)
            show_extras = False
        else:
            Y = pp[ix_p][ix_v][ix_c][site.mu][:, :, ix_m]

        y = np.mean(Y, axis=0)
        hdi_local = compute_hdi(Y)
        y1, y2 = hdi_local[:, 0], hdi_local[:, 1]
        ax.plot(x, y, color=color, alpha=alpha_line)
        if alpha > 0:
            ax.fill_between(x, y1, y2, color=color, alpha=alpha, linewidth=0)

        if show_extras:
            df_local = deepcopy(df_data)
            ind1 = (df_local['visit_index'] == ix_v) & (df_local['participant_index'] == ix_p)
            df_local = df_local[ind1]

            x_local = df_local[model_hbmep.intensity].values
            y_local = df_local[model_hbmep.response[ix_m]].values
            ax.plot(
                x_local,
                y_local,
                color=color,
                marker='o',
                markeredgecolor='w',
                markerfacecolor=color,
                linestyle='None',
                markeredgewidth=1,
                markersize=4
            )

            filtered_data = data.loc[
                (data['participant_index'] == ix_p) &
                (data['visit_index'] == ix_v) &
                (data['cxsc_index'] == ix_i),
                [str_intensity, str_intensity + 'Pct']
            ]
            if filtered_data[str_intensity].unique().size > 0:
                x_arrow = filtered_data[str_intensity].unique()[0]
                x_text = filtered_data[str_intensity + 'Pct'].unique()[0]

            else:
                x_arrow = np.array(np.nan)
                x_text = ''

            if visit_in_text:
                str_visit = f' (V{ix_v})'
            else:
                str_visit = ''
            if ylim is None:
                ylim = ax.get_ylim()
            else:
                ax.set_ylim(ylim)
            dy = np.diff(ylim)[0]
            xlim = ax.get_xlim()
            dx = np.diff(xlim)[0]
            off_arrow = dy * 0.05
            len_arrow = dy * 0.20
            # str_text = f'{x_text:0.0f}%'
            str_text = f'Pairing intensity: {x_arrow:0.1f}{units}{str_visit}'
            ax.text(x_arrow, off_arrow*2.5, str_text, fontsize=4, color=color_pairing, rotation="vertical", horizontalalignment='right', verticalalignment='bottom')
            if x_arrow.size:
                ax.arrow(
                    x_arrow, len_arrow+off_arrow, 0, -len_arrow, color=color_pairing,
                    length_includes_head=True,
                    head_width=dx*0.04, head_length=dy*0.05, zorder=4
                )

            y_threshold = np.mean(pa)
            ax.vlines(y_threshold, ylim[0], ylim[1], linewidth=1, linestyle='--', color=color)
            str_text = f'Threshold: {y_threshold:0.1f}{units}'
            ax.text(y_threshold, ylim[1], str_text, fontsize=4, rotation="vertical", horizontalalignment='right',
                    verticalalignment='top', color=color)
            ax.set_ylim(ylim)
            ax.set_xlabel(f'{str_pre.capitalize()} intensity ({units})')
            ax.set_ylabel(f'MEP size ({mepsize_units})')
            ax.set_box_aspect(1)  # Forces a square axes box
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)


    def plot_rc_individuals(dhbmep, str_intensity, df_template, pp, mepsizetype='auc', color_by='muscle', mepsize_units=None, vec_visit=(0,), dhbmep_v1=None, ylim=None, skip=False):
        if skip:
            return None
        fig_width = 1 + 16.6 * CMTI * num_muscles / 5
        fig_height = 5 * CMTI
        ix_i = mapping.get_inverse('intensity', 'supra-sub')

        for ix_p in range(len(mapping.get('participant'))):
            ix_c = mapping.get_inverse('condition', mapping.get('participant_condition', ix_p))
            do_plot = np.all([pp[ix_p][ix_v][ix_c] is not None for ix_v in vec_visit])
            if do_plot:
                fig, axs = plt.subplots(
                    1,
                    num_muscles,
                    figsize=(fig_width, fig_height),
                    squeeze=False,
                    constrained_layout=True,
                    dpi=300
                )

                str_p = mapping.get('participant', ix_p)
                if np.shape(vec_visit)[-1] == 1:
                    str_v = f'V{vec_visit[0]}'
                else:
                    str_v = f'V0-1'

                if ylim is None:
                    es_ylim = ''
                else:
                    es_ylim = '_zoom'

                fig.figure_name = f"rc_{str_p}_{str_intensity}_P{ix_p}{str_v}I{ix_i}"
                for ix_m in range(num_muscles):
                    ax = axs[0, ix_m]
                    str_muscle = mapping.get('muscle', ix_m)
                    ix_c = mapping.get_inverse('condition', mapping.get('participant_condition', ix_p))

                    for ix_v in vec_visit:
                        c_muscle = colors[vec_muscle_color == str_muscle, :]
                        c_condition = colors_condition[ix_c, :]
                        c_visit = colors_visit[ix_v, :]
                        if color_by == 'muscle':
                            c = c_muscle
                        elif color_by == 'condition':
                            c = c_condition
                        elif color_by == 'visit':
                            c = c_visit
                        else:
                            raise Exception(f'Unknown color by {color_by}')

                        if ix_v == 0:
                            model_hbmep, hbmep_ps, hbmep_unpack, df_data = dhbmep['model'], dhbmep['ps'], dhbmep['unpack'], dhbmep['data']
                        elif ix_v == 1:
                            model_hbmep, hbmep_ps, hbmep_unpack, df_data = dhbmep_v1['model'], dhbmep_v1['ps'], dhbmep_v1['unpack'], dhbmep_v1['data']

                        hbmep_unpack_pv = np.array([[int(x) for x in hbmep_unpack['participant_visit'][y].split('_')] for y in range(len(hbmep_unpack['participant_visit']))])
                        case_ps_f0 = (hbmep_unpack_pv[:, 0] == ix_p) & (hbmep_unpack_pv[:, 1] == ix_v)
                        case_ps_f1 = mapping.get(model_hbmep.features[1], ix_p) == hbmep_unpack[
                            model_hbmep.features[1]]
                        pa = hbmep_ps[site.a][:, case_ps_f0, case_ps_f1, ix_m].ravel()  # TODO: confirm this muscle indexing
                        plot_rc(ax, ix_p, ix_v, ix_c, ix_i, ix_m, model_hbmep, str_intensity, df_template, df_data, pp, pa, mepsize_units=mepsize_units, color=c, ylim=ylim, visit_in_text=True)
                        if mepsizetype == 'pkpk':
                            ax.axhline(50e-3, linestyle='--', linewidth=1, color='k')

                    if (str_muscle != 'auc_target') and (ylim is None):
                        str_text = rf'{str_muscle}'
                        ax.text(
                            0.05, 0.95, str_text, color=c_muscle,
                            transform=ax.transAxes, ha='left', va='bottom', fontsize=6,
                            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                        )
                write_figure(fig, d_analysis, show, extra_dir='rc_' + mepsizetype + '_' + color_by + '_V' + '_'.join(map(str, vec_visit)) + es_ylim)


    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        plot_rc_individuals(hbmep_tss, 'TSCSInt', df_template_tss, pp_tss, skip=skip)
        plot_rc_individuals(hbmep_tms,'TMSInt', df_template_tms, pp_tms, skip=skip)
        plot_rc_individuals(hbmep_tss,'TSCSInt', df_template_tss, pp_tss, color_by='condition', skip=skip)
        plot_rc_individuals(hbmep_tms,'TMSInt', df_template_tms, pp_tms, color_by='condition', skip=skip)

# %%
    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        try:
            for str_stim in ['TMSInt', 'TSCSInt']:
                print(f'Attempting to generate pkpk RCs: {str_stim}')
                hbmep_temp_pkpk = load_and_process_hbmep(str_stim, model_name, feature1, mapping, cfg, mepsizetype='pkpk')
                hbmep_temp_pkpk_short = load_and_process_hbmep(str_stim, model_name, feature1, mapping, cfg, mepsizetype='pkpk', short=True)

                pp_temp_pkpk, df_template_temp_pkpk = make_prediction(hbmep_temp_pkpk_short, str_stim, num_conditions, num_visits, num_participants, mapping,
                                                                      max_intensity=max_intensity, compute_full_pp=compute_full_pp)

                print(f'Attempting to plot pkpk RCs: {str_stim}')
                plot_rc_individuals(hbmep_temp_pkpk, str_stim, df_template_temp_pkpk, pp_temp_pkpk, mepsizetype='pkpk',
                                    mepsize_units='mV', color_by='muscle', skip=skip)
        except:
            print('Failed to plot pk-pk recruitment curves.')


# %%
    def plot_rc_conditions(dhbmep, str_intensity, df_template, pp, alpha=0.2, skip=False):
        if skip:
            return None
        fig_width = 1 + 16.6 * CMTI * num_muscles / 5
        fig_height = 5 * CMTI
        ix_v, ix_i = 0, mapping.get_inverse('intensity', 'supra-sub')

        fig, axs = plt.subplots(
            1,
            num_muscles,
            figsize=(fig_width, fig_height),
            squeeze=False,
            constrained_layout=True,
            sharey=True,
            dpi=300
        )
        es = '_CI' if alpha>0 else ''
        fig.figure_name = f"rc_{str_intensity}_V{ix_v}I{ix_i}{es}"

        for ix_m in range(num_muscles):
            ax = axs[0, ix_m]
            str_muscle = mapping.get('muscle', ix_m)
            c_muscle = colors[vec_muscle_color == str_muscle, :]
            plot_rc_condition_ax(ax, dhbmep, df_template, ix_i, ix_m, ix_v, pp, str_intensity, alpha=alpha)
            str_text = rf'{str_muscle}'
            ax.text(
                0.05, 0.95, str_text, color=c_muscle,
                transform=ax.transAxes, ha='left', va='bottom', fontsize=6,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
            )

        write_figure(fig, d_analysis, show)


    def plot_rc_condition_ax(ax, dhbmep, df_template, ix_i, ix_m, ix_v,
                    pp, str_intensity, alpha=0.3, alpha_line=0.95, show_average_p=False):
        model_hbmep, hbmep_ps, hbmep_unpack, df_data = dhbmep['model'], dhbmep['ps'], dhbmep['unpack'], dhbmep['data']
        hbmep_unpack_pv = np.array([[int(x) for x in hbmep_unpack['participant_visit'][y].split('_')] for y in
                                    range(len(hbmep_unpack['participant_visit']))])
        if str_intensity == 'TSCSInt':
            units = 'mA'
        elif str_intensity == 'TMSInt':
            units = '% MSO'
        for ix_p in range(len(mapping.get('participant'))):
            ix_c = mapping.get_inverse('condition', mapping.get('participant_condition', ix_p))
            if pp[ix_p][ix_v][ix_c] is not None:
                c = colors_condition[ix_c, :]
                case_ps_f0 = (hbmep_unpack_pv[:, 0] == ix_p) & (hbmep_unpack_pv[:, 1] == ix_v)
                case_ps_f1 = mapping.get(model_hbmep.features[1], ix_p) == hbmep_unpack[
                    model_hbmep.features[1]]
                pa = hbmep_ps[site.a][:, case_ps_f0, case_ps_f1, ix_m].ravel()  # TODO: confirm this muscle indexing
                plot_rc(ax, ix_p, ix_v, ix_c, ix_i, ix_m, model_hbmep, str_intensity, df_template, df_data, pp, pa,
                        color=c, alpha=alpha, alpha_line=alpha_line,show_extras=False)

            ax.set_xlabel(f'Stimulation intensity ({units})')
            ax.set_ylabel(f'MEP size (µV⋅s)')
            ax.set_box_aspect(1)  # Forces a square axes box
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if show_average_p:
                ix_c = mapping.get_inverse('condition', 'Uninjured')
                c = colors_condition[ix_c, :]
                plot_rc(ax, None, ix_v, ix_c, ix_i, ix_m, model_hbmep, str_intensity, df_template, df_data, pp, pa,
                        color=c, alpha=alpha, alpha_line=0.95, show_extras=False)
                ix_c = mapping.get_inverse('condition', 'SCI')
                c = colors_condition[ix_c, :]
                plot_rc(ax, None, ix_v, ix_c, ix_i, ix_m, model_hbmep, str_intensity, df_template, df_data, pp, pa,
                        color=c, alpha=alpha, alpha_line=0.95, show_extras=False)


    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        plot_rc_conditions(hbmep_tss,'TSCSInt', df_template_tss, pp_tss, skip=skip)
        plot_rc_conditions(hbmep_tms,'TMSInt', df_template_tms, pp_tms, skip=skip)


    # %%
    def plot_intensity_relations(mode='TSS', str_keep='All', squish_muscles=False, skip=False, hdidelta_a_cutoff=None, replace_with_summary=None, mark_participants=False, impute_missing_with_manual=False):
        """
        """
        if skip: return None
        if cfg['DATA_OPTIONS']['type'] == 'intraoperative': return None
        if squish_muscles and num_muscles == 1: return None

        color2 = "#9AAFB7"
        color3 = "#000000"

        str_p_hl = 'SCA09'

        is_sci = np.array(
            [(mapping.get('participant_condition', i) == 'SCI') for i in range(num_participants)])
        is_marked_participant = np.array(
            [(mapping.get('participant', i) == str_p_hl) for i in range(num_participants)])
        # Extract data.
        if mode == 'TSS':
            units = 'mA'
            ix_mode = 1
            p = np.arange(1, 0.5, -0.2)
            str_intensity = 'TSCSInt'
            mode_alt = 'TSCS'
            hbmep_type, df_template, pp = hbmep_tss, df_template_tss, pp_tss
            str_pre = 'spinal'
        elif mode == 'TMS':
            units = '% MSO'
            ix_mode = 0
            p = np.arange(1, 1.5, +0.2)
            str_intensity = 'TMSInt'
            mode_alt = mode
            hbmep_type, df_template, pp = hbmep_tms, df_template_tms, pp_tms
            str_pre = 'cortical'
        else:
            raise Exception('ddd')

        x_array_th, y_array_fac, z_array_int = extract_s_vs_threshold(posterior_samples_grouped, muscles, hbmep_type, mapping, mask_muscle, data, hdidelta_a_cutoff=hdidelta_a_cutoff)
        df = get_participant_summary(data, mapping, posterior_samples_grouped, hbmep_tms, hbmep_tss, mask_muscle, target_muscle, hdidelta_a_cutoff=hdidelta_a_cutoff, ix_v=0)

        if impute_missing_with_manual:
            case_nan = np.isnan(x_array_th)
            print(f'Using manually estimated thresholds for {np.sum(case_nan)} missing hbMEP {str_intensity} values. For: ')
            print([mapping.get('participant', ix_p_) for ix_p_ in np.where(case_nan.flatten())[0]])
            x_array_th[case_nan] = df[str_intensity].values[case_nan.flatten()]

        if squish_muscles:
            num_columns = 1
            is_sci = np.broadcast_to(is_sci[:, np.newaxis], x_array_th.shape)
            z_array_int = np.broadcast_to(z_array_int[:, ix_mode][:, np.newaxis], x_array_th.shape)
            is_sci = is_sci.ravel()[:, np.newaxis]
            x_array_th = x_array_th.ravel()[:, np.newaxis]
            y_array_fac = y_array_fac.ravel()[:, np.newaxis]
            z_array_int_local = z_array_int.ravel()[:, np.newaxis]

        else:
            num_columns = num_muscles
            z_array_int_local = z_array_int[:, [ix_mode]]  # Use second column

        ix_p_hl, ix_v_hl, ix_i_hl = mapping.get_inverse('participant', str_p_hl), 0, mapping.get_inverse('intensity', 'supra-sub')
        ix_c_hl = mapping.get_inverse('condition', mapping.get('participant_condition', ix_p_hl))
        str_target_hl, ix_target_hl = get_target(target_muscle, muscles, ix_v_hl, ix_p_hl, ix_i_hl)

        def add_tally(ax, plotted_participant_list):
            str_text = tally_participants(df, plotted_participant_list)
            ax.text(
                0.95, 0.5, str_text,
                transform=ax.transAxes, ha='right', va='top', fontsize=6,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
            )

        # Define figure dimensions with the top row half as high as the bottom row.
        for ix_m in range(num_columns):
            str_muscle = mapping.get('muscle', ix_m)

            x_array_th_local = x_array_th[:, [ix_m]]
            if replace_with_summary:
                if replace_with_summary == 'RMT':
                    str_column = f'{mode_alt}_RMT'
                    es_replacement = '_RMT_instead_of_a'
                elif replace_with_summary == site.a:
                    str_column = f'{site.a}_{mode.lower()}_target'
                    es_replacement = '_a_from_summary'
                x_array_th_local = df[str_column].values.reshape(-1, 1)
            else:
                es_replacement = ''
            y_array_fac_pct = pctchg(y_array_fac[:, [ix_m]])

            mask = np.isnan(x_array_th_local)
            y_array_fac_pct[mask] = np.nan
            z_array_int_local[mask] = np.nan

            if str_keep == "All":
                es_keep = ''
            else:
                assert not (squish_muscles), "!"
                es_keep = '_' + str_keep
                mask_cond_local = np.array(
                    [(mapping.get('participant_condition', i) == str_keep) for i in range(num_participants)])
                x_array_th_local[~mask_cond_local, :] = np.nan
                y_array_fac_pct[~mask_cond_local, :] = np.nan
                z_array_int_local[~mask_cond_local] = np.nan

            fig_width = 17.6 * CMTI
            fig_height = 10 * CMTI
            fig, axs = plt.subplots(2, 4,
                                     figsize=(fig_width, fig_height),
                                     gridspec_kw={'height_ratios': [1, 1]},
                                     squeeze=False,
                                     constrained_layout=True,
                                     dpi=300)

            es_squish = '_squish' if squish_muscles else ''
            es_cutoff = f'_cut{hdidelta_a_cutoff:0.0f}' if hdidelta_a_cutoff else ''
            es_marked = f'_marked' if mark_participants else ''
            fig.figure_name = "intensity_relations_" + mode + '_' + site.a + '_' + str_muscle + es_keep + es_squish + es_cutoff + es_replacement + es_marked

            fig.delaxes(axs[0, 0])

            norm_array_pct = (z_array_int_local / x_array_th_local) * 100

            # Compute common x-limits for columns 0, 1, and 3.
            common_x_max = max(np.nanmax(x_array_th_local), np.nanmax(z_array_int_local))
            common_x_max = np.ceil(common_x_max / 10) * 10
            # common_x_min = min(np.nanmin(x_array_th_local), np.nanmin(z_array_int_local))
            # common_x_min = np.floor(common_x_min / 10) * 10
            common_xlim = (0, common_x_max)

            ax = axs[1, 0]
            if str_target_hl is None:
                c_hl = 'k'
            else:
                if str_muscle == 'auc_target':
                    c_hl = colors[vec_muscle_color == str_target_hl, :]
                else:
                    c_hl = colors[vec_muscle_color == str_muscle, :]
            hbmep_unpack_pv = np.array([[int(x) for x in hbmep_type['unpack']['participant_visit'][y].split('_')] for y in
                                        range(len(hbmep_type['unpack']['participant_visit']))])
            case_ps_f0 = (hbmep_unpack_pv[:, 0] == ix_p_hl) & (hbmep_unpack_pv[:, 1] == ix_v_hl)
            case_ps_f1 = mapping.get(hbmep_type['model'].features[1], ix_p_hl) == hbmep_type['unpack'][
                hbmep_type['model'].features[1]]
            pa = hbmep_type['ps'][site.a][:, case_ps_f0, case_ps_f1, ix_m].ravel()  # TODO: confirm this muscle indexing
            plot_rc(ax, ix_p_hl, ix_v_hl, ix_c_hl, ix_i_hl, ix_m, hbmep_type['model'], str_intensity, df_template, hbmep_type['data'], pp, pa, color=c_hl)
            ax.text(
                0.95, 0.05, h_alias(str_p_hl),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=4,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
            )

            ax.set_xlim(common_xlim)
            ms_sci = 2
            ms_marked = 3
            ix_sci_color = mapping.get_inverse('condition', 'SCI')
            color_sci = colors_condition[ix_sci_color, :]

            ax = axs[1, 1]
            for p_ in p:
                ax.plot([0, common_xlim[1]],
                        [0*p_, common_xlim[1]*p_], '--', linewidth=1, color=np.ones((1, 3)) * 0.7)
            ax.plot([0, common_xlim[1]],
                            [0, common_xlim[1]], '-', linewidth=1, color=np.ones((1, 3))*0.7)
            ax.plot(x_array_th_local, z_array_int_local, 'o', markerfacecolor=color3,
                            markeredgecolor='w', markersize=4, alpha=0.7)
            ax.plot(x_array_th_local[is_sci], z_array_int_local[is_sci], 'o', markeredgewidth=0,
                    markerfacecolor=color_sci, markeredgecolor='w', markersize=ms_sci, alpha=1.0)
            try:
                # Breaks under some condition...
                ax.plot(x_array_th_local[is_marked_participant], z_array_int_local[is_marked_participant], '*', markeredgewidth=0,
                    markerfacecolor='r', markeredgecolor='w', markersize=ms_marked, alpha=1.0)
            except:
                print(1)

            _, mask = plot_with_fit(ax, x_array_th_local, z_array_int_local, color='b', linestyle='--', add_offset=False, flip_pr_text_location=True, show_data=False)
            plotted_participant_list = [mapping.get('participant', ix_p) for ix_p in list(range(len(mask))) if mask[ix_p]]
            add_tally(ax, plotted_participant_list)

            pct_pavth = (z_array_int_local / x_array_th_local) * 100
            pct_pavth_mea = np.nanmean(pct_pavth)
            pct_pavth_std = np.nanstd(pct_pavth)
            pct_pavth_count = np.sum(np.isfinite(pct_pavth))
            pct_pavth_sem = pct_pavth_std / np.sqrt(pct_pavth_count)

            str_text = rf'(P.int/Th=${pct_pavth_mea:0.1f}±{pct_pavth_sem:0.1f}$%, SD={pct_pavth_std:0.1f}%)'
            ax.text(
                1.0, 0.75, str_text,
                transform=ax.transAxes, ha='right', va='top', fontsize=4,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
            )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_xlabel(f'{str_pre.capitalize()} threshold ({units})', fontsize=8, color=color_threshold)
            ax.set_ylabel(f'{str_pre.capitalize()}\npairing intensity ({units})', fontsize=8, color=color_pairing)
            ax.set_xlim(common_xlim)
            ax.set_ylim(common_xlim)
            ax.set_aspect('equal', adjustable='box')

            ax = axs[0, 1]
            plot_with_fit(ax, z_array_int_local, y_array_fac_pct, color=color_pairing, flip_pr_text_location=True)
            ax.plot(z_array_int_local[is_sci], y_array_fac_pct[is_sci], 'o', markeredgewidth=0,
                    markerfacecolor=color_sci, markeredgecolor='w', markersize=ms_sci, alpha=1.0)
            ax.plot(z_array_int_local[is_marked_participant], y_array_fac_pct[is_marked_participant], '*', markeredgewidth=0,
                    markerfacecolor='r', markeredgecolor='w', markersize=ms_marked, alpha=1.0)
            ax.set_box_aspect(1)
            ax.set_xlim(common_xlim)
            ax.set_xlabel(f'Pairing intensity ({units})', fontsize=8, color=color_pairing)
            ax.set_ylabel('% Facilitation', fontsize=8)

            ax = axs[1, 2]
            _, mask = plot_with_fit(ax, x_array_th_local, y_array_fac_pct, color=color_threshold, flip_pr_text_location=True)
            plotted_participant_list = [mapping.get('participant', ix_p) for ix_p in list(range(len(mask))) if mask[ix_p]]
            ax.plot(x_array_th_local[is_sci], y_array_fac_pct[is_sci], 'o', markeredgewidth=0,
                    markerfacecolor=color_sci, markeredgecolor='w', markersize=ms_sci, alpha=1.0)
            ax.plot(x_array_th_local[is_marked_participant], y_array_fac_pct[is_marked_participant], '*', markeredgewidth=0,
                    markerfacecolor='r', markeredgecolor='w', markersize=ms_marked, alpha=1.0)
            add_tally(ax, plotted_participant_list)
            ax.set_xlim(common_xlim)
            ax.set_box_aspect(1)  # Forces a square axes box
            ax.set_xlabel(f'{str_pre.capitalize()} threshold ({units})', fontsize=8, color=color_threshold)
            ax.set_ylabel('% Facilitation', fontsize=8)
            if mark_participants:
                for ix_p in range(num_participants):
                    ax.text(x_array_th_local[ix_p], y_array_fac_pct[ix_p], mapping.get('participant', ix_p), fontsize=4)

            ax = axs[1, 3]
            _, mask = plot_with_fit(ax, norm_array_pct, y_array_fac_pct, color=color2, flip_pr_text_location=True)
            plotted_participant_list = [mapping.get('participant', ix_p) for ix_p in list(range(len(mask))) if mask[ix_p]]
            add_tally(ax, plotted_participant_list)
            ax.plot(norm_array_pct[is_sci], y_array_fac_pct[is_sci], 'o', markeredgewidth=0,
                    markerfacecolor=color_sci, markeredgecolor='w', markersize=ms_sci, alpha=1.0)
            ax.plot(norm_array_pct[is_marked_participant], y_array_fac_pct[is_marked_participant], '*', markeredgewidth=0,
                    markerfacecolor='r', markeredgecolor='w', markersize=ms_marked, alpha=1.0)
            ax.set_xlabel(f'% of {str_pre} threshold', fontsize=8)
            ax.set_ylabel('% Facilitation', fontsize=8)
            if mark_participants:
                for ix_p in range(num_participants):
                    ax.text(norm_array_pct[ix_p], y_array_fac_pct[ix_p], mapping.get('participant', ix_p), fontsize=4)
            if np.nanmean(norm_array_pct) > 0:
                if np.nanmin(norm_array_pct) > 100:
                    ax.set_xlim(left=100)
            else:
                ax.set_xlim(left=0)

            ax.set_box_aspect(1)

            write_figure(fig, d_analysis, show)


    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        plot_intensity_relations(mode='TSS', hdidelta_a_cutoff=h_hdidelta_a_cutoff(), mark_participants=False, skip=skip)
        plot_intensity_relations(mode='TMS', hdidelta_a_cutoff=h_hdidelta_a_cutoff(), mark_participants=False, skip=skip)


    # %%
    def plot_scivariables(df_in, column_types, es='', skip=skip):
        """
        Plots histograms and bar charts for SCI participants.
        """
        if skip: return None

        df = get_participant_summary(df_in, mapping, posterior_samples_grouped, hbmep_tms, hbmep_tss, mask_muscle, target_muscle, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), ix_v=0)

        # Separate column names based on type.
        numeric_cols = [col for col, is_cat in column_types.items() if not is_cat]
        categorical_cols = [col for col, is_cat in column_types.items() if is_cat]

        n_cols = np.max(np.array([len(numeric_cols), len(categorical_cols)]))
        fig_width = 11.6 * CMTI * (1/3) * n_cols
        fig_height = 10 * CMTI
        fig, axs = plt.subplots(2, n_cols, figsize=(fig_width, fig_height),
                                squeeze=False, constrained_layout=True, dpi=300)
        fig.figure_name = "scivariables" + es

        df_demo_sci = df[df['participant_condition'] == 'SCI']

        # Plot histograms for numeric columns.
        for i, col in enumerate(numeric_cols):
            ax = axs[0, i]
            ax.hist(df_demo_sci[col].dropna(), bins=10, color='skyblue', edgecolor='black')
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.set_box_aspect(1)

        # Plot bar charts for categorical columns.
        for i, col in enumerate(categorical_cols):
            ax = axs[1, i]
            counts = df_demo_sci[col].value_counts().sort_index()
            errors = np.sqrt(counts.values)  # Poisson error approximation
            ax.bar(counts.index.astype(str), counts.values, color='salmon',
                   yerr=errors, capsize=3, ecolor='salmon')
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_box_aspect(1)

        write_figure(fig, d_analysis, show)

    column_types_main = {  # whether they are categories or cont.
        'MotorLevelRebinned': True,
        'AIS': True,
        'age_sci_rebinned': True,
        'time_post_injury': False,
    }
    column_types_mmt = {x:False for x in
                        ['MMT_TS_C5_BB', 'MMT_TS_C7_TB', 'MMT_TS_C6_ECR', 'MMT_TS_XX_FCR', 'MMT_TS_T1_ADM' ,'MMT_TS_XX_APB', 'MMT_TS_XX_FDI', 'MMT_TS_L4_TA']
                        }

    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive') and ('co' in str(p_model)):
        plot_scivariables(data, column_types=column_types_main, skip=skip)
        # plot_scivariables(data, column_types=column_types_mmt, es='_mmt', skip=skip)


# %%
    def plot_scivariables_relation(data, ordinate_data, column_types, mode='TMS', es='', site_flag=None, hdidelta_a_cutoff=None, skip=False):
        if skip: return None
        site_flag = site.a if site_flag is None else site_flag

        if mode == 'TSS':
            ix_mode = 1
            p = np.arange(1, 0.5, -0.1)
            str_intensity = 'TSCSInt'
            hbmep_type, df_template, pp = hbmep_tss, df_template_tss, pp_tss
            str_pre = 'spinal'

        elif mode == 'TMS':
            ix_mode = 0
            p = np.arange(1, 1.5, +0.1)
            str_intensity = 'TMSInt'
            hbmep_type, df_template, pp = hbmep_tms, df_template_tms, pp_tms
            str_pre = 'cortical'
        else:
            raise Exception('ddd')

        if site_flag == site.a:
            str_hbmep_param = 'threshold'
            if mode == 'TSS':
                units = 'mA'
            elif mode == 'TMS':
                units = '% MSO'
        elif site_flag == site.H:
            str_hbmep_param = 'saturation'
            units = 'muVs (check)'
        elif site_flag == site.ell:
            str_hbmep_param = 'smoothness'
            units = 'a.u.'
        else:
            raise Exception('not coded')

        if ordinate_data == 'x_array_th_local':
           ylabel = f'{str_pre.capitalize()} {str_hbmep_param} ({units})'
           str_mode = '_' + mode
        elif ordinate_data == 'y_array_fac_pct':
            ylabel = f'% Facilitation'
            str_mode = ''

        df = get_participant_summary(data, mapping, posterior_samples_grouped, hbmep_tms, hbmep_tss, mask_muscle, target_muscle, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), ix_v=0)

        x_array_th, y_array_fac, z_array_int = extract_s_vs_threshold(posterior_samples_grouped, muscles, hbmep_type, mapping, mask_muscle, data, site_flag=site_flag, hdidelta_a_cutoff=hdidelta_a_cutoff)
        num_columns = num_muscles
        z_array_int_local = z_array_int[:, [ix_mode]]  # Use second column

        for ix_m in range(num_columns):
            str_muscle = mapping.get('muscle', ix_m)
            c = colors[vec_muscle_color == str_muscle, :]

            x_array_th_local = x_array_th[:, [ix_m]]
            y_array_fac_pct = pctchg(y_array_fac[:, [ix_m]])


            df_demo = deepcopy(df)
            df_demo['x_array_th_local'] = x_array_th_local
            df_demo['y_array_fac_pct'] = y_array_fac_pct
            df_demo['z_array_int_local'] = z_array_int_local

            col_types = column_types

            numeric_cols = [col for col, is_cat in col_types.items() if not is_cat]
            categorical_cols = [col for col, is_cat in col_types.items() if is_cat]
            n_cols = np.max(np.array([len(numeric_cols), len(categorical_cols)]))

            fig_width = 11.6 * CMTI * (1/3) * n_cols
            fig_height = 10 * CMTI
            fig, axs = plt.subplots(2, n_cols, figsize=(fig_width, fig_height),
                                    squeeze=False, constrained_layout=True, dpi=300)
            es_cutoff = f'_cut{hdidelta_a_cutoff:0.0f}' if hdidelta_a_cutoff else ''
            fig.figure_name = "scivar_relation_" + ordinate_data + str_mode + es + '_' + str_muscle + es_cutoff

            # Filter for SCI participants.
            df_demo_sci = df_demo[df_demo['participant_condition'] == 'SCI']

            for i, col in enumerate(numeric_cols):
                ax = axs[0, i]
                x, y = df_demo_sci[col].to_numpy(), df_demo_sci[ordinate_data].to_numpy()
                plot_with_fit(ax, x, y,
                              xlabel=col,
                              ylabel=ylabel,
                              color=c
                              )

                ax.set_box_aspect(1)

            # Plot bar charts for categorical columns.
            for i, col in enumerate(categorical_cols):
                ax = axs[1, i]
                if col == 'age_binned':
                    w = 5 * 0.9
                else:
                    w = 0.9

                grouped = df_demo_sci.groupby(col)[ordinate_data].agg(['mean', 'sem']).sort_index()
                ax.bar(grouped.index, grouped['mean'], color=c, width=w, yerr=grouped['sem'], capsize=3, ecolor=c)

                df_anova = df_demo_sci[[col, ordinate_data]].dropna()
                model = ols(f'{ordinate_data} ~ C({col})', data=df_anova).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                p_value = anova_table.loc[f"C({col})", "PR(>F)"]
                str_text = rf'$p = {p_value:0.2f}$'
                ax.text(
                    0.95, 0.95, str_text,
                    transform=ax.transAxes, ha='right', va='top', fontsize=6,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )

                ax.set_xlabel(col)
                ax.set_ylabel(ylabel)
                ax.set_box_aspect(1)

            write_figure(fig, d_analysis, show)


    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive') and ('co' in str(p_model)):
        plot_scivariables_relation(data, ordinate_data='y_array_fac_pct', column_types=column_types_main, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), skip=skip)
        # plot_scivariables_relation(data, ordinate_data='x_array_th_local', column_types=column_types_main, mode='TSS', hdidelta_a_cutoff=h_hdidelta_a_cutoff(), skip=skip)
        # plot_scivariables_relation(data, ordinate_data='x_array_th_local', column_types=column_types_main, mode='TMS', hdidelta_a_cutoff=h_hdidelta_a_cutoff(), skip=skip)

        # plot_scivariables_relation(data, ordinate_data='x_array_th_local', column_types=column_types_main, mode='TSS', es='_H', site_flag=site.H, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), skip=skip)
        # plot_scivariables_relation(data, ordinate_data='x_array_th_local', column_types=column_types_main, mode='TMS', es='_H', site_flag=site.H, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), skip=skip)

        # %%
    def plot_scivariables_relation_nice(data, ordinate_data, column_types, column_rename, mode='TMS', es='', site_flag=None,
                                   hdidelta_a_cutoff=None, skip=False):
        if skip: return None
        site_flag = site.a if site_flag is None else site_flag

        if mode == 'TSS':
            ix_mode = 1
            p = np.arange(1, 0.5, -0.1)
            str_intensity = 'TSCSInt'
            hbmep_type, df_template, pp = hbmep_tss, df_template_tss, pp_tss
            str_pre = 'spinal'

        elif mode == 'TMS':
            ix_mode = 0
            p = np.arange(1, 1.5, +0.1)
            str_intensity = 'TMSInt'
            hbmep_type, df_template, pp = hbmep_tms, df_template_tms, pp_tms
            str_pre = 'cortical'
        else:
            raise Exception('ddd')

        if site_flag == site.a:
            str_hbmep_param = 'threshold'
            if mode == 'TSS':
                units = 'mA'
            elif mode == 'TMS':
                units = '% MSO'
        elif site_flag == site.H:
            str_hbmep_param = 'saturation'
            units = 'muVs (check)'
        elif site_flag == site.ell:
            str_hbmep_param = 'smoothness'
            units = 'a.u.'
        else:
            raise Exception('not coded')

        if ordinate_data == 'x_array_th_local':
            ylabel = f'{str_pre.capitalize()} {str_hbmep_param} ({units})'
            str_mode = '_' + mode
        elif ordinate_data == 'y_array_fac_pct':
            ylabel = f'% Facilitation'
            str_mode = ''

        df = get_participant_summary(data, mapping, posterior_samples_grouped, hbmep_tms, hbmep_tss, mask_muscle,
                                     target_muscle, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), ix_v=0)

        x_array_th, y_array_fac, z_array_int = extract_s_vs_threshold(posterior_samples_grouped, muscles,
                                                                      hbmep_type, mapping, mask_muscle, data,
                                                                      site_flag=site_flag,
                                                                      hdidelta_a_cutoff=hdidelta_a_cutoff)
        num_columns = num_muscles
        z_array_int_local = z_array_int[:, [ix_mode]]  # Use second column
        # Define figure dimensions with the top row half as high as the bottom row.
        for ix_m in range(num_columns):
            str_muscle = mapping.get('muscle', ix_m)
            c = colors[vec_muscle_color == str_muscle, :]

            x_array_th_local = x_array_th[:, [ix_m]]
            y_array_fac_pct = pctchg(y_array_fac[:, [ix_m]])

            df_demo = deepcopy(df)
            df_demo['x_array_th_local'] = x_array_th_local
            df_demo['y_array_fac_pct'] = y_array_fac_pct
            df_demo['z_array_int_local'] = z_array_int_local

            col_types = column_types

            numeric_cols = [col for col, is_cat in col_types.items() if not is_cat]
            categorical_cols = [col for col, is_cat in col_types.items() if is_cat]
            n_cols = len(numeric_cols) + len(categorical_cols)

            fig_width = 17.6 * CMTI
            fig_height = 5 * CMTI
            fig, axs = plt.subplots(1, n_cols, figsize=(fig_width, fig_height),
                                    squeeze=False, constrained_layout=True, dpi=300)
            es_cutoff = f'_cut{hdidelta_a_cutoff:0.0f}' if hdidelta_a_cutoff else ''
            fig.figure_name = "scivariables_relation_nice_" + ordinate_data + str_mode + es + '_' + str_muscle + es_cutoff

            # Filter for SCI participants.
            df_demo_sci = df_demo[df_demo['participant_condition'] == 'SCI']

            for i, col_item in enumerate(col_types.items()):
                col, is_cat = col_item[0], col_item[1]
                ax = axs[0, i]
                if is_cat:
                    if col == 'age_binned':
                        w = 5 * 0.9
                    else:
                        w = 0.9

                    grouped = df_demo_sci.groupby(col)[ordinate_data].agg(['mean', 'sem', 'count']).sort_index()
                    ax.bar(grouped.index, grouped['mean'], color=c, width=w, yerr=grouped['sem'], capsize=3, ecolor=c)

                    # Show n =
                    for idx, (x, y, n) in enumerate(zip(grouped.index, grouped['mean'], grouped['count'])):
                        ax.text(
                            x, y + grouped['sem'].iloc[idx] + 0.02 * grouped['mean'].max(),  # slightly above error bar
                            f'n = {n}', ha='center', va='bottom', fontsize=6
                        )

                    df_anova = df_demo_sci[[col, ordinate_data]].dropna()
                    model = ols(f'{ordinate_data} ~ C({col})', data=df_anova).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    p_value = anova_table.loc[f"C({col})", "PR(>F)"]
                    str_text = rf'$p = {p_value:0.2f}$'
                    ax.text(
                        0.95, 0.95, str_text,
                        transform=ax.transAxes, ha='right', va='top', fontsize=6,
                        bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                    )
                else:
                    x, y = df_demo_sci[col].to_numpy(), df_demo_sci[ordinate_data].to_numpy()
                    plot_with_fit(ax, x, y,
                                  xlabel=col,
                                  ylabel=ylabel,
                                  color=c
                                  )
                ax.set_ylim([-2.5, 40])
                ax.set_xlabel(column_rename[col])
                ax.set_ylabel(ylabel)
                ax.set_box_aspect(1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            write_figure(fig, d_analysis, show)

    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive') and ('co' in str(p_model)):
            column_rename = {
                'MotorLevelRebinned': 'Motor Level',
                'AIS': 'AIS',
                'age_sci_rebinned': 'Age (years)',
                'time_post_injury': 'Time post injury (years)',
            }
            plot_scivariables_relation_nice(data, ordinate_data='y_array_fac_pct', column_types=column_types_main, column_rename=column_rename,
                                       hdidelta_a_cutoff=h_hdidelta_a_cutoff(), skip=skip)


    # %%
    def plot_facilitation_single(
            ax, ix_p, ix_c, ix_m, ix_v, str_muscle,
            m, w, s, pi_candidate,
            zero_m=False,
            xlim=None, ylim=None, xlabel=None, ylabel=None, c=None,show_data=False, show_ci=True):
        """
        Plot facilitation curves (and optionally per-run data) for a single participant/condition/target/visit
        into the provided `ax`.
        """
        c = 'k' if c is None else c
        assert ix_m is not None, "ix_m should not be None - probably an incorrect muscle name upstream"
        assert ix_p is not None, "ix_p should not be None"

        m_local = m[:, :, ix_p, ix_c, ix_m]
        w_local = w[:, :, ix_p, ix_c, ix_m]
        s_local = s[:, :, ix_p, ix_c, ix_m]

        if zero_m:
            m_local = m_local * 0

        y_hdi = np.array([hdi_g(x_, m_local, w_local, s_local) for x_ in pi_candidate])
        y_cen = np.array([cen_g(x_, m_local, w_local, s_local) for x_ in pi_candidate])

        y_hdi = pctchg(y_hdi)
        y_cen = pctchg(y_cen)

        ax.plot(pi_candidate, y_cen, color=c)
        if show_ci:
            ax.fill_between(pi_candidate, y_hdi[:, 0], y_hdi[:, 1], color=c, alpha=0.3)

        if ylim is not None:
            ax.plot([0, 0], ylim, 'k--', alpha=0.3)
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.plot(xlim, [0, 0], 'k--', alpha=0.3)
            ax.set_xlim(xlim)

        if xlabel is None:
            ax.set_xlabel('PI (ms)')
        if ylabel is None:
            ax.set_ylabel('% Facilitation')

        if show_data:
            # Optional: show actual run-by-run data (median for repeated points)
            spi_list = []
            x1ox0_list = []
            num_runs = mask_run.shape[0]
            for ix_r in range(num_runs):
                has_run = mask_run[ix_r, ix_v, ix_p, ix_c, :, ix_m]

                if np.any(has_run):
                    # Filter data
                    case_data = (
                            (data['run_index'] == ix_r)
                            & (data['visit_index'] == ix_v)
                            & (data['participant_index'] == ix_p)
                            & (data['cxsc_index'] == ix_c)
                    )
                    filtered_data = data.loc[case_data, str_muscle].values
                    spi_local = pi[case_data]
                    time_local = time[case_data].reshape(-1)

                    baseline = posterior_samples['a'][:, ix_r, ix_v, ix_p, ix_c, ix_m].mean()
                    if cfg['MODEL_OPTIONS']['use_b']:
                        baseline += posterior_samples['b'][:, ix_r, ix_v, ix_p, ix_c, ix_m].mean() * time_local

                    x1ox0_local = pctchg(filtered_data / (base ** baseline))
                    spi_list.append(spi_local)
                    x1ox0_list.append(x1ox0_local)

            if len(spi_list) > 0:
                spi_flat = np.concatenate(spi_list).flatten()
                x1ox0_flat = np.concatenate(x1ox0_list).flatten()
                df = pd.DataFrame({'spi': spi_flat, 'x1ox0': x1ox0_flat})
                result = df.groupby('spi')['x1ox0'].median().reset_index()
                ax.plot(result['spi'], result['x1ox0'], 'o', color=c, alpha=0.1)


    def plot_participant_facilitation(pi_candidate, skip=False):
        """
        Wrapper that sets up the figure(s), loops over participants/muscles/visits,
        and calls `plot_facilitation_single` for each subplot.
        """
        if skip:
            return None

        ix_i = mapping.get_inverse('intensity', 'supra-sub')
        ix_v = 0

        s = posterior_samples_grouped['s'][:, :, ix_v, :, :, ix_i, :]
        mask_local = np.any(mask_muscle[:, ix_v, :, :, ix_i, :], axis=0)
        s = np.where(mask_local[None, None, ...], s, np.nan)

        m = posterior_samples_grouped['c']
        w = posterior_samples_grouped['w']
        mask_local = np.any(mask_muscle, axis=(0, 1, 4))
        m = np.where(mask_local[None, None, ...], m, np.nan)
        w = np.where(mask_local[None, None, ...], w, np.nan)

        # Order participants by the size of s (for the target)
        s_stack = np.zeros(num_participants) * np.nan
        for ipx in range(num_participants):
            str_target, ix_target = get_target(target_muscle, muscles, ix_v, ipx, ix_i)
            if ix_target is None:
                continue

            if np.any(mask_visit[ix_v, ipx, :, ix_i, ix_target]):
                s_local = s[:, :, ipx, :, ix_target]
                s_stack[ipx] = np.nanmean(s_local)
        sorted_p = np.flip(np.argsort(s_stack))

        if mapping.get('muscle', 0) == 'auc_target':
            n_plots = num_muscles
        else:
            n_plots = num_muscles + 1

        xlim = [-15, 15]
        n_r, n_c = optimal_division(num_participants, priority='columns')

        for ix_m in range(n_plots):
            str_muscle = mapping.get('muscle', ix_m)

            fig, axs = plt.subplots(n_r, n_c, figsize=(3 * n_c, 3 * n_r), sharex=True, sharey=False)
            fig.figure_name = f'fac_participants_{str_muscle}'

            axs = np.reshape(axs, -1)

            if str_muscle == 'auc_target':
                str_muscle = None
            if str_muscle is None:
                str_muscle = 'Target'

            for ix_plot, ix_p in enumerate(sorted_p):
                ax = axs[ix_plot]

                str_participant = mapping.get('participant', ix_p)
                str_alias = mapping.get('alias', ix_p)
                ax.set_title(str_alias + ' ' + str_participant)

                # Decide y-limits by data type
                if cfg['DATA_OPTIONS']['type'] == 'intraoperative':
                    if ix_plot < n_c:
                        ylim = [-150, 7000]
                    elif ix_plot < n_c * 2:
                        ylim = [-150, 7000]
                    else:
                        ylim = [-150, 1250]
                else:
                    ylim = [-100, +100]

                # Figure out the condition index, muscle/target index
                ix_c = mapping.get_inverse('condition', mapping.get('participant_condition', ix_p))
                for idx_i, cxsc in enumerate(vec_cxsc):
                    # If it's the last plot, use the actual target muscle from get_target
                    if ix_m == n_plots - 1:
                        str_target, ix_target = get_target(target_muscle, muscles, ix_v, ix_p, idx_i)
                        if ix_target is None:
                            continue
                    else:
                        str_target = str_muscle
                        ix_target = ix_m

                    c = colors[vec_muscle_color == str_target, :]
                    if np.any(mask_visit[ix_v, ix_p, :, idx_i, ix_target]):
                        plot_facilitation_single(
                            ax=ax, ix_p=ix_p, ix_c=ix_c, ix_m=ix_target, ix_v=ix_v, str_muscle=str_target,
                            m=m, w=w, s=s, pi_candidate=pi_candidate,
                            xlim=xlim, ylim=ylim, c=c, show_data=False)

            write_figure(fig, d_analysis, show)


    def plot_specific_participant_facilitation(pi_candidate, str_participant, str_muscle, ix_v, ix_i, figsize=None, c=None, show_ci=True, xlim=None, ylim=None, skip=False):
        """
        Wrapper that sets up the figure(s), loops over participants/muscles/visits,
        and calls `plot_facilitation_single` for each subplot.
        """
        if skip:
            return None

        s = posterior_samples_grouped['s'][:, :, ix_v, :, :, ix_i, :]
        mask_local = np.any(mask_muscle[:, ix_v, :, :, ix_i, :], axis=0)
        s = np.where(mask_local[None, None, ...], s, np.nan)

        m = posterior_samples_grouped['c']
        w = posterior_samples_grouped['w']
        mask_local = np.any(mask_muscle, axis=(0, 1, 4))
        m = np.where(mask_local[None, None, ...], m, np.nan)
        w = np.where(mask_local[None, None, ...], w, np.nan)

        if figsize is None:
            fig_width = 4 * CMTI
            fig_height = 6 * CMTI
            figsize = (fig_width, fig_height)

        fig, axs = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=False)
        fig.figure_name = f'fac_participants_{str_participant}{str_muscle}'
        axs = np.reshape(axs, -1)

        ax = axs[0]

        ix_p = mapping.get_inverse('participant', str_participant)
        ix_m = mapping.get_inverse('muscle', str_muscle)
        ix_c = mapping.get_inverse('condition', mapping.get('participant_condition', ix_p))

        assert ix_p is not None, "ix_p should not be None"
        assert ix_m is not None, "ix_m should not be None - probably an incorrect muscle name upstream"

        if c is None:
            c = colors[vec_muscle_color == str_muscle, :]

        plot_facilitation_single(
            ax=ax, ix_p=ix_p, ix_c=ix_c, ix_m=ix_m, ix_v=ix_v, str_muscle=str_muscle,
            m=m, w=w, s=s, pi_candidate=pi_candidate,
            zero_m=True,
            ylabel=None, xlim=xlim, ylim=ylim, c=c, show_data=False, show_ci=show_ci)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        write_figure(fig, d_analysis, show)

    plot_participant_facilitation(pi_candidate, skip=skip)


    # %%
    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive') and (num_muscles > 1):
        plot_specific_participant_facilitation(pi_candidate=np.linspace(-15, 15, 301),
                                              str_participant='SCA05',
                                              str_muscle='APB', ix_v=0, ix_i=mapping.get_inverse('intensity', 'supra-sub'),
                                              figsize=(4.0*CMTI, 4.7*CMTI), show_ci=False, xlim=[-15, 15], ylim=[-5, 25],
                                              skip=skip)


    # %%
    if mapping.get('muscle', 0) == 'auc_target':
        n_plots = num_muscles
    else:
        n_plots = num_muscles + 1
    Z = generate_curves(pi_candidate, num_intensity, num_visits, n_plots, num_participants, target_muscle, muscles,
                        mask_visit, posterior_samples_grouped, mapping, mask_muscle)
    Z_non_zero_pi = generate_curves(pi_candidate, num_intensity, num_visits, n_plots, num_participants, target_muscle, muscles,
                        mask_visit, posterior_samples_grouped, mapping, mask_muscle, zero_pi=False)


    def plot_fac_averaged(Z, pi_candidate, intensity_condition, es='', skip=False):
        """
        """
        if skip:
            return None
        ix_i = mapping.get_inverse('intensity', intensity_condition)
        ix_v = 0
        Z_squeezed = Z[:, ix_i, ix_v, ...]  # 0 intensity, 0 visit
        Z_squeezed = pctchg(Z_squeezed)

        if cfg['DATA_OPTIONS']['type'] == 'intraoperative':
            n_r, n_c = 1, 5
            ylim = [-150, 1500]
            xlim = [0, 0.015]
        else:
            n_r, n_c = 1, 6
            ylim = [-25, +45]
            xlim = [0, 0.22]

        fig_width = 19 * CMTI
        fig_height = 8 * n_r * CMTI

        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.figure_name = f'fac_averaged_{intensity_condition}{es}'
        gs = gridspec.GridSpec(1, n_c, figure=fig)

        for ix_m in range(n_plots):
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[ix_m], width_ratios=[4, 1])
            ax1 = fig.add_subplot(inner_gs[0])
            ax2 = fig.add_subplot(inner_gs[1])

            str_muscle = mapping.get('muscle', ix_m)
            # ax = axs[ix_m]
            if (ix_m == n_plots - 1):
                c = colors[vec_muscle_color == 'auc_target', :]
                str_x = 'Target'
            else:
                c = colors[vec_muscle_color == str_muscle, :]
                str_x = str_muscle
            # ax1.set_title(str_x, color=c)
            ax1.text(
                0.05, 0.95,
                str_x,
                transform=ax1.transAxes, ha='left', va='center', fontsize=8, color=c,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
            )

            z = np.nanmean(Z_squeezed, axis=-1)[:, ix_m]
            std = np.nanstd(Z_squeezed, axis=-1)[:, ix_m]
            n_valid = np.sum(~np.isnan(Z_squeezed), axis=-1)[:, ix_m]
            sem = std / np.sqrt(n_valid)

            above_bounds = dict()
            for ix_p in range(num_participants):
                if np.all(np.isnan(Z_squeezed[:, ix_m, ix_p])):
                    continue
                str_p = mapping.get('participant', ix_p)
                case_target = [x for x in np.unique(target_muscle[:, ix_v, ix_p, ix_i]) if x != '']
                str_target = case_target[0][1:]
                if ('auc_target' == str_muscle):
                    c_ = c
                elif (str_target == str_muscle):
                    c_ = colors[vec_muscle_color == 'auc_target', :]
                    # c_ = 'k''  # usual coloring
                else:
                    c_ = 'k'

                ax1.plot(pi_candidate, Z_squeezed[:, ix_m, ix_p].squeeze(), color=c_, alpha=0.05)
                pk_max = Z_squeezed[:, ix_m, ix_p].max()
                if pk_max > ylim[1]:
                    above_bounds[str_p] = float(pk_max)

            ax1.plot(pi_candidate, z, color=c, label=f'Mean')

            ax1.set_xlabel('PI (ms)')

            xticks = ax1.get_xticks()
            xticks_labels = [f'{int(xtick):+d}' if xtick != 0 else '0' for xtick in xticks]
            ax1.set_xticks(xticks)  # just to suppress a warning
            ax1.set_xticklabels(xticks_labels)

            if ix_m == 0:
                ax1.set_ylabel('% Facilitation')
            else:
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])

            ax1.set_ylim(ylim)
            ax1.set_xlim([pi_candidate[0], pi_candidate[-1]])
            for spine in ['top', 'right']:
                ax1.spines[spine].set_visible(False)
                # ax1.spines[spine].set_color('none')  # Alternatively use rgba(0, 0, 0, 0)
                # ax1.spines[spine].set_linewidth(0)  # Set width to zero

            if (ix_m < n_plots - 1) or (str_muscle == 'auc_target'):
                n_conditions = posterior_samples_grouped['loc_s'].shape[2]
                for ix_c in range(n_conditions):
                    ls = '-'

                    ps_ = np.array(posterior_samples_grouped['loc_s'])[:, :, ix_c, ix_i, ix_m]
                    ps = pctchg(ps_)
                    ps = ps.flatten()
                    ax2.set_yticks([])
                    ax2.set_xticks([0, xlim[-1]])
                    xtickl = [f'{a:0.2f}' for a in ax2.get_xticks()]
                    xtickl[0] = 0
                    ax2.set_xticklabels(xtickl)
                    ax2.set_yticklabels([])
                    ax2.tick_params(left=False)  # Alternative way to remove ticks
                    ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)

                    sns.kdeplot(y=ps, color=c, linestyle=ls, ax=ax2, fill=False)
                    sns.kdeplot(y=ps, color=c, linestyle=ls, ax=ax2, fill=True, clip=(0, np.max(ps)), linewidth=0)

                    p = np.mean(ps > 0, axis=0)
                    p_str = f"Probability = {p:0.2f} "
                    y_text = np.percentile(ps, 98)
                    if y_text > np.mean(ylim):
                        va = 'top'
                    else:
                        va = 'bottom'
                    ax2.text(0,  y_text, f'{p_str}', fontsize=5.5, ha='left', va=va, rotation=-90)

            ax2.set_ylim(ylim)
            ax2.set_xlim(xlim)
            for spine in ['top', 'right']:
                ax2.spines[spine].set_visible(False)
                ax2.spines[spine].set_color('none')  # Alternatively use rgba(0, 0, 0, 0)
                ax2.spines[spine].set_linewidth(0)  # Set width to zero

            if len(above_bounds) > 0:
                print(f'Above bounds participants (% fac.) for {str_muscle}:')
                print(above_bounds)
                for ix, key in enumerate(above_bounds.keys()):
                    ax1.text(
                        1.00, 0.99-0.03*ix,
                        f'{above_bounds[key]:0.0f}%',
                        transform=ax1.transAxes, ha='right', va='top', fontsize=4,
                        bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                    )

            str_max = f'Max of avg.: {np.max(z):0.1f}%'
            ax1.text(
                0.5, 0.9,
                str_max,
                transform=ax1.transAxes, ha='center', va='top', fontsize=4,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
            )

        write_figure(fig, d_analysis, show)

        return ylim


    for ix_intensity in range(len(mapping.get('intensity'))):
        str_intensity = mapping.get('intensity', ix_intensity)
        ylim_ = plot_fac_averaged(Z,pi_candidate, str_intensity, skip=skip)
        # plot_fac_averaged(Z_non_zero_pi, pi_candidate, str_intensity, es='_non-zero-pi', skip=skip)

        if mapping.get('intensity', ix_intensity) == 'supra-sub':
            ylim = ylim_


    # %%
    def plot_fac_averaged_condition_inner(ax, str_condition_expanded, muscle, Z_squeezed, pi_candidate,
                                          ylim=(-25, 100), alpha_line=0.1, show_mean=False, rotate=False):
        allowable_conditions = list(mapping.get('condition').values())
        allowable_conditions.append('Merged')
        assert str_condition_expanded in allowable_conditions, "???"

        # Plot individual participant curves.
        for ix_p in range(num_participants):
            cond_participant = mapping.get('participant_condition', ix_p)
            if str_condition_expanded == 'Merged' or cond_participant == str_condition_expanded:
                ix_c = mapping.get_inverse('condition', cond_participant)
                c_ = colors_condition[ix_c, :]
                # Swap the x and y arguments when rotated.
                if rotate:
                    ax.plot(Z_squeezed[:, muscle, ix_p].squeeze(), pi_candidate, color=c_, alpha=alpha_line)
                else:
                    ax.plot(pi_candidate, Z_squeezed[:, muscle, ix_p].squeeze(), color=c_, alpha=alpha_line)

        # Plot mean curves if requested.
        if show_mean:
            if str_condition_expanded == 'Merged':
                conditions_of_interest = list(mapping.get('condition').values())
            else:
                conditions_of_interest = [cond_participant]

            for cond_label in conditions_of_interest:
                ix_c = mapping.get_inverse('condition', cond_label)
                relevant_indices = [
                    ix_p for ix_p in range(num_participants)
                    if mapping.get('participant_condition', ix_p) == cond_label
                ]
                X = np.nanmean(Z_squeezed[:, muscle, relevant_indices], axis=-1)
                c_ = colors_condition[ix_c, :]
                if rotate:
                    ax.plot(X, pi_candidate, color=c_, alpha=0.95, label=f'Mean {cond_label}')
                else:
                    ax.plot(pi_candidate, X, color=c_, alpha=0.95, label=f'Mean {cond_label}')

        # Adjust axis labels and limits based on rotation.
        if rotate:
            ax.set_xlabel('% Facilitation')
            ax.set_ylabel('PI (ms)')
            ax.set_xlim(ylim)
        else:
            ax.set_xlabel('PI (ms)')
            ax.set_ylabel('% Facilitation')
            ax.set_ylim(ylim)

        # Remove top and right spines.
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.spines[spine].set_color('none')
            ax.spines[spine].set_linewidth(0)

        return ax


    def plot_fac_averaged_condition(Z, pi_candidate, intensity_condition, ylim=(-15, 80), skip=False):
        if skip:
            return None

        ix_i = mapping.get_inverse('intensity', intensity_condition)
        ix_v = 0
        Z_squeezed = Z[:, ix_i, ix_v, ...]
        Z_squeezed = pctchg(Z_squeezed)

        n_r, n_c = num_muscles, 3
        fig_width = (1.1 + 2.75 * n_c) * CMTI * 1.5
        fig_height = 8 * (n_r + float(n_plots > 1)) * CMTI

        fig, axs = plt.subplots(n_r, n_c, figsize=(fig_width, fig_height), sharex=True, sharey=True, squeeze=False)
        fig.figure_name = f'fac_averaged_condition_{intensity_condition}'

        for ix_m in range(num_muscles):
            for ix_col in range(n_c):
                if ix_col == 0:
                    str_condition_expanded = 'Uninjured'
                elif ix_col == 1:
                    str_condition_expanded = "SCI"
                elif ix_col == 2:
                    str_condition_expanded = 'Merged'
                plot_fac_averaged_condition_inner(axs[ix_m, ix_col], str_condition_expanded, ix_m, Z_squeezed, pi_candidate, ylim=ylim)

        fig.tight_layout()

        write_figure(fig, d_analysis, show)


    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        plot_fac_averaged_condition(Z, pi_candidate, 'supra-sub', ylim=ylim, skip=skip)


    # %%
    def plot_condition_kde(ax, ps_red_sca, ps_red_sci, color_sca, color_scs,
                           alpha=0.2, common_faclim=None, rotate=True):
        ps_red_sca_pct = pctchg(ps_red_sca.flatten())
        ps_red_sci_pct = pctchg(ps_red_sci.flatten())
        ps_red_delta = (ps_red_sci - ps_red_sca).flatten()
        ps_red_delta_pct = pctchg(ps_red_delta.flatten())

        p_diff = np.mean(ps_red_delta > 0, axis=0)
        p_sca = np.mean(ps_red_sca.flatten() > 0, axis=0)
        p_sci = np.mean(ps_red_sci.flatten() > 0, axis=0)
        p_str_difference = f"Diff Pr. = {p_diff:0.2f} "
        p_str_sca = f"U%F>0: Pr. = {p_sca:0.2f}, fac:{ps_red_sca_pct.mean():0.1f}% "
        p_str_sci = f"SCI%F>0: Pr. = {p_sci:0.2f}, fac:{ps_red_sci_pct.mean():0.1f}% "

        e_sca = compute_hdi(ps_red_sca_pct)
        e_scs = compute_hdi(ps_red_sci_pct)

        # Set up rotation-specific parameters.
        plot_dim = 'y' if rotate else 'x'
        ref_line_func = ax.axhline if rotate else ax.axvline
        limit_setter = ax.set_ylim if rotate else ax.set_xlim
        tick_setter = ax.set_xticklabels if rotate else ax.set_yticklabels
        facilitation_setter = ax.set_ylabel if rotate else ax.set_xlabel
        density_setter = ax.set_xlabel if rotate else ax.set_ylabel


        # Draw reference line at zero.
        ref_line_func(0, color='k', linestyle='--', linewidth=0.5)

        # Plot KDEs for both conditions.
        sns.kdeplot(**{plot_dim: ps_red_sca_pct}, color=color_sca, ax=ax, fill=False)
        sns.kdeplot(**{plot_dim: ps_red_sca_pct}, color=color_sca, ax=ax, fill=True,
                    linewidth=0, alpha=alpha)
        sns.kdeplot(**{plot_dim: ps_red_sci_pct}, color=color_scs, ax=ax, fill=False)
        sns.kdeplot(**{plot_dim: ps_red_sci_pct}, color=color_scs, ax=ax, fill=True,
                    linewidth=0, alpha=alpha)

        line_limit = ax.get_xlim()[1] if rotate else ax.get_ylim()[1]

        facilitation_setter('% Facilitation')
        density_setter('Density')

        if rotate:
            tick_vals = [f'{a:0.2f}' for a in ax.get_xticks()]
            ax.plot(np.ones(2) * 1.00 * line_limit, e_sca, '-', color=color_sca)
            ax.plot(np.ones(2) * 1.03 * line_limit, e_scs, '-', color=color_scs)

        else:
            tick_vals = [f'{a:0.2f}' for a in ax.get_yticks()]
            ax.plot(e_sca, np.ones(2) * 1.00 * line_limit, '-', color=color_sca)
            ax.plot(e_scs, np.ones(2) * 1.03 * line_limit, '-', color=color_scs)
        #
        if common_faclim:
            limit_setter(common_faclim)
        tick_setter(tick_vals)

        ax.text(
            0.95, 0.99,
            f'{p_str_difference}\n{p_str_sca}\n{p_str_sci}',
            transform=ax.transAxes, ha='right', va='top', fontsize=6,
            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    def plot_condition_diff(ax, ps_red_sca, ps_red_sci, muscle_color, str_muscle):
        ps_red_delta = (ps_red_sci - ps_red_sca).flatten()
        ps_red_delta_pct = pctchg(ps_red_delta.flatten())
        ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
        sns.kdeplot(y=ps_red_delta_pct, color=muscle_color, ax=ax, fill=False)
        sns.kdeplot(y=ps_red_delta_pct, color=muscle_color, ax=ax, fill=True, clip=(0, np.max(ps_red_delta_pct)), linewidth=0)
        y_limit = ax.get_xlim()[1]
        e_diff = compute_hdi(ps_red_delta)
        ax.plot(np.ones(2) * 1.00 * y_limit, e_diff, '-', color=muscle_color)
        p = np.mean(ps_red_delta > 0, axis=0)
        p_str = f"Probability = {p:0.2f} "
        ax.set_title(f'SCS - SCA, {str_muscle}')
        ax.text(
            0.95, 0.05, f'{p_str}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=4,
            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
        )
        ax.set_xlabel('Density')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        xtickl = [f'{a:0.2f}' for a in ax.get_xticks()]
        xtickl[0] = 0
        ax.set_xticklabels(xtickl)


    def plot_fac_condition_comparison(skip=False):
        if skip:
            return None
        if posterior_samples_grouped['loc_s'].shape[2] == 1:
            return None

        ix_sci = mapping.get_inverse('condition', 'SCI')
        ix_sca = mapping.get_inverse('condition', 'Uninjured')
        ps_loc_s_scs = posterior_samples_grouped['loc_s'][:, :, [ix_sci], :, :]
        ps_loc_s_sca = posterior_samples_grouped['loc_s'][:, :, [ix_sca], :, :]

        ix_i = mapping.get_inverse('intensity', 'supra-sub')
        n_plots = num_muscles
        n_r, n_c = 2, n_plots
        fig_width = (1.1 + 2.75 * n_c) * CMTI * 1.5
        fig_height = 8 * (n_r + float(n_plots > 1)) * CMTI
        fig, axs = plt.subplots(n_r, n_c, figsize=(fig_width, fig_height), sharex=True, sharey=True, squeeze=False)
        fig.figure_name = 'fac_condition_difference'

        for ix_m in range(num_muscles):
            str_muscle = mapping.get('muscle', ix_m)

            ps_red_sca = ps_loc_s_sca[:, :, :, ix_i, ix_m].squeeze()
            ps_red_scs = ps_loc_s_scs[:, :, :, ix_i, ix_m].squeeze()

            ax_top = axs[0, ix_m]
            plot_condition_kde(ax_top, ps_red_sca, ps_red_scs, colors_condition[ix_sca, :], colors_condition[ix_sci, :])

            ax_bottom = axs[1, ix_m]
            c = colors[vec_muscle_color == str_muscle, :]
            plot_condition_diff(ax_bottom, ps_red_sca, ps_red_scs, c, str_muscle)

        fig.tight_layout()

        write_figure(fig, d_analysis, show)


    if cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        plot_fac_condition_comparison(skip=skip)


    # %%
    def plot_condition_comparisons(df, pi_candidate, Z, posterior_samples_grouped, dhbmep_tms, dhbmep_tss, df_template_tms,
                                   df_template_tss, pp_tms, pp_tss, individualdemo='UEMS', hdidelta_a_cutoff=None, skip=False):
        if skip:
            return None
        if posterior_samples_grouped['loc_s'].shape[2] == 1:
            return None

        fig_width = 17.6 * CMTI
        fig_height = 15 * CMTI
        ix_v, ix_i, ix_m = 0, mapping.get_inverse('intensity', 'supra-sub'), 0
        fig, axs = plt.subplots(
            3, 3,
            figsize=(fig_width, fig_height),
            squeeze=False,
            gridspec_kw={'width_ratios': [4.5, 1, 4.5]},
            constrained_layout=True,
            dpi=300
        )

        es_cutoff = f'_cut{hdidelta_a_cutoff:0.0f}' if hdidelta_a_cutoff else ''
        fig.figure_name = 'condition_comparison_merged' + es_cutoff

        df = get_participant_summary(df, mapping, posterior_samples_grouped, dhbmep_tms, dhbmep_tss, mask_muscle,
                                     target_muscle, hdidelta_a_cutoff=h_hdidelta_a_cutoff(), ix_v=0)
        df_demo_sci = df[df['participant_condition'] == 'SCI']
        demo_score = df_demo_sci[individualdemo].to_numpy()
        individualdemo_lims = [0, np.ceil(np.max(demo_score)/5) * 5]

        alpha = 0.05
        ix_i = mapping.get_inverse('intensity', 'supra-sub')
        for ix_row in range(2):
            if ix_row == 0:
                hbmep_type, str_intensity, df_template, pp, pp_avg = \
                    dhbmep_tss, 'TSCSInt', df_template_tss, pp_tss, pp_tss_avg
                x_array_th, y_array_fac_pct, _ = extract_s_vs_threshold(posterior_samples_grouped, muscles, dhbmep_tss, mapping,
                                                              mask_muscle, data, hdidelta_a_cutoff=hdidelta_a_cutoff)
                units = 'mA'
                str_pre_stim = 'spinal'
                common_intensity_lim = [0, 80]
            elif ix_row == 1:
                hbmep_type, str_intensity, df_template, pp, pp_avg = \
                    dhbmep_tms, 'TMSInt', df_template_tms, pp_tms, pp_tms_avg
                x_array_th, y_array_fac_pct, _ = extract_s_vs_threshold(posterior_samples_grouped, muscles,
                                                                            dhbmep_tms,
                                                                            mapping, mask_muscle, data, hdidelta_a_cutoff=hdidelta_a_cutoff)
                units = '% MSO'
                str_pre_stim = 'cortical'
                common_intensity_lim = [0, 80]
            else:
                raise Exception("?")
            assert str_intensity in hbmep_type['es'], "Some sort of TMS/TSS mismatch!"

            x_array_th_local = x_array_th[:, ix_m]
            y_array_fac_pct = pctchg(y_array_fac_pct[:, ix_m])

            ix_sci_hbmep = hbmep_type['unpack']['participant_condition'] == 'SCI'
            ps_sci = hbmep_type['ps']['a_loc'][:, ix_sci_hbmep, ix_m].ravel()
            ix_sca_hbmep = hbmep_type['unpack']['participant_condition'] == 'Uninjured'
            ps_sca = hbmep_type['ps']['a_loc'][:, ix_sca_hbmep, ix_m].ravel()

            ix_sci = mapping.get_inverse('condition', 'SCI')
            ix_sca = mapping.get_inverse('condition', 'Uninjured')

            ax = axs[ix_row, 0]
            alpha_line = 0.1
            plot_rc_condition_ax(ax, hbmep_type, df_template, ix_i, ix_m, ix_v, pp,
                                 str_intensity, alpha=0, alpha_line=alpha_line, show_average_p=False)
            for ix_c in [0, 1]:
                plot_rc(ax, 0, 0, ix_c, 0, ix_m, hbmep_type['model'], str_intensity, df_template, hbmep_type['data'], pp_avg, None,
                    color=colors_condition[ix_c, :], alpha=0.0, alpha_line=0.95, show_extras=False)
            ax.set_xlim(common_intensity_lim)

            ax = axs[ix_row, 1]
            plot_hbmep_distribution(ax=ax, ps_sci=ps_sci, ps_sca=ps_sca, str_intensity=str_intensity, alpha=alpha, rotate=True)
            # ax.set_box_aspect(1)
            ax.set_ylim(common_intensity_lim)

            ax = axs[ix_row, 2]
            plot_with_fit(ax, demo_score, x_array_th_local[df_demo_sci.index],
                         ylabel=f'{str_pre_stim.capitalize()} threshold ({units})',
                         xlabel=individualdemo,
                          show_n=True,
                          color=colors_condition[ix_sci, :]
                         )
            ax.set_xlim(individualdemo_lims)
            ax.set_ylim(common_intensity_lim)
            ax.set_box_aspect(1)

        ax = axs[2, 0]
        common_faclim = (-25, 100)
        Z_squeezed = deepcopy(Z[:, ix_i, ix_v, ...])
        Z_squeezed = pctchg(Z_squeezed)
        plot_fac_averaged_condition_inner(ax, 'Merged', ix_m, Z_squeezed, pi_candidate,
                                          ylim=common_faclim, alpha_line=alpha_line, rotate=False, show_mean=True)
        ax.set_ylim(common_faclim)
        ax.set_box_aspect(1)

        ax = axs[2, 1]
        ps_loc_s_scs = posterior_samples_grouped['loc_s'][:, :, [ix_sci], :, :]
        ps_loc_s_sca = posterior_samples_grouped['loc_s'][:, :, [ix_sca], :, :]
        ps_red_sca = ps_loc_s_sca[:, :, :, ix_i, ix_m].squeeze()
        ps_red_scs = ps_loc_s_scs[:, :, :, ix_i, ix_m].squeeze()
        plot_condition_kde(ax, ps_red_sca, ps_red_scs, colors_condition[ix_sca, :], colors_condition[ix_sci, :],
                           alpha=alpha, common_faclim=common_faclim, rotate=True)
        ax.set_ylim(common_faclim)
        # ax.set_box_aspect(1)

        ax = axs[2, 2]
        plot_with_fit(ax, demo_score, y_array_fac_pct[df_demo_sci.index],
                     xlabel=individualdemo,
                     ylabel='% Facilitation',
                     show_n=True,
                     color=colors_condition[ix_sci, :]
                     )
        ax.set_xlim(individualdemo_lims)
        ax.set_ylim(common_faclim)
        ax.set_box_aspect(1)

        write_figure(fig, d_analysis, show)


    if (cfg['DATA_OPTIONS']['type'] == 'noninvasive') and (num_muscles == 1):
        plot_condition_comparisons(data, pi_candidate, Z, posterior_samples_grouped,
                                   hbmep_tms, hbmep_tss, df_template_tms, df_template_tss, pp_tms, pp_tss,
                                   hdidelta_a_cutoff=h_hdidelta_a_cutoff(), skip=skip)


    # %% Correlation between effect strengths across muscles
    def plot_across_muscle_correlation_scatters(skip=False):
        """
        Plot correlation scatters across muscles, showing r
        with significance asterisks, fit lines, and custom spines.
        """
        if skip:
            return None

        ix_i = mapping.get_inverse('intensity', 'supra-sub')
        ix_v = 0
        s = posterior_samples_grouped['s'][:, :, ix_v, :, :, ix_i, :]
        mask_local = np.any(mask_muscle[:, ix_v, :, :, ix_i, :], axis=0)
        s = np.where(mask_local[None, None, :, :, :], s, np.nan)
        s_mea = np.mean(s, axis=(0, 1))

        # get the condition out
        vec_ix_c = [
            mapping.get_inverse('condition', mapping.get('participant_condition', i))
            for i in range(s_mea.shape[0])
        ]
        s_mea_reduced = s_mea[np.arange(s_mea.shape[0]), vec_ix_c, :]

        num_columns = s_mea.shape[2]
        fig_width = 8.0 * CMTI
        fig_height = 8.0 * CMTI
        n_panels = num_columns - 1

        fig, axs = plt.subplots(n_panels, n_panels, figsize=(fig_width, fig_height), squeeze=False)
        fig.figure_name = 'correlation_muscle_scatter'

        xmins = np.full(axs.shape, np.nan)
        xmaxs = np.full(axs.shape, np.nan)
        ymins = np.full(axs.shape, np.nan)
        ymaxs = np.full(axs.shape, np.nan)

        # Iterate over pairs (lower triangle only: i > j)
        for i in range(1, num_columns):
            for j in range(0, i):
                ax = axs[i - 1, j]

                s_mea_1 = s_mea_reduced[:, i]
                s_mea_2 = s_mea_reduced[:, j]

                for ix_p in range(num_participants):
                    c = color_pairing
                    ax.plot(
                        s_mea_2[ix_p],
                        s_mea_1[ix_p],
                        'o',
                        markersize=3,
                        alpha=0.5,
                        markerfacecolor=color_pairing,
                        markeredgecolor='w'
                    )

                x_lo, x_hi = ax.get_xlim()
                y_lo, y_hi = ax.get_ylim()
                xmins[i - 1, j] = x_lo
                xmaxs[i - 1, j] = x_hi
                ymins[i - 1, j] = y_lo
                ymaxs[i - 1, j] = y_hi

        # Compute unified min/max for each column/row
        xmins = np.full_like(xmins, np.nanmin(xmins, axis=0, keepdims=True))
        xmaxs = np.full_like(xmaxs, np.nanmax(xmaxs, axis=0, keepdims=True))
        ymins = np.full_like(ymins, np.nanmin(ymins, axis=1, keepdims=True))
        ymaxs = np.full_like(ymaxs, np.nanmax(ymaxs, axis=1, keepdims=True))

        for i in range(1, num_columns):
            for j in range(0, i):
                ax = axs[i - 1, j]
                ax.set_xlim(xmins[i - 1, j], xmaxs[i - 1, j])
                ax.set_ylim(ymins[i - 1, j], ymaxs[i - 1, j])

        for i in range(1, num_columns):
            for j in range(0, i):
                ax = axs[i - 1, j]
                s_mea_1 = s_mea_reduced[:, i]
                s_mea_2 = s_mea_reduced[:, j]
                valid_mask = ~np.isnan(s_mea_1) & ~np.isnan(s_mea_2)

                y_clean = s_mea_1[valid_mask]
                x_clean = s_mea_2[valid_mask]

                x_fit, y_fit, stats = linear_model(x_clean, y_clean, xmins[i - 1, j], xmaxs[i - 1, j])
                ax.plot(x_fit, y_fit, '-', color=color_pairing)

                str_text = rf'$r = {stats["corr_value"]:.2f}^{{{stats["corr_star"]}}}$'
                ax.text(
                    0.95, 0.05, str_text,
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=4,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )

        for i in range(1, num_columns):
            for j in range(0, i):
                ax = axs[i - 1, j]
                c = colors[vec_muscle_color == mapping.get('muscle', j), :]
                ax.spines['bottom'].set_color(c)
                c = colors[vec_muscle_color == mapping.get('muscle', i), :]
                ax.spines['left'].set_color(c)

        for row_i in range(n_panels):
            for col_j in range(n_panels):
                ax = axs[row_i, col_j]

                # Hide upper-triangle panels
                if col_j >= row_i + 1:
                    ax.set_visible(False)
                    continue

                ax.spines['top'].set(visible=False)
                ax.spines['right'].set(visible=False)
                ax.spines['bottom'].set(visible=True, linewidth=2)
                ax.spines['left'].set(visible=True, linewidth=2)

                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

                if row_i < n_panels - 1:
                    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                if col_j > 0:
                    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        fig.tight_layout()

        write_figure(fig, d_analysis, show)


    if num_muscles > 1:
        plot_across_muscle_correlation_scatters(skip=skip)


    # %% Correlation between threshold across muscles
    def plot_across_muscle_correlation_scatters_threshold(dhbmep, skip=False):
        """
        Plot correlation scatters across muscles, showing r
        with significance asterisks, fit lines, and custom spines.
        """
        if skip:
            return None

        model_hbmep, hbmep_ps, hbmep_unpack = dhbmep['model'], dhbmep['ps'], dhbmep['unpack']
        ix_i = mapping.get_inverse('intensity', 'supra-sub')
        ix_v = 0

        th_mat = np.zeros((num_participants, num_muscles))
        hbmep_unpack_pv = np.array([[int(x) for x in hbmep_unpack['participant_visit'][y].split('_')] for y in
                                    range(len(hbmep_unpack['participant_visit']))])
        for ix_p in range(num_participants):
            case_ps_f0 = (hbmep_unpack_pv[:, 0] == ix_p) & (hbmep_unpack_pv[:, 1] == ix_v)
            case_ps_f1 = mapping.get(model_hbmep.features[1], ix_p) == hbmep_unpack[
            model_hbmep.features[1]]
            pa = hbmep_ps[site.a][:, case_ps_f0, case_ps_f1, :]
            th_mat[ix_p, :] = pa.mean(axis=(0, 1))

        num_columns = num_muscles
        fig_width = 8.0 * CMTI
        fig_height = 8.0 * CMTI
        n_panels = num_columns - 1

        fig, axs = plt.subplots(n_panels, n_panels, figsize=(fig_width, fig_height), squeeze=False)
        fig.figure_name = 'correlation_muscle_scatter_threshold'

        xmins = np.full(axs.shape, np.nan)
        xmaxs = np.full(axs.shape, np.nan)
        ymins = np.full(axs.shape, np.nan)
        ymaxs = np.full(axs.shape, np.nan)

        # Iterate over pairs (lower triangle only: i > j)
        for i in range(1, num_columns):
            for j in range(0, i):
                ax = axs[i - 1, j]

                s_mea_1 = th_mat[:, i]
                s_mea_2 = th_mat[:, j]

                for ix_p in range(num_participants):
                    c = color_pairing
                    ax.plot(
                        s_mea_2[ix_p],
                        s_mea_1[ix_p],
                        'o',
                        markersize=3,
                        alpha=0.5,
                        markerfacecolor=color_pairing,
                        markeredgecolor='w'
                    )

                x_lo, x_hi = ax.get_xlim()
                y_lo, y_hi = ax.get_ylim()
                xmins[i - 1, j] = x_lo
                xmaxs[i - 1, j] = x_hi
                ymins[i - 1, j] = y_lo
                ymaxs[i - 1, j] = y_hi

        xmins = np.full_like(xmins, np.nanmin(xmins, axis=0, keepdims=True))
        xmaxs = np.full_like(xmaxs, np.nanmax(xmaxs, axis=0, keepdims=True))
        ymins = np.full_like(ymins, np.nanmin(ymins, axis=1, keepdims=True))
        ymaxs = np.full_like(ymaxs, np.nanmax(ymaxs, axis=1, keepdims=True))

        for i in range(1, num_columns):
            for j in range(0, i):
                ax = axs[i - 1, j]
                ax.set_xlim(xmins[i - 1, j], xmaxs[i - 1, j])
                ax.set_ylim(ymins[i - 1, j], ymaxs[i - 1, j])

        for i in range(1, num_columns):
            for j in range(0, i):
                ax = axs[i - 1, j]
                s_mea_1 = th_mat[:, i]
                s_mea_2 = th_mat[:, j]
                valid_mask = ~np.isnan(s_mea_1) & ~np.isnan(s_mea_2)

                y_clean = s_mea_1[valid_mask]
                x_clean = s_mea_2[valid_mask]

                x_fit, y_fit, stats = linear_model(x_clean, y_clean, xmins[i - 1, j], xmaxs[i - 1, j])
                ax.plot(x_fit, y_fit, '-', color=color_pairing)

                str_text = rf'$r = {stats["corr_value"]:.2f}^{{{stats["corr_star"]}}}$'
                ax.text(
                    0.95, 0.05, str_text,
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=4,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )

        for i in range(1, num_columns):
            for j in range(0, i):
                ax = axs[i - 1, j]
                c = colors[vec_muscle_color == mapping.get('muscle', j), :]
                ax.spines['bottom'].set_color(c)
                c = colors[vec_muscle_color == mapping.get('muscle', i), :]
                ax.spines['left'].set_color(c)

        for row_i in range(n_panels):
            for col_j in range(n_panels):
                ax = axs[row_i, col_j]

                if col_j >= row_i + 1:
                    ax.set_visible(False)
                    continue

                ax.spines['top'].set(visible=False)
                ax.spines['right'].set(visible=False)
                ax.spines['bottom'].set(visible=True, linewidth=2)
                ax.spines['left'].set(visible=True, linewidth=2)

                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

                if row_i < n_panels - 1:
                    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                if col_j > 0:
                    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Reduce whitespace more aggressively
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        fig.tight_layout()

        write_figure(fig, d_analysis, show)


    if (num_muscles > 1) and cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        plot_across_muscle_correlation_scatters_threshold(hbmep_tss, skip=skip)


    # %%
    variables_to_plot = ['scale_w', 'scale_s', 'scale_loc_s']  # pop_bell1_mean
    plot_posteriors(cfg, posterior_samples, variables_to_plot, mapping, None, d_analysis / 'dist_pop.png', show=show, skip=skip)


   # %%
    print(f'Done with:\n{d_analysis}')


#%%
if __name__ == '__main__':
    main()