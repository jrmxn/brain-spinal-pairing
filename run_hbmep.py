import os
import pickle
import importlib
import inspect
from copy import deepcopy
import numpy as np
from pathlib import Path
import time
import random
from joblib import Parallel, delayed
import arviz as az
from src.bsp.core.filter import filter_ni
from hbmep.config import Config as ConfigHBMEP
from hbmep.model.utils import Site as site

import config

if __name__ == "__main__":
    model_module = importlib.import_module("src.bsp.recruitment.model")

    # Reload configuration
    importlib.reload(config)
    cfg = config.CONFIG
    cfg['DATA_OPTIONS']['response_transform'] = 'linear'  # ensure linear, not log
    es_version = cfg['DATA_OPTIONS']['es']

    # Create list of relevant model classes
    model_list = [
        func for func in dir(model_module)
        if inspect.isclass(getattr(model_module, func)) and func.startswith('RectifiedLogistic')
    ]
    model_list = [model for model in model_list if model in ['RectifiedLogistic', 'RectifiedLogisticCo_a']]

    # Filter data once
    ie_only = False
    dfo, mapping, mep, mep_channel = filter_ni(cfg, overwrite=True, ie_only=ie_only, es='hbmep_')
    dfo = dfo.reset_index(drop=True).sort_index()

    # Load hbMEP config
    toml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "bsp", "recruitment", "hbmep_config.toml")
    cfg_hbmep = ConfigHBMEP(toml_path=toml_path)

    render_curves = True  # only on re-run though
    all_responses = [
        ['auc_target'],
        ['ECR', 'FCR', 'APB', 'ADM', 'FDI']
    ]
    all_use_condition_features = [False, True]
    all_ix_models = range(len(model_list))
    all_small_flags = [True, False]
    all_ix_stim_types = [0, 1]
    all_s50parameterization = [False]
    all_smooth = [True]
    all_size_type = ['auc', 'pkpk']
    all_mixture = [True]


    def run_experiment(
            smooth,
            s50parameterization,
            response,
            use_condition_feature,
            ix_model,
            small,
            ix_stim_type,
            dfo,
            mep,
            mep_channel,
            cfg_hbmep,
            model_list,
            model_module,
            es_version,
            size_type,
            use_mixture,
            render_curves
    ):
        """

        """

        str_model = model_list[ix_model]

        # Skip
        if use_condition_feature:
            if "Co" not in str_model:
                return
        else:
            if "Co" in str_model:
                return

        if size_type == 'pkpk':
            if (s50parameterization) or ("Co" in str_model) or (len(response) > 1):
                return

        delay = random.uniform(0, 5)
        print(f'Delaying start by {delay:0.2f}s')
        time.sleep(delay)

        dfo_reduced = dfo.copy()

        if size_type == 'auc':
            pass
        elif size_type == 'pkpk':
            muscles_pkpk = {col.replace('_pkpk', '') for col in dfo_reduced.columns if col.endswith('_pkpk')}
            rename_dict = {}
            for muscle in muscles_pkpk:
                pkpk_col = f"{muscle}_pkpk"
                if pkpk_col in dfo_reduced.columns:
                    rename_dict[pkpk_col] = muscle
                if muscle in dfo_reduced.columns:
                    rename_dict[muscle] = muscle + "_auc"
            dfo_reduced = dfo_reduced.rename(columns=rename_dict)

        # Determine stimulation type
        stim_type = 'TMS' if ix_stim_type == 0 else 'TSS'
        stim_type_alt = 'TMS' if ix_stim_type == 0 else 'TSCS'

        # Filter data for this stimulation type
        case_rc_type = dfo_reduced['stim_type'] == stim_type
        mep_stype = mep[:, :, case_rc_type]
        dfo_reduced = dfo_reduced[case_rc_type].copy()
        dfo_reduced = dfo_reduced.reset_index(drop=True).sort_index()

        mep_indexing = np.array([list(mep_channel).index(channel) for channel in response if channel in mep_channel])
        mep_stype = mep_stype[mep_indexing, :, :]

        # Add participant_visit feature
        dfo_reduced['participant_visit'] = (
                dfo_reduced['participant_index'].astype(str)
                + '_'
                + dfo_reduced['visit_index'].astype(str)
        )

        # Clean and preprocess the data
        dfo_reduced[response] = dfo_reduced[response].apply(
            lambda col: col.map(lambda x: np.nan if x is not None and x <= 0 else x)
        )

        # Optionally exclude participants with any missing data
        exclude_participants_with_missing = False
        if exclude_participants_with_missing:
            rows_to_drop = np.any(dfo_reduced[response].isna(), axis=1)
            mep_stype = mep_stype[:, :, ~rows_to_drop]
            dfo_reduced = dfo_reduced[~rows_to_drop]

        dfo_reduced = dfo_reduced.reset_index(drop=True).sort_index()
        mep_stype = np.transpose(mep_stype, (2, 1, 0))

        # Update the hbMEP configuration
        cfg_hbmep.INTENSITY = f'{stim_type_alt}Int'
        cfg_hbmep.RESPONSE = response
        if use_condition_feature:
            cfg_hbmep.FEATURES = ["participant_visit", "participant_condition"]
        else:
            # Just a dummy filler if no condition feature is desired
            cfg_hbmep.FEATURES = ["participant_visit", "participant_filler"]

        # Build / configure model
        h_model = getattr(model_module, str_model)
        model = h_model(config=cfg_hbmep)

        # Flags
        model.use_mixture = use_mixture
        model.smooth = smooth
        model.s50parameterization = s50parameterization
        model.visit = 0
        model.size_type = size_type
        if small:
            model.small = True
            model.mcmc_params = {
                "num_warmup": 4000,
                "num_samples": 500,
                "thinning": 50,
                "num_chains": 4,
            }

        build_dir_local = str(Path(config.BASE_DIR).parent / 'hbmep' / f"hbmep_{es_version}{model.subname}")
        model.build_dir = build_dir_local

        # Load data into the model
        df, encoder_dict = model.load(df=dfo_reduced)
        hbmep_unpack = {
            cfg_hbmep.FEATURES[0]: encoder_dict[cfg_hbmep.FEATURES[0]].classes_,
            cfg_hbmep.FEATURES[1]: encoder_dict[cfg_hbmep.FEATURES[1]].classes_
        }
        # encoder_dict['participant_condition'].inverse_transform([0])

        # Mask
        hbmep_unpack_pv = np.array([[int(x) for x in hbmep_unpack['participant_visit'][y].split('_')] for y in
                                    range(len(hbmep_unpack['participant_visit']))])
        n_curves = len(hbmep_unpack_pv)
        n_conditions = hbmep_unpack[cfg_hbmep.FEATURES[1]].shape[0]
        desired_local_shape = (1, 1, n_conditions, len(cfg_hbmep.RESPONSE))
        mask = np.zeros((1, n_curves, n_conditions, len(cfg_hbmep.RESPONSE)), dtype=bool)
        for ix_f in range(n_curves):
            f = hbmep_unpack_pv[ix_f]
            ix_p, ix_v = f[0], f[1]
            case_df = (df['participant_index'] == ix_p) & (df['visit_index'] == ix_v)
            df_local = df[case_df].copy()
            mask_local = np.array(np.isfinite(df_local[response])).astype(bool)
            mask_local = np.any(mask_local, axis=0, keepdims=True)
            mask_local = np.reshape(mask_local, (1, 1, 1, -1))
            mask_local = np.broadcast_to(mask_local, desired_local_shape).copy()
            if use_condition_feature:
                # this has been transformed by hbmep... so it's now index instead of a string
                # n.b. that if it's not a feature it does not get transformed...!
                ix_condition_hbmep = df_local['participant_condition'].unique()[0]
                mask_local[:, :, np.delete(np.arange(mask_local.shape[2]), ix_condition_hbmep), :] = 0
            mask[:, ix_f, :, :] = mask_local

        # Inference path
        inference_path = Path(model.build_dir) / 'inference.pkl'
        overwrite = False

        # Run or load inference
        render_curves_ = deepcopy(render_curves)
        if overwrite or not inference_path.exists():
            model.plot(df=df, encoder_dict=encoder_dict)
            mcmc, posterior_samples = model.run_inference(df=df)

            # Save the model and inference results
            with open(inference_path, "wb") as f:
                pickle.dump((model, mcmc, posterior_samples, hbmep_unpack, mask, df), f)

            # model.print_summary(samples=posterior_samples)
            render_curves_ = False  # if False, just rely on subsequent re-render
        else:
            if render_curves_:
                with open(inference_path, "rb") as f:
                    model, mcmc, posterior_samples, hbmep_unpack, mask, df = pickle.load(f)

        if render_curves_:
            try:
                if site.outlier_prob in posterior_samples.keys():
                    posterior_samples[site.outlier_prob] = posterior_samples[site.outlier_prob] * 0

                prediction_df = model.make_prediction_dataset(df=df, num_points=100)
                posterior_predictive = model.predict(
                    df=prediction_df, posterior_samples=posterior_samples
                )

                # Optionally render curves
                model.mep_window = [-0.25, 0.25]
                model.mep_size_window = [0.005, 0.09]
                model.mep_response = response
                if build_dir_local != model.build_dir:  # switched computers
                    model.build_dir = str(Path(build_dir_local).parent / Path(model.build_dir).parts[-1])
                print(f'Rendering to: {model.build_dir}')
                model.render_recruitment_curves(
                    df=df,
                    encoder_dict=encoder_dict,
                    posterior_samples=posterior_samples,
                    prediction_df=prediction_df,
                    posterior_predictive=posterior_predictive,
                    mep_matrix=mep_stype
                )

                model.render_predictive_check(
                    df=df,
                    encoder_dict=encoder_dict,
                    prediction_df=prediction_df,
                    posterior_predictive=posterior_predictive
                )

                inference_data = az.from_numpyro(mcmc)

                divergences = np.sum(mcmc.get_extra_fields()["diverging"])
                with open(Path(model.build_dir) / 'divergence.txt', 'w') as file:
                    file.write(
                        f'Number of divergences: {divergences.item()}.')

                print('Generating and saving HDI summary...')
                summary = az.summary(inference_data, hdi_prob=0.95)
                summary.to_csv(Path(model.build_dir) / 'summary.csv')
                print('Done.')

                # df.to_csv('/home/mcintosh/dnc_df.csv', index=False)
                # np.save('/home/mcintosh/dnc_mep.npy', mep_stype)

            except Exception as exc:
                print(f"Error printing: {exc}")


    n_jobs = 4
    Parallel(n_jobs=n_jobs)(
        delayed(run_experiment)(
            smooth_val,
            s50parameterization_val,
            response_val,
            use_condition_feature_val,
            ix_model_val,
            small_val,
            ix_stim_type_val,
            dfo,
            mep,
            mep_channel,
            cfg_hbmep,
            model_list,
            model_module,
            es_version,
            size_type,
            use_mixture,
            render_curves
        )
        for smooth_val in all_smooth
        for s50parameterization_val in all_s50parameterization
        for use_mixture in all_mixture
        for response_val in all_responses
        for size_type in all_size_type
        for small_val in all_small_flags
        for use_condition_feature_val in all_use_condition_features
        for ix_model_val in all_ix_models
        for ix_stim_type_val in all_ix_stim_types
    )
