import sys
from pathlib import Path
import pickle
import arviz as az
from numpyro.infer import log_likelihood
import toml
import re

# To allow importing from project root when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import BASE_DIR, DATA_FOLDER
from config_analysis import CONFIG_ANALYSIS
from src.bsp.core.filter import filter_data as filter
import src.bsp.core.model as model_module


def main():
    o_models = CONFIG_ANALYSIS.get("o_model", [])
    if not isinstance(o_models, list):
        o_models = [o_models]

    if not o_models:
        print("No models found in CONFIG_ANALYSIS['o_model']")
        return

    idatas = {}

    for o_model in o_models:
        print(f"\n=============================================")
        print(f"Processing model: {o_model}")
        d_analysis = BASE_DIR / o_model
        p_model = d_analysis / "mcmc_model.pkl"
        cfg_file_path = d_analysis / "config.toml"

        if not p_model.exists():
            print(f"Skipping {o_model}: mcmc_model.pkl not found at {p_model}")
            continue
        if not cfg_file_path.exists():
            print(f"Skipping {o_model}: config.toml not found at {cfg_file_path}")
            continue

        cfg = toml.load(cfg_file_path)
        cfg["DATA_FOLDER"] = DATA_FOLDER
        
        # Handle anterior condition from check_posteriors.py
        is_anterior = False
        if 'anterior' in cfg['DATA_OPTIONS']['es']:
            cfg['DATA_FOLDER']['intraoperative'] = cfg['DATA_FOLDER']['intraoperative'].with_name('np_anterior_2025-09-09')
            is_anterior = True

        match = re.search(r"(model[^_]+)", o_model)
        if match:
            model_version = match.group(1)
        else:
            print(f"Could not determine model version from {o_model}")
            continue

        muscles_from_cfg = cfg["DATA_OPTIONS"]["response"]

        if cfg["DATA_OPTIONS"]["type"] == "intraoperative":
            es = ""
        else:
            es = "_" + o_model
            
        print("Loading data...")
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
        assert muscles == muscles_from_cfg, "Muscles from config do not match mapping"

        num_muscles = len(muscles)
        response_obs = data[muscles].values.reshape(-1, num_muscles)

        model = getattr(model_module, model_version)
        print("Loading MCMC samples...")
        with open(p_model, 'rb') as f:
            mcmc = pickle.load(f)

        posterior_samples = mcmc.get_samples(group_by_chain=False)

        print("Computing log_likelihood...")
        # Note: the arguments to model are: cpi, time, response_obs, run_index, visit_index, participant_index, descriptor_index, intensity_index, average_count, model_options
        log_lik = log_likelihood(
            model, posterior_samples, pi, time, response_obs, run_index, visit_index, 
            participant_index, condition_index, cxsc_index, average_count, cfg["MODEL_OPTIONS"]
        )

        print("Generating InferenceData object...")
        idata = az.from_numpyro(posterior=mcmc, log_likelihood={"y": log_lik["y"]})
        idatas[o_model] = idata
        
        # Optionally print individual LOO
        print(f"LOO for {o_model}:")
        print(az.loo(idata))

    if len(idatas) > 0:
        # Group idatas by the shape/size of their log likelihood data
        # because az.compare requires the number of observations to be identical.
        from collections import defaultdict
        groups = defaultdict(dict)
        for name, idata in idatas.items():
            num_obs = idata.log_likelihood['y'].size
            groups[num_obs][name] = idata
            
        print("\n=============================================")
        for num_obs, group_idatas in groups.items():
            print(f"\nComparing models with {num_obs} observations...")
            if len(group_idatas) > 1:
                try:
                    comp = az.compare(group_idatas, ic="loo")
                    print(comp)
                    # Optionally save comparison to CSV
                    output_csv = BASE_DIR / f"loo_comparison_obs{num_obs}.csv"
                    comp.to_csv(output_csv)
                    print(f"Comparison saved to {output_csv}")
                except Exception as e:
                    print(f"Failed to compare models with {num_obs} observations: {e}")
            else:
                print("Only one model in this group. Cannot perform comparison.")
    else:
        print("\nNo models were processed successfully.")


if __name__ == "__main__":
    main()
