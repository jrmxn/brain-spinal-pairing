from pathlib import Path
import numpy as np
import numpyro
from numpyro import set_host_device_count
from numpyro.infer import MCMC, NUTS
import jax.random as random
import arviz as az
import importlib
import shutil
from config import CONFIG as cfg
from src.bsp.core.filter import filter_data as filter_data
from src.bsp.core.utils import get_hash, save_model, load_model, write_cfg_to_toml
numpyro.enable_x64()


def main(cfg, model_version='model00'):
    numpyro.enable_x64()
    set_host_device_count(cfg['MCMC_OPTIONS']['num_chains'])
    cfg_file_path = write_cfg_to_toml(cfg)
    log_file_name = '_' + cfg_file_path.parts[-1].replace('.toml', '')

    data, mapping, _, _ = filter_data(cfg, es=log_file_name)

    model_module = importlib.import_module(f"src.bsp.core.model")
    model = getattr(model_module, model_version)

    response = cfg['DATA_OPTIONS']['response']
    y = data[response].values.reshape(-1, len(response))
    spi = data['SPI_target'].values.reshape(-1, 1)
    if "obs" not in cfg['MODEL_OPTIONS']['mask']:
        nan_mask = (np.isnan(y).any(axis=1)) | (np.isnan(spi).any(axis=1))
        spi = spi[~nan_mask]
        y = y[~nan_mask]
        data = data[~nan_mask]

    run_index = data['run_index'].values
    visit_index = data['visit_index'].values
    participant_index = data['participant_index'].values
    condition_index = data['condition_index'].values
    cxsc_index = data['cxsc_index'].values
    time = data['time'].values.reshape(-1, 1)
    average_count = data['average_count'].values.reshape(-1, 1)

    num_participants = len(np.unique(participant_index))

    rng_key = random.PRNGKey(0)
    active_hash = get_hash(cfg_file_path, model, data)
    num_warmup, num_samples = cfg['MCMC_OPTIONS']['num_warmup'], cfg['MCMC_OPTIONS']['num_samples']
    num_chains, thinning = cfg['MCMC_OPTIONS']['num_chains'], cfg['MCMC_OPTIONS']['thinning']

    data_type = cfg['DATA_OPTIONS']['type']
    es = cfg['DATA_OPTIONS']['es']
    cfg['BASE_DIR'].mkdir(parents=True, exist_ok=True)
    o_mcmc = f'{es}{data_type}_{model.__name__}_{"_".join(response)}_n{num_participants}_{active_hash}'
    d_saved_model = Path(cfg['BASE_DIR'] / o_mcmc)
    p_saved_model = d_saved_model / 'mcmc_model.pkl'
    p_data_resaved = p_saved_model.with_stem('data').with_suffix('.parquet')
    p_saved_model_alt = p_saved_model.parent.parent / 'KEEP' / p_saved_model.parent.name / p_saved_model.name
    if not p_saved_model.exists() and p_saved_model_alt.exists():
        p_saved_model = p_saved_model_alt
        d_saved_model = p_saved_model_alt.parent
    p_cfg = p_saved_model.with_stem('config').with_suffix('.toml')

    if p_saved_model.exists():
        file_size = p_saved_model.stat().st_size
        if file_size < 1:
            p_saved_model.unlink()

    if cfg['LOAD_SAVED_MODEL'] and p_saved_model.exists():
        print('Loading pre-generated mcmc.')
        mcmc = load_model(p_saved_model)
        # mcmc.print_summary()
        compute_summary = False
    else:
        print(f'Generating:')
        print(f'{p_saved_model.parent.name}')
        d_saved_model.mkdir(exist_ok=True)
        shutil.move(str(cfg_file_path), str(p_cfg))
        sampler = NUTS(model, target_accept_prob=cfg['SAMPLER_OPTIONS']['target_accept_prob'], max_tree_depth=10)
        mcmc = MCMC(sampler, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, thinning=thinning)
        mcmc.run(rng_key, spi, time, y, run_index, visit_index, participant_index, condition_index, cxsc_index, average_count, cfg['MODEL_OPTIONS'])
        save_model(mcmc, p_saved_model)
        data.to_parquet(p_data_resaved, engine='pyarrow', index=False)
        compute_summary = True
    p_saved_model.with_stem('touch').with_suffix('').touch()

    if compute_summary:
        # mcmc.print_summary(prob=0.95)
        inference_data = az.from_numpyro(mcmc)

        divergences = np.sum(mcmc.get_extra_fields()["diverging"])
        with open(d_saved_model / 'divergence.txt', 'w') as file:
            file.write(
                f'Number of divergences: {divergences.item()}. Number of observations: {np.sum(np.isfinite(y)).item()}')

        print('Generating and saving HDI summary...')
        summary = az.summary(inference_data, hdi_prob=0.95)
        summary.to_csv(d_saved_model / 'summary.csv')
        print('Done.')


if __name__ == '__main__':
    main(cfg)
