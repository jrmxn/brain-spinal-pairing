from pathlib import Path
import numpy as np
import toml
import importlib
from config import BASE_DIR, DATA_FOLDER
from config_analysis import CONFIG_ANALYSIS
from src.bsp.core.filter import filter_data as filter
from src.bsp.core.filter import filter_ni as filter_ni  # just for mep plots
from copy import deepcopy
from config import CONFIG as cfg_main

model_module = importlib.import_module(f"src.bsp.core.model")


if __name__ == '__main__':

    use_post = False
    if use_post:
        ix_model = CONFIG_ANALYSIS['ix_model']
        o_model = CONFIG_ANALYSIS['o_model'][ix_model]
        f_model = 'mcmc_model.pkl'
        p_model = None
        for path in Path(BASE_DIR).rglob(f'**/{o_model}/{f_model}'):
            p_model = path
            break
        cfg_file_path = p_model.with_stem('config').with_suffix('.toml')
        cfg = toml.load(cfg_file_path)
        cfg['DATA_FOLDER'] = DATA_FOLDER
    else:
        cfg = cfg_main


    cfg_local = deepcopy(cfg)

    # cfg_local['DATA_OPTIONS']['type'] = 'intraoperative'
    cfg_local['DATA_OPTIONS']['type'] = 'noninvasive'

    if cfg_local['DATA_OPTIONS']['type'] == 'noninvasive':
        cfg_local['DATA_OPTIONS']['response'] = ['auc_target']
        # cfg_local['DATA_OPTIONS']['response_transform'] = 'linear'
        ie_only = False
        data, mapping, mep, mep_ch = filter_ni(cfg_local, overwrite=True, ie_only=ie_only, es=f"_justload_{'_'.join(cfg_local['DATA_OPTIONS']['response'])}")

    elif cfg_local['DATA_OPTIONS']['type'] == 'intraoperative':
        cfg_local['DATA_OPTIONS']['intraoperative'][0] = 'global_target'
        cfg_local['DATA_OPTIONS']['visit'] = 'mcintosh2024'
        cfg_local['DATA_OPTIONS']['response'] = ['auc_target']
        data, mapping, mep, mep_ch = filter(cfg_local, overwrite=True, es=f"_justload_{'_'.join(cfg_local['DATA_OPTIONS']['response'])}")

    muscles = cfg_local['DATA_OPTIONS']['response']


    mep_indexing = np.array([list(mep_ch).index(channel) for channel in muscles if channel in mep_ch])
    mep_stype = mep[mep_indexing, :, :]

    np.sum(np.all(np.isnan(mep_stype), axis=1), axis=1)/mep_stype.shape[2]