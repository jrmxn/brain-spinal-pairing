from pathlib import Path
import importlib.util
import importlib.metadata

vs = importlib.metadata.version('brain-spinal-pairing')

# Default configurations
DATA_FOLDER = {
                'noninvasive': Path.home() / 'pairing' / 'data' / 'noninvasive',
                'intraoperative': None
              }

BASE_DIR = Path(Path.home() / 'pairing' / 'proc')

LOAD_SAVED_MODEL = True

intraoperative_opt = ['global_average',  # adjust_for_spi: None or 'average' or 'global_average'
                        None,  # augment: None or 'replicate_minimum_spi_as_extreme' or 'replicate_sccx_only_spi_as_extreme'
                      ]

DATA_OPTIONS = {
                'type': 'noninvasive',  # 'noninvasive', 'intraoperative'
                'participants': 'all',  # 'all' or ['SCA01', 'SCA02', ...]
                'visit': 'all', # 'all' for ni or 'mcintosh2024' for intraop
                'intensities': ['supra-sub', 'sub-sub'],  # 'supra-sub', 'sub-sub', 'supra-zero', 'sub-zero', 'zero-sub',
                'response': ['ECR', 'FCR', 'APB', 'ADM', 'FDI'],  # ['auc_target'], ['APB', 'ADM', 'FDI'], exclude Biceps, Triceps
                'response_transform': "log2",
                'intraoperative': intraoperative_opt,
                'es': 'v' + vs.replace('.', 'p') + '_'
                }

MCMC_OPTIONS = {'num_warmup': 7500, 'num_samples': 5000, 'num_chains': 4, 'thinning': 2}

SAMPLER_OPTIONS = {'target_accept_prob': 0.95}

MODEL_OPTIONS = {
    'obs': 'normal',
    'mask': ['obs', 'run', 'visit', 'participant', 'condition', 'cxsc'],
    'scale_c_prior': 0.5,
    'c_limits': 2.0,
    'use_b': DATA_OPTIONS['type'] == 'noninvasive',
}

NI_RECOMPUTE_RUNS = [  # n.b. these are annotated by original visits
    {"participant": "SCA01", "visit": 1},
    {"participant": "SCA02", "visit": 1},
    {"participant": "SCS02", "visit": 1},  # looks like a single run otherwise
    {"participant": "SCS09", "visit": 1},
    {"participant": "SCS10", "visit": 1},
    {"participant": "SCS15", "visit": 1}
]

NI_EXCLUDE = [  # n.b. these are annotated by the RENAMED visits
    {"participant": "SCA01", "visit": 1, "run": "all", "muscles": ["FCR"]},  # overlapping stim art. (T: APB)
    {"participant": "SCA03", "visit": 1, "run": "all", "muscles": ["ECR"]},  # response totally absent (T: APB)
    {"participant": "SCA05", "visit": 1, "run": 2, "muscles": ["ADM"]}, # discontinuity in drift + response totally absent (T: APB)
    {"participant": "SCA06", "visit": 1, "run": 3, "datetime_posix": [5827.59719995117], "muscles": ["ECR", "FCR", "APB", "ADM", "FDI"]}, # discontinuity - but it's just the last sample (T: APB <--!!!)
    {"participant": "SCS05", "visit": 1, "run": "all", "muscles": ["ADM", "FDI"]},  # response totally absent (T: APB)
    {"participant": "SCS06", "visit": 1, "run": "all", "muscles": ["ECR", "FCR"]},  # stim art. (T: FDI)
    {"participant": "SCS08", "visit": 1, "run": "all", "muscles": ["ECR", "FCR"]},  # overlapping stim art. (T: APB)
    {"participant": "SCS10", "visit": 1, "run": "all", "muscles": ["APB", "ADM", "FDI"]}, # response totally absent (T: ECR)
    {"participant": "SCS13", "visit": 1, "run": "all", "muscles": ["APB"]},  # response totally absent (T: ECR)
    {"participant": "SCS15", "visit": 1, "run": "all", "muscles": ["ECR"]},
    {"participant": "SCS16", "visit": 1, "run": "all", "muscles": ["APB", "ADM", "FDI"]},  # response totally absent (T: ECR)
]

IO_EXCLUDE = [
    {"participant": "cornptio028", "set_group": 'gpr-001', "set_sequence": "all", "muscles": ["ADM"]},
    {"participant": "cornptio022", "set_group": 'gpr-001', "set_sequence": "all", "muscles": ["ADM"]},
]

CONFIG = {
    'DATA_FOLDER': DATA_FOLDER,
    'BASE_DIR': BASE_DIR,
    'LOAD_SAVED_MODEL': LOAD_SAVED_MODEL,
    'DATA_OPTIONS': DATA_OPTIONS,
    'MCMC_OPTIONS': MCMC_OPTIONS,
    'SAMPLER_OPTIONS': SAMPLER_OPTIONS,
    'MODEL_OPTIONS': MODEL_OPTIONS,
    'NI_EXCLUDE': NI_EXCLUDE,
    'NI_RECOMPUTE_RUNS': NI_RECOMPUTE_RUNS,
    'IO_EXCLUDE': IO_EXCLUDE,
}

# Attempt to import dnc_config.py and overwrite default values
dnc_config_path = Path(__file__).parent / 'dnc_config.py'
if dnc_config_path.exists():
    spec = importlib.util.spec_from_file_location("dnc_config", dnc_config_path)
    dnc_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dnc_config)

    # Overwrite CONFIG with values from dnc_config if they exist
    for key in CONFIG:
        if hasattr(dnc_config, key):
            CONFIG[key] = getattr(dnc_config, key)

    # If needed, add or overwrite individual variables as well
    DATA_FOLDER = CONFIG['DATA_FOLDER']
    BASE_DIR = CONFIG['BASE_DIR']


# # Example dnc_config.py
# # This can be used to set custom folders without modifying this config directly (since it is version controlled)
# from pathlib import Path
# import config
#
# cfg = config.CONFIG
#
# # Custom Data Folder Paths
# DATA_FOLDER = {
#     'noninvasive': Path('/mnt/hdd3/va_data_temp/proc/preproc_tables/2025-04-11_cmct_v0p0p1'),
#     'intraoperative': Path('/home/mcintosh/Cloud/DataPort/2024-01-00_human_scs_working/preproc_hb/np_posterior_2025-04-04'),
# }
#
# BASE_DIR = Path('/mnt/hdd3/va_data_temp/proc/bsp')
#
# print('Changing data folders to:')
# print(DATA_FOLDER)
# print('Changing proc folder to:')
# print(BASE_DIR)
# print('-----------')
