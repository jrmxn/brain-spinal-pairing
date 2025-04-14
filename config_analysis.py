from pathlib import Path
import importlib.util

CONFIG_ANALYSIS = {
    'version': 'vxx',
    'overwrite': True, # only matters if from run_analysis.py
    'show': False,  #
    'skip': True,
    'rl_model': [''],
    'compute_full_pp': True,
    'ix_rl_model': None,
    'ix_model': None,  #
    'o_model': [
'vxx_placeholder',
],
}

# Attempt to import dnc_config.py and overwrite default values
dnc_config_path = Path(__file__).parent / 'dnc_config_analysis.py'
if dnc_config_path.exists():
    spec = importlib.util.spec_from_file_location("dnc_config_analysis", dnc_config_path)
    dnc_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dnc_config)
    CONFIG_ANALYSIS_DNC = getattr(dnc_config, 'CONFIG_ANALYSIS')
    # Overwrite CONFIG with values from dnc_config if they exist
    for key in CONFIG_ANALYSIS_DNC:
        CONFIG_ANALYSIS[key] = CONFIG_ANALYSIS_DNC[key]


# Example dnc_config_analysis.py:
# CONFIG_ANALYSIS = {
#     'version': 'v06',
#     'overwrite': True,
#     'show': True,  #
#     'skip': False,
#     'rl_model': [''],
#     'compute_full_pp': True,
#     'ix_model': 0,
#     'ix_rl_model': 0,
#     'o_model': [
#         'vxx_noninvasive_model00_auc_target_n31_aaaaaaaaa',
#     ],
# }
