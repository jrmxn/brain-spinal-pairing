from src.bsp.analysis import check_posteriors
from config_analysis import CONFIG_ANALYSIS
from config import BASE_DIR, DATA_OPTIONS
from pathlib import Path
import re
from joblib import Parallel, delayed
import time
import psutil
max_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
v = CONFIG_ANALYSIS['version']


def custom_sort_key(model_name):
    """
    """
    noninvasive_priority = (v + '_' + 'noninvasive' in model_name) or ('intraoperative' in model_name)
    auc_target_priority = 'auc_target' in model_name
    model00_priority = 'model00_' in model_name


    # Sorting logic: Higher priority should come first, so we use negative values to sort descending
    return (-noninvasive_priority, -auc_target_priority, -model00_priority)

if v == "":
    v = DATA_OPTIONS['es'].split('_')[0]
list_paths = Path(BASE_DIR).rglob(f'**/{v}_*/')
list_paths = [lp for lp in list_paths if 'KEEP' not in str(lp)]
list_o_model = [lp.stem for lp in list_paths]
pattern = re.compile(r'_r\d{5}_')
list_o_model = [o_model for o_model in list_o_model if not pattern.search(o_model)]
list_o_model = sorted(list_o_model, key=custom_sort_key, reverse=False)
list_rl_model = CONFIG_ANALYSIS['rl_model']


def run_function(o_model, rl_model, overwrite, delay=0):
    time.sleep(delay)
    print('------------')
    print(o_model)
    try:
        check_posteriors.main(o_model=o_model, rl_model=rl_model, overwrite=overwrite)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    print('------------')


def run_jobs(n_jobs=3, skip_parallel=False):
    overwrite = False #  CONFIG_ANALYSIS['overwrite']
    step = 1
    vec_delay = list(range(0, step*len(list_o_model), step))
    if skip_parallel:
        for o_model in list_o_model:
            for rl_model in list_rl_model:
                run_function(o_model, rl_model, overwrite)
    else:
        Parallel(n_jobs=n_jobs)(
            delayed(run_function)
            (list_o_model[ix], list_rl_model[ixrl], overwrite, delay=0)
            for ixrl in range(len(list_rl_model))
            for ix in range(len(list_o_model))
        )


n_jobs = int(max_memory_gb/25)
if n_jobs < 1: n_jobs = int(1)
n_jobs = int(1)
run_jobs(n_jobs=n_jobs, skip_parallel=False)

