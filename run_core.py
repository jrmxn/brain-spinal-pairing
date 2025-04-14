import importlib
import inspect
import time
import random
from joblib import Parallel, delayed
import config
import argparse

from src.bsp.core import hbie

parser = argparse.ArgumentParser(description="Run the core script with options.")
parser.add_argument("--n_jobs", type=int, default=int(1), help="Number of parallel jobs to run (default: 1).") # 1 or the prompt for data download won't work
parser.add_argument("--functions_list", nargs='+', default=None, help="List of functions to run. If not provided, all functions starting with 'model' will be used.")
parser.add_argument("--exp_list", nargs='+', default=None, help="List of data to run.")

args = parser.parse_args()

# Load the model module and function list
model_module = importlib.import_module("src.bsp.core.model")
if args.functions_list is None:
    functions_list = [func for func in dir(model_module) if inspect.isfunction(getattr(model_module, func)) and func.startswith('model')]
else:
    functions_list = args.functions_list

exp_list_full = ['ni_target', 'ni', 'io_mcintosh2024_cgtarget', 'io_mcintosh2024_cgtarget_mtarget', 'ni_noexc', 'ni_target_without_b', 'io_mcintosh2024_cgtarget_mtarget_with_b']  # 'ni_noexc'
if args.exp_list is None:
    exp_list = ['ni_target', 'ni']
else:
    exp_list = args.exp_list

n_jobs = args.n_jobs

def run_model(f, e, sleep_range=(0, 15), make_plots=False):
    importlib.reload(config)
    cfg = config.CONFIG
    if ('ni' == e) or ('ni_noexc' == e):
        cfg['DATA_OPTIONS']['type'] = 'noninvasive'
        cfg['DATA_OPTIONS']['es'] = cfg['DATA_OPTIONS']['es']
        cfg['DATA_OPTIONS']['response'] = ['ECR', 'FCR', 'APB', 'ADM', 'FDI']
        cfg['DATA_OPTIONS']['intensities'] = ['supra-sub', 'sub-sub']
        if 'ni_noexc' == e:
            if 'co' in f: return
            cfg['NI_EXCLUDE'] = []
            cfg['DATA_OPTIONS']['es'] = cfg['DATA_OPTIONS']['es'] + 'noexc_'

    elif ('ni_target' == e) or ('ni_target_without_b' == e):
        cfg['DATA_OPTIONS']['type'] = 'noninvasive'
        cfg['DATA_OPTIONS']['es'] = cfg['DATA_OPTIONS']['es']
        cfg['DATA_OPTIONS']['response'] = ['auc_target']
        cfg['DATA_OPTIONS']['intensities'] = ['supra-sub', 'sub-sub']
        if 'ni_target_without_b' == e:
            cfg['MODEL_OPTIONS']['use_b'] = False
            cfg['DATA_OPTIONS']['es'] = cfg['DATA_OPTIONS']['es'] + 'withoutb_'

    elif 'io_mcintosh2024_cgtarget_mfull' == e:
        cfg['DATA_OPTIONS']['type'] = 'intraoperative'
        cfg['DATA_OPTIONS']['es'] = cfg['DATA_OPTIONS']['es'] + 'mcgtarget_'
        cfg['DATA_OPTIONS']['response'] = ['Biceps', 'Triceps', 'ECR', 'FCR', 'APB', 'ADM']
        cfg['DATA_OPTIONS']['intensities'] = ['supra-sub']
        cfg['DATA_OPTIONS']['visit'] = 'mcintosh2024'
        cfg['DATA_OPTIONS']['intraoperative'][0] = 'global_target'  # as in average across partcipants (global) target muscle
        cfg['MODEL_OPTIONS']['use_b'] = False

    elif 'io_mcintosh2024_cgtarget' == e:
        cfg['DATA_OPTIONS']['type'] = 'intraoperative'
        cfg['DATA_OPTIONS']['es'] = cfg['DATA_OPTIONS']['es'] + 'mcgtarget_'
        cfg['DATA_OPTIONS']['response'] = ['ECR', 'FCR', 'APB', 'ADM']
        cfg['DATA_OPTIONS']['intensities'] = ['supra-sub']
        cfg['DATA_OPTIONS']['visit'] = 'mcintosh2024'
        cfg['DATA_OPTIONS']['intraoperative'][0] = 'global_target'  # as in average across partcipants (global) target muscle
        cfg['MODEL_OPTIONS']['use_b'] = False
        cfg['MODEL_OPTIONS']['scale_c_prior'] = 1.0

    elif ('io_mcintosh2024_cgtarget_mtarget' == e) or ('io_mcintosh2024_cgtarget_mtarget_with_b' == e):
        cfg['DATA_OPTIONS']['type'] = 'intraoperative'
        cfg['DATA_OPTIONS']['es'] = cfg['DATA_OPTIONS']['es'] + 'mcgtarget_'
        cfg['DATA_OPTIONS']['visit'] = 'mcintosh2024'
        cfg['DATA_OPTIONS']['response'] = ['auc_target']
        cfg['DATA_OPTIONS']['intensities'] = ['supra-sub']
        cfg['DATA_OPTIONS']['intraoperative'][0] = 'global_target'  # as in average across partcipants (global) target muscle
        cfg['MODEL_OPTIONS']['use_b'] = False
        cfg['MODEL_OPTIONS']['scale_c_prior'] = 1.0
        if 'io_mcintosh2024_cgtarget_mtarget_with_b' == e:
            cfg['MODEL_OPTIONS']['use_b'] = True
            cfg['DATA_OPTIONS']['es'] = cfg['DATA_OPTIONS']['es'] + 'withb_'

    if (cfg['DATA_OPTIONS']['type'] == 'intraoperative') and ('co' in f):
        print('SKIPPING condition model, for intraoperative data!')
        return

    try:
        delay = random.uniform(sleep_range[0], sleep_range[1])
        print(f'Delaying start by {delay:0.2f}s')
        time.sleep(delay)
        hbie.main(cfg, model_version=f)
    except Exception as ex:
        print(f'MODEL CRASHED. Moving on... ({ex})')
        pass


def run_jobs(functions_list, exp_list, n_jobs=1):
    Parallel(n_jobs=n_jobs)(
        delayed(run_model)(f, e)
        for e in exp_list
        for f in functions_list
    )
    return None


if __name__ == '__main__':

    if False:
        # debugging
        print('WARNING - DEBUG LINE IS ON!!!!')
        exp_list, functions_list, n_jobs = ['ni'], ['model00'], 1

    run_jobs(functions_list, exp_list, n_jobs)
