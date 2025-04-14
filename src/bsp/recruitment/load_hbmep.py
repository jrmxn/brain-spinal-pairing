from pathlib import Path
import pickle
from hbmep.model.utils import Site as site

source_path = Path("dnc_hbmep_TMS/", "inference.pkl")
with open(source_path, "rb") as f:
    model_hbmep, mcmc_hbmep, ps_hbmep = pickle.load(f)
print(ps_hbmep[site.a])