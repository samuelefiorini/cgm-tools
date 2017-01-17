"""KF experiments development."""
from cgmtools import utils
from cgmtools import plotting
from cgmtools.forecast import kf
import datetime
import numpy as np
import pickle as pkl

###############################################################################

# Load full data set from pickle file (see data_wrangler.py)
dfs_full = pkl.load(open('data/dfs_py3.pkl', 'rb'))

# Keep only patients with more than `THRESHOLD` days of CGM acquisition
_threshold = datetime.timedelta(days=3.5)  # default
dfs = utils.filter_patients(dfs_full, _threshold)

burn_in = 300  # burn-in samples used to learn the best order via cv
n_splits = 15
ph = 18  # prediction horizon

# Get patients list
patients = list(dfs.keys())

for idx in patients:
    df = utils.gluco_extract(dfs[idx], return_df=True)

    # Learn the best order via cv
    lambda2_range = np.logspace(-12, -4, 10)
    sigma2_range = np.linspace(1, 40, 10)
    out = kf.grid_search(df, lambda2_range, sigma2_range, burn_in=burn_in,
                         n_splits=15, return_mean_vld_error=True,
                         verbose=False)
    lambda2, sigma2, mse = out
    print("[{}]:\tBest lambda {:2.12f}, sigma {:2.0f}".format(idx, lambda2,
                                                              sigma2))

    
