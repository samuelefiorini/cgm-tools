"""KF experiments development."""
from cgmtools import utils
from cgmtools import plotting
from cgmtools.forecast import kf
import datetime
import numpy as np
import os
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

# State-space model:
# transition matrix (double integration model)
F = np.array([[2, -1], [1, 0]])
# measures matrix
H = np.array([1, 0])

# Get patients list
patients = list(dfs.keys())

for idx in patients:
    df = utils.gluco_extract(dfs[idx], return_df=True)

    # Learn the best order via cv
    # lambda2_range = np.logspace(-12, -4, 10)
    lambda2_range = np.logspace(-12, -4, 3)
    sigma2_range = np.linspace(1, 40, 3)
    # sigma2_range = np.linspace(1, 40, 10)
    out = kf.grid_search(df, lambda2_range, sigma2_range, burn_in=burn_in,
                         n_splits=15, F=F, H=H,
                         return_mean_vld_error=True,
                         return_initial_state_mean=True,
                         return_initial_state_covariance=True,
                         verbose=False)
    lambda2, sigma2, mse, X0, P0 = out
    print("[{}]:\tBest lambda {:2.12f}, sigma {:2.0f}".format(idx, lambda2,
                                                              sigma2))
    Q = np.array([[lambda2, 0], [0, 0]])  # transition_covariance
    R = sigma2  # observation (co)variance

    df = df.iloc[burn_in:]  # don't mix-up training/test

    _kf = kf.cgmkalmanfilter(F=F, Q=Q, R=R, X0=X0, P0=P0)
    errs, forecast = kf.online_forecast(df, _kf, H, ph=18, lambda2=lambda2,
                                        sigma2=sigma2, verbose=True)
    # Save results reports
    error_summary = utils.forecast_report(errs)
    print(error_summary)
    import matplotlib.pyplot as plt
    plotting.cgm(df, forecast['ts'], title='Patient '+idx,
                 savefig=False)
    plotting.residuals(df, forecast['ts'], skip_first=burn_in,
                       skip_last=ph, title='Patient '+idx,
                       savefig=False)
    plt.show()
    break

    # # dump it into a pkl
    # pkl.dump(error_summary, open(os.path.join('KFresults', idx+'.pkl'), 'wb'))
    #
    # try:
    #     # Plot signal and its fit
    #     plotting.cgm(df, forecast['ts'], title='Patient '+idx,
    #                  savefig=True)
    #
    #     # Plot residuals
    #     plotting.residuals(df, forecast['ts'], skip_first=burn_in,
    #                        skip_last=ph, title='Patient '+idx,
    #                        savefig=True)
    # except:
    #     print("Plotting failed for patient {}".format(idx))
