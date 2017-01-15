#!/usr/bin/env python3
"""[cgm-tools] Run ARIMA experiments.

Run ARIMA experiments:
    - load the data
    - filter the patients with less than `THRESHOLD` CGM days
    - identify the best (p, d, q) ARIMA order via cross-validation
    - fit moving-window ARIMA model with the best order
        * evaluate 30/60/90' error (mean absolute and root mean square)
        * evaluate 1-step-ahead residuals and Durbin Watson statistic
    - dump the results in a pickle file
"""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import argparse
from cgmtools import utils
from cgmtools import plotting
from cgmtools.forecast import arima
import datetime
import pickle as pkl
import warnings


def main(args):
    """Run ARIMA experiments."""

    # Load full data set from pickle file (see data_wrangler.py)
    dfs_full = pkl.load(open(args.data_folder, 'rb'))

    # Keep only patients with more than `THRESHOLD` days of CGM acquisition
    _threshold = args.threshold
    if _threshold is None:
        _threshold = datetime.timedelta(days=3.5)  # default
    dfs = utils.filter_patients(dfs_full, _threshold)

    # ----------------- TEST ----------------------------- #
    # select patient
    idx = list(dfs.keys())[100]
    df = utils.gluco_extract(dfs[idx], return_df=True)


    # learn the best order
    out = arima.grid_search(df, burn_in=144, n_splits=8, p_bounds=(1, 4),
                            d_bounds=(1, 2), q_bounds=(1, 4), ic_score='AIC',
                            return_order_rank=True, return_final_index=True,
                            verbose=True)
    # out = arima.grid_search(df, burn_in=300, n_splits=15, p_bounds=(1, 4),
    #                         d_bounds=(1, 2), q_bounds=(1, 4), ic_score='AIC',
    #                         return_order_rank=True, return_final_index=True,
    #                         verbose=True)

    opt_order, order_rank, final_index = out

    #opt_order = (2, 1, 1)
    print("Order rank:\n{}".format(order_rank))

    # Window-size and prediction horizon
    w_size = 30
    ph = 18

    # Try the order from best to worst
    for order in order_rank:
        p, d, q = order
        try:
            # perform moving-window arma
            errs, forecast = arima.moving_window(df, w_size=w_size, ph=ph,
                                                 p=p, d=d, q=q,
                                                 start_params=None,
                                                 verbose=True)
            print('ARIMA(%d, %d, %d) done' % p, d, q)
            break  # greedy beahior: take the first that works
        except Exception as e:
            warnings.warn('arima.moving_window raised %s' % e)
            warnings.warn('ARIMA(%d, %d, %d) failed' % p, d, q)

    print(forecast['ts'])

    # plot results
    import numpy as np
    import matplotlib; matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    plotting.cgm(df.index, df.as_matrix(), title='Patient ')
    plt.plot(df.index, forecast['ts'], linestyle='dashed', label='forecast')
    plt.legend(bbox_to_anchor=(1.2, 1.0))
    MAE_6 = np.mean(errs['err_6'])
    MAE_12 = np.mean(errs['err_12'])
    MAE_18 = np.mean(errs['err_18'])
    RMSE_6 = np.linalg.norm(errs['err_6']) / np.sqrt(len(errs['err_6']))
    RMSE_12 = np.linalg.norm(errs['err_12']) / np.sqrt(len(errs['err_12']))
    RMSE_18 = np.linalg.norm(errs['err_18']) / np.sqrt(len(errs['err_18']))
    print("MAE (30') = {:2.3f}\t|\tMAE (60') = {:2.3f}\t|\tMAE (90') = {:2.3f}".format(MAE_6, MAE_12, MAE_18))
    print("RMSE (30') = {:2.3f}\t|\tRMSE (60') = {:2.3f}\t|\tRMSE (90') = {:2.3f}".format(RMSE_6, RMSE_12, RMSE_18))
    plt.savefig('fits.png')


    # In[14]:

    residuals = df.as_matrix()[w_size:-ph].ravel() - forecast['ts'][w_size:-ph]
    fig = plt.figure(figsize=(12, 4))
    plt.plot(df.index[w_size:-ph], residuals)
    plt.title('Durbin-Watson: {:.3f}'.format(sm.stats.durbin_watson(residuals)))
    plt.savefig('res.png')



######################################################################


def parsing():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='[cgm-tools] ARIMA runner')
    parser.add_argument("data_folder", help='The folder that contains the '
                        'input data as pkl file (see data_wrangler.py)')
    parser.add_argument('--threshold', metavar='threshold', action='store',
                        help='exclude patients with less than threshold days '
                        'of CGM data', default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    ARGS = parsing()
    main(ARGS)
