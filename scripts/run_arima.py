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
import os
import warnings; warnings.filterwarnings('ignore')


def main(args):
    """Run ARIMA experiments."""

    ### TODO: deleteme ###
    # List all completed patients
    completed = list(filter(lambda x: x.endswith('.pkl'),
                     os.listdir('/home/samu/projects/glicemie/experiments/cgm-tools/scripts')))
    completed = [x[-3]+'.csv' for x in completed]
    ### TODO: deleteme ###


    # Load full data set from pickle file (see data_wrangler.py)
    dfs_full = pkl.load(open(args.data_folder, 'rb'))

    # Keep only patients with more than `THRESHOLD` days of CGM acquisition
    _threshold = args.threshold
    if _threshold is None:
        _threshold = datetime.timedelta(days=3.5)  # default
    dfs = utils.filter_patients(dfs_full, _threshold)

    # ----------------- TEST ----------------------------- #
    # Experiment parameters
    burn_in = 300  # burn-in samples used to learn the best order via cv
    n_splits = 15
    # burn_in = 144  # burn-in samples used to learn the best order via cv
    # n_splits = 8
    w_size = 36  # Window-size
    ph = 18  # prediction horizon

    # Get patients list
    patients = list(dfs.keys())

    for count, idx in enumerate(patients):
        if idx not in completed:
            print("Evaluating patient: {} ({}/{}) ...".format(idx,
                                                              count,
                                                              len(patients)))
            df = utils.gluco_extract(dfs[idx], return_df=True)

            # Learn the best order via cv
            out = arima.grid_search(df, burn_in=burn_in, n_splits=n_splits,
                                    p_bounds=(1, 4), d_bounds=(1, 2), q_bounds=(1, 4),
                                    ic_score='AIC', return_order_rank=True,
                                    return_final_index=True, verbose=False)
            opt_order, order_rank, final_index = out

            print("Order rank:\n{}".format(order_rank))

            df = df.iloc[burn_in:]  # don't mix-up training/test

            errs = None
            # Try the order from best to worst
            for order in order_rank:
                p, d, q = order
                try:  # perform moving-window arma
                    print('Using ARIMA({}, {}, {}) ...'.format(p, d, q))
                    errs, forecast = arima.moving_window(df, w_size=w_size, ph=ph,
                                                         p=p, d=d, q=q,
                                                         start_params=None,
                                                         verbose=False)
                    print('ARIMA({}, {}, {}) success'.format(p, d, q))
                    break  # greedy beahior: take the first that works
                except Exception as e:
                    print('ARIMA({}, {}, {}) failure'.format(p, d, q))
                    print('arima.moving_window raised the following exception')
                    print(e)

            if errs is not None:
                # Save results reports
                error_summary = utils.forecast_report(errs)
                print(error_summary)
                # dump it into a pkl
                pkl.dump(error_summary, open(idx+'.pkl', 'wb'))

                try:
                    # Plot signal and its fit
                    plotting.cgm(df, forecast['ts'], title='Patient '+idx,
                                 savefig=True)

                    # Plot residuals
                    plotting.residuals(df, forecast['ts'], skip_first=w_size,
                                       skip_last=ph, title='Patient '+idx,
                                       savefig=True)
                except:
                    print("Plotting failed for patient {}".format(idx))
        else:
            print("{} already completed".format(idx))





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
