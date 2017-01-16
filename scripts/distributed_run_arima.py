"""[cgm-tools] Distributed (master-slave) arima fitting tool."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import argparse
from cgmtools import utils
from cgmtools import plotting
from cgmtools.forecast import arima
from collections import deque
import datetime
from mpi4py import MPI
# import os
import pickle as pkl
import warnings; warnings.filterwarnings('ignore')
# import shutil

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NAME = MPI.Get_processor_name()


# MAX_RESUBMISSIONS = 2
# constants to use as tags in communications
DO_WORK = 100
EXIT = 200

# VERBOSITY
# VERBOSITY = 1

# Experiment parameters
burn_in = 300  # burn-in samples used to learn the best order via cv
n_splits = 15
# burn_in = 144  # burn-in samples used to learn the best order via cv
# n_splits = 8
w_size = 36  # Window-size
ph = 18  # prediction horizon


def master(dfs):
    """Distribute ARIMA fitting with mpi4py."""
    # RUN PIPELINES
    n_procs = COMM.Get_size()
    print(NAME + ": start running slaves", n_procs, NAME)
    # queue = deque(list(enumerate(pipes)))
    patients_queue = deque(list(dfs.keys()))

    # Dictionary to save the output for each patient
    patients_out = dict()
    count = 0
    n_jobs = len(patients_queue)

    # seed the slaves by sending work to each processor
    for rankk in range(1, min(n_procs, n_jobs)):
        patient_tuple = patients_queue.popleft()  # TODO it's hust a key
        COMM.send(patient_tuple, dest=rankk, tag=DO_WORK)
        print(NAME + ": send to rank", rankk)

    # loop until there's no more work to do. If queue is empty skips the loop.
    while patients_queue:
        # receive result from slave
        status = MPI.Status()
        patient_id, error_summary, forecast = COMM.recv(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        patients_out[patient_id] = {'summary': error_summary,
                                    'forecast': forecast}
        count += 1
        # --- Save the results --- #
        error_summary = patients_out[patient_id]['summary']
        forecast = patients_out[patient_id]['forecast']
        # Dump patient summary into a pkl
        pkl.dump(error_summary, open(patient_id+'.pkl', 'wb'))
        # Plot signal and its fit
        # dfs and patient_out share the same indexes
        plotting.cgm(dfs[patient_id], forecast['ts'],
                     title='Patient '+patient_id, savefig=True)
        # Plot residuals
        plotting.residuals(dfs[patient_id], forecast['ts'], skip_first=w_size,
                           skip_last=ph, title='Patient '+patient_id,
                           savefig=True)
        # -- submit again -- #
        patient_tuple = patients_queue.popleft()
        # send to the same slave new work
        COMM.send(patient_tuple, dest=status.source, tag=DO_WORK)

    # there's no more work to do, so receive all the results from the slaves
    for rankk in range(1, min(n_procs, n_jobs)):
        # print(NAME + ": master - waiting from", rankk)
        status = MPI.Status()
        patient_id, error_summary, forecast = COMM.recv(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        patients_out[patient_id] = {'summary': error_summary,
                                    'forecast': forecast}
        count += 1

    # tell all the slaves to exit by sending an empty message with the EXIT_TAG
    for rankk in range(1, n_procs):
        # print(NAME + ": master - killing", rankk)
        COMM.send(0, dest=rankk, tag=EXIT)

    # print(NAME + ": terminating master")
    return patients_out


def _worker(df):
    # Learn the best order via cv
    out = arima.grid_search(df, burn_in=burn_in, n_splits=n_splits,
                            p_bounds=(1, 4), d_bounds=(1, 2), q_bounds=(1, 4),
                            ic_score='AIC', return_order_rank=True,
                            return_final_index=True, verbose=False)
    opt_order, order_rank, final_index = out

    print("Order rank:\n{}".format(order_rank))

    df = df.iloc[burn_in:]  # don't mix-up training/test

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

    # Save results reports
    error_summary = utils.forecast_report(errs)

    return error_summary, forecast


def slave(dfs):
    """Fit the ARIMA model."""
    try:
        while True:
            status_ = MPI.Status()
            idx = COMM.recv(source=0, tag=MPI.ANY_TAG, status=status_)
            # check the tag of the received message
            if status_.tag == EXIT:
                return
            # do the work
            print(NAME + ": slave received", RANK, idx)
            df = utils.gluco_extract(dfs[idx], return_df=True)
            try:
                out = _worker(df)
            except:
                out = (None, None, None)  # fit failed for current patient
            COMM.send((idx, out[0], out[1]), dest=0, tag=0)

    except StandardError as exc:
        print("Quitting ... TB:", str(exc))


def main(args):
    """Run ARIMA experiments on multiple machines."""
    # TODO: everybody loads the data here
    # Load full data set from pickle file (see data_wrangler.py)
    dfs_full = pkl.load(open(args.data_folder, 'rb'))

    # Keep only patients with more than `THRESHOLD` days of CGM acquisition
    _threshold = args.threshold
    if _threshold is None:
        _threshold = datetime.timedelta(days=3.5)  # default
    dfs = utils.filter_patients(dfs_full, _threshold)

    if RANK == 0:
        patients_out = master(dfs)
    else:
        slave(dfs)

    # if IS_MPI_JOB:
    # Wait for all jobs to end
    COMM.barrier()

    if RANK == 0:
        pkl.dump(patients_out, open('full_output.pkl', 'wb'))

        # # Save results reports
        # for idx in patients_out.keys():
        #
        #     error_summary = patients_out[idx]['summary']
        #     forecast = patients_out[idx]['forecast']
        #
        #     # Dump patient summary into a pkl
        #     pkl.dump(error_summary, open(idx+'.pkl', 'wb'))
        #
        #     # Plot signal and its fit
        #     # dfs and patient_out share the same indexes
        #     plotting.cgm(dfs[idx], forecast['ts'], title='Patient '+idx,
        #                  savefig=True)
        #
        #     # Plot residuals
        #     plotting.residuals(dfs[idx], forecast['ts'], skip_first=w_size,
        #                        skip_last=ph, title='Patient '+idx,
        #                        savefig=True)

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
