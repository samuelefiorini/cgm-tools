#!/usr/bin/env python
# This program uses a master slave approach to consume a queue
# of elaborations

import time
import datetime

import numpy as np

from mpi4py import MPI
from collections import deque

from cgmtools import utils
from cgmtools.forecast import lstm
# import datetime
import pickle as pkl
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
# import time
import os


# constants to use as tags in communications
DO_WORK = 100
EXIT = 200

# VERBOSITY
VERBOSITY = 1


def master(inputs):
    """
    dispatch the work among processors.
    queue is a list of input
    return a list of timings
    """
    procs_ = comm.Get_size()
    queue = deque(inputs)  # deque to enable popping from the left

    timings = np.zeros(procs_)
    count = 0
    # seed the slaves by sending work to each processor
    for rank in range(1, procs_):
        input_file = queue.popleft()
        comm.send(input_file, dest=rank, tag=DO_WORK)

    # loop until there's no more work to do. If queue is empty skips the loop.
    while queue:
        input_file = queue.popleft()
        # receive result from slave
        status = MPI.Status()
        elapsed_time = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        timings[status.source] += elapsed_time
        count += 1
        # send to the same slave new work
        comm.send(input_file, dest=status.source, tag=DO_WORK)

    # there's no more work to do, so receive all the results from the slaves
    for rank in range(1, procs_):
        status = MPI.Status()
        elapsed_time = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        timings[status.source] += elapsed_time
        count += 1

    # tell all the slaves to exit by sending an empty message with the EXIT_TAG
    for rank in range(1, procs_):
        comm.send(0, dest=rank, tag=EXIT)

    return count, timings


def slave():
    """
    slave will spawn the work processes
    """
    while True:
        status_ = MPI.Status()
        input_name = comm.recv(source=0, tag=MPI.ANY_TAG, status=status_)
        # check the tag of the received message
        if status_.tag == EXIT:
            return
        # do the work
        result = worker(input_name)  # result contains the times required for execution
        comm.send(result, dest=0, tag=0)


def worker(idx):
    """
    spawn the work process
    """
    import os
    my_rank_ = comm.Get_rank()

    t1_ = time.time()
    burn_in = 300  # burn-in samples used to learn the best order via cv
    w_size = 36

    # print("Evaluating patient {}".format(idx))
    # Train/test split
    df = utils.gluco_extract(dfs[idx], return_df=True)
    train_df0 = df.iloc[:burn_in]
    test_df0 = df.iloc[burn_in:]

    # preprocess the dataset
    # BEWARE! Do not use the trainig set to learn the scaling parameters
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_df0)
    test_data = scaler.transform(test_df0)

    # Create LSTM suitable {X, Y} dataset
    X_tr, Y_tr = lstm.create_XY_dataset(train_data, window_size=w_size)
    X_ts, Y_ts = lstm.create_XY_dataset(test_data, window_size=w_size)

    # Create LSTM model
    # model = lstm.create_model(n_units=4)

    # Create cross-validated LSTM model
    param_grid = {'n_units': [4, 8, 16]}
    keras_regressor = KerasRegressor(build_fn=lstm.create_model,
                                     batch_size=1,
                                     verbose=0,
                                     nb_epoch=50)
    model = GridSearchCV(keras_regressor, param_grid=param_grid)

    tic = time.time()
    # Fit the model
    # model.fit(X_tr, Y_tr, nb_epoch=50, batch_size=1, verbose=1)
    model.fit(X_tr, Y_tr)
    print("Fitting time: {} seconds".format(time.time() - tic))

    # Predict the ph and save the errors
    tic = time.time()
    errs, forecast = lstm.online_forecast(X_ts, Y_ts, model, scaler, ph=18,
                                          verbose=True)
    print("Predicting time: {} seconds".format(time.time() - tic))
    error_summary = utils.forecast_report(errs)
    print(error_summary)
    pkl.dump(error_summary, open(os.path.join(ROOT, 'results', idx+'.pkl', 'wb')))
    model.save(os.path.join(ROOT, 'results', idx+'_model_.h5'))

    # -- Plotting -- #
    try:
        import statsmodels.api as sm
        import numpy as np
        import matplotlib; matplotlib.use('agg')
        import matplotlib.pyplot as plt
        Y_pred_tr = model.predict(X_tr)
        Y_pred_ts = model.predict(X_ts)  # maybe its just forecast['ts']
        Y_pred_tr_plot = scaler.inverse_transform(Y_pred_tr)
        Y_pred_ts_plot = scaler.inverse_transform(Y_pred_ts)
        plt.figure(figsize=(10, 6), dpi=300)
        plt.subplot(211)
        plt.plot(df.index, df.values, label='real cgm')
        plt.plot(df.index[w_size:burn_in], Y_pred_tr_plot.ravel(), '--',
                 label='y_tr')
        plt.plot(df.index[burn_in+w_size:], Y_pred_ts_plot.ravel(), '--',
                 label='y_tr')
        plt.legend()

        residuals = Y_pred_ts_plot.ravel() - df.values[burn_in+w_size:].ravel()
        mae = np.mean(residuals)
        rmse = np.sqrt(np.mean(residuals ** 2))
        DW = sm.stats.durbin_watson(residuals)

        plt.subplot(212)
        plt.plot(df.index[burn_in:-w_size], residuals)
        plt.title("MAE {:2.5f} | RMSE {:2.5f} | DW {:2.5f}".format(mae, rmse, DW))
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT, 'results', idx+'.png'))
    except:
        print('Plotting failed')

    # Do the work
    # time.sleep(2)

    t2_ = time.time()

    if VERBOSITY:
        print(' ---> processor %s has calculated for %s' % (my_rank_, t2_-t1_))
    return t2_ - t1_


###############################################################################
#                                                                             #
# --------------------------------- main ------------------------------------ #
#                                                                             #
###############################################################################


if __name__ == '__main__':
    """Run with
    ```mpirun -np 4 python master_slave.py```
    """
    # import sys

    comm = MPI.COMM_WORLD
    procs = comm.Get_size()
    my_rank = comm.Get_rank()

    t1 = time.time()
    # timing
    if my_rank == 0:
        print('*'*80)
        print('%s -- Calculation started' % datetime.datetime.utcnow())
        print('*'*80)
        print(procs)

    # -------------- Load data ---------------------- #
    ROOT = '/home/jdoe/shared/glicemie/LSTM'
    # Load full data set from pickle file (see data_wrangler.py)
    dfs_full = pkl.load(open(os.path.join(ROOT, 'dfs_py3.pkl'), 'rb'))

    # Keep only patients with more than `THRESHOLD` days of CGM acquisition
    _threshold = datetime.timedelta(days=3.5)  # default
    dfs = utils.filter_patients(dfs_full, _threshold)


    # n_splits = 15
    # ph = 18  # prediction horizon


    # Get patients list
    patients = list(dfs.keys())
    # --------------------------------------------------- #

    elapsed_times = None
    if my_rank == 0:
        # get the input list
        # input_files = get_input_names(sys.argv[1])
        # input_files = ['arg' + str(i) for i in range(10)]
        input_files = patients
        print('Number of elaborations: %d' % len(input_files))
        number_of_elaborations, elapsed_times = master(input_files)
        print('Number of elaborations: %d' % number_of_elaborations)
    else:
        slave()

    comm.Barrier()

    if my_rank == 0:
        t2 = time.time()

        for i, time in enumerate(elapsed_times):
            print(i, time)

        print('*'*80)
        print('%s -- Calculation ended after %s seconds' % (datetime.datetime.utcnow(), t2-t1))
print('*'*80)
