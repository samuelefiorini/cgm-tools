#!/usr/bin/env python3
"""[cgm-tools] Run LSTM experiments.

Run LSTM experiments:
    - load the data
    - filter the patients with less than `THRESHOLD` CGM days
    - identify the best number of units in the LSTM hidden-layer
      via cross-validation (TODO, now it's fixed as 4)
    - fit moving-window LSTM model with the best number of units
        * evaluate 30/60/90' error (mean absolute and root mean square)
        * evaluate 1-step-ahead residuals and Durbin Watson statistic
    - dump the results in a pickle file
"""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################
from cgmtools import utils
from cgmtools.forecast import lstm
import datetime
from keras.wrappers.scikit_learn import KerasRegressor
import pickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import time
###############################################################################

# Load full data set from pickle file (see data_wrangler.py)
dfs_full = pkl.load(open('../../data/dfs_py3.pkl', 'rb'))

# Keep only patients with more than `THRESHOLD` days of CGM acquisition
_threshold = datetime.timedelta(days=3.5)  # default
dfs = utils.filter_patients(dfs_full, _threshold)

burn_in = 300  # burn-in samples used to learn the best order via cv
# n_splits = 15
ph = 18  # prediction horizon
w_size = 36

# Get patients list
patients = list(dfs.keys())

# Iterate on the patients
for count, idx in enumerate(patients):
    print("Evaluating patient {}/{}".format(count, len(patients)))
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
                                     verbose=1,
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
    pkl.dump(error_summary, open(idx+'.pkl', 'wb'))
    model.save(idx+'_model_.h5')

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
        plt.savefig(idx+'.png')
    except:
        print('Plotting failed')
