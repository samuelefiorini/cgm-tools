""""[cgm-tools] CGM forecast via LSTM network."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
import numpy as np

__all__ = ['forecast', 'online_forecast', 'create_XY_dataset',
           'create_model']


def online_forecast(test_data, test_labels, model, scaler, ph=18,
                    verbose=False):
    """Recursively predict an input CGM time-series unsing a fitted LSTM.

    This function recursively predicts input CGM time series and
    evaluates 30/60/90 mins (absolute) error.

    Parameters
    -------------------
    test_data : array of float, returned by create_XY_dataset in first position
                of size (1, window_size, 1)
    test_labels : array of float, returned by create_XY_dataset in second
                  position fo size (1,)
    model : keras.model, the (compiled) LSTM to use to forecast returned by
            `create_model`
    scaler : sklearn.preprocessing, the (fitted) preprocessing object used on
             the input data
    ph : number, the prediction horizon. It must be ph > 0
         (default=18, i.e. 90 mins (1.5 hours))
    verbose : bool, print debug messages each 200 iterations (default=False)

    Returns
    -------------------
    errs : dictionary, errors at 30/60/90 mins ('err_18', 'err_12', 'err_6')
    forecast : dictionary, time-series prediction ['ts'], with std_dev
               ['sigma'] and confidence interval ['conf_int'].
               The output has the same length of the input, but the first
               `w_size` elements are set to 0.
    """
    # Argument check
    if ph <= 0:
        raise Exception('ph must be at least 1')

    errs_dict = {'err_18': [], 'err_12': [], 'err_6': []}
    forecast_dict = {'ts': [], 'sigma': [], 'conf_int': []}
    w_size = test_data.shape[1]

    for t in range(test_data.shape[0] - ph):
        if t % 200 == 0 and verbose:
            print("Forecasting t = {}/{}".format(t, test_data.shape[0]))
        _X_ts_next = test_data[t].reshape(1, w_size, 1)

        y_pred = forecast(model, n_steps=ph, test_point=_X_ts_next)

    # Get back to original dimensions
    y_pred = scaler.inverse_transform(y_pred)
    y_future_real = scaler.inverse_transform(test_labels[t:t + ph])
    abs_pred_err = np.abs(y_pred - y_future_real)

    # Save errors
    errs_dict['err_18'].append(abs_pred_err[17])
    errs_dict['err_12'].append(abs_pred_err[11])
    errs_dict['err_6'].append(abs_pred_err[5])

    forecast_dict['ts'] = y_pred

    return errs_dict, forecast_dict


def forecast(model=None, n_steps=1, test_point=None):
    """Forecast n_steps-ahead using the input LSTM.

    Parameters
    -------------------
    model : keras.model, the (fitted) LSTM to use to forecast
    n_steps : number, the prediction horizon (default = 1)
    test_data : array of float, the next test point of shape
                (1, window-size, 1), windows-size is the same
                used in create_XY_dataset

    Returns
    -------------------
    y_pred : array of float, the `n_steps` predicted future values
    """
    if n_steps <= 0:
        raise Exception('n_steps must be at least 1')

    # Init the prediction vector
    y_pred = np.zeros(n_steps)

    # Start with the current test point
    _next_test_point = test_point
    w_size = _next_test_point.shape[1]

    # Iterate on the prediction horizon
    for step in range(n_steps):
        # Forecast one-step-ahead
        y_pred[step] = model.predict(_next_test_point)
        # Shift one-step-ahead the window
        _next_test_point = np.reshape(np.append(_next_test_point[:, 1:, :],
                                                y_pred[step]), (1, w_size, 1))
    return y_pred


def create_XY_dataset(data, window_size=1):
    """Create suitable {X,Y} couples suitable for LSTM training/test.

    This function has been adapted from:
    [http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/]

    Parameters
    -------------------
    data : input time-series data of size (n_samples, 1)
    window_size : number, the window-size

    Returns
    -------------------
    X : array of float, input data of size (n_samples, window_size, 1)
    Y : array of float, output data of size (n_samples,)
    """
    X, Y = [], []
    for i in range(len(data) - window_size):
        tmp = data[i:(i + window_size), 0]
        X.append(tmp)
        Y.append(data[i + window_size, 0])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y


def create_model(n_units=4, input_dim=1):
    """Create a single hidden-layer LSTM network.

    This function creates a single layer LSTM network using Keras.
    This model has an LSTM layer followed by a one-dimensional Dense
    layer. The loss function is 'mean_squared_error' and the optimizer is
    'adam'.

    Parameters
    -------------------
    n_units : number, the number of units of the LSTM layer (default = 4)
    input_dim : number, the dimensionality of the input (default = 1)

    Returns
    -------------------
    model : keras.model, the (compiled) LSTM to use to forecast
    """
    # create and compile the LSTM network
    model = Sequential()
    model.add(LSTM(n_units, input_dim=input_dim))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
