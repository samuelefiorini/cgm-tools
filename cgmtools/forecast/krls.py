""""[cgm-tools] CGM forecast via Kernel Ridge Regression."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import numpy as np


def online_forecast(test_data, test_labels, model, scaler, ph=18,
                    verbose=False):
    """Recursively predict an input CGM time-series unsing a fitted LSTM.

    This function recursively predicts input CGM time series and
    evaluates 30/60/90 mins (absolute) error.

    This function is basically the same as forecast.lstm.online_forecast,
    the only difference is the shape of the data.

    Parameters
    -------------------
    test_data : array of float, returned by create_XY_dataset in first position
                of size (1, window_size, 1)
    test_labels : array of float, returned by create_XY_dataset in second
                  position fo size (1,)
    model : sklearn.kernel_ridge.KernelRidge, class instance
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

    for t in range(test_data.shape[0] - ph):
        if t % 200 == 0 and verbose:
            print("Forecasting t = {}/{}".format(t, test_data.shape[0]))
        _X_ts_next = test_data[t]

        y_pred = forecast(model, n_steps=ph, test_point=_X_ts_next)

        # Get back to original dimensions
        y_pred = scaler.inverse_transform(y_pred)
        y_future_real = scaler.inverse_transform(test_labels[t:t + ph])
        abs_pred_err = np.abs(y_pred - y_future_real)

        # Save errors
        errs_dict['err_18'].append(abs_pred_err[17])
        errs_dict['err_12'].append(abs_pred_err[11])
        errs_dict['err_6'].append(abs_pred_err[5])

        forecast_dict['ts'].append(y_pred[0])

    # Return numpy.array
    forecast_dict['ts'] = np.array(forecast_dict['ts'])
    return errs_dict, forecast_dict


def forecast(model=None, n_steps=1, test_point=None):
    """Forecast n_steps-ahead using the input Kernel Ridge regressor.

    Parameters
    -------------------
    model : sklearn.kernel_ridge.KernelRidge, the (fitted) Kernel Ridge
            regressor to use to forecast
    n_steps : number, the prediction horizon (default = 1)
    test_data : array of float, the next test point of shape
                (window-size,), windows-size is the same
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
    w_size = _next_test_point.shape[0]

    # Iterate on the prediction horizon
    for step in range(n_steps):
        # Forecast one-step-ahead
        y_pred[step] = model.predict(_next_test_point)
        # Shift one-step-ahead the window
        _next_test_point = np.reshape(np.append(_next_test_point[1:],
                                                y_pred[step]), (w_size,))
    return y_pred
