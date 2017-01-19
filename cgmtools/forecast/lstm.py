""""[cgm-tools] CGM forecast via LSTM network."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import numpy as np

__all__ = ['forecast']


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
