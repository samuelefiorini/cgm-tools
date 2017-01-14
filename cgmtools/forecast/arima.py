""""[cgm-tools] CGM forecast via ARIMA model."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import warnings

__all__ = ['moving_window', 'grid_search']


def moving_window(df, w_size=30, ph=18, p=2, d=1, q=1,
                  start_params=None, verbose=False):
    """Fit a moving-window ARIMA model with fixed window size.

    This function tries to fit a moving-window AIRMA model on input
    time-series. In case of failure, due to numpy.linalg.linalg.LinAlgError,
    the function returns NaNs.

    Parameters
    -------------------
    df : DataFrame, the output returned by gluco_extract(..., return_df=True)
    w_size : number, the window size (default=30, i.e. 150 mins (2.5 hours))
    ph : number, the prediction horizon. It must be w_size > ph
         (default=18, i.e. 90 mins (1.5 hours))
    p : number, AR order (default=2)
    d : number, I order (default=1)
    q : numnber, MA order (default=1)
    start_params : array of length p + q, the starting parameters for the
                   ARIMA model (default=None)
    verbose : bool, print debug messages each 100 iterations (default=False)

    Returns
    -------------------
    errs : dictionary, errors at 30/60/90 mins ('err_18', 'err_12', 'err_6':)
    forecast : dictionary, time-series prediction ['ts'], with std_dev
               ['sigma'] and confidence interval ['conf_int'].
               The output has the same length of the input, but the first
               `w_size` elements are set to 0.
    """
    # Argument check
    if w_size < ph:
        raise NotImplementedError('The length of the window size %d should be '
                                  'larger than the prediction horizon '
                                  '%d' % (w_size, ph))
    n_samples = df.shape[0]

    # Absolute prediction error at 30/60/90 minutes
    errs = {'err_18': [], 'err_12': [], 'err_6': []}
    # 1 step-ahead predictions
    forecast = {'ts': [0] * w_size, 'sigma': [], 'conf_int': []}

    # Move the window across the signal
    for w_start in range(n_samples - (w_size + ph - 1)):
        w_end = w_start + w_size
        y = df.iloc[w_start:w_end]
        # BEWARE: y is a time-indexed pandas DataFrame

        # Fit the model and forecast the next ph steps
        try:
            model = sm.tsa.ARIMA(y, (p, d, q)).fit(trend='nc', #method='css',
                                                   start_params=start_params,
                                                   solver='cg', maxiter=500,
                                                   disp=0)
            y_pred, std_err, conf_int = model.forecast(ph)

            # Update the starting parameters for the next iter (warm restart)
            start_params = model.params.copy()
        except np.linalg.linalg.LinAlgError as e:
            # warnings.warn("CRITICAL: %s" % e)
            print("*************************************")
            print("CRITICAL: %s" % e)
            print(start_params)
            print("*************************************")
            return np.nan, np.nan

        if (w_end + ph) < n_samples:
            # Save the 1-step ahead prediction (for plotting reasons)
            forecast['ts'].append(y_pred[0])
            forecast['sigma'].append(std_err[0])
            forecast['conf_int'].append(conf_int[0])

            # Evaluate the errors
            y_future_real = df.iloc[w_end:w_end + ph].values.ravel()
            abs_pred_err = np.abs(y_pred - y_future_real)

            # Save errors
            errs['err_18'].append(abs_pred_err[17])
            errs['err_12'].append(abs_pred_err[11])
            errs['err_6'].append(abs_pred_err[5])

            if (w_start % 200) == 0 and verbose:
                print("[:{}]\nErrors: 30' = {:2.3f}\t|\t60' = "
                      "{:2.3f}\t|\t90' = {:2.3f}".format(w_end,
                                                         errs['err_6'][-1],
                                                         errs['err_12'][-1],
                                                         errs['err_18'][-1]))
                print(model.params)
        else:
            forecast['ts'].extend(y_pred)
            forecast['sigma'].extend(std_err)
            forecast['conf_int'].extend([_ for _ in conf_int])

    # Return numpy.array
    forecast['ts'] = np.array(forecast['ts'])
    forecast['sigma'] = np.array(forecast['sigma'])

    print(w_start)
    print(w_end)
    return errs, forecast


def grid_search(df, burn_in=300, n_splits=15, p_bounds=(2, 8),
                d_bounds=(1, 2), q_bounds=(2, 4), ic_score='AIC',
                return_final_index=False, verbose=False):
    """Find the best ARIMA parameters via grid search cross-validation.

    This function perform a grid search of the optimal (p, d, q)
    parameters of the ARIMAsklearn.model_selection.TimeSeriesSplit
    on input data. The index to optimize can be either AIC or BIC.

    Parameters
    -------------------
    df : DataFrame, the output returned by gluco_extract(..., return_df=True)
    burn_in : number, the number of samples at the beginning of the time-series
              that should be splitted to perform grid search (default = 300)
    n_splits : number, the number of splits of the time-series cross-validation
               schema (default=15). Your prediction horizon will be
               `floor(n_samples / (n_splits + 1))`
    p_bounds : tuple, the AR parameters range organized as (min_p, max_p)
               (default = (4, 8))
    d_bounds : tuple, the I parameters range organized as (min_d, max_d)
               (default = (1, 2))
    q_bounds : tuple, the MA parameters range organized as (min_q, max_q)
               (default = (2, 4))
    ic_score : str, this can be either 'AIC' (default) or 'BIC'
    return_final_index : bool, return the final index as second argument
                         (default=False)
    verbose : bool, print debug messages (default=False)

    Returns
    -------------------
    optimal_order : tuple, the obtained optimal order (p, d, q) for the ARIMA
                    model
    final_index : array_like, the index obtained by the sum of the
                  cross-validation out-of-samples (validation) prediction with
                  the chosen information criteria
    """
    n_samples = df.shape[0]

    # Argument check
    if n_samples < burn_in:
        raise Exception('The number of burn in samples %d should be '
                        'smaller than the total number of samples '
                        '%d' % (burn_in, n_samples))
    if ic_score not in ['AIC', 'BIC']:
        warnings.warn('Information criterion %s not understood, '
                      'using default.' % ic_score)
        ic_score = 'AIC'

    # Isolate the burn in samples
    time_series = df.iloc[:burn_in]

    # Get the parameter bounds & ranges
    min_p, max_p = p_bounds
    min_d, max_d = d_bounds
    min_q, max_q = q_bounds
    p_range = np.arange(min_p, max_p + 1)
    d_range = np.arange(min_d, max_d + 1)
    q_range = np.arange(min_q, max_q + 1)

    # Parameter grid definition
    param_grid = ParameterGrid({'p': p_range,  # AR order
                                'd': d_range,  # I order
                                'q': q_range})  # MA order

    # Time-series cross validation split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize the cross-validation error tensor of size
    # (len(p_range), len(d_range), len(q_range))
    mean_vld_error = np.zeros((len(p_range), len(d_range), len(q_range)))
    std_vld_error = np.zeros_like(mean_vld_error)

    # Initialize the information criteria tensor of size
    # (len(p_range), len(d_range), len(q_range))
    mean_ic_score = np.zeros_like(mean_vld_error)
    std_ic_score = np.zeros_like(mean_ic_score)

    # Iterate trough the model parameters (p, d, q)
    for param in param_grid:
        if verbose: print('trying params {} ...'.format(param))
        p, d, q = param['p'], param['d'], param['q']
        order = (p, d, q)
        start_params = None  # use warm restart

        # i, j, k index will be used to access the mean_vld_error tensor
        i, j, k = p - min_p, d - min_d, q - min_q

        # Init the vld_error vector for the current order
        vld_error = np.zeros(n_splits)
        ic_score = np.zeros_like(vld_error)

        # Iterate through the CV splits
        for cv_count, (tr_index, vld_index) in enumerate(tscv.split(time_series)):
            try:
                y_tr, y_vld = time_series.iloc[tr_index], time_series.iloc[vld_index]

                # Fit the model on the training set and forecast the
                # validation set
                model = sm.tsa.ARIMA(y_tr, order).fit(trend='nc', #method='css',
                                                      start_params=start_params,
                                                      solver='cg', maxiter=500,
                                                      disp=0)
                y_pred, _, _ = model.forecast(len(y_vld))
                start_params = model.params.copy()  # warm restart

                # Save the current vld error (in terms of mean squared error)
                _current_vld_err = mean_squared_error(y_pred, y_vld)

                # Save the specified information criteria
                _current_ic_score = model.aic if ic_score is 'AIC' else model.bic
            except Exception as e:
                if verbose: warnings.warn('statsmodels.tsa.arima_model.ARIMA '
                                          'raised:\n%s' % e)
                _current_vld_err = np.nan
                _current_ic_score = np.nan

            # Save vld error and ic score
            vld_error[cv_count] = _current_vld_err
            ic_score[cv_count] = _current_ic_score

        # Save mean and standard deviation of cross-validation error
        # (excluding NaNs)
        mean_vld_error[i, j, k] = np.nanmean(vld_error)
        std_vld_error[i, j, k] = np.nanstd(vld_error)

        # Save mean and standard deviation of cross-validation information
        # criteria (excluding NaNs)
        mean_ic_score[i, j, k] = np.nanmean(ic_score)
        std_ic_score[i, j, k] = np.nanstd(ic_score)

    # Get the optimal orders from the score that we want to optimize
    final_index = mean_ic_score + mean_vld_error
    _ip, _id, _iq = np.where(final_index == np.nanmin(final_index))

    # Re-convert the indexes of the tensor in one of the input ARIMA order
    p_opt = _ip[0] + min_p
    d_opt = _id[0] + min_d
    q_opt = _iq[0] + min_q

    if return_final_index:
        return (p_opt, d_opt, q_opt), final_index
    else:
        return (p_opt, d_opt, q_opt)
