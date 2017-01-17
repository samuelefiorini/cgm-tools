""""[cgm-tools] CGM forecast via Kalman filter."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit
from pykalman import KalmanFilter
import warnings


__all__ = ['cgmkalmanfilter', 'forecast', 'grid_search']


def cgmkalmanfilter(X0=None, P0=None, F=None, Q=None, X=None, R=None,
                    random_state=None, em_vars=None, **kwargs):
    """Kalman Filter wrapper that uses compact names notation.

    Unspecified KalmanFilter arguments are:
    - transition_offsets
    - observation_offsets
    - n_dim_state
    - n_dim_obs
    their access granted via **kwargs
    """
    return KalmanFilter(transition_matrices=F,
                        observation_matrices=X,
                        transition_covariance=Q,
                        observation_covariance=R,
                        initial_state_mean=X0,
                        initial_state_covariance=P0,
                        random_state=random_state,
                        em_vars=em_vars, **kwargs)


def forecast(kf=None, n_steps=1, X_old=None, P_old=None, H=None,
             y=None, return_first_kf=False):
    """Forecast n_steps-ahead using the input Kalman filter.

    Parameters
    -------------------
    kf : pykalman.standard.KalmanFilter, the (fitted) KF to use
         to perform forecast
    n_steps : number, the prediction horizon (default = 1)
    H : array of float, the acquisition model
    y : array of float, the observation until the last acquired sample
    return_first_kf : bool, return the KF used to perform
                      the first one-step-ahead prediction

    Returns
    -------------------
    y_pred : array of float, the `n_steps` predicted future values
    X_new : array of float, the one-step-updated state matrix
    P_new : array of float, the one-step-updated state covariance matrix
    kf_out : pykalman.standard.KalmanFilter, the KF used to perform
             the first one-step-ahead prediction (if return_first_kf=True)
    """
    if n_steps <= 0:
        raise Exception('n_steps must be at least 1')

    # Init predictions
    y_pred = np.zeros(n_steps)

    # Perform state estimation until the end of y
    X_new, P_new = kf.filter(y)

    # perform one-step-ahead prediction
    X_new, P_new = kf.filter_update(filtered_state_mean=X_new[-1],
                                    filtered_state_covariance=P_new[-1],
                                    observation=y[-1])
    y_pred[0] = np.dot(H.reshape(1, 2), X_new.reshape(2, 1))[0][0]

    if n_steps < 2:
        # Multiple return
        ret = [y_pred, X_new, P_new]
        if return_first_kf:
            ret.append(kf)
        return ret
    else:
        P_old = P_new.copy()
        X_old = X_new.copy()

        # copy the KF to perform recursive forecast
        _kf = copy.deepcopy(kf)
        _y_curr = y.copy()
        for t in range(1, n_steps - 1):
            _y_curr = y_pred[t]  # append the predicted y
            X_new, P_new = _kf.filter_update(filtered_state_mean=X_old,
                                             filtered_state_covariance=P_old,
                                             observation=_y_curr)
            y_pred[t] = np.dot(H.reshape(1, 2), X_new.reshape(2, 1))[0][0]
            P_old = P_new.copy()
            X_old = X_new.copy()

        # Multiple return
        ret = [y_pred, X_new, P_new]
        if return_first_kf:
            ret.append(kf)
        return ret


def grid_worker(l2, s2, F, H, tscv, time_series, count, jobs_dump):
    """Grid-search worker."""
    Q = np.array([[l2, 0], [0, 0]])  # transition_covariance
    R = s2  # observation (co)variance

    # Init the vld_error vector for the current order
    vld_error = np.zeros(tscv.n_splits)

    # Iterate through the CV splits
    for cv_count, (tr_index, vld_index) in enumerate(tscv.split(time_series)):
        if cv_count == 0:  # init X0 and P0 via EM on the first chunk of data
            y_0 = time_series.iloc[np.hstack((tr_index,
                                              vld_index))].values.ravel()
            # Init KalmanFilter object
            kf = cgmkalmanfilter(F=F, Q=Q, R=R, X0=None, P0=None)
            kf.em(y_0, em_vars=('initial_state_mean',
                                'initial_state_covariance'))
        else:
            y_tr = time_series.iloc[tr_index].values.ravel()
            y_vld = time_series.iloc[vld_index].values.ravel()
            y_pred, X_new, P_new, kf = forecast(kf=kf,
                                                n_steps=len(y_vld),
                                                H=H, y=y_tr,
                                                return_first_kf=True)
            # Save vld error
            vld_error[cv_count] = mean_squared_error(y_pred, y_vld)
    jobs_dump[count] = (l2, s2, vld_error)


def grid_search(df, lambda2_range, sigma2_range, burn_in=300, n_splits=15,
                return_mean_vld_error=False, verbose=False):
    """Find the best Kalman filter parameters via grid search cross-validation.

    This function perform a grid search of the optimal (lambda2, r)
    parameters of the pykalman.KalmanFilter on input data where:

    transition_matrix      -> F = [[2,-1], [1, 0]] (double-integrated
                                                    random-walk model)
    transition_covariance  -> Q = [[lambda2, 0], [0, 0]]
    observation_covariance -> R = [sigma2]
    observation_model      -> H = [1, 0]

    as in [1]. In this function lambda2 and sigma2 are not estimated
    using the Bayesian framework described in [1], but they are
    obtained via cross-validation. The optimization is ran on ...


    Parameters
    -------------------
    df : DataFrame, the output returned by gluco_extract(return_df=True)
    burn_in : number, the number of samples at the beginning of the
              time-series that should be splitted to perform grid search
              (default = 300)
    n_splits : number, the number of splits of the time-series
               cross-validation schema (default=15). Your prediction
               horizon will be `floor(n_samples / (n_splits + 1))`
    [....]
    return_mean_vld_error : bool, return the average validation error
                           (default=False)
    verbose : bool, print debug messages (default=False)

    Returns
    -------------------
    [...]

    References
    -------------------
    [1] Facchinetti, Andrea, Giovanni Sparacino, and Claudio Cobelli.
    "An online self-tunable method to denoise CGM sensor data."
    IEEE Transactions on Biomedical Engineering 57.3 (2010): 634-641.
    """
    n_samples = df.shape[0]

    # Argument check
    if n_samples < burn_in:
        raise Exception('The number of burn in samples %d should be '
                        'smaller than the total number of samples '
                        '%d' % (burn_in, n_samples))

    import multiprocessing as mp
    # State-space model:
    # transition matrix (double integration model)
    F = np.array([[2, -1], [1, 0]])
    # measures matrix
    H = np.array([1, 0])

    # Isolate the burn in samples
    time_series = df.iloc[:burn_in]

    # Parameter grid definition
    # see state covariance and noise variance parameters
    param_grid = ParameterGrid({'lambda2': lambda2_range,
                                'sigma2': sigma2_range})

    # Time-series cross validation split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize the cross-validation error matrix of size
    # (len(lambda2_range), len(sigma2_range))
    mean_vld_error = np.zeros((len(lambda2_range), len(sigma2_range)))
    std_vld_error = np.zeros_like(mean_vld_error)

    # Positions dictionary
    d_lambda = dict(zip(lambda2_range, np.arange(len(lambda2_range))))
    d_sigma = dict(zip(sigma2_range, np.arange(len(sigma2_range))))

    # Iterate trough the parameters lambda2, sigma2
    # i, j index will be used to access the mean_vld_error matrix
    jobs = []
    manager = mp.Manager()
    jobs_dump = manager.dict()
    for count, param in enumerate(param_grid):
        if verbose: print('trying params {} ...'.format(param))
        l2, s2 = param['lambda2'], param['sigma2']

        proc = mp.Process(target=grid_worker,
                          args=(l2, s2, F, H, tscv, time_series,
                                count, jobs_dump))
        jobs.append(proc)
        proc.start()
        if verbose: print("Job: %d submitted", count)

    # Collect results
    count = 0
    for proc in jobs:
        proc.join()
        count += 1
    if verbose: print("%d jobs collected", count)

    for count in jobs_dump.keys():
        # Save mean and standard deviation of cross-validation error
        # (excluding NaNs)
        l2, s2, vld_error = jobs_dump[count]
        i, j, = d_lambda[l2], d_sigma[s2]
        mean_vld_error[i, j] = np.nanmean(vld_error)
        std_vld_error[i, j] = np.nanstd(vld_error)

    # Get the optimal orders from the score that we want to optimize
    final_index = mean_vld_error
    i_opt, j_opt, = np.argwhere(final_index == np.nanmin(mean_vld_error))[0]

    # Multiple returns
    ret = [lambda2_range[i_opt], sigma2_range[j_opt]]
    if return_mean_vld_error:
        ret.append(mean_vld_error)
    return ret

# def online_forecast()
