""""[cgm-tools] Data handling utilities."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import datetime
import numpy as np
import pandas as pd


def filter_patients(dfs, threshold):
    """Filter patients with less than `THRESHOLD` days of CGM acquisition.

    Parameters
    -------------------
    dfs : dictionary of pandas.DataFrame, obtained from data_wrangler.py
    threshold : datetime.timedelta, CGM monitoring inclusion criterion

    Returns
    -------------------
    dfs_out : dictionary of pandas.DataFrame, containing only the patients that
              match the inclusion criterion
    """
    # Init list of patients that satisfy inclusion criterion
    ok_keys = []

    # Iterate on the patients
    for k in dfs.keys():
        df = dfs[k]
        time, gluco = gluco_extract(df)
        try:  # TODO: improve here
            delta = time[-1] - time[0]
            if delta > threshold: ok_keys.append(k)
        except:
            pass

    # Filter short time-series
    dfs_out = {k: dfs[k] for k in ok_keys}

    return dfs_out


def gluco_extract(df, return_df=False):
    """Extract glucose trend and time axis of a given patient.

    Parameters
    -------------------
    df : pandas DataFrame, tabular info coming from Medtronic Diabetes iPro
         Data Export File (v1.0.1) (see data_wrangler.py)
    return_df : bool, when this flag is set to true, the function returns a
                single pandas DataFrame containing the glucose level indexed by
                times

    Returns
    -------------------
    time : array of datetimes, the glucose measuring time span
           (if return_df=False)
    gluco : array of float, the correspondent glucose level
            (if return_df=False)
    df_tg : pandas.DataFrame, gluco indexed by times (if return_df=True)
    """
    # Select glucose, time and date
    column = 'Sensor Glucose (mg/dL)'
    _gluco = df[column]
    _time = df['Time']
    _date = df['Date']

    # Create a new data frame and drop missing values
    _df = pd.concat((_gluco, _time, _date), axis=1).dropna()

    # Parse time axis
    time = []
    for (d, h) in zip(_df['Date'], _df['Time']):
        _d = datetime.datetime.strptime(d, '%d/%m/%y').date()
        _h = datetime.datetime.strptime(h, '%H:%M:%S').time()
        time.append(datetime.datetime.combine(_d, _h))

    # Extract glucose level
    gluco = _df[column]

    if return_df:
        return pd.DataFrame(data=gluco.values,
                            index=pd.DatetimeIndex(time),
                            columns=[column])
    else:
        return time, gluco


def root_mean_squared(x):
    """Evaluate root mean squared error of a given list of errors.

    This function is NaNs insensitive.

    Parameters
    -------------------
    x : array of floats, the array of absolute errors

    Returns
    -------------------
    rmse : float, the root mean squared error
    """
    idx = np.where(map(lambda x: not x, np.isnan(x)))[0]  # filter NaNs
    return np.linalg.norm(x[idx]) / np.sqrt(np.count_nonzero(idx))


def forecast_report(errors):
    """Build a report showing mean absolute and root mean squared errors.

    Parameters
    -------------------
    errors : dictionary, errors at 30/60/90 mins ('err_18', 'err_12', 'err_6')
             (see forecast.arima.moving_window())

    Returns
    -------------------
    report : pandas DataFrame, absolute mean and root mean squared errors at
             30/60/90 min report
    """
    report = pd.DataFrame(index=['MAE', 'RMSE'], columns=[30, 60, 90])
    report.loc['MAE'] = [np.nanmean(errors['err_6']),
                         np.nanmean(errors['err_12']),
                         np.nanmean(errors['err_18'])]
    report.loc['RMSE'] = [root_mean_squared(errors['err_6']),
                          root_mean_squared(errors['err_12']),
                          root_mean_squared(errors['err_18'])]
    return report
