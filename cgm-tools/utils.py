""""[cgm-tools] Data handling utilities."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################


import datetime
import pandas as pd


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
