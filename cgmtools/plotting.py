"""[cgm-tools] Plotting utilities."""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import matplotlib; matplotlib.use('agg')
from seaborn import plt
import statsmodels.api as sm

__all__ = ['cgm', 'autocorrelation']


def cgm(time, gluco, hypo=70, hyper=126,
        title="Patiend ID glucose level"):
    """Plot the CGM signal on an input time span.

    Parameters
    -------------------
    time : array of datetimes, the horizontal axis
    gluco : array of float, the correspondent glucose
            level on the vertical axis
    hypo : number, hypoglicaemia threshold in
           mg/dl (default is 70, i.e. 3.9 mmol/l)
    hyper : number, hyperglicaemia threshold
            in mg/dl (default is 126, i.e. 7 mmol/l)
    title : string, the title of the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.hlines(hypo, time[0], time[-1], linestyles='dashed',
               label='hypoglicaemia')
    plt.hlines(hyper, time[0], time[-1], linestyles='dotted',
               label='hyperglicaemia')
    plt.ylim([10, 410])
    plt.plot_date(time, gluco, '-', label='glucose level')
    plt.title(title)
    plt.ylabel('mg/dL')
    plt.xticks(rotation='vertical')
    plt.legend()


def autocorrelation(ts, lags=None):
    """Plot the (partial) autocorrelation of a given time-series.

    Parameters
    -------------------
    ts : array of float, time-series values
    lags : array of int, lag values on horizontal axis.
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=ax2)
    plt.suptitle('Lags: {}'.format(lags))

###############################################################################

if __name__ == '__main__':
    pass  # no action
