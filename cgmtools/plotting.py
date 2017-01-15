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


def cgm(df, gluco_fit=None, hypo=70, hyper=126,
        title="Patiend ID CGM", savefig=False):
    """Plot the CGM signal on an input time span.

    Parameters
    -------------------
    df : DataFrame, the output returned by gluco_extract(..., return_df=True)
    gluco_fit : array of float, the results of a fitted model (optional)
    hypo : number, hypoglicaemia threshold in
           mg/dl (default is 70, i.e. 3.9 mmol/l)
    hyper : number, hyperglicaemia threshold
            in mg/dl (default is 126, i.e. 7 mmol/l)
    title : string, the title of the plot (optional)
    savefig : bool, if True save title.png
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hlines(hypo, df.index[0], df.index[-1], linestyles='dashed',
               label='hypoglicaemia')
    plt.hlines(hyper, df.index[0], df.index[-1], linestyles='dotted',
               label='hyperglicaemia')
    plt.ylim([10, 410])
    plt.plot_date(df.index, df.as_matrix(), '-', label='real CGM')
    if gluco_fit is not None:
        plt.plot(df.index, gluco_fit, '--', label='predicted CGM')
    plt.title(title)
    plt.ylabel('mg/dL')
    plt.xticks(rotation='vertical')
    plt.legend(bbox_to_anchor=(1.1, 1.0))
    if savefig: plt.savefig(title+'_fit.png')


def autocorrelation(ts, lags=None):
    """Plot the (partial) autocorrelation of a given time-series.

    Parameters
    -------------------
    ts : array of float, time-series values
    lags : array of int, lag values on horizontal axis.
    """
    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=ax2)
    plt.suptitle('Lags: {}'.format(lags))


def residuals(df, forecast, skip_first=0, skip_last=0,
              title="Patiend ID CGM", savefig=False):
    """Plot the input residuals.

    Parameters
    -------------------
    df : DataFrame, the output returned by gluco_extract(..., return_df=True)
    forecast : array of float, the prediction for the given time-series
    skip_first : number, the number of initial samples to exclude
    skip_last : number, the number of final samples to exclude
    title : string, the title of the plot (optional)
    savefig : bool, if True save title.png
    """
    # Evaluate the residuals (exclude learning and open loop ph samples)
    residuals = df.as_matrix()[skip_first:-skip_last].ravel() - forecast[skip_first:-skip_last]

    plt.figure(figsize=(12, 4), dpi=300)
    plt.plot(df.index[skip_first:-skip_last], residuals)
    DW = sm.stats.durbin_watson(residuals)
    plt.title('Durbin-Watson: {:.3f}'.format(DW))
    if savefig: plt.savefig(title+'_residuals.png')

###############################################################################

if __name__ == '__main__':
    pass  # no action
