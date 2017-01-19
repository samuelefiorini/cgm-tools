
# coding: utf-8

# In[15]:

from cgmtools import utils
# from cgmtools import plotting
import datetime
import numpy as np
import pickle as pkl

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

import matplotlib.pyplot as plt


# In[16]:

# Load full data set from pickle file (see data_wrangler.py)
dfs_full = pkl.load(open('../experiments/cgm-tools/data/dfs_py3.pkl', 'rb'))

# Keep only patients with more than `THRESHOLD` days of CGM acquisition
_threshold = datetime.timedelta(days=3.5)  # default
dfs = utils.filter_patients(dfs_full, _threshold)


# In[17]:

# Get patients list
patients = list(dfs.keys())
idx = patients[1]
df = utils.gluco_extract(dfs[idx], return_df=True)


# In[18]:

# Training (burn-in) / Test split
burn_in = 300  # burn-in samples used to learn the best model parameters
train_df0 = df.iloc[:burn_in]
test_df0 = df.iloc[burn_in:]


# In[19]:

# preprocess the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_df0)
test_data = scaler.transform(test_df0) # BEWARE! Do not use the trainig set to learn the scaling parameters


# In[20]:

# convert an array of values into a dataset matrix
def create_XY_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
w_size = 36 # look back
X_tr, Y_tr = create_XY_dataset(train_data, w_size)
X_ts, Y_ts = create_XY_dataset(test_data, w_size)


# In[21]:

# reshape input to be [samples, time steps, features]
X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], 1))
X_ts = np.reshape(X_ts, (X_ts.shape[0], X_ts.shape[1], 1))


# In[22]:

print(X_tr.shape)
print(X_ts.shape)
print(df.shape)


# In[27]:

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=X_tr.shape[-1]))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_tr, Y_tr, nb_epoch=100, batch_size=1, verbose=0)


# In[28]:

Y_pred_tr = model.predict(X_tr)
Y_pred_ts = model.predict(X_ts)


# In[29]:

Y_pred_tr_plot = scaler.inverse_transform(Y_pred_tr)
Y_pred_ts_plot = scaler.inverse_transform(Y_pred_ts)


# In[30]:

plt.figure(figsize=(10, 6), dpi=300)
plt.subplot(211)
plt.plot(df.index, df.values, label='real cgm')
plt.plot(df.index[w_size:burn_in], Y_pred_tr_plot.ravel(), '--', label='y_tr')
plt.plot(df.index[burn_in+w_size:], Y_pred_ts_plot.ravel(), '--', label='y_tr')
plt.legend()

residuals = Y_pred_ts_plot.ravel() - df.values[burn_in+w_size:].ravel()
mae = np.mean(residuals)
rmse = np.sqrt(np.mean(residuals ** 2))
DW = sm.stats.durbin_watson(residuals)

plt.subplot(212)
plt.plot(df.index[burn_in:-w_size], residuals)
plt.title("MAE {:2.5f} | RMSE {:2.5f} | DW {:2.5f}".format(mae, rmse, DW))
plt.tight_layout();


# In[39]:

X_ts[0, :, 0]


# In[128]:

X_ts[1, :, 0]


# In[130]:

import warnings
warnings.filterwarnings('ignore')


# In[ ]:

# def forecast(model=None, n_steps=1, X=None):
#     if n_steps <= 0:
#         raise Exception('n_steps must be at least 1')
n_steps = 18
# Init predictions

errs_dict = {'err_18': [], 'err_12': [], 'err_6': []}

for t in range(X_ts.shape[0] - n_steps):
    if t % 200 == 0: print("Forecasting t = {}/{}".format(t, X_ts.shape[0]))
    _X_ts_next = X_ts[t].reshape(1, w_size, 1)

    # Forecast next ph samples
    # --------------------------------------------------------------------- #
    y_pred = np.zeros(n_steps)
    for step in range(n_steps):
        # Perform one-step-ahead
        y_pred[step] = model.predict(_X_ts_next)
         # shift ahead the window
#         _X_ts_next = np.append(_X_ts_next[:, 1:, 0], y_pred[step]).reshape(1, w_size, 1)
        _X_ts_next = np.reshape(np.append(_X_ts_next[:, 1:, :], y_pred[step]), (1, w_size, 1))
    # --------------------------------------------------------------------- #

    y_pred = scaler.inverse_transform(y_pred)  # get back to original dimensions
    y_future_real = scaler.inverse_transform(Y_ts[t:t+n_steps])
    abs_pred_err = np.abs(y_pred - y_future_real)

    # Save errors
    errs_dict['err_18'].append(abs_pred_err[17])
    errs_dict['err_12'].append(abs_pred_err[11])
    errs_dict['err_6'].append(abs_pred_err[5])

report = utils.forecast_report(errs_dict)
print(report)


# In[133]:

#            30       60       90
# MAE   21.9983  58.5689   79.121
# RMSE  35.3848  93.0235  116.726


# In[101]:

y_pred


# In[76]:

np.append(X_ts[0, 1:, 0], 0.22262774)


# In[78]:

X_ts[0, :, 0]


# In[79]:

X_ts[1, :, 0]


# In[64]:

X_ts.shape


# In[ ]:
