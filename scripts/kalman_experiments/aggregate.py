import numpy as np
import pandas as pd
import pickle as pkl
import os

files = list(filter(lambda x: x.endswith('.pkl'), os.listdir('.')))
dfs = dict()
errors = pd.DataFrame(columns=['MAE30', 'MAE60', 'MAE90',
                               'RMSE30', 'RMSE60', 'RMSE90'],
                      index=np.arange(len(files)))
for i, f in enumerate(files):
    dfs[f] = pkl.load(open(f, 'rb'))
    errors.loc[i] = dfs[f].values.ravel()

print('Error summary:\n-mean')
print(errors.mean())
print('-std')
print(errors.std())
