#!/usr/bin/env python3
"""LSTM data wrangler.

Prepare a fat traning / test split.
"""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################
from cgmtools import utils
import datetime
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
import time
###############################################################################

# Load full data set from pickle file (see data_wrangler.py)
dfs_full = pkl.load(open('../data/dfs_py3.pkl', 'rb'))

# Keep only patients with more than `THRESHOLD` days of CGM acquisition
_threshold = datetime.timedelta(days=3.5)  # default
dfs = utils.filter_patients(dfs_full, _threshold)

burn_in = 300  # burn-in samples used to learn the best order via cv
# n_splits = 15
ph = 18  # prediction horizon
w_size = 36

# Get patients list
patients = list(dfs.keys())

print(patients)
