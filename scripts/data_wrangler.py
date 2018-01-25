#!/usr/bin/env python3
"""[cgm-tools] Data wrangler.

data_wrangler.py is a Python scripts that converts
Medtronic Diabetes iPro Data Export File (v1.0.1)
to pandas DataFrame objects that can optionally be
saved as pickle files
"""
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################

import argparse
import numpy as np
import os
import pandas as pd
import pickle as pkl
import sys


def Excel2DF(root, tabularfiles, show_failed=False):
    """Load and transform each .csv data file."""
    failed = []
    out = {}
    if tabularfiles[0].endswith('.csv'):
        csvfiles = tabularfiles
        for csvfile in csvfiles:
            filename = os.path.join(root, csvfile)
            try:
                df = pd.read_csv(filename,
                                 encoding='UTF-16LE', sep='\t',
                                 skiprows=11, index_col=0, header=0,
                                 lineterminator=os.linesep)
                out[csvfile] = df
            except:
                failed.append(csvfile)
    else:
        xlsxfiles = tabularfiles
        for xlsxfile in xlsxfiles:
            filename = os.path.join(root, xlsxfile)
            try:
                df = pd.read_excel(filename,
                                   encoding='UTF-16LE',
                                   skiprows=11, index_col=0, header=0)
                out[xlsxfile] = df
            except Exception as e:
                print(e)
                failed.append(xlsxfile)

    if show_failed:
        print("Import failed for {} files:".format(len(failed)))
        for f in failed:
            print("\t- {}".format(f))

    return out


def main(args):
    """Main data wrangling routine."""
    # Check that the input folder contains .csv files
    files = os.listdir(args.data_folder)
    csvfiles = list(filter(lambda x: x.endswith('.csv'), files))
    xlsxfiles = list(filter(lambda x: x.endswith('.xlsx'), files))
    if len(csvfiles) == 0 and len(xlsxfiles) == 0:
        raise Exception("No .csv/.xlsx files in {}".format(args.data_folder))
        sys.exit(-1)
    tabularfiles = csvfiles if len(csvfiles) > 0 else xlsxfiles

    dfs = Excel2DF(root=args.data_folder, tabularfiles=tabularfiles, show_failed=True)

    # Dump content in a pickle file if required
    if args.output:
        with open(args.output+'.pkl', 'wb') as f:
            pkl.dump(dfs, f)

######################################################################


def parsing():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Data wrangling utility for '
                                                 'Medtronic Diabetes iPro '
                                                 'Data Export File (v1.0.1)')
    parser.add_argument("data_folder", help='The folder that contains all'
                                            'the exported csv files')
    parser.add_argument('-o', '--output', metavar='output', action='store',
                        help='output pickle filename containing a dictionary'
                        'with filename as key and its content as value',
                        default='dfs')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ARGS = parsing()
    main(ARGS)
