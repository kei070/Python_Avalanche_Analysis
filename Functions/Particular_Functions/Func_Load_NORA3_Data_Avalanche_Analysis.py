#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for loading the NorCP predictor data to use with a statisitcal model to predict avalanches.
"""

# imports
import os
import sys
import glob
import numpy as np
import pandas as pd

# set scripts directory and import self-written functions
script_direc = "/PATH_TO_SCTIPTS/Python_Avalanche_Analysis/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)
from Functions.General_Functions.Func_DatetimeSimple import date_dt2

# define the function
def load_feats_nora3(reg_codes, sel_mon=[12, 1, 2, 3, 4, 5], start_date=None, end_date=None, dropna=True,
                     data_path_par=""):

    """
    Parameters:
        reg_codes      List of integers. Region code(s) of the avalanche regions.
        data_path_par  Parent data path for the predictors/features.
        dropna         Boolean. If True (default) all lines containing any NaN will be removed.

    More information on the parameters:
        reg_code = 3009  # Nord-Troms
        reg_code = 3010  # Lyngen
        reg_code = 3011  # Tromsoe
        reg_code = 3012  # Soer-Troms
        reg_code = 3013  # Indre Troms
    """

    # loop over the region codes and load the data
    data = []

    for reg_code in reg_codes:

        # set region name according to region code
        if reg_code == 3009:
            region = "NordTroms"
        elif reg_code == 3010:
            region = "Lyngen"
        elif reg_code == 3011:
            region = "Tromsoe"
        elif reg_code == 3012:
            region = "SoerTroms"
        elif reg_code == 3013:
            region = "IndreTroms"
        elif reg_code == "AllReg":
            region = "AllRegions"
        # end if elif

        # NORA3 data path
        data_path = data_path_par

        print(data_path)

        ## load the training and test data
        data_fns = sorted(glob.glob(data_path + f"/NORA3_Avalanche_Predictors_*_{region}.csv"))

        print(data_path + f"/NORA3_Avalanche_Predictors_*_{region}.csv")

        # select and load a dataset
        data_i = 0
        data_fn = data_fns[data_i]
        df = pd.read_csv(data_fn)

        # convert the date column to datetime format
        dates = pd.to_datetime(df.date)
        df.date = dates
        df.set_index(dates, inplace=True)


        # generate a column with the region number for identification
        df["reg_code"] = np.repeat(reg_code, repeats=len(df))

        # reduce the data to the requested period
        if start_date != None:
            print(f"Applying start date {start_date}")
            df = df[df.index >= date_dt2(start_date)]
        if end_date != None:
            print(f"Applying end date: {end_date}")
            df = df[df.index <= date_dt2(end_date)]
        # end if

        # append the data to the list for later concatenation
        data.append(df[df["date"].dt.month.isin(sel_mon)])
    # end for reg_code

    if dropna:
        return pd.concat(data).dropna()
    else:
        return pd.concat(data)
    # end if else

# end def


