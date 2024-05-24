#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for loading the NorCP predictor data to use with a statisitcal model to predict avalanches.
"""

# imports
import glob
import numpy as np
import pandas as pd


# define the function
def load_feats_norcp(reg_codes, scen="historical", period="", model="EC-EARTH", dropna=True, h_low=400, h_hi=900,
                     data_path_par="/"):

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
        # data_path = data_path_par + region + "/" + scen + period + "/" + model + "/"
        data_path = data_path_par + f"Between{h_low}_and_{h_hi}m/{scen}{period}/{model}/"
        print(data_path)

        ## load the training and test data
        data_fns = sorted(glob.glob(data_path +
                                    f"/NorCP_Avalanche_Predictors_MultiCellMean_{region}.csv"))

        # select and load a dataset
        data_i = 0
        data_fn = data_fns[data_i]
        df = pd.read_csv(data_fn)

        # convert the date column to datetime format
        df.date = pd.to_datetime(df.date)

        # generate a column with the region number for identification
        df["reg_code"] = np.repeat(reg_code, repeats=len(df))

        # append the data to the list for later concatenation
        data.append(df)

    # end for reg_code

    if dropna:
        return pd.concat(data).dropna()
    else:
        return pd.concat(data)
    # end if else

# end def


