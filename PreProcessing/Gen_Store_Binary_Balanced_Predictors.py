#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate and store the binary and balanced predicors data.
"""


#%% import
import os
import sys


#%% set scripts directory and import self-written functions
script_direc = "/PATH_TO_FUNCTIONS/Python_NORA_Data/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)

from Functions.Particular_Functions.Func_Prep_Data_Avalanche_Analysis import load_feats_binary_risk


#%% set a number for the n-th permutation
try:
    perm = int(sys.argv[1])

    reg_code = int(sys.argv[2])  # if code is 0 then use all regions

    h_low = int(sys.argv[3])
    h_hi = int(sys.argv[4])
except:
    perm = 0
    reg_code = 3013  # use 0 for all regions combined
    h_low = 400
    h_hi = 900
# end try except


#%% set the region codes
if reg_code == 0:
    reg_codes = [3009, 3010, 3011, 3012, 3013]
else:
    reg_codes = [reg_code]
# end if else

print(f"\nGenerating binary predictor data: permutation {perm}, reg_code: {reg_codes}\n")


#%% select features and exposure
sel_feats = ["s1", "s2", "s3", "s4", "s5", "s6", "s7",
             "r1", "r2", "r3", "r4", "r5", "r6", "r7",
             "t2", "t3", "t4", "t5",
             "tmax2", "tmax3", "tmax4", "tmax5",
             "total_prec_sum",
             "wspeed_max", "wspeed_mean", "wind_direction",
             "w2", "w3", "w4", "w5",
             "wmax2", "wmax3", "wmax4", "wmax5",
             "t_mean", "t_min", "t_max", "t_range",
             "dtemp1", "dtemp2", "dtemp3",
             "dtempd1", "dtempd2", "dtempd3",
             "pdd", "ftc",
             "wdrift", "wdrift3", "wdrift_2", "wdrift3_2", "wdrift_3", "wdrift3_3"]
exposure = None


#%% add exposure part to the model name
add_expos = ""
if exposure == "west":
    add_expos = "_WestExpos"
elif exposure == "east":
    add_expos = "_EastExpos"
# end if elif


#%% load the data
data_path_par = "/PATH_TO_AVALANCHE_PREDICTORS/"
features = load_feats_binary_risk(reg_codes=reg_codes, exposure=exposure, sel_feats=sel_feats, out_type="dataframe",
                                  plot_box=False, h_low=h_low, h_hi=h_hi,
                                  data_path_par=data_path_par + "Avalanche_Predictors/")


#%% create the folder
data_path = data_path_par + f"Avalanche_Predictors_Binary/Between{h_low}_and_{h_hi}m/Permutation_{perm:02}/"
os.makedirs(data_path, exist_ok=True)


#%% store the data as csv files
if len(reg_codes) == 1:

    print("Storing data for individual region in")
    print(data_path)

    # set region name according to region code
    if reg_codes[0] == 3009:
        region = "NordTroms"
    elif reg_codes[0] == 3010:
        region = "Lyngen"
    elif reg_codes[0] == 3011:
        region = "Tromsoe"
    elif reg_codes[0] == 3012:
        region = "SoerTroms"
    elif reg_codes[0] == 3013:
        region = "IndreTroms"
    # end if elif

    # training data balanced
    features["train_balanced"].to_csv(data_path +
                                      f"/Train_Features_Binary_Balanced_Between{h_low}_{h_hi}m_{reg_codes[0]}_" +
                                      f"{region}.csv")

    # training data all
    features["train_all"].to_csv(data_path +
                                 f"/Train_Features_Binary_All_Between{h_low}_{h_hi}m_{reg_codes[0]}_" +
                                 f"{region}.csv")

    # test data balanced
    features["test_balanced"].to_csv(data_path +
                                     f"/Test_Features_Binary_Balanced_Between{h_low}_{h_hi}m_{reg_codes[0]}_" +
                                     f"{region}.csv")

    # test data all
    features["test_all"].to_csv(data_path +
                                f"/Test_Features_Binary_All_Between{h_low}_{h_hi}m_{reg_codes[0]}_" +
                                f"{region}.csv")
elif len(reg_codes) == 5:

    print("Storing data for all regions in")
    print(data_path)

    # training data balanced
    features["train_balanced"].to_csv(data_path +
                                      f"/Train_Features_Binary_Balanced_Between{h_low}_{h_hi}m_AllReg.csv")

    # training data all
    features["train_all"].to_csv(data_path +
                                 f"/Train_Features_Binary_All_Between{h_low}_{h_hi}m_AllReg.csv")

    # test data balanced
    features["test_balanced"].to_csv(data_path +
                                     f"/Test_Features_Binary_Balanced_Between{h_low}_{h_hi}m_AllReg.csv")

    # test data all
    features["test_all"].to_csv(data_path +
                                f"/Test_Features_Binary_All_Between{h_low}_{h_hi}m_AllReg.csv")
# end if elif


