#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the individual features for avalanche risk prediction with logistic regression model.
"""

#%% imports
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#%% set scripts directory and import self-written functions
script_direc = "/PATH_TO_FUNCTIONS/Python_Avalanche_Analysis/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)

from Functions.General_Functions.Func_Model_Fidelity_Metrics import mod_metrics


#%% set number of shuffles
try:
    nshuffle = int(sys.argv[7])
except:
    nshuffle = 50
# end try except


#%% read in parameters
try:
    reg_code =  sys.argv[1]
    first_feat = sys.argv[2]
except:
    reg_code = "AllReg"
    first_feat = "s3"
# end try except

# set the height thresholds
try:
    h_low = sys.argv[3]  # 200
    h_hi = sys.argv[4]  # 600
except:
    h_low = 400
    h_hi = 900
# end try except


#%% switch training and test data
try:
    switch = sys.argv[5]
except:
    switch = "False"
# end try except

if switch == "False":
    train_s = "Train"
    test_s = "Test"
elif switch == "True":
    train_s = "Test"
    test_s = "Train"
# end if


#%% set permutation
try:
    perm = int(sys.argv[6])
except:
    perm = 0
# end try except


#%% read the region code
# reg_code = 3009  # Nord-Troms
# reg_code = 3010  # Lyngen
# reg_code = 3011  # Tromsoe
# reg_code = 3012  # Soer-Troms
# reg_code = 3013  # Indre Troms
# reg_code = "AllReg"

if reg_code != "AllReg":
    reg_code = int(reg_code)
# end if


#%% features

# first_feat = "s3"
second_feat = ["s1", "s2", "s3", "s4", "s5", "s6", "s7",
               "r1", "r2", "r3", "r4", "r5", "r6", "r7",
               "t2", "t3", "t4", "t5",
               "wspeed_mean", "wspeed_max", "wind_direction",
               "w2", "w3", "w4", "w5",
               "wmax2", "wmax3", "wmax4", "wmax5",
               "tmax2", "tmax3", "tmax4", "tmax5",
               "total_prec_sum",
               "t_mean", "t_min", "t_max", "t_range",
               "dtemp1", "dtemp2", "dtemp3",
               "dtempd1", "dtempd2", "dtempd3",
               "pdd", "ftc",
               "wdrift", "wdrift3", "wdrift_2", "wdrift3_2", "wdrift_3", "wdrift3_3"]
second_feat.remove(first_feat)


#%% set region name according to region code
region = "All"
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
# end if elif


#%% handle the chosen permutation
p_dir = ""
if perm > -1:
    p_dir = f"Permutation_{perm:02}"
# end if


#%% set paths
data_path = f"/PATH_TO_AVALANCHE_PREDICTORS/Between{h_low}_and_{h_hi}m/{p_dir}/"
out_path = f"/PATH_TO_ACCURACY/Two_Features/{first_feat}/Between{h_low}_and_{h_hi}m/{region}/"
# --> out_path will be generated


#%% load the data
try:
    train_df = pd.read_csv(glob.glob(data_path +
                                     f"{train_s}_Features_Binary_Balanced_Between{h_low}_{h_hi}m_{reg_code}*.csv")[0])
    test_df = pd.read_csv(glob.glob(data_path +
                                    f"{test_s}_Features_Binary_Balanced_Between{h_low}_{h_hi}m_{reg_code}*.csv")[0])

    test_all_df = pd.read_csv(glob.glob(data_path +
                                        f"{test_s}_Features_Binary_All_Between{h_low}_{h_hi}m_{reg_code}*.csv")[0])
    train_all_df = pd.read_csv(glob.glob(data_path +
                                        f"{train_s}_Features_Binary_All_Between{h_low}_{h_hi}m_{reg_code}*.csv")[0])
except IndexError:
    sys.exit(f"No predictors available for {reg_code} between {h_low} and {h_hi} m.")
# end try except


#%% load the data

# set decision tree parameters
max_depth = 2
min_leaf_samp = 5  # minimum number of samples that a leaf must contain

# accuracy means over all shuffels
acc_train_m = []
acc_test_m = []
acc_test_all_m = []

# accuracy standard deviations over all shuffles
acc_train_s = []
acc_test_s = []
acc_test_all_s = []

fid_test_d_m = {"feature":second_feat, "far":[], "mfr":[]}
fid_test_all_d_m = {"feature":second_feat, "far":[], "mfr":[]}
fid_train_d_m = {"feature":second_feat, "far":[], "mfr":[]}
fid_test_d_s = {"feature":second_feat, "far":[], "mfr":[]}
fid_test_all_d_s = {"feature":second_feat, "far":[], "mfr":[]}
fid_train_d_s = {"feature":second_feat, "far":[], "mfr":[]}

for sel_feat in second_feat:

    fid_test_d = {"feature":second_feat, "rpc":[], "tss":[], "far":[], "pod":[], "pon":[], "hss":[], "pss":[], "mfr":[]}
    fid_test_all_d = {"feature":second_feat, "rpc":[], "tss":[], "far":[], "pod":[], "pon":[], "hss":[], "pss":[],
                      "mfr":[]}
    fid_train_d = {"feature":second_feat, "rpc":[], "tss":[], "far":[], "pod":[], "pon":[], "hss":[], "pss":[],
                   "mfr":[]}

    acc_train = []
    acc_test = []
    acc_test_all = []

    sel_feats = [first_feat, sel_feat]

    for shuffles in np.arange(nshuffle):

        odata_x = pd.concat([train_df[sel_feats], test_df[sel_feats]])
        odata_y = pd.concat([train_df[f"{train_s.lower()}_y_balanced"], test_df[f"{test_s.lower()}_y_balanced"]])

        odata_x_all = pd.concat([train_all_df[sel_feats], test_all_df[sel_feats]])
        odata_y_all = pd.concat([train_all_df[f"{train_s.lower()}_y"], test_all_df[f"{test_s.lower()}_y"]])

        train_x, test_x, train_y, test_y = train_test_split(odata_x, odata_y, test_size=0.33, shuffle=True)
        train_x_all, test_x_all, train_y_all, test_y_all = train_test_split(odata_x, odata_y, test_size=0.33,
                                                                            shuffle=True)


        #% define the decision tree model
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf_samp)
        model.fit(train_x, train_y)


        #% make probability predictions with the model
        pred_prob = model.predict(test_x)
        pred_train_prob = model.predict(train_x)

        pred = (model.predict(test_x) > 0.5).astype(int)  # binary prediction
        pred_train = (model.predict(train_x) > 0.5).astype(int)  # binary prediction

        pred_prob_test = model.predict(test_x)
        pred_test = (model.predict(test_x) > 0.5).astype(int)  # binary prediction

        pred_prob_test_all = model.predict(test_x_all)
        pred_test_all = (model.predict(test_x_all) > 0.5).astype(int)  # binary prediction


        #% accuracy score
        acc_train.append(accuracy_score(train_y, pred_train))
        acc_test.append(accuracy_score(test_y, pred_test))
        acc_test_all.append(accuracy_score(test_y_all, pred_test_all))


        #% calculate the confusion matrix
        m_train = mod_metrics(metrics.confusion_matrix(train_y, pred_train))
        m_test = mod_metrics(metrics.confusion_matrix(test_y, pred_test))
        m_test_all = mod_metrics(metrics.confusion_matrix(test_y_all, pred_test_all))


        # from the confusion matrix calculate the fidelity metrics
        fid_train_d["tss"].append(m_train["TSS"])
        fid_test_d["tss"].append(m_test["TSS"])
        fid_test_all_d["tss"].append(m_test_all["TSS"])

        fid_train_d["rpc"].append(m_train["RPC"])
        fid_test_d["rpc"].append(m_test["RPC"])
        fid_test_all_d["rpc"].append(m_test_all["RPC"])

        fid_train_d["pod"].append(m_train["POD"])
        fid_test_d["pod"].append(m_test["POD"])
        fid_test_all_d["pod"].append(m_test_all["POD"])

        fid_train_d["pon"].append(m_train["PON"])
        fid_test_d["pon"].append(m_test["PON"])
        fid_test_all_d["pon"].append(m_test_all["PON"])

        fid_train_d["hss"].append(m_train["HSS"])
        fid_test_d["hss"].append(m_test["HSS"])
        fid_test_all_d["hss"].append(m_test_all["HSS"])

        fid_train_d["pss"].append(m_train["PSS"])
        fid_test_d["pss"].append(m_test["PSS"])
        fid_test_all_d["pss"].append(m_test_all["PSS"])

        fid_train_d["far"].append(m_train["FAR"])
        fid_test_d["far"].append(m_test["FAR"])
        fid_test_all_d["far"].append(m_test_all["FAR"])

        fid_train_d["mfr"].append(m_train["MFR"])
        fid_test_d["mfr"].append(m_test["MFR"])
        fid_test_all_d["mfr"].append(m_test_all["MFR"])

    # end for shuffle

    acc_train = np.array(acc_train)
    acc_test = np.array(acc_test)
    acc_test_all = np.array(acc_test_all)

    # calculate mean and std for the accuracy
    acc_train_m.append(np.mean(acc_train))
    acc_test_m.append(np.mean(acc_test))
    acc_test_all_m.append(np.mean(acc_test_all))
    acc_train_s.append(np.std(acc_train))
    acc_test_s.append(np.std(acc_test))
    acc_test_all_s.append(np.std(acc_test_all))

    fid_train_d["far"] = np.array(fid_train_d["far"])
    fid_test_d["far"] = np.array(fid_test_d["far"])
    fid_test_all_d["far"] = np.array(fid_test_all_d["far"])

    fid_train_d["mfr"] = np.array(fid_train_d["mfr"])
    fid_test_d["mfr"] = np.array(fid_test_d["mfr"])
    fid_test_all_d["mfr"] = np.array(fid_test_all_d["mfr"])

    # calculate mean and std for the the other fidelity metrics
    fid_train_d_m["far"].append(np.mean(fid_train_d["far"]))
    fid_test_d_m["far"].append(np.mean(fid_test_d["far"]))
    fid_test_all_d_m["far"].append(np.mean(fid_test_all_d["far"]))

    fid_train_d_s["far"].append(np.std(fid_train_d["far"]))
    fid_test_d_s["far"].append(np.std(fid_test_d["far"]))
    fid_test_all_d_s["far"].append(np.std(fid_test_all_d["far"]))

    fid_train_d_m["mfr"].append(np.mean(fid_train_d["mfr"]))
    fid_test_d_m["mfr"].append(np.mean(fid_test_d["mfr"]))
    fid_test_all_d_m["mfr"].append(np.mean(fid_test_all_d["mfr"]))

    fid_train_d_s["mfr"].append(np.std(fid_train_d["mfr"]))
    fid_test_d_s["mfr"].append(np.std(fid_test_d["mfr"]))
    fid_test_all_d_s["mfr"].append(np.std(fid_test_all_d["mfr"]))

# end for sel_feats


#%% convert the dictionaries into dataframes for later storage
fid_train_m = pd.DataFrame(fid_train_d_m)
fid_test_m = pd.DataFrame(fid_test_d_m)
fid_test_all_m = pd.DataFrame(fid_test_all_d_m)
fid_train_s = pd.DataFrame(fid_train_d_s)
fid_test_s = pd.DataFrame(fid_test_d_s)
fid_test_all_s = pd.DataFrame(fid_test_all_d_s)


#%% set the accuracies up in a dataframe and store them in a .csv files
os.makedirs(out_path, exist_ok=True)

out_df = pd.DataFrame({"feature":second_feat, "train":acc_train_m, "test":acc_test_m, "test_all":acc_test_all_m,
                       "train_std":acc_train_s, "test_std":acc_test_s, "test_all_std":acc_test_all_s})

out_df.to_csv(out_path + f"DT_Two_Feature_Accuracy-{first_feat}.csv")


#%% also store the other model metrics
fid_train_m.to_csv(out_path + f"DT_Two_Feature_Metrics_Train-{first_feat}.csv")
fid_test_m.to_csv(out_path + f"DT_Two_Feature_Metrics_Test-{first_feat}.csv")
fid_test_all_m.to_csv(out_path + f"DT_Two_Feature_Metrics_Test_All-{first_feat}.csv")
fid_train_s.to_csv(out_path + f"DT_Two_Feature_Metrics_Std_Train-{first_feat}.csv")
fid_test_s.to_csv(out_path + f"DT_Two_Feature_Metrics_Std_Test-{first_feat}.csv")
fid_test_all_s.to_csv(out_path + f"DT_Two_Feature_Metrics_Std_Test_All-{first_feat}.csv")

