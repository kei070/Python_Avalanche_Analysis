"""
Generate a logistic regression for the avalanche data and store it.
"""


#%% imports
import os
import sys
import glob
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from joblib import dump
from sklearn.model_selection import train_test_split


#%% set the height thresholds
h_low = 400
h_hi = 900


#%% set permutation
perm = 0


#%% handle the chosen permutation
p_dir = ""
if perm > -1:
    p_dir = f"Permutation_{perm:02}"
# end if


#%% set paths
data_path = f"/PATH_TO_AVALANCHE_PREDICTORS/Between{h_low}_and_{h_hi}m/{p_dir}/"
lr_path = f"/PATH_TO_LOGISTIC_MODEL/Auto_TrainTest_Split/Between{h_low}_and_{h_hi}m/{p_dir}/"
# --> lr_path will be generated


#%% read the region code
# reg_code = 3009  # Nord-Troms
# reg_code = 3010  # Lyngen
# reg_code = 3011  # Tromsoe
# reg_code = 3012  # Soer-Troms
# reg_code = 3013  # Indre Troms
# reg_code = "AllReg"
try:
    reg_code = sys.argv[1]
except:
    reg_code = "AllReg"
# end try except


#%% select features and exposure
try:
    feat1 = sys.argv[2]  # "s3"
    feat2 = sys.argv[3]  # "wspeed_max"
except:
    feat1 = "s3"
    feat2 = "wspeed_max"
# end try except

sel_feats = [feat1, feat2, "t5"]  # , "pdd"]  # "t_mean"]  # , "wind_direction"]


#%% load the data
train_df = pd.read_csv(glob.glob(data_path + f"Train_Features_Binary_Balanced_*_{reg_code}*.csv")[0])
test_df = pd.read_csv(glob.glob(data_path + f"Test_Features_Binary_Balanced_*_{reg_code}*.csv")[0])

train_all_df = pd.read_csv(glob.glob(data_path + f"Train_Features_Binary_All_*_{reg_code}*.csv")[0])
test_all_df = pd.read_csv(glob.glob(data_path + f"Test_Features_Binary_All_*_{reg_code}*.csv")[0])

odata_x = pd.concat([train_df[sel_feats], test_df[sel_feats]])
odata_y = pd.concat([train_df["train_y_balanced"], test_df["test_y_balanced"]])

odata_x_all = pd.concat([train_all_df[sel_feats], test_all_df[sel_feats]])
odata_y_all = pd.concat([train_all_df["train_y"], test_all_df["test_y"]])

train_x, test_x, train_y, test_y = train_test_split(odata_x, odata_y, test_size=0.33, shuffle=True)
train_x_all, test_x_all, train_y_all, test_y_all = train_test_split(odata_x, odata_y, test_size=0.33, shuffle=True)


#%% define the logistic regression model
max_iter = 1000
model = linear_model.LogisticRegression(max_iter=max_iter)
model.fit(train_x, train_y)


#%% make probability predictions with the model
pred_prob = model.predict(test_x)
pred_train_prob = model.predict(train_x)

pred = (model.predict(test_x) > 0.5).astype(int)  # binary prediction
pred_train = (model.predict(train_x) > 0.5).astype(int)  # binary prediction

pred_prob_test = model.predict(test_x)
pred_test = (model.predict(test_x) > 0.5).astype(int)  # binary prediction

pred_prob_test_all = model.predict(test_x_all)
pred_test_all = (model.predict(test_x_all) > 0.5).astype(int)  # binary prediction


#%% accuracy score
acc_train = accuracy_score(train_y, pred_train)
acc_test = accuracy_score(test_y, pred_test)
acc_test_all = accuracy_score(test_y_all, pred_test_all)


#%% print the accuracies
print("Accuracies:")
print(f"training data: {acc_train}")
print(f"test data:     {acc_test}")
print(f"all test data: {acc_test_all}")


#%% store the logistic regression model
os.makedirs(lr_path, exist_ok=True)
dump(model, f"{lr_path}/LogReg_{reg_code}_Between{h_low}_and_{h_hi}m_Preds-{'-'.join(sel_feats)}.joblib")

