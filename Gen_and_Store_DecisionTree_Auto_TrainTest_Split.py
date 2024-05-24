"""
Generate a neural network for the avalanche data with Keras and store it.
"""


#%% imports
import os
import sys
import glob
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn import tree as sktree
from sklearn.model_selection import train_test_split


#%% set permutation
perm = 0


#%% set the height thresholds
h_low = 400
h_hi = 900


#%% handle the chosen permutation
p_dir = ""
if perm > -1:
    p_dir = f"Permutation_{perm:02}"
# end if


#%% set paths
data_path = f"/PATH_TO_AVALANCHE_PREDICTORS/Between{h_low}_and_{h_hi}m/{p_dir}/"
dt_path = f"/PATH_TO_DECISION_TREE/Between{h_low}_and_{h_hi}m/{p_dir}/"
# --> dt_path will be generated


#%% read the region code
# reg_code = 3009  # Nord-Troms
# reg_code = 3010  # Lyngen
# reg_code = 3011  # Tromsoe
# reg_code = 3012  # Soer-Troms
# reg_code = 3013  # Indre Troms
reg_code = "AllReg"  #  sys.argv[1]


#%% select features and exposure
feat1 = "s3"  # sys.argv[2]  # "s3"
feat2 = "wspeed_max"  # sys.argv[3]  # "wspeed_max"

sel_feats = [feat1, feat2]  # , "pdd"]  # , "t5"]  # , "wind_direction"]

pl_feats = {feat1:"s3", feat2:"w_max"}  #, "pdd":"pdd"} # , "t5":"t5"}  # "w_mean"}


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


#%% define the model

# set decision tree parameters
max_depth = len(sel_feats)
min_leaf_samp = 5

acc_test_l = []
acc_test_all_l = []
acc_train_l = []


# define the decision tree model
model = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, min_samples_leaf=min_leaf_samp)
model.fit(train_x, train_y)

# perform the prediction
pred_prob = model.predict(test_x)
pred_train_prob = model.predict(train_x)

pred = (model.predict(test_x) > 0.5).astype(int)  # binary prediction
pred_train = (model.predict(train_x) > 0.5).astype(int)  # binary prediction

pred_prob_test = model.predict(test_x)
pred_test = (model.predict(test_x) > 0.5).astype(int)  # binary prediction

pred_prob_test_all = model.predict(test_x_all)
pred_test_all = (model.predict(test_x_all) > 0.5).astype(int)  # binary prediction


#%% evaluate the model
acc_train = accuracy_score(train_y, pred_train)
print(f'Accuracy balanced training data: {(acc_train * 100)} %')

acc_test = accuracy_score(test_y, pred_test)
print(f'Accuracy balanced test data: {(acc_test * 100)} %')

acc_test_all = accuracy_score(test_y_all, pred_test_all)
print(f'Accuracy all test data: {(acc_test_all * 100)} %\n')


#%% store the model
os.makedirs(dt_path, exist_ok=True)
dump(model, f"{dt_path}/DT_{reg_code}_Between{h_low}_and_{h_hi}m_Preds-{'-'.join(sel_feats)}.joblib")


#%% store the visual representation of the DT
sktree.export_graphviz(model,
                       out_file=dt_path + f"dtree_{reg_code}_Between{h_low}_and_{h_hi}m_" +
                       f"{max_depth}depth_{min_leaf_samp}leaf_Preds-{'-'.join(sel_feats)}.dot",
                       feature_names=list(pl_feats.values()),
                       class_names=["0", "1"], label="all",
                       rounded=True, filled=True)


#%%
# sktree.plot_tree(model, feature_names=list(pl_feats.values()), class_names=["0", "1"], filled=True)