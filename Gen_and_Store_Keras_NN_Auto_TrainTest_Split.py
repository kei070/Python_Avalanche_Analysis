"""
Generate a neural network for the avalanche data with Keras and store it.

NOTE for the training on the date from all regions: The file from all regions is a concatenation of the files from each
individual region (i.e., the data is averaged only over the individual regions, respectively, and NOT over all regions
combined). That means that the models is successively trained on all regions.
"""


#%% imports
import os
import sys
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import HeNormal
from sklearn.model_selection import train_test_split


#%% which permuation? (random; for now choose a number between 0 and 5)
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


#%% generate the path to store the neural network
nn_path = f"/PATH_TO_NEURAL_NETWORK/Between{h_low}_and_{h_hi}m/{p_dir}/"  # --> will be generated
os.makedirs(nn_path, exist_ok=True)


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

sel_feats = [feat1, feat2, "t5"]  # , "pdd"]  # "t_mean"]  # , "t_max"]  # , "wind_direction"]


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


#%% define the keras model
acc_test_l = []
acc_test_all_l = []
acc_train_l = []

# define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
for rounds in np.arange(3):
    model = Sequential()  # build the model sequentially, kind of 'by hand'

    # first layer
    model.add(Dense(8, input_shape=(len(sel_feats),), activation='relu', kernel_initializer=HeNormal()))
    Dropout(0.3)
    # first hidden layer
    model.add(Dense(12, activation='relu', kernel_initializer=HeNormal()))
    Dropout(0.3)
    # model.add(Dense(20, activation='relu'))

    # model.add(Dense(20, activation='relu'))

    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(8, activation='relu'))

    # the output layer has one node and uses the sigmoid activation function
    model.add(Dense(1, activation='sigmoid'))

    # for multiclass problems: apparently you need as many nodes on the output layer as you have classes; i.e. for the
    # avalanche risk levels where we have 5 risk levels we need 5 nodes in the output layer
    # further for multiclass problems: to be able to predict multiple classes the output layer must use the "softmax"
    #                                  activation function


    #% compile the keras model

    # we use the loss function "binary_crossentropy" which is suited for binary classification
    # we further use the "Adam" algorithm to implement the stochastic gradient descent
    # as a metric to evaluate the model we store the "accuracy" score
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # --> for a multiclass problem use loss='categorical_crossentropy'


    #% fit the keras model on the dataset

    # the batch_size argument set after how many sample the weights are updated
    # one epoch is completed after the whole dataset has been used once (i.e., all samples have been used once)
    # note that this means on epoch usually contains at least one or more batches
    # print("Training the neural network...")
    history = model.fit(train_x, train_y, epochs=400, batch_size=20, verbose=0, callbacks=[early_stopping],
                        shuffle=True, validation_data=(test_x, test_y))
    # print("...training done.")


    #% evaluate the keras model
    # the first value is the "loss" of the model which we (apparently) are not interested in
    _, acc_train = model.evaluate(train_x, train_y, verbose=0)
    print(f'Accuracy balanced training data round {rounds+1}: {(acc_train * 100)} %')

    _, acc_test = model.evaluate(test_x, test_y, verbose=0)
    print(f'Accuracy balanced test data round {rounds+1}: {(acc_test * 100)} %')

    _, acc_test_all = model.evaluate(test_x_all, test_y_all, verbose=0)
    print(f'Accuracy all test data round {rounds+1}: {(acc_test_all * 100)} %\n')

    # add values to the lists
    acc_test_l.append(acc_test)
    acc_test_all_l.append(acc_test_all)
    acc_train_l.append(acc_train)

    # store the model
    model.save(nn_path + f"NeuralNet_{reg_code}_Between{h_low}_and_{h_hi}m_Preds-{'-'.join(sel_feats)}_{rounds}.keras")

# end for rounds


#%% convert lists to arrays
acc_test_a = np.array(acc_test_l)
acc_test_all_a = np.array(acc_test_all_l)
acc_train_a = np.array(acc_train_l)


#%% aggregate the accuracy results in a dataframe
# pd.DataFrame({"acc_test":acc_test_a, "acc_test_all":acc_test_all_a, "acc_train":acc_train_a}).to_csv(out_path +
#              f"Accuracies_NN_{reg_code}_{'-'.join(sel_feats)}.csv")



