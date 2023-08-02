import os
import pandas as pd
import numpy as np
from functools import reduce
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from ML_util import *
from ML_Analysis_OptimalStimuli import LoadSelectiveData


def ml_train(data, boolGaze=None):
    if boolGaze == None:
        boolGazeData = data
    else:
        boolGazeData = data[data['gaze_hit']==boolGaze]
    x_train_data = thisData.take_x(boolGazeData).to_numpy()
    y_train_data = thisData.take_y(boolGazeData).to_numpy()
    # warnings.filterwarnings('ignore')
    rf_model = RandomForestClassifier(random_state=0)  # Random Forest
    rf_model.fit(x_train_data, y_train_data)
    return rf_model

def ml_validate(model, data, boolGaze=None):
    if boolGaze == None:
        boolGazeData = data
    else:
        boolGazeData = data[data['gaze_hit']==boolGaze]
    x_valid_data = thisData.take_x(boolGazeData).to_numpy()
    y_valid_data = thisData.take_y(boolGazeData).to_numpy()
    predict_y = model.predict(x_valid_data)
    thisConfusionMatrix = confusion_matrix(y_valid_data, predict_y)
    visualize_cm(thisConfusionMatrix, 'RF', ' ')
    for i in range(13):
        # print("====================")
    #     thisConfusionMatrix = confusion_matrix(y_valid_data, predict_y)
    #     print(f"FAR: {BiometricEvaluation(thisConfusionMatrix, i, 'FAR')}")
        print(f" participant {i} FRR: {BiometricEvaluation(thisConfusionMatrix, i, 'FRR')}")

def ml_test_baseline(model, data, boolGaze=None):
    if boolGaze == None:
        boolGazeData = data
    else:
        boolGazeData = data[data['gaze_hit']==boolGaze]
    total_x = thisData.take_x(boolGazeData).to_numpy()
    total_y = thisData.take_y(boolGazeData).to_numpy()
    stack_index, stack_y = stack_ydata_from_same_combinations(total_y, 3)
    results = latefusion(model, total_x, stack_index, stack_y)
    return results

path = "data/BlueMediumRarePupil_task1-1.csv"
thisData = LoadSelectiveData(path)

train_data_ratio = 0.6
valid_data_ratio = 0.2
test_data_ratio = 0.2

trb, vdb, teb = thisData.split_data(train_data_ratio, valid_data_ratio, test_data_ratio)

# nohit_model = ml_train(thisData.get_data().iloc[trb], boolGaze=0)
# onhit_model = ml_train(thisData.get_data().iloc[trb], boolGaze=1)
total_model = ml_train(thisData.get_data().iloc[trb])

results = ml_test_baseline(total_model, thisData.get_data().iloc[teb])
thisConfusionMatrix = results["cm_multiply"]
visualize_cm(normalize_cm(thisConfusionMatrix), 'rf', 'total')

# for i in range(13):
#     print("====================")
#     print(f"participant {i}")
#     print(f"Accuracy for verification: {accuracyMeasurementForVerification(thisConfusionMatrix, i)}")
#     print(f"FAR: {BiometricEvaluation(thisConfusionMatrix, i, 'FAR')}")
#     print(f"FRR: {BiometricEvaluation(thisConfusionMatrix, i, 'FRR')}")
