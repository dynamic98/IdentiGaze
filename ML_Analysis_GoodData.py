import os
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.ensemble import RandomTreesEmbedding
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
import seaborn as sns
import json
from PersonalTraitGaze import LoadBlueRawData


class TakeLogFromCM(LoadBlueRawData):
    def __init__(self, datapath: str) -> None:
        super().__init__(datapath)
        self.log_dict = self.take_log_from_cm()

    def take_log_from_cm(self):
        log_dict = {f"true_{i}":{f"pred_{j}":[] for j in range(1,14)} for i in range(1,14)}
        return log_dict

    def mlanalysis(self, x_data, y_data):
        warnings.filterwarnings('ignore')
        rf_model = RandomForestClassifier(random_state=0)  # Random Forest
        
        clf2 = clone(rf_model)
        X = x_data.to_numpy()
        Y = y_data.to_numpy()
        kf = StratifiedKFold(n_splits=10)

        for _, (train, test) in enumerate(kf.split(X,Y)):
            train_x = X[train]
            train_y = Y[train]
            test_x = X[test]
            test_y = Y[test]
            clf2.fit(train_x, train_y)
            predict_y = clf2.predict(test_x)
            for i in tqdm(range(len(test))):
                this_predict = predict_y[i]
                this_gt = test_y[i]
                this_index = self.get_indexlist()[test[i]]
                this_meta = self.take_meta(this_index)
                self.log_dict[f"true_{this_gt}"][f"pred_{this_predict}"].append(this_meta)
    
    def get_log_dict(self):
        return self.log_dict


if __name__=='__main__':
    # print(take_log_from_cm())
    path = 'data/blue_mediumdata_task1.csv'
    # path = 'data/blue_rawdata_task2.csv'
    # mydata = MakeDataDouble(path, 1)
    mydata = TakeLogFromCM(path)
    x = mydata.take_x()
    y = mydata.take_y()
    mydata.mlanalysis(x,y)
    with open("ml-results/medium/RF_CM_meta.json", "w") as f:
        json.dump(mydata.get_log_dict(), f, default=str)