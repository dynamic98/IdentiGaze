from ast import Load
import os
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.model_selection import KFold, cross_validate, cross_val_predict, LeaveOneGroupOut, GroupKFold, StratifiedGroupKFold, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score
from sklearn import tree
from sklearn.ensemble import RandomTreesEmbedding
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
import json


class LoadEMVIC:
    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        self.data = data
        self.exceptlist = ['class']

    def set_domain_except(self, *args):
        self.exceptlist = self.exceptlist + list(args)

    def take_x(self):
        x_data = self.data.loc[:,~self.data.columns.isin(self.exceptlist)]
        return x_data

    def take_y(self):
        return self.data['class']
    
    def get_data(self):
        return self.data

def mlanalysis(train: LoadEMVIC, test: LoadEMVIC):
    warnings.filterwarnings('ignore')
    base_model = DummyClassifier(strategy='most_frequent', random_state=0)  # ZeroR
    svc_model = SVC()  # SVM
    knn_model = KNeighborsClassifier()  # k-Nearest Neighbors
    lr_model = LogisticRegression(C=1, random_state=0)  # Logistic Regression
    dt_model = DecisionTreeClassifier()  # Decision Tree
    rf_model = RandomForestClassifier(random_state=0)  # Random Forest
    ab_model = AdaBoostClassifier()  # AdaBoost
    nb_model = GaussianNB()  # Naive Bayse
    
    clf_names = ['ZeroR', 'DecisionTree', 'kNN', 'NaiveBayes', 'SVM', 'LogisticRegression', 'AdaBoost', 'RandomForest']
    classifiers = [base_model, dt_model, knn_model, nb_model, svc_model, lr_model, ab_model, rf_model]
    results = {}
    for n, clf in enumerate(classifiers):
        print("==================================")
        print(clf_names[n])
        clf2 = clone(clf)
        kf = KFold(n_splits=10)

        f1 = []
        precision = []
        accuracy = []

        train_x = train.take_x()
        train_y = train.take_y()
        test_x = test.take_x()
        test_y = test.take_y()

        class_amount = len(list(set(train_y.tolist())))
        cm_added = np.zeros((class_amount,class_amount))

        clf2.fit(train_x, train_y)
        y_pred = clf2.predict(test_x)
        y_true = test_y
        cm_added = np.add(cm_added, confusion_matrix(y_true, y_pred))
        f1.append(f1_score(y_true, y_pred, average=None).tolist())
        precision.append(precision_score(y_true, y_pred, average=None).tolist())
        accuracy.append(accuracy_score(y_true, y_pred).tolist()) 
            # print(confusion_matrix(y_true, y_pred))
        results[clf_names[n]] = {'confusion_matrix': cm_added.tolist(), 'precision': precision, 'accuracy': accuracy, 'f1':f1}
        print(cm_added)
        print(precision)
        print(accuracy)
        print(f1)

    return results    


if __name__ == '__main__':
    traindata = 'data/emvic/train.csv'
    testdata = 'data/emvic/test.csv'
    # data = 'data/rawdata_task2.csv'
    mytraindata = LoadEMVIC(traindata)
    mytestdata = LoadEMVIC(traindata)
    results = mlanalysis(mytraindata, mytestdata)

    with open("ml-results/first-emvic_result.json", "w") as f:
        json.dump(results, f)
        # f.write(results)