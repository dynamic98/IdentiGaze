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


def task_encoding(task):
    task_dict = {'shape':1, 'size':2, 'orientation':3, 'hue':4, 'brightness':5}
    encoded_task = task_dict[task]
    return encoded_task

def similarity_encoding(similarity):
    if '#' in similarity:
        similarity = similarity.split('#')[1]
    similarity = list(map(int,similarity.split('-')))
    difference = abs(similarity[0]-similarity[1])
    return difference

def confusion_matrix_scorer(clf, X, y):
     y_pred = clf.predict(X)
     cm = confusion_matrix(y, y_pred)
     return {'tn': cm[0, 0], 'fp': cm[0, 1],
             'fn': cm[1, 0], 'tp': cm[1, 1]}

def decide_label(data, participant):
    if data == participant:
        return 1
    else:
        return 0

def euclidean_distance(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)



class LoadData:
    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        data['Task Encoding'] = data.apply(lambda x: task_encoding(x['task']), axis=1)
        data['Similarity Encoding'] = data.apply(lambda x: similarity_encoding(x['level']), axis=1)
        data = data.dropna(axis=0)
        data = data.sample(frac=1, random_state=1).reset_index(drop=True)
        self.decide_task(path)
        self.data = data
        self.exceptlist = ['participant', 'task', 'level']

    def set_domain_except(self, *args):
        addition_exceptlist = []
        frames = self.get_frame()
        for arg in args:
            if arg == 'x' or arg == 'y' or arg == 'd':
                addition_exceptlist = addition_exceptlist + [f"{arg}{i}" for i in range(1,frames+1)]
            else:
                if arg in self.data.columns:
                    addition_exceptlist.append(arg)
                else:
                    print(f"{arg} is not definded in columns of dataset")
        self.exceptlist = self.exceptlist + addition_exceptlist

    def take_x(self):
        x_data = self.data.loc[:,~self.data.columns.isin(self.exceptlist)]
        return x_data

    def take_y(self):
        return self.data['participant']
    
    def get_data(self):
        return self.data

    def get_frame(self):
        if self.task == 1:
            return 60
        elif self.task == 2:
            return 84

    def take_one(self, participant: int):
        data = self.data.copy()
        data['Label'] = data.apply(lambda x: decide_label(x['participant'], participant), axis=1)
        return data['Label']

    def decide_task(self, path:str):
        if path.endswith('task1.csv'):
            self.task = 1
        elif path.endswith('task2.csv'):
            self.task = 2
        else:
            raise Exception("Task name is undefined.")

    def calculate_distance(self):
        frames = self.get_frame()
        for i in range(1, frames+1):
            self.data[f"d{i}"] = self.data.apply(lambda x: euclidean_distance(x[f"x{i}"],x["cnt_x"],x[f"y{i}"],x["cnt_y"]), axis=1)

    def savedata(self, filename:str):
        self.data.to_csv(os.path.join("data", filename))


def mlanalysis(x_data, y_data):
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
        X = x_data.to_numpy()
        Y = y_data.to_numpy()
        kf = KFold(n_splits=10)
        class_amount = len(list(set(Y.tolist())))
        cm_added = np.zeros((class_amount,class_amount))
        f1 = []
        precision = []
        accuracy = []
        for i, (train, test) in enumerate(kf.split(X)):
            train_x = X[train]
            train_y = Y[train]
            test_x = X[test]
            test_y = Y[test]
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
    data = 'data/rawdata_distance_task1.csv'
    # data = 'data/rawdata_task2.csv'
    mydata = LoadData(data)
    # print(mydata.data['x1'])
    # mydata.calculate_distance()
    # mydata.savedata("rawdata_distance_task1.csv")

    mydata.set_domain_except('x','y')

    total_results = {}
    x = mydata.take_x()
    y = mydata.take_y()
    total_results = mlanalysis(x,y)

    # # y = mydata.take_y()
    # for i in range(1, 14):
    #     x = mydata.take_x()
    #     y = mydata.take_one(i)

    #     # print(mydata.take_y())
    #     results = mlanalysis(x, y)
    #     total_results[i] = results

    with open("ml-results/second-distance_result.json", "w") as f:
        json.dump(total_results, f)
        # f.write(results)