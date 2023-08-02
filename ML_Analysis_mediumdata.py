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
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.ensemble import RandomTreesEmbedding
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
import json
from PersonalTraitGaze import LoadBlueRawData

class LoadMediumData(LoadBlueRawData):
    def __init__(self, datapath: str) -> None:
        super().__init__(datapath)
        data = self.get_data()
        data['Task Encoding'] = data.apply(lambda x: task_encoding(x['task']), axis=1)
        data['Similarity Encoding'] = data.apply(lambda x: similarity_encoding(x['level']), axis=1)
        self.data = data
        self.log_dict = self.take_log_from_cm()

    def take_log_from_cm(self):
        log_dict = {f"true_{i}":{f"pred_{j}":[] for j in range(1,14)} for i in range(1,14)}
        return log_dict
    
    def mlanalysis(self):
        x_data = self.take_x()
        y_data = self.take_y()

        warnings.filterwarnings('ignore')
        clf = RandomForestClassifier(random_state=0)  # Random Forest
        results = {}
        print("==================================")
        X = x_data.to_numpy()
        Y = y_data.to_numpy()
        kf = StratifiedKFold(n_splits=10)
        cm_added = np.zeros((13,13))
        f1 = []
        precision = []
        accuracy = []
        for _, (train, test) in enumerate(kf.split(X, Y)):
            clf2 = clone(clf)
            train_x = X[train]
            train_y = Y[train]
            test_x = X[test]
            test_y = Y[test]
            clf2.fit(train_x, train_y)
            predict_y = clf2.predict(test_x)
            
            cm_added = np.add(cm_added, confusion_matrix(test_y, predict_y))
            f1.append(f1_score(test_y, predict_y, average=None).tolist())
            precision.append(precision_score(test_y, predict_y, average=None).tolist())
            accuracy.append(accuracy_score(test_y, predict_y).tolist()) 

        results['rf'] = {'confusion_matrix': cm_added.tolist(), 'precision': precision, 'accuracy': accuracy, 'f1':f1}
            # visualize_cm(cm_added, clf_name=clf_names[n], path='ml-results/medium',title="stack1")
            # print(precision)
            # print(accuracy)
            # print(f1)
        return results    
    
    def get_log_dict(self):
        return self.log_dict

def visualize_cm(cm, clf_name:str, path:str, title:str):
    length_cm = cm.shape[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(list(range(1,length_cm+1))))
    disp.plot(include_values=False)
    plt.title(f'{title}_{clf_name}')
    plt.savefig(os.path.join(path,f'{title}_confusion_matrix_{clf_name}.png'))
    plt.close()

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



if __name__ == '__main__':
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'

    data = 'data/blue_medium_data_task1.csv'
    # data = 'data/blue_medium_data_task2.csv'
    mydata = LoadMediumData(data)
    # print(mydata.get_data().columns.to_list())
    # print(mydata.exceptlist)
    results = mydata.mlanalysis()
    print(results)

    # x = mydata.take_x()
    # y = mydata.take_y()

    # print(mydata.take_y())
    # results = mlanalysis(x, y)
    # print(results)
    with open("ml-results/medium/second-mediumresult.json", "w") as f:
        json.dump(results, f)
        # f.write(results)