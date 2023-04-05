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
from sklearn import tree
from sklearn.ensemble import RandomTreesEmbedding
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
import json

def together(task, block='B-C'):
    path = 'data/AOI_HitScoring/features+target/'
    whole_df = pd.DataFrame()
    for file in os.listdir(path):
        if file == '.DS_Store':
            continue
        if file.endswith(f"{task}.csv"):
            df = pd.read_csv(os.path.join(path, file))
            sliced_df = df[df['Block']==block]
            whole_df = pd.concat([whole_df, sliced_df], ignore_index=True)
    return whole_df

def processing(data, participant):
    useful = ['User', 'Fixation Count', 'Saccade Count', 'Fixation Duration', 'Saccade Duration',
              'Average Pupil Left', 'Average Pupil Right', 'Saccade Velocity',
              'Saccade Amplitude', 'Target Hit', 'task', 'simiarity']
    

    data['simiarity'] = data.apply(lambda x: x['simiarity'][-3:], axis=1)
    data['Average Fixation Duration'] = data.apply(lambda x: average(x['Fixation Duration']), axis=1)
    data['Average Saccade Duration'] = data.apply(lambda x: average(x['Saccade Duration']), axis=1)
    data['Average Average Pupil Left'] = data.apply(lambda x: average(x['Average Pupil Left']), axis=1)
    data['Average Average Pupil right'] = data.apply(lambda x: average(x['Average Pupil Right']), axis=1)
    data['Average Saccade Velocity'] = data.apply(lambda x: average(x['Saccade Velocity']), axis=1)
    data['Average Saccade Amplitude'] = data.apply(lambda x: average(x['Saccade Amplitude']), axis=1)
    data['Task Encoding'] = data.apply(lambda x: encoding(x['task'], x['simiarity']), axis=1)
    data['Label'] = data.apply(lambda x: decide_label(x['User'], participant), axis=1)

    df = data.loc[:,['Fixation Count', 'Saccade Count', 'Average Fixation Duration', 'Average Saccade Duration', 
                     'Average Average Pupil Left', 'Average Average Pupil right', 'Average Saccade Velocity', 
                     'Average Saccade Amplitude', 'Task Encoding', 'Label']]
    return df

def sample_data(data):
    true_df = data[data['Label']==1]
    num_true = true_df.count()[0]
    false_df = data[data['Label']==0].sample(n=num_true, random_state=1)
    whole_df = pd.concat([true_df, false_df], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
    return whole_df

def average(datalist):
    data = datalist.strip("[]")
    if len(data)!= 0:
        data = list(map(float, data.split(",")))
        return sum(data)/len(data)
    else:
        return 0

def decide_label(data, participant):
    if data == participant:
        return 1
    else:
        return 0

def encoding(task, similarity):
    task_dict = {'shape':1, 'size':2, 'orientation':3, 'hue':4, 'brightness':5}
    similarity = list(map(int,similarity.split('-')))
    difference = abs(similarity[0]-similarity[1])
    encoded_task = task_dict[task]
    return encoded_task*10+difference

def take_x(data):
    domain = ['Fixation Count', 'Saccade Count', 'Average Fixation Duration', 'Average Saccade Duration', 
              'Average Average Pupil Left', 'Average Average Pupil right', 'Average Saccade Velocity', 
              'Average Saccade Amplitude', 'Task Encoding']
    return data.loc[:,domain]

def take_y(data):
    return data.loc[:,'Label']


def mlanalysis(data):
    warnings.filterwarnings('ignore')
    base_model = DummyClassifier(strategy='most_frequent', random_state=0)  # ZeroR
    svc_model = SVC()  # SVM
    knn_model = KNeighborsClassifier()  # k-Nearest Neighbors
    lr_model = LogisticRegression(C=1, random_state=0)  # Logistic Regression
    dt_model = DecisionTreeClassifier()  # Decision Tree
    rf_model = RandomForestClassifier(random_state=0)  # Random Forest
    ab_model = AdaBoostClassifier()  # AdaBoost
    nb_model = BernoulliNB()  # Naive Bayse
    
    clf_names = ['ZeroR', 'DecisionTree', 'kNN', 'NaiveBayes', 'SVM', 'LogisticRegression', 'AdaBoost', 'RandomForest']
    classifiers = [base_model, dt_model, knn_model, nb_model, svc_model, lr_model, ab_model, rf_model]
    results = {}
    for n, clf in enumerate(classifiers):
        # print("==================================")
        print(clf_names[n])
        clf2 = clone(clf)
        X = take_x(data).to_numpy()
        Y = take_y(data).to_numpy()
        scoring = ['accuracy', 'f1']
        scores = cross_validate(clf2, X, Y, cv=10, scoring=scoring)
        f1_value = scores['test_f1']
        acc_value = scores['test_accuracy']
        # print(f1_value)
        average_acc = sum(acc_value)/len(acc_value)
        average_f1 = sum(f1_value)/len(f1_value)
        # print("Cross-validation of F1", sum(f1_value)/len(f1_value))
        results[clf_names[n]] = {'average f1': average_f1, 'average acc': average_acc}
    
    return results
        


if __name__ == '__main__':
    # path = 'data/AOI_HitScoring/features+target/'
    # together(0.7)
    participants = ['chungha','dongik','eunhye','In-Taek','jooyeong','juchanseo','junryeol','juyeon',
                    'myounghun','songmin','sooyeon','woojinkang','yeogyeong']
    tasks = [0.5, 0.7]
    results = {}
    for task in tasks:
        print("------------------------")
        print(task)
        print("------------------------")
        data = together(task)
        results[task]={}
        for participant in participants:
            print(participant)
            print("------------------------")
            train_data = sample_data(processing(data, participant))
            result = mlanalysis(train_data)
            results[task][participant] = result
    with open("ml-results/first-result.json", "w") as f:
        json.dump(results, f)

    # data = together(0.7)
    # # print(data.columns)
    # train_data = sample_data(processing(data, 'chungha'))
    # print(take_y(train_data))
