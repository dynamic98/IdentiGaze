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
import matplotlib.pyplot as plt
import seaborn as sns
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

class LoadData:
    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        data['Task Encoding'] = data.apply(lambda x: task_encoding(x['task']), axis=1)
        data['Similarity Encoding'] = data.apply(lambda x: similarity_encoding(x['level']), axis=1)
        data = data.dropna(axis=0)
        data = data.sample(frac=1, random_state=1).reset_index(drop=True)

        self.data = data
        self.exceptlist = ['bbx_x1','bbx_x2','bbx_y1','bbx_y2','task','level','participant','shape_target','shape_distractor',
                           'set_size','target_size','distractor_size','target_color_b','target_color_g','target_color_r',
                           'distractor_color_b','distractor_color_g','distractor_color_r','target_orientation','distractor_orientation']

    def set_domain_except(self, *args):
        self.exceptlist = self.exceptlist + list(args)

    def take_x(self):
        x_data = self.data.loc[:,~self.data.columns.isin(self.exceptlist)]
        return x_data

    def take_y(self):
        return self.data['participant']
    
    def get_data(self):
        return self.data

class MakeDataDouble:
    def __init__(self, path, amount, random_state=1) -> None:
        data = pd.read_csv(path)
        data['Task Encoding'] = data.apply(lambda x: task_encoding(x['task']), axis=1)
        data['Similarity Encoding'] = data.apply(lambda x: similarity_encoding(x['level']), axis=1)
        data = data.dropna(axis=0)
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self.exceptlist = ['bbx_x1','bbx_x2','bbx_y1','bbx_y2','task','level','participant','shape_target','shape_distractor',
                           'set_size','target_size','distractor_size','target_color_b','target_color_g','target_color_r',
                           'distractor_color_b','distractor_color_g','distractor_color_r','target_orientation','distractor_orientation']
        self.data = self.stack_rows(data, amount)
        self.data = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    def set_domain_except(self, *args):
        self.exceptlist = self.exceptlist + list(args)

    def take_x(self):
        x_data = self.data.loc[:,~self.data.columns.isin(self.exceptlist)]
        return x_data

    def take_y(self):
        return self.data['participant']
    
    def stack_rows(self, data, amount):
        participants = list(range(1,14))
        total_dataframe = pd.DataFrame()
        for participant in participants:
            this_data = data.loc[data['participant']==participant].reset_index()
            total_amount = this_data.index.size
            block_size = total_amount//amount
            dataframe_block = pd.DataFrame()
            for i in range(amount):
                small_block = this_data.iloc[i*block_size:(i+1)*block_size].reset_index()
                small_block = small_block.loc[:,~small_block.columns.isin(self.exceptlist)]
                dataframe_block = pd.concat([dataframe_block, small_block], axis=1)
            dataframe_block['participant'] = participant
            total_dataframe = pd.concat([total_dataframe, dataframe_block], ignore_index=True)
        return total_dataframe
        
    def get_data(self):
        return self.data

def mlanalysis_stacked(x_data, y_data, stack):
    warnings.filterwarnings('ignore')
    base_model = DummyClassifier(strategy='most_frequent', random_state=0)  # ZeroR
    svc_model = SVC(probability=True)  # SVM
    knn_model = KNeighborsClassifier()  # k-Nearest Neighbors
    lr_model = LogisticRegression(C=1, random_state=0)  # Logistic Regression
    dt_model = DecisionTreeClassifier()  # Decision Tree
    rf_model = RandomForestClassifier(random_state=0)  # Random Forest
    ab_model = AdaBoostClassifier()  # AdaBoost
    nb_model = GaussianNB()  # Naive Bayse
    
    # clf_names = ['ZeroR', 'DecisionTree', 'kNN', 'NaiveBayes', 'SVM', 'LogisticRegression', 'AdaBoost', 'RandomForest']
    # classifiers = [base_model, dt_model, knn_model, nb_model, svc_model, lr_model, ab_model, rf_model]

    clf_names = ['ZeroR', 'DecisionTree', 'kNN', 'NaiveBayes', 'SVM', 'AdaBoost', 'RandomForest']
    classifiers = [base_model, dt_model, knn_model, nb_model, svc_model, ab_model, rf_model]

    # results = {}
    for n, clf in enumerate(classifiers):
        clf2 = clone(clf)
        X = x_data.to_numpy()
        Y = y_data.to_numpy()
        kf = StratifiedKFold(n_splits=10)
        cm_added_plus = np.zeros((13,13))
        cm_added_multiply = np.zeros((13,13))
        cm_added_max = np.zeros((13,13))
        print(clf_names[n])
        acc = {'plus':[], 'multi':[],'max':[]}
        precision = {'plus':[], 'multi':[],'max':[]}
        f1 = {'plus':[], 'multi':[],'max':[]}

        for i, (train, test) in enumerate(kf.split(X,Y)):
            train_x = X[train]
            train_y = Y[train]
            test_x = X[test]
            test_y = Y[test]
            clf2.fit(train_x, train_y)
            # print(test_x[0])
            results = latefusion(clf2, test_x, test_y, stack)
            cm_added_plus = np.add(cm_added_plus, results['cm_plus'])
            cm_added_multiply = np.add(cm_added_multiply, results['cm_multiply'])
            cm_added_max = np.add(cm_added_max, results['cm_max'])
            acc['plus'].append(results['acc_plus'])
            acc['multi'].append(results['acc_multiply'])
            acc['max'].append(results['acc_max'])
            precision['plus'].append(results['precision_plus'])
            precision['multi'].append(results['precision_multiply'])
            precision['max'].append(results['precision_max'])
            f1['plus'].append(results['f1_plus'])
            f1['multi'].append(results['f1_multiply'])
            f1['max'].append(results['f1_max'])

        visualize_cm(cm_added_plus, clf_name=clf_names[n], path='ml-results/latefusion',title='plus')
        visualize_cm(cm_added_multiply, clf_name=clf_names[n], path='ml-results/latefusion',title='multiply')
        visualize_cm(cm_added_max, clf_name=clf_names[n], path='ml-results/latefusion',title='max')
        print(f"Accuracy-> plus: {calc_mean(acc['plus'])}, multiply: {calc_mean(acc['multi'])}, max: {calc_mean(acc['max'])}")
        print(f"Precision-> plus: {calc_mean(precision['plus'])}, multiply: {calc_mean(precision['multi'])}, max: {calc_mean(precision['max'])}")
        print(f"f1 score-> plus: {calc_mean(f1['plus'])}, multiply: {calc_mean(f1['multi'])}, max: {calc_mean(f1['max'])}")

def calc_mean(value_list:list):
    return round((sum(value_list)/len(value_list)), 2)

def visualize_cm(cm, clf_name:str, path:str, title:str):
    length_cm = cm.shape[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(list(range(1,length_cm+1))))
    disp.plot(include_values=False)
    plt.title(f'{title}_{clf_name}')
    plt.savefig(os.path.join(path,f'{title}_confusion_matrix_{clf_name}.png'))
    plt.close()

def latefusion(clf, x_data, y_data, stack):
    label_dict = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[]}
    for index,label in enumerate(y_data):
        label_dict[label].append(index)
    
    # stack_x = [[] for i in range(stack)]
    stack_y = []
    stack_index = [[] for i in range(stack)]

    for label in label_dict:
        index_list = label_dict[label]
        stack_size = len(index_list)//stack
        for i in range(stack):
            stack_index[i].extend(index_list[i*stack_size:(i+1)*stack_size])
        stack_y.extend([label for k in range(stack_size)])

    stack_index = np.array(stack_index)
    _, total_stack_size = stack_index.shape
    
    plus_predict = []
    multiply_predict = []
    max_predict = []
    probabilities = clf.predict_proba(x_data)
    
    for i in range(total_stack_size):
        # probabilities = clf.predict_proba(x_data[stack_index[:,i]])
        plus = np.zeros((1,13))
        multiply = np.ones((1,13))
        maximum = [0 for k in range(13)]
        for j in range(stack):
            this_probabilities = probabilities[stack_index[j,i]]
            plus = plus + np.array(this_probabilities)
            multiply = multiply * np.array(this_probabilities)
            maximum = list(map(max, maximum, this_probabilities))
        plus_predict.append(np.argmax(plus)+1)
        multiply_predict.append(np.argmax(multiply)+1)
        max_predict.append(np.argmax(maximum)+1)

    cm_plus = confusion_matrix(stack_y, plus_predict)
    cm_multiply = confusion_matrix(stack_y, multiply_predict)
    cm_max = confusion_matrix(stack_y, max_predict)

    acc_plus = accuracy_score(stack_y, plus_predict)
    acc_multiply = accuracy_score(stack_y, multiply_predict)
    acc_max = accuracy_score(stack_y, max_predict)

    precision_plus = precision_score(stack_y, plus_predict, average='macro')
    precision_multiply = precision_score(stack_y, multiply_predict, average='macro')
    precision_max = precision_score(stack_y, max_predict, average='macro')

    f1_plus = f1_score(stack_y, plus_predict, average='macro')
    f1_multiply = f1_score(stack_y, multiply_predict, average='macro')
    f1_max = f1_score(stack_y, max_predict, average='macro')

    results = {
        'cm_plus':cm_plus, 'cm_multiply':cm_multiply, 'cm_max':cm_max,
        'acc_plus':acc_plus, 'acc_multiply':acc_multiply, 'acc_max':acc_max,
        'precision_plus':precision_plus, 'precision_multiply':precision_multiply, 'precision_max':precision_max,
        'f1_plus':f1_plus, 'f1_multiply':f1_multiply, 'f1_max':f1_max
    }

    return results

if __name__ == '__main__':
    path = 'data/blue_rawdata_task1.csv'
    # path = 'data/blue_rawdata_task2.csv'
    # mydata = MakeDataDouble(path, 1)
    mydata = LoadData(path)
    mydata.get_data()

    x = mydata.take_x()
    y = mydata.take_y()
    mlanalysis_stacked(x,y, stack=4)
