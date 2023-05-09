import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.svm import SVC
import json
import numpy as np
import os
import pandas as pd


def visualize_cm(cm, clf, task:str):
    length_cm = cm.shape[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(list(range(1,length_cm+1))))
    disp.plot(include_values=False)
    plt.title(clf)
    plt.savefig(os.path.join('ml-results',f'{task}_confusion_matrix_{clf}.png'))
    plt.close()


def data_plot(data, metric, task:str):
    # print(participant)
    clf_names = ['ZeroR', 'DecisionTree', 'kNN', 'NaiveBayes', 'SVM', 'LogisticRegression', 'AdaBoost', 'RandomForest']
    plot_data = pd.DataFrame()
    for clf in clf_names:
        df_data = pd.DataFrame({"Model": clf, f"Mean {metric}":np.average(np.array(data[clf][f'{metric}']))}, index=[0])
        # print(data)
        plot_data = pd.concat([plot_data, df_data], ignore_index=True)
        participant = "Whole"
    zeror_value = plot_data.loc[0,f'Mean {metric}']
    plt.figure(figsize=(13,5))
    plots = sns.barplot(data = plot_data, x="Model", y=f"Mean {metric}")
    plots.set_title(f"{metric}")
    plots.set_ylim(0,1)
    plots.hlines(zeror_value, 0, 7, colors='black', linestyles="--")
    for p in plots.patches:
        plots.text(p.get_x() + p.get_width() / 2, # x 좌표
                p.get_y() + p.get_height() , # y 좌표
                f"{p.get_height():.2f}", # 값
                ha='center') # 가운데 정렬
    plt.savefig(os.path.join('ml-results', f"{task}_{metric}.png"))
    plt.close()

# def visualize_precision(data):



if __name__ == '__main__':
    with open("ml-results/second-distance_result.json", "r") as f:
        data = json.load(f)
    clf_names = ['ZeroR', 'DecisionTree', 'kNN', 'NaiveBayes', 'SVM', 'LogisticRegression', 'AdaBoost', 'RandomForest']
    domain = ['confusion_matrix', 'precision', 'accuracy', 'f1']
    data_plot(data, 'precision', 'distance_2')
    data_plot(data, 'accuracy', 'distance_2')
    data_plot(data, 'f1', 'distance_2')
    for clf in clf_names:
        cm = np.array(data[clf]['confusion_matrix'])
        # print(np.average(precision))
        visualize_cm(cm, clf, 'distance_2')
        

