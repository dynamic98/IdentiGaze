import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

class ML_Result:
    def __init__(self, file) -> None:
        path = 'ml-results'
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
        self.data = data

    def get_data(self):
        return self.data
    
    def plot(self, task, participant):
        clf_names = ['ZeroR', 'DecisionTree', 'kNN', 'NaiveBayes', 'SVM', 'LogisticRegression', 'AdaBoost', 'RandomForest']
        this_data = self.data[task][participant]
        plot_data = pd.DataFrame()
        for clf in clf_names:
            data = pd.DataFrame({"Task":task,"Participant":participant,"Model": clf, "Mean Accuracy":this_data[clf]['average acc']}, index=[0])
            # print(data)
            plot_data = pd.concat([plot_data, data], ignore_index=True)
        
        # print(plot_data)
        plt.figure(figsize=(10,5))
        plots = sns.barplot(data = plot_data, x="Model", y="Mean Accuracy")
        plots.set_title(f"{participant} in task time : {task}")
        plots.set_ylim(0,1)
        plt.show()

if __name__ == '__main__':
    myML = ML_Result('first-result.json')
    # print(myML.get_data())
    participants = ['chungha','dongik','eunhye','In-Taek','jooyeong','juchanseo','junryeol','juyeon',
                    'myounghun','songmin','sooyeon','woojinkang','yeogyeong']
    tasks = ['0.5', '0.7']
    for task in tasks:
        for participant in participants:
            myML.plot(task, participant)