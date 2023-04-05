import os
from tkinter.filedialog import test
import numpy as np
import pandas as pd


class SingleStimuliData:
    def __init__(self, datapath, filename) -> None:
        self.dataname = filename
        self.datapath = datapath
        df = pd.read_csv(os.path.join(self.datapath, filename), sep='\t', low_memory=False)
        starttime = self.task_start_time(df)
        self.df = df.iloc[starttime:-1].copy()
        self.task = df.loc[starttime]['Event value']
        self.participant = df['Participant name'][0]
        self.date = df['Recording date'][0]
        self.time = df['Recording start time'][0].split('.')[0]

    def get_df(self):
        return self.df

    def save_df(self, abc_list):
        savepath = os.path.join(os.getcwd(), 'data', 'data_processed')
        if not os.path.exists(os.path.join(savepath, self.participant)):
            os.makedirs(os.path.join(savepath, self.participant))
        this_path = os.path.join(savepath, self.participant,f"{self.date}_{self.time}_task{self.task}")
        if not os.path.exists(this_path):
            os.makedirs(this_path)
            for i in range(100):
                start, end = abc_list[i]
                df = self.df.loc[start:end+1]
                if len(df.index.to_list())==0:
                    print(self.dataname, self.participant, self.date, self.time)
                df.to_csv(os.path.join(this_path, f"{i+1}.tsv"), sep='\t', index=False)
        else:
            raise Exception("Path is already existed. f{this_path}")
    
    def analysis_df(self, abc_list):
        # print(len(abc_list))
        print(self.dataname, self.participant, self.date, self.time)
        print(self.get_df().index)
        for i in range(100):
            start, end = abc_list[i]
            df = self.get_df().loc[start:end+1]
            if len(df.index.to_list())==0:
                print(start, end)
                



    def get_stimuli_list(self):
        a_list = self.slice_stimuli('a')
        b_list = self.slice_stimuli('b')
        c_list = self.slice_stimuli('c')

        if len(a_list) != 100:
            if a_list[1]<b_list[0]:
                del a_list[0]

        if len(c_list) != 100:
            if c_list[0]<a_list[0]:
                del c_list[0]

        if len(a_list)!= 100 or len(b_list)!=100 or len(c_list)!=100:
            raise Exception("Event size error")

        abc_list = []
        for i in range(100):
            if i!=99:
                abc_list.append([a_list[i], a_list[i+1]])
            else:
                abc_list.append([a_list[i], c_list[i]])

        return abc_list

    def slice_stimuli(self, keyboardevent):
        df_event = self.df[self.df['Event']=='KeyboardEvent'].copy()
        df_event = df_event[df_event['Event value']==keyboardevent]
        event_list = df_event.index.to_list()
        stimuli_list = []
        for event in event_list:
            if len(stimuli_list)>0 and (event == stimuli_list[-1]+1 or event == stimuli_list[-1]+2):
                del stimuli_list[-1]
                stimuli_list.append(event)
            else:
                stimuli_list.append(event)
        return stimuli_list
        # print(df_event)

    def task_start_time(self, df):
        starttime = df[df['Event value'].isin(['1','2'])].index.to_list()
        if len(starttime)==1:
            return starttime[0]
        else:
            # print(f"Caution! {self.dataname} has a multiple task name. please check it.")
            return starttime[-1]


class MultiStimuliData:
    def __init__(self, filename) -> None:
        self.dataname = filename
        self.datapath = os.path.join(os.getcwd(),'data','data_export')
        self.df = pd.read_csv(os.path.join(self.datapath, filename), sep='\t')

    def task_start_time(self, df):
        starttime = df[df['Event value'].isin(['1','2'])].index.to_list()
        print(starttime)

    def separate_data(self, time1, time2):
        df1 = self.df.iloc[time1:time2]
        df2 = self.df.iloc[time2:-1]
        return df1, df2
        

if __name__ == '__main__':
    # print(os.path.join(os.getcwd(),'data','data_export'))
    datapath = os.path.join(os.getcwd(),'data','data_export')
    for file in os.listdir(os.path.join('data', 'data_export')):
        if file == '.DS_Store':
            continue
        test_data = SingleStimuliData(datapath, file)
        abc_list = test_data.get_stimuli_list()
        test_data.save_df(abc_list)

    # print(df[df['Event value']=='a']['Event value'])
