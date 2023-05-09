import os
import pandas as pd
import json
import numpy as np

class TaskStimuliData:
    def __init__(self, datapath, filename) -> None:
        self.datapath = datapath
        self.filename = filename
        self.df = pd.read_csv(os.path.join(self.datapath, self.filename))
        self.user = self.df['User'][0]
        self.task = self.df['Task'][0]
        self.blocklist = ['A1-B','B-C','C-A2','A1-A2']

    def get_df(self, trial=False, block=False):
        if type(trial)!='list':
            print(type(trial))


        if trial==False and block==False:
            return self.df
        elif trial==False and block!=False:
            return self.df[self.df['Block']==block]
        elif trial!=False and block==False:
            return self.df[self.df['Trial']==trial]
        else:
            return self.df[(self.df['Trial']==trial)&(self.df['Block']==block)]

class HitStimuliAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.timestart = self.data['Recording timestamp [μs]'][0]

    def differential(self, column_name):
        self.data[f'diff_{column_name}'] = self.data.apply(lambda x: self.divide(x[f'{column_name}'], x['Recording timestamp [μs]']), axis=1)
        return self.data[f'diff_{column_name}']
    
    def divide(self, a, b):
        if a == np.nan or b == np.nan or b == 0:
            return np.nan
        else:
            return a/b




class MetaAnalysis:
    def __init__(self, data) -> None:
        self.data = data
        self.task = self.data['task']
        self.target_stimuli = ['shape_target','target_size','target_color','target_orientation']
        self.distractor_stimuli = ['shape_distractor','distractor_size','distractor_color','distractor_orientation']


    def get_data(self):
        return self.data

    def get_level(self):
        if self.task == 'shape':
            target = self.data['shape_target']
            distractor = self.data['shape_distractor']

        elif self.task == 'size':
            level_mapping = {30:1, 40:2, 50:3, 60:4, 70:5}
            target = level_mapping[self.data['target_size']]
            distractor = level_mapping[self.data['distractor_size']]

        elif self.task == 'hue':
            level_mapping = {(62,101,127):'r#1',(62,82,127):'r#2', (62,62,127):'r#3', (82,62,127):'r#4', (101,62,127):'r#5',
                             (62,127,88):'y#1', (62,127,107):'y#2', (62,127,127):'y#3', (62,107,127):'y#4',(62,88,127):'y#5',
                             (101,127,62):'g#1',(82,127,62):'g#2', (62,127,62):'g#3', (62,127,82):'g#4', (62,127,101):'g#5',
                             (127,62,101):'b#1',(127,62,82):'b#2', (127,62,62):'b#3', (127,82,62):'b#4', (127,101,62):'b#5'}
            target = level_mapping[tuple(self.data['target_color'])]
            distractor = level_mapping[tuple(self.data['distractor_color'])][-1]

        elif self.task == 'brightness':
            level_mapping = {(51, 51, 51): 1, (102, 102, 102):2, (153, 153, 153):3, (204, 204, 204):4, (255, 255, 255):5}
            target = level_mapping[tuple(self.data['target_color'])]
            distractor = level_mapping[tuple(self.data['distractor_color'])]

        elif self.task == 'orientation':
            level_mapping = {-30:1, -15:2, 0:3, 15:4, 30:5}
            target = level_mapping[self.data['target_orientation']]
            distractor = level_mapping[self.data['distractor_orientation']]

        return f"{target}-{distractor}"

    def get_task(self):
        return self.task
    
    def get_cnt(self):
        return self.data['target_cnt']

    def get_aoi(self):
        return self.data['area']


if __name__ == '__main__':


    # examplejson = {'task': 'hue',
    #                 'shape_target': 'orientation',
    #                 'shape_distractor': 'orientation',
    #                 'set_size': 6,
    #                 'target_cnt': [1235, 375],
    #                 'target_size': 50,
    #                 'distractor_size': 50,
    #                 'target_color': [82, 127, 62],
    #                 'distractor_color': [82, 127, 62],
    #                 'target_orientation': 15,
    #                 'distractor_orientation': 0,
    #                 'features': 'features/In-Taek_2023-03-20_10\uf02216\uf02256_task1.csv'}
    
    # {"task": "size", "shape_target": 1, "shape_distractor": 1, "set_size": 6, "target_cnt": [1245, 483], 
    # "target_size": 60, "distractor_size": 30, "target_color": [82, 127, 62], "distractor_color": [82, 127, 62], 
    # "target_orientation": 'null', "distractor_orientation": 'null'},

    # myMeta = MetaAnalysis(examplejson)
    # print(myMeta.get_level())

    # hit_path = os.path.join(os.getcwd(), 'data', 'features_hit')
    # participatns = ['chungha','dongik','eunhye','In-Taek','jooyeong','juchanseo','junryeol','juyeon',
                    # 'myounghun','songmin','sooyeon','woojinkang','yeogyeong']
    # filelist = os.listdir(hit_path)
    # data = TaskStimuliData(hit_path, filelist[0])
    # print(data.get_df(trial=1, block='A1-B'))
    # print(data.get_df(block='B-C'))
    # for file in filelist:
        # TaskStimuliData(hit_path, file)

    data = pd.read_csv('data/AOI_HitScoring/IdentiGaze_Processed Data/chungha/2023-03-20_14_07_20_task2/1_hit.csv')
    myhit = HitStimuliAnalysis(data)
    print(myhit.differential('Recording timestamp [μs]'))
    