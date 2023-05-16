from operator import index
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
import json

from SingleGazeAnalysis import MetaAnalysis


def decide_session_and_task(idx, foldername):
    session = f'{idx//2+1} session'
    taskname = foldername.split('_')[-1]
    if taskname == 'task2':
        task = 'task 0.7'
    elif taskname == 'task1':
        task = 'task 0.5'
    else:
        raise Exception(f'{taskname} is not acceptable')
    
    return session, task

def feature_load(log_path):
    loglist = os.listdir(log_path)
    log = [i for i in loglist if i.endswith("aoi.json")][0]
    with open(os.path.join(log_path, log), 'r') as f:
        log_json = json.load(f)
    return log_json

def slice_stimuli(df, task):
    df_event = df[df['Event']=='KeyboardEvent'].copy()
    df_event_b = df_event[df_event['Event value']=='b']
    # df_event_c = df_event[df_event['Event value']=='c']
    event_b = df_event_b.index.to_list()[-1]
    # event_c = df_event_c.index.to_list()[0]
    if task == 'task 0.7':
        frame = 84
    elif task == 'task 0.5':
        frame = 60
    bc_stimuli = df.iloc[event_b+1:event_b+1+frame,:]
    return bc_stimuli

def get_gazeXY(df: pd.DataFrame):
    x = 'Gaze point X [DACS px]'
    y = 'Gaze point Y [DACS px]'
    x_data = df[x]
    x_data = x_data.fillna(method='bfill')
    x_data = x_data.fillna(method='ffill')
    x_data = x_data.to_list()
    
    y_data = df[y]
    y_data = y_data.fillna(method='bfill')
    y_data = y_data.fillna(method='ffill')
    y_data = y_data.to_list()
    xy_dict = {}
    for i in range(len(x_data)):
        xy_dict[f'x{i+1}'] = x_data[i]
        xy_dict[f'y{i+1}'] = y_data[i]
    return xy_dict

def get_gazeXY_for_heatmap(df: pd.DataFrame):
    x = 'Gaze point X [DACS px]'
    y = 'Gaze point Y [DACS px]'
    x_data = df[x]
    x_data = x_data.fillna(method='bfill')
    x_data = x_data.fillna(method='ffill')
    x_data = x_data.to_list()
    
    y_data = df[y]
    y_data = y_data.fillna(method='bfill')
    y_data = y_data.fillna(method='ffill')
    y_data = y_data.to_list()
    xy_list = []
    for i in range(len(x_data)):
        xy_list.append([int(x_data[i]), int(y_data[i]),1])
    return xy_list

def bool_hit(df:pd.DataFrame):
    hit_data = df['gaze hit']
    if 'hit' in hit_data.tolist():
        return 1
    else:
        return 0

if __name__ == '__main__':
    participant_dict = {'chungha': '8', 'dongik': '7', 'eunhye': '1', 'In-Taek': '5', 'jooyeong': '13', 'juchanseo': '3', 'junryeol': '11', 
                        'juyeon': '4', 'myounghun': '9', 'songmin': '10', 'sooyeon': '6', 'woojinkang': '2', 'yeogyeong': '12'}
    
    processed_datadir_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data'
    logdir_path = 'data/AOI_HitScoring/IdentiGaze_Data'
    example_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data/chungha'
    whole_dataframe_task1 = pd.DataFrame()
    whole_dataframe_task2 = pd.DataFrame()
    for participant in participant_dict:
        p_tasklist = sorted(os.listdir(os.path.join(processed_datadir_path, participant)))
        p_tasklist.remove('.DS_Store')
        print(participant)
        for idx, foldername in tqdm(enumerate(p_tasklist)):
            session, task = decide_session_and_task(idx, foldername)
            feature_log_path = os.path.join(logdir_path, f'P{participant_dict[participant]}_{participant}',session, task)
            log_json = feature_load(feature_log_path)
            for iter in range(1,101):
                dataname = f'{iter}_hit.csv'
                data_df = pd.read_csv(os.path.join(processed_datadir_path, participant, foldername, dataname), index_col=0)
                meta = MetaAnalysis(log_json[f'{iter}'])
                task_target = meta.get_task()
                level = meta.get_level()
                cnt_x , cnt_y = meta.get_cnt()
                bbx_x1, bbx_y1, bbx_x2, bbx_y2 = meta.get_aoi()
                bc_stimuli = slice_stimuli(data_df, task)
                hit = bool_hit(bc_stimuli)

                data_dict = get_gazeXY(bc_stimuli)
                data_dict['gaze_hit'] = hit
                data_dict['cnt_x'] = cnt_x
                data_dict['cnt_y'] = cnt_y
                data_dict['bbx_x1'] = bbx_x1
                data_dict['bbx_x2'] = bbx_x2
                data_dict['bbx_y1'] = bbx_y1
                data_dict['bbx_y2'] = bbx_y2
                data_dict['task'] = task_target
                data_dict['level'] = level
                data_dict['participant'] = int(participant_dict[participant])
                meta_data = meta.get_data()
                data_dict['shape_target'] = meta_data['shape_target']
                data_dict['shape_distractor'] = meta_data['shape_distractor']
                data_dict['set_size'] = meta_data['set_size']
                data_dict['target_size'] = meta_data['target_size']
                data_dict['distractor_size'] = meta_data['distractor_size']
                data_dict['target_color_b'] = meta_data['target_color'][0]
                data_dict['target_color_g'] = meta_data['target_color'][1]
                data_dict['target_color_r'] = meta_data['target_color'][2]
                data_dict['distractor_color_b'] = meta_data['distractor_color'][0]
                data_dict['distractor_color_g'] = meta_data['distractor_color'][1]
                data_dict['distractor_color_r'] = meta_data['distractor_color'][2]

                if meta_data['distractor_orientation'] == None:
                    data_dict['target_orientation'] = 0
                    data_dict['distractor_orientation'] = 0
                else:
                    data_dict['distractor_orientation'] = meta_data['distractor_orientation']
                    data_dict['target_orientation'] = meta_data['target_orientation']
                this_df = pd.DataFrame(data_dict, index=[0])
                if task == 'task 0.7':
                    whole_dataframe_task2 = pd.concat([whole_dataframe_task2, this_df])
                elif task == 'task 0.5':
                    whole_dataframe_task1 = pd.concat([whole_dataframe_task1, this_df])
    
    whole_dataframe_task1.to_csv('data/blue_rawdata_task1.csv', index=False)
    whole_dataframe_task2.to_csv('data/blue_rawdata_task2.csv', index=False)


                
            

