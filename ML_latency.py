from operator import index
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
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
    df_event_c = df_event[df_event['Event value']=='c']
    event_b = df_event_b.index.to_list()[-1]
    event_c = df_event_c.index.to_list()[0]
    if task == 'task 0.7':
        frame = 84
    elif task == 'task 0.5':
        frame = 60
    # bc_stimuli = df.iloc[event_b+1:event_b+1+frame,:]
    bc_stimuli = df.iloc[event_b+1:event_c,:]
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
    hit_data = df['hardcore_hit']
    # if 'hit' in hit_data.tolist():
    if 1 in hit_data.tolist():
        return 1
    else:
        return 0

def get_hit(df: pd.DataFrame):
    hit_col = 'hardcore_hit'
    hit_data = df[hit_col]
    hit_data = hit_data.to_list()
    
    hit_dict = {}
    for i in range(len(hit_data)):
        hit_dict[i+1] = hit_data[i]
    return hit_dict

def gaze_plot(data_dict, cnt_x, cnt_y, save_path=None, bbox=None):
    plt.figure(figsize=(15,10))
    plt.xlim((0, 1920))
    plt.ylim((0, 1080))

    x_list = []
    y_list = []
    for i in range(1, T+1):
        x = data_dict[f'x{i}']
        y = data_dict[f'y{i}']
        x_list.append(x)
        y_list.append(y)
        
    plt.axhline(1080/2,0,1, color='lightgray', linestyle='--', linewidth=1)
    plt.axvline(1920/2,0,1, color='lightgray', linestyle='--', linewidth=1)
    plt.scatter(x_list,y_list,c=range(1,T+1), linewidth = 2,marker='o', alpha=0.5, cmap="Blues", label='gaze point')
    plt.scatter(cnt_x,cnt_y,color='red', linewidth = 2,marker='o', label='target')
    plt.scatter(1920/2,1080/2,color='green', linewidth = 2,marker='o', label='start')
    # for i, (x, y) in enumerate(zip(x_list, y_list)):
        # plt.annotate(str(i), (x, y))
    plt.annotate('1', (x_list[0], y_list[0]))
    plt.annotate(str(T), (x_list[-1], y_list[-1]))
    ax = plt.gca()
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # plt.plot(bbox[0], bbox[1],color='black', linestyle='solid', linewidth = 2,marker='o')
    # plt.plot(bbox[2], bbox[3],color='black', linestyle='solid', linewidth = 2,marker='o')
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        print(save_path.split('/')[-1])
        plt.title(save_path.split('/')[-1])
        plt.savefig(save_path+'_plot.png')

def get_slope_mean_std(start_idx, end_idx, cnt_x, cnt_y):
    slope_list = []
    last_idx = int(sorted(hit_dict.keys())[-1])
    if end_idx > last_idx:
        end_idx = last_idx
    for i in range(start_idx, end_idx+1):
        hit = hit_dict[i]

        x = data_dict[f'x{i}']
        y = data_dict[f'y{i}']
        dist = np.sqrt((cnt_x - x)**2 + (cnt_y-y)**2)
        
        if i>=2:
            xp = data_dict[f'x{i-1}']
            yp = data_dict[f'y{i-1}']
            distp = np.sqrt((cnt_x - xp)**2 + (cnt_y-yp)**2)
        
            slope = (distp-dist) / (i-1-i)
            slope_list.append(slope)
            
    mean = np.mean(slope_list)
    std = np.std(slope_list)
    return mean, std

def get_dist_mean_std(start_idx, end_idx, cnt_x, cnt_y):
    dist_list = []

    last_idx = int(sorted(hit_dict.keys())[-1])
    if end_idx > last_idx:
        end_idx = last_idx
    for i in range(start_idx, end_idx+1):

        x = data_dict[f'x{i}']
        y = data_dict[f'y{i}']
        dist = np.sqrt((cnt_x - x)**2 + (cnt_y-y)**2)
        
        dist_list.append(dist)
            
    mean = np.mean(dist_list)
    std = np.std(dist_list)
    return mean, std

def get_perceive_idx(data_dict,hit_dict,plot=False, save_path=None):

    if os.path.exists(save_path+'.txt'):
        os.remove(save_path+'.txt')
    
    try:
        # hit_idx = [idx for idx in hit_dict.keys() if hit_dict[idx] == 'hit'][0]
        hit_idx = [idx for idx in hit_dict.keys() if hit_dict[idx] == 1][0]
    except:
        hit_idx = -1 # no_hit
    

    slope_list = []
    dist_list = []
    perceive_idx = -1
    patience = 2
    for i in range(1, T+1):
        hit = hit_dict[i]

        x = data_dict[f'x{i}']
        y = data_dict[f'y{i}']
        dist = np.sqrt((cnt_x - x)**2 + (cnt_y-y)**2)   # calculate dist
        dist_list.append(dist)
        
        
        if i > 1:
            xp = data_dict[f'x{i-1}']
            yp = data_dict[f'y{i-1}']
            distp = np.sqrt((cnt_x - xp)**2 + (cnt_y-yp)**2)
        
            slope = (distp-dist) / (i-1-i)  # calculate slope
            slope_list.append(slope)
            
            # after_dist_mean, after_dist_std = get_slope_mean_std(i+2, T, cnt_x, cnt_y)
            if save_path != None:
                with open(save_path+'.txt', 'a') as f:
                    f.write(f"{i}-th point -- dist: {dist}\t slope: {slope}\t {hit}\n")
                    # f.write(f"{i}-th point -- dist: {dist}\t slope: {slope}\t {hit}\t after_slope_mean, std : {after_dist_mean, after_dist_std}\n")
            # print(f"{i}-th point -- dist: {dist}\t slope: {slope}\t {hit}\t after_slope_mean, std : {after_slope_mean, after_slope_std}\n")
            
            
            if slope < SLOPE_THRESHOLD :
                patience -= 1
            else:
                patience = 2


            if perceive_idx == -1 and patience == 0 and i-2<hit_idx:
                print("find!!!")
                perceive_idx = i-2
                # if len(perceive_idx) ==0:
                #     perceive_idx.append(i)       # before patience count (2)
                # if(i-2-perceive_idx[-1]>patience):
                #     perceive_idx.append(i)       # before patience count (2)
                # patience=2
            
    
    # with open(save_path+'.txt', 'a') as f:
    #     f.write(str(perceive_idx))
    # if len(perceive_idx) != 0:
    #     perceive_idx = perceive_idx[-1]     # last idx
    # else:
    #     perceive_idx=-1
                
            
    if plot:
        plt.figure(figsize=(30,5))
        plt.xticks(range(1,T+1))
        plt.scatter(range(2, T+1), slope_list,color='blue', linestyle='solid', linewidth = 2,marker='o', label='slope')
        plt.scatter(range(1, T+1), dist_list,color='green', linestyle='solid', linewidth = 2,marker='o', label='distance')
        plt.vlines([perceive_idx-0.1], ymin=[0], ymax=[np.max(dist_list)], colors=['green'])
        plt.vlines([hit_idx], ymin=[0], ymax=[np.max(dist_list)], colors=['red'])
        plt.hlines([0], xmin=[0], xmax=[T+1], colors=['black'])
      
        print(f"perceive_idx: {perceive_idx}")
        plt.legend()
        if save_path != None:
            plt.title(save_path.split('/')[-1])
            plt.savefig(save_path+'_dist.png')
        else:
            plt.show()

    return perceive_idx


if __name__ == '__main__':
    SLOPE_THRESHOLD = -30
    participant_dict = {'chungha': '8', 'dongik': '7', 'eunhye': '1', 'In-Taek': '5', 'jooyeong': '13', 'juchanseo': '3', 'junryeol': '11', 
                        'juyeon': '4', 'myounghun': '9', 'songmin': '10', 'sooyeon': '6', 'woojinkang': '2', 'yeogyeong': '12'}
    
    processed_datadir_path = 'AOI_HitScoring/IdentiGaze_Processed Data_ver2'
    logdir_path = 'AOI_HitScoring/IdentiGaze_data'
    example_path = 'AOI_HitScoring/IdentiGaze_Processed Data/chungha'
    whole_dataframe_task1 = pd.DataFrame()
    whole_dataframe_task2 = pd.DataFrame()
    for participant in participant_dict:
        p_tasklist = sorted(os.listdir(os.path.join(processed_datadir_path, participant)))
        # p_tasklist.remove('.DS_Store')
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
                bbx_x1, bbx_y1, bbx_x2, bbx_y2 = meta.get_real_bbx()
                bc_stimuli = slice_stimuli(data_df, task)
                hit = bool_hit(bc_stimuli)

                
                

                data_dict = get_gazeXY(bc_stimuli)
                hit_dict = get_hit(bc_stimuli)
                T = len(bc_stimuli)
                print(f"total row count (=T): {T}" )
                print(f"target poistion: {cnt_x}, {cnt_y}")

                gaze_plot(data_dict, cnt_x, cnt_y, f'data/plot/{participant}_{session}_{task}_{iter}_{hit}', bbox = (bbx_x1, bbx_y1, bbx_x2, bbx_y2))

                # percevie_idx = get_perceive_idx(data_dict, hit_dict, plot=True, save_path=f'data/plot/{participant}_{session}_{task}_{iter}_{hit}')
                percevie_idx = get_perceive_idx(data_dict, hit_dict, plot=False, save_path=None)




                # data_dict = get_gazeXY(bc_stimuli)
                # data_dict['gaze_hit'] = hit
                # data_dict['cnt_x'] = cnt_x
                # data_dict['cnt_y'] = cnt_y
                # data_dict['bbx_x1'] = bbx_x1
                # data_dict['bbx_x2'] = bbx_x2
                # data_dict['bbx_y1'] = bbx_y1
                # data_dict['bbx_y2'] = bbx_y2
                # data_dict['task'] = task_target
                # data_dict['level'] = level
                # data_dict['participant'] = int(participant_dict[participant])
                # meta_data = meta.get_data()
                # data_dict['shape_target'] = meta_data['shape_target']
                # data_dict['shape_distractor'] = meta_data['shape_distractor']
                # data_dict['set_size'] = meta_data['set_size']
                # data_dict['target_size'] = meta_data['target_size']
                # data_dict['distractor_size'] = meta_data['distractor_size']
                # data_dict['target_color_b'] = meta_data['target_color'][0]
                # data_dict['target_color_g'] = meta_data['target_color'][1]
                # data_dict['target_color_r'] = meta_data['target_color'][2]
                # data_dict['distractor_color_b'] = meta_data['distractor_color'][0]
                # data_dict['distractor_color_g'] = meta_data['distractor_color'][1]
                # data_dict['distractor_color_r'] = meta_data['distractor_color'][2]

                # if meta_data['distractor_orientation'] == None:
                #     data_dict['target_orientation'] = 0
                #     data_dict['distractor_orientation'] = 0
                # else:
                #     data_dict['distractor_orientation'] = meta_data['distractor_orientation']
                #     data_dict['target_orientation'] = meta_data['target_orientation']
                # this_df = pd.DataFrame(data_dict, index=[0])
                # if task == 'task 0.7':
                #     whole_dataframe_task2 = pd.concat([whole_dataframe_task2, this_df])
                # elif task == 'task 0.5':
                #     whole_dataframe_task1 = pd.concat([whole_dataframe_task1, this_df])
    
    # whole_dataframe_task1.to_csv('data/blue_rawdata_task1.csv', index=False)
    # whole_dataframe_task2.to_csv('data/blue_rawdata_task2.csv', index=False)


            