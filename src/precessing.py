import os
import pandas as pd
from utils import feature_processing
from tqdm import tqdm
import json
from numpyencoder import NumpyEncoder


def getUserTaskDirs(user_dir):
    task_dirs = os.listdir(os.path.join(DATA_DIR, user_dir))
    task_dirs = [dir for dir in task_dirs if '._' not in dir]
    try:
        task_dirs.remove('.DS_Store')
    except:
        pass
    return task_dirs



def make_directory(user, task):
    if os.path.exists(os.path.join(f'features/{user}/{task}')):
        return
    if not os.path.exists(os.path.join(f'features/{user}/')):
        os.mkdir(os.path.join(f'features/{user}/'))
    if not os.path.exists(os.path.join(f'features/{user}/{task}')):
        os.mkdir(os.path.join(f'features/{user}/{task}'))
    if not os.path.exists(os.path.join(f'features/{user}/{task}/1')):
        for i in range(1,101):
            os.mkdir(os.path.join(f'features/{user}/{task}/{i}'))



def process_task(user, session, task):
    feat_df = pd.DataFrame(columns=['User','Task','Trial','Block',
                                    'Fixation Count','Saccade Count',
                                    'Fixation Duration','Saccade Duration',
                                    'Average Pupil Left','Average Pupil Right',
                                    'Saccade Velocity','Saccade Amplitude'])
    
    # load NASA-TLX scores
    user_id = f'{name2id[user]}_{user}'
    session_id = f'{session} session'
    task_num = task.split('_')[-1]
    if task_num == 'task1': task_id = 'task 0.5'
    if task_num == 'task2': task_id = 'task 0.7'
    json_files = [file for file in os.listdir(f'IdentiGaze_data/{user_id}/{session_id}/{task_id}/') if file.endswith('.json')]
    json_file = [json_file for json_file in json_files if 'feat' not in json_file][0]
    # print(json_file)
        
    with open(f'IdentiGaze_data/{user_id}/{session_id}/{task_id}/{json_file}', 'r') as f:
        data = json.load(f)
        

    i = 0
    for trial in tqdm(range(1,101)):

        # load raw gaze dataframe 
        df = pd.read_csv(os.path.join(DATA_DIR, user, task, f'{trial}.tsv'), delimiter='\t')
        data[str(trial)]['features'] = f'features/{user}_{task}.csv'


        # df = df.drop_duplicates(['Event value'], keep='first')
        
        if trial != 100:
            # spilit dataframe by event value
            A1, A2 = df[df['Event value'] == 'a'].index.values[:2]
            B = df[df['Event value'] == 'b'].index.values[0]
            C = df[df['Event value'] == 'c'].index.values[0]

            assert(A2-A1 !=1)
            # print(A1, A2, B, C)

            df_A1_B = df.iloc[A1+1: B]
            df_B_C = df.iloc[B+1: C]
            df_C_A2 = df.iloc[C+1: A2]
            df_A1_A2 = df.iloc[A1+1: A2]
            # print(df_A1_B)
            # print(df_B_C)
            # print(df_C_A2)  
            # print(df_A1_A2) 

            # A1 ~ B
            fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list = feature_processing(df=df_A1_B, sample_size=120)
            feat_df.loc[i] = [user, task_num, trial, 'A1-B', fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list]
            i+=1
            # data[str(trial)]['features']['A1-B'] = {
            #     'FC': fixation_count, 
            #     'SC': saccade_count, 
            #     'FD': fd_list,
            #     'SD': sd_list, 
            #     'PDL': pd_left, 
            #     'PDR': pd_right,
            #     'SV': sv_list,
            #     'SA': sa_list
            # }
            
            # B ~ C
            fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list = feature_processing(df=df_B_C, sample_size=120)
            feat_df.loc[i] = [user, task_num, trial, 'B-C', fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list]
            i+=1
            # data[str(trial)]['features']['B-C'] = {
            #     'FC': fixation_count, 
            #     'SC': saccade_count, 
            #     'FD': fd_list,
            #     'SD': sd_list, 
            #     'PDL': pd_left, 
            #     'PDR': pd_right,
            #     'SV': sv_list,
            #     'SA': sa_list
            # }

            # C ~ A2
            fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list = feature_processing(df=df_C_A2, sample_size=120)
            feat_df.loc[i] = [user, task_num, trial, 'C-A2', fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list]
            i+=1
            # data[str(trial)]['features']['C-A2'] = {
            #     'FC': fixation_count, 
            #     'SC': saccade_count, 
            #     'FD': fd_list,
            #     'SD': sd_list, 
            #     'PDL': pd_left, 
            #     'PDR': pd_right,
            #     'SV': sv_list,
            #     'SA': sa_list
            # }
            # A1 ~ A2
            fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list = feature_processing(df=df_A1_A2, sample_size=120)
            feat_df.loc[i] = [user, task_num, trial, 'A1-A2', fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list]
            i+=1
            # data[str(trial)]['features']['A1-A2'] = {
            #     'FC': fixation_count, 
            #     'SC': saccade_count, 
            #     'FD': fd_list,
            #     'SD': sd_list, 
            #     'PDL': pd_left, 
            #     'PDR': pd_right,
            #     'SV': sv_list,
            #     'SA': sa_list
            # }

        if trial == 100:
            A1 = df[df['Event value'] == 'a'].index.values[0]
            B = df[df['Event value'] == 'b'].index.values[0]
            C = df[df['Event value'] == 'c'].index.values[0]
            # print(A1, A2, B, C)
            # print(df)
            
            df_A1_B = df.iloc[A1+1: B]
            df_B_C = df.iloc[B+1: C]
            df_A1_C = df.iloc[A1+1: C]
            
            # A1 ~ B
            fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list = feature_processing(df=df_A1_B, sample_size=120)
            feat_df.loc[i] = [user, task_num, trial, 'A1-B', fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list]
            i+=1
            # data[str(trial)]['features']['A1-B'] = {
            #     'FC': fixation_count, 
            #     'SC': saccade_count, 
            #     'FD': fd_list,
            #     'SD': sd_list, 
            #     'PDL': pd_left, 
            #     'PDR': pd_right,
            #     'SV': sv_list,
            #     'SA': sa_list
            # }
            
            # B ~ C
            fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list = feature_processing(df=df_B_C, sample_size=120)
            feat_df.loc[i] = [user, task_num, trial, 'B-C', fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list]
            i+=1
            # data[str(trial)]['features']['B-C'] = {
            #     'FC': fixation_count, 
            #     'SC': saccade_count, 
            #     'FD': fd_list,
            #     'SD': sd_list, 
            #     'PDL': pd_left, 
            #     'PDR': pd_right,
            #     'SV': sv_list,
            #     'SA': sa_list
            # }

            # A1 ~ C
            fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list = feature_processing(df=df_A1_C, sample_size=120)
            feat_df.loc[i] = [user, task_num, trial, 'A1-C', fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list]
            i+=1
            # data[str(trial)]['features']['A1-C'] = {
            #     'FC': fixation_count, 
            #     'SC': saccade_count, 
            #     'FD': fd_list,
            #     'SD': sd_list, 
            #     'PDL': pd_left, 
            #     'PDR': pd_right,
            #     'SV': sv_list,
            #     'SA': sa_list
            # }



    # print(feat_df)
    feat_df.to_csv(f'features/{user}_{task}.csv', index=None, sep=',')   
    new_json_file = json_file.rstrip('.json') + '_feat.json'
    with open(f'IdentiGaze_data/{user_id}/{session_id}/{task_id}/{new_json_file}', 'w') as f:
        json.dump(data, f)





DATA_DIR = 'IdentiGaze_Processed Data'
user_dirs = os.listdir(DATA_DIR)

user_dirs.remove('.DS_Store')
user_dirs.sort()
len(user_dirs)

id2name= {'P3' : 'juchanseo',
 'P5' : 'In-Taek',
 'P1' : 'eunhye',
 'P4' : 'juyeon',
 'P8' : 'chungha',
 'P9' : 'myounghun',
 'P10' : 'songmin',
 'P7' : 'dongik',
 'P2' : 'woojinkang',
 'P11' : 'junryeol',
 'P6' : 'sooyeon',
 'P12' : 'yeogyeong',
 'P13' : 'jooyeong'}

name2id = {id2name[id]:id for id in id2name.keys()}
for user in ['junryeol']:
    task_dirs = getUserTaskDirs(user)
    task_dirs.sort()

    for i, task in enumerate(task_dirs):

        session = (i // 2) + 1

        
        print(f'{user} - session {session} - {task}')
        
        process_task(user, session, task)

        

