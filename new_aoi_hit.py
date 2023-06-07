import os
import json
import pandas as pd
from tqdm import tqdm


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

def bool_hit(fx, fy, bbx_x1, bbx_y1, bbx_x2, bbx_y2):
    if (bbx_x1<=fx<=bbx_x2) and (bbx_y1<=fy<=bbx_y2):
        return 1
    else:
        return 0
    
def add_data_hit(df:pd.DataFrame, bbx:list):
    bbx_x1, bbx_y1, bbx_x2, bbx_y2 = bbx
    raw_x = 'Gaze point X [DACS px]'
    raw_y = 'Gaze point Y [DACS px]'
    df['hardcore_hit'] = df.apply(lambda x: bool_hit(x[raw_x], x[raw_y],bbx_x1,bbx_y1,bbx_x2,bbx_y2), axis=1)
    return df

def dir_check_create(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    participant_dict = {'chungha': '8', 'dongik': '7', 'eunhye': '1', 'In-Taek': '5', 'jooyeong': '13', 'juchanseo': '3', 'junryeol': '11', 
                        'juyeon': '4', 'myounghun': '9', 'songmin': '10', 'sooyeon': '6', 'woojinkang': '2', 'yeogyeong': '12'}
    
    processed_datadir_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data'
    logdir_path = 'data/AOI_HitScoring/IdentiGaze_Data'
    example_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data/chungha'
    save_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data_ver2'
    whole_dataframe_task1 = pd.DataFrame()
    whole_dataframe_task2 = pd.DataFrame()
    for participant in participant_dict:
        p_tasklist = sorted(os.listdir(os.path.join(processed_datadir_path, participant)))
        p_tasklist.remove('.DS_Store')
        dir_check_create(os.path.join(save_path, participant))
        print(participant)
        for idx, foldername in tqdm(enumerate(p_tasklist)):
            session, task = decide_session_and_task(idx, foldername)
            feature_log_path = os.path.join(logdir_path, f'P{participant_dict[participant]}_{participant}',session, task)
            log_json = feature_load(feature_log_path)
            dir_check_create(os.path.join(save_path, participant, foldername))
            for iter in range(1,101):
                dataname = f'{iter}_hit.csv'
                data_df = pd.read_csv(os.path.join(processed_datadir_path, participant, foldername, dataname), index_col=0)
                meta = MetaAnalysis(log_json[f'{iter}'])
                task_target = meta.get_task()
                level = meta.get_level()
                cnt_x , cnt_y = meta.get_cnt()
                bbx_list = meta.get_real_bbx()
                data_df = add_data_hit(data_df, bbx_list)
                data_df.to_csv(os.path.join(save_path, participant, foldername, dataname))
