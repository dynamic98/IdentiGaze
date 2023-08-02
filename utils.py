
from metric_revised import *
import os


id2name= {
    'P3' : 'juchanseo',
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
    'P13' : 'jooyeong'
}
name2id = {id2name[id]:id for id in id2name.keys()}




def feature_processing(df, save_path=None, sample_size=120, baseline=None):
    """calculate gaze features and save them to save_path"""


    fixation_count = Fixation_Count(df)
    saccade_count = Saccade_Count(df)
    fd_list = Fixation_Duration(df)
    sd_list = Saccade_Duration(df)
    # fr_list, _ = fixation_count_per_block(df, sample_size)
    # sr_list, _ = saccade_count_per_block(df,sample_size)
    sv_list, sa_list = Saccade_Velocity_Amplitude(df)
    pd_left, pd_right, _ = Pupil_Diameter_per_block(df, sample_size)
    pd_avg = []
    for l, r in zip(pd_left, pd_right):
        if l != 0 and r != 0:
            pd_avg.append((l + r) / 2)
        elif l == 0:
            pd_avg.append(r)
        elif r == 0:
            pd_avg.append(l)
        elif l == 0 and r == 0:
            pd_avg.append(0)

    if baseline is not None:
        new_pd_avg = []
        for p in pd_avg:
            p = p - baseline
            new_pd_avg.append(p)
        # save_feature(save_path, new_pd_avg, 'PD')
    
    # save_feature(save_path, fd_list, 'FD')
    # save_feature(save_path, sd_list, 'SD')
    # save_feature(save_path, fr_list, 'FR')
    # save_feature(save_path, sr_list, 'SR')
    # save_feature(save_path, sv_list, 'SV')
    # save_feature(save_path, sa_list, 'SA')
    

    # if df['gaze hit'].isin(['hit']).sum() != 0:
    #     hit = 1
    # else:
    #     hit = 0
    return fixation_count, saccade_count, fd_list, sd_list, pd_left, pd_right, sv_list, sa_list

def save_feature(save_path, result, feature_name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    new_df = pd.DataFrame()
    new_df[feature_name] = result
    new_df.to_csv(os.path.join(save_path,feature_name +'.csv'), header=False, index=False)
    # print('complete to save - {0}.csv'.format(feature_name))
    return

