from operator import index
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import warnings
import json
import math
from scipy.fftpack import dct

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
    # elif task == 'task 0.5':
    #     frame = 60
    # bc_stimuli = df.iloc[event_b+1:event_b+1+frame,:]
    bc_stimuli = df.iloc[event_b+1:event_c,:]
    return bc_stimuli

def first_fixation_time(df: pd.DataFrame, meta, task):
    x = 'Fixation point X [DACS px]'
    y = 'Fixation point Y [DACS px]'
    fx_data = df[x].to_list()
    fy_data = df[y].to_list()
    bbx_x1, bbx_y1, bbx_x2, bbx_y2 = meta.get_aoi()
    df_size = df.index.size
    fft_index = False
    if task == 'task 0.7':
        max_fft = 0.7
    elif task == 'task 0.5':
        max_fft = 0.5
    for i in range(df_size):
        if (bbx_x1<=fx_data[i]<=bbx_x2) and (bbx_y1<=fy_data[i]<=bbx_y2):
            fft_index = i
            break
        else:
            continue
    if fft_index:
        fft = max_fft*(i/len(fx_data))
        return fft
    else:
        return max_fft

def fixation_duration(df: pd.DataFrame, meta, task):
    if task == 'task 0.7':
        max_duration = 0.7
    elif task == 'task 0.5':
        max_duration = 0.5

    fx = 'Fixation point X [DACS px]'
    fy = 'Fixation point Y [DACS px]'
    total_size = df.index.size
    fixation_duration_list = []
    target_hit_list = []
    eyetype = 'Eye movement type'
    eyeindex = 'Eye movement type index'
    fixation_df = df[df[eyetype]=='Fixation'][eyeindex]
    fixation_list = list(set(fixation_df.to_list()))
    bbx_x1, bbx_y1, bbx_x2, bbx_y2 = meta.get_real_bbx()

    for fixation in fixation_list:
        this_fixation_df = df[(df[eyetype]=='Fixation')&(df[eyeindex]==fixation)]
        fixation_duration_list.append(this_fixation_df.index.size)
        if (bbx_x1<=this_fixation_df[fx].iloc[0]<=bbx_x2) and (bbx_y1<=this_fixation_df[fy].iloc[0]<=bbx_y2):
            target_hit_list.append(1)
        else:
            target_hit_list.append(0)
    
    target_duration = []
    nontarget_duration = []

    for i, value in enumerate(target_hit_list):
        if value == 1:
            target_duration.append(fixation_duration_list[i])
        else:
            nontarget_duration.append(fixation_duration_list[i])

    total_duration_statistic = get_list_statistic(fixation_duration_list)
    target_duration_statistic = get_list_statistic(target_duration)
    nontarget_duration_statistic = get_list_statistic(nontarget_duration)

    gaze_data = extend_list(total_duration_statistic, target_duration_statistic, nontarget_duration_statistic)
    gaze_data = [max_duration*i/total_size for i in gaze_data]
    return gaze_data

def reaction_time(df: pd.DataFrame, task):
    if task == 'task 0.7':
        max_reactiontime = 0.7
    elif task == 'task 0.5':
        max_reactiontime = 0.5
    total_size = df.index.size

    eyetype = 'Eye movement type'
    eyeindex = 'Eye movement type index'
    fixation_df = df[df[eyetype]=='Fixation'][eyeindex]
    fixation_list = list(set(fixation_df.to_list()))
    if len(fixation_list)>=2:
        reaction_fixation = fixation_list[1]
        eyeindex_list = df[eyeindex].to_list()
        eyetype_list = df[eyetype].to_list()
        for i in range(total_size):
            if (reaction_fixation == eyeindex_list[i]) and (eyetype_list[i]=='Fixation'):
                reactiontime_now = i
                break
            else:
                reactiontime_now = False

        if reactiontime_now:
            return max_reactiontime*(reactiontime_now/total_size)
        else:
            return max_reactiontime
    else:
        return max_reactiontime

def average(data_list):
    return (sum(data_list)/len(data_list))

def get_list_statistic(data_list):
    if len(data_list)>0:
        average_value = average(data_list)
        max_value = max(data_list)
        min_value = min(data_list)
    else:
        average_value = 0
        max_value = 0
        min_value = 0
    return [average_value, max_value, min_value]

def extend_list(*arg):
    result = []
    for i in arg:
        result.extend(i)
    return result

def get_fixation_saccade_XY(df: pd.DataFrame):
    fx = 'Fixation point X [DACS px]'
    fy = 'Fixation point Y [DACS px]'
    gx = 'Gaze point X [DACS px]'
    gy = 'Gaze point Y [DACS px]'
    eyetype = 'Eye movement type'

    fx_data = df[fx]
    fx_data = fx_data.fillna(method='bfill')
    fx_data = fx_data.fillna(method='ffill')
    fx_data = fx_data.to_list()
    
    fy_data = df[fy]
    fy_data = fy_data.fillna(method='bfill')
    fy_data = fy_data.fillna(method='ffill')
    fy_data = fy_data.to_list()

    gx_data = df[gx]
    gx_data = gx_data.fillna(method='bfill')
    gx_data = gx_data.fillna(method='ffill')
    gx_data = gx_data.to_list()

    gy_data = df[gy]
    gy_data = gy_data.fillna(method='bfill')
    gy_data = gy_data.fillna(method='ffill')
    gy_data = gy_data.to_list()

    eyetype_data = df[eyetype].to_list()
    xy_dict = {}
    for i in range(len(fx_data)):
        xy_dict[f'type{i+1}'] = eyetype_data[i]
        if eyetype_data[i] == 'EyesNotFound':
            xy_dict[f'x{i+1}'] = np.nan
            xy_dict[f'y{i+1}'] = np.nan
        elif eyetype_data[i] == 'Fixation':
            xy_dict[f'x{i+1}'] = fx_data[i]
            xy_dict[f'y{i+1}'] = fy_data[i]
        elif eyetype_data[i] == 'Saccade':
            xy_dict[f'x{i+1}'] = gx_data[i]
            xy_dict[f'y{i+1}'] = gy_data[i]

    return xy_dict

def get_gazeXY(df: pd.DataFrame):
    x = 'Gaze point X [DACS px]'
    y = 'Gaze point Y [DACS px]'
    x_data = df[x]
    x_data = x_data.fillna(method='bfill')
    x_data = x_data.fillna(method='ffill')
    x_data = x_data.to_list()
    frame = 84
    
    y_data = df[y]
    y_data = y_data.fillna(method='bfill')
    y_data = y_data.fillna(method='ffill')
    y_data = y_data.to_list()
    xy_dict = {}
    for i in range(frame):
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

def pupil(df: pd.DataFrame):
    pupilDiameter = 'Pupil diameter filtered [mm]'
    pupilData = df[pupilDiameter]
    pupilData = pupilData.fillna(method='bfill')
    pupilData = pupilData.fillna(method='ffill')
    pupilData = pupilData.to_list()
    pupilStatistic = get_list_statistic(pupilData)
    return pupilStatistic

def bool_hit(df:pd.DataFrame):
    hit_data = df['hardcore_hit']
    if 1 in hit_data.to_list():
        return 1
    else:
        return 0

def velocity(df:pd.DataFrame, task):
    total_size = df.index.size
    if task == 'task 0.7':
        tick = 0.7/total_size
    elif task == 'task 0.5':
        tick = 0.5/total_size
    gx = 'Gaze point X [DACS px]'
    gy = 'Gaze point Y [DACS px]'
    gxlist_1 = df[gx].to_list()
    gylist_1 = df[gy].to_list()
    target_hit = df['hardcore_hit'].to_list()
    gxlist_2 = gxlist_1[1:]
    gylist_2 = gylist_1[1:]
    gxlist_1 = gxlist_1[:-1]
    gylist_1 = gylist_1[:-1]
    total_velocity = []
    hit_velocity = []
    nohit_velocity = []
    onhit_velocity = []
    for j in range(len(gxlist_1)):
        p = [gxlist_1[j], gylist_1[j]]
        q = [gxlist_2[j], gylist_2[j]]
        this_distance = math.dist(p,q)
        this_velocity = this_distance/tick
        total_velocity.append(this_velocity)
        if (target_hit[j]==0)&(target_hit[j+1]==1):
            hit_velocity.append(this_velocity)
        elif (target_hit[j]==0)&(target_hit[j+1]==0):
            nohit_velocity.append(this_velocity)
        elif (target_hit[j]==1)&(target_hit[j+1]==1):
            onhit_velocity.append(this_velocity)
    mfcc_data = mfcc(total_velocity)
   
    total_velocity_statistic = get_list_statistic(total_velocity)
    hit_velocity_statistic = get_list_statistic(hit_velocity)
    nohit_velocity_statistic = get_list_statistic(nohit_velocity)
    onhit_velocity_statistic = get_list_statistic(onhit_velocity)
    velocity_data = extend_list(total_velocity_statistic, hit_velocity_statistic, nohit_velocity_statistic, onhit_velocity_statistic, mfcc_data)

    # plt.bar(list(range(len(gxlist_1))), total_velocity)
    # plt.show()
    return velocity_data
 
def saccade_velocity(df:pd.DataFrame, meta, task):
    total_size = df.index.size
    if task == 'task 0.7':
        tick = 0.7/total_size
    elif task == 'task 0.5':
        tick = 0.5/total_size
    eyetype = 'Eye movement type'
    eyeindex = 'Eye movement type index'
    gx = 'Gaze point X [DACS px]'
    gy = 'Gaze point Y [DACS px]'
    saccade_list = list(set(df[df[eyetype]=='Saccade'][eyeindex].to_list()))
    this_saccade_velocity = []
    hit_velocity = []
    nohit_velocity = []
    if len(saccade_list)>1:
        for i in saccade_list:
            this_df = df[(df[eyetype]=='Saccade')&(df[eyeindex]==i)]
            gxlist_1 = this_df[gx].to_list()
            gylist_1 = this_df[gy].to_list()
            gxlist_2 = gxlist_1[1:]
            gylist_2 = gylist_1[1:]
            gxlist_1 = gxlist_1[:-1]
            gylist_1 = gylist_1[:-1]
            if len(gxlist_1)<1:
                break
            distance = []
            for j in range(len(gxlist_1)):
                p = [gxlist_1[j], gylist_1[j]]
                q = [gxlist_2[j], gylist_2[j]]
                this_distance = math.dist(p,q)
                distance.append(this_distance)
            this_velocity = sum(distance)/(len(gxlist_1)*tick)
            this_saccade_velocity.append(this_velocity)
            if bool_hit(this_df):
                hit_velocity.append(this_velocity)
            else:
                nohit_velocity.append(this_velocity)
    
    if len(this_saccade_velocity)>0:
        total_velocity_average = average(this_saccade_velocity)
        total_velocity_min = min(this_saccade_velocity)
        total_velocity_max = max(this_saccade_velocity)
    else:
        total_velocity_average = 0
        total_velocity_min = 0
        total_velocity_max = 0

    if len(hit_velocity)>0:
        hit_velocity_average = average(hit_velocity)
        hit_velocity_min = min(hit_velocity)
        hit_velocity_max = max(hit_velocity)
    else:
        hit_velocity_average = 0
        hit_velocity_min = 0
        hit_velocity_max = 0
    
    if len(nohit_velocity)>0:
        nohit_velocity_average = average(nohit_velocity)
        nohit_velocity_min = min(nohit_velocity)
        nohit_velocity_max = max(nohit_velocity)
    else:
        nohit_velocity_average = 0
        nohit_velocity_min = 0
        nohit_velocity_max = 0
    gaze_data = [total_velocity_average, total_velocity_max, total_velocity_min, hit_velocity_average, hit_velocity_max, hit_velocity_min, nohit_velocity_average, nohit_velocity_max, nohit_velocity_min]
    return gaze_data   

    # for i in fixation_list:
    #     fixation_df_index.append(df[(df[eyetype]=='Fixation')&(df[eyeindex]==i)].index.to_list())

    # for i in saccade_df_index:

def rotated_path(df:pd.DataFrame, meta):
    # print(df.columns)
    # 1080 * 1920 
    gaze_point_x=df['Gaze point X [DACS px]']
    gaze_point_y=df['Gaze point Y [DACS px]']

    target_x, target_y = meta.get_cnt()
    target_x=target_x-960
    target_y=target_y-540
    if(target_x!=0):
        target_slope=2*math.pi-(math.atan2(target_y, target_x))
        if target_slope<0:
            target_slope+=2*math.pi

        real_point_x=[]
        real_point_y=[]
        new_point_x=[]
        new_point_y=[]
        for i in range(len(gaze_point_x)):
            x= gaze_point_x.iloc[i]-960
            y= gaze_point_y.iloc[i]-540
            real_point_x.append(x)
            real_point_y.append(y)
            new_x= x * math.cos(target_slope) - y*math.sin(target_slope)
            new_y= x * math.sin(target_slope) + y*math.cos(target_slope)
            new_point_x.append(new_x)
            new_point_y.append(new_y)

        new_target_x= target_x * math.cos(target_slope) - target_y*math.sin(target_slope)
        new_target_y= target_x * math.sin(target_slope) + target_y*math.cos(target_slope)

        # plt.subplot(221)        
        # plt.xlim([-540, 540])
        # plt.ylim([-960, 960])

        # plt.scatter(target_x, target_y, label='target')

        # plt.scatter(new_target_x, new_target_y, label='nomalized target')

        # plt.plot(real_point_x, real_point_y, label='Gaze point')
        # plt.plot(new_point_x, new_point_y, label='Nomalized Gaze point')
        # plt.legend(['Target', 'Nomalized Target', 'Gaze Point', 'Nomalized Gaze Point'], loc='center left', bbox_to_anchor=(1, 0.5))
        # # plt.show()

        # plt.subplot(222)
        # # plt.ylim([-new_target_x, new_target_x])
        # plt.plot(new_point_x)
        # plt.legend(['approaching scalar value'], loc='center left', bbox_to_anchor=(1, 0.5))
        # # plt.show()

        # plt.subplot(223)
        # # plt.ylim([-new_target_x, new_target_x])
        # plt.plot(new_point_y)
        # plt.legend(['directional value +/-'], loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()
        return new_point_x, new_point_y
    else:
        real_point_x=[]
        real_point_y=[]
        for i in range(len(gaze_point_x)):
            x= gaze_point_x.iloc[i]-960
            y= gaze_point_y.iloc[i]-540
            real_point_x.append(x)
            real_point_y.append(y)
        return real_point_x, real_point_y

def hammingwindow(array):
    array_length = len(array)
    frames = array*np.array([0.54-0.46*np.cos((2*np.pi*n)/(array_length -1)) for n in range(array_length)])
    # plt.subplot(2,1,1)
    # plt.bar(list(range(array_length)),array)
    # plt.subplot(2,1,2)
    # plt.bar(list(range(array_length)),frames)
    # plt.show()
    return frames

def DFT(array, bin=512):
    dft_frames = np.fft.rfft(array, bin)
    mag_frames = np.absolute(dft_frames)
    pow_frames = ((1.0/bin)*((mag_frames)**2))
    return mag_frames
    # return pow_frames


def mfcc(array, num_ceps = 12):
    nfilt = 40
    NFFT = 512
    sample_rate = 120
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    hammedArray = hammingwindow(array)
    frames = DFT(hammedArray, bin=NFFT)
    filter_banks = np.dot(frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks) # Numerical Stability
    # print(len(filter_banks))
    # results = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]
    results = dct(filter_banks, norm='ortho')[1:(num_ceps+1)]
    # print(results)
    # results = dct(filter_banks, type=2, axis=1, norm='ortho')
    return results

def get_hit(df: pd.DataFrame):
    hit_col = 'hardcore_hit'
    hit_data = df[hit_col]
    hit_data = hit_data.to_list()
    
    hit_dict = {}
    for i in range(len(hit_data)):
        hit_dict[i+1] = hit_data[i]
    return hit_dict

def gaze_plot(data_dict, cnt_x, cnt_y, T, save_path=None, bbox=None):
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

def get_slope_mean_std(data_dict, hit_dict, start_idx, end_idx, cnt_x, cnt_y):
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

def get_dist_mean_std(data_dict, hit_dict, start_idx, end_idx, cnt_x, cnt_y):
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

def get_perceive_idx(data_dict, hit_dict, cnt_x, cnt_y, T, SLOPE_THRESHOLD, plot=False, save_path=None):
    if save_path:
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
            
            # after_dist_mean, after_dist_std = get_slope_mean_std(data_dict, hit_dict,i+2, T, cnt_x, cnt_y)
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
                # print("find!!!")
                perceive_idx = i-2
                # if len(perceive_idx) ==0:
                #     perceive_idx.append(i)       # before patience count (2)
                # if(i-2-perceive_idx[-1]>patience):
                #     perceive_idx.append(i)       # before patience count (2)
                # patience=2
            
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

def jspirit_func(df:pd.DataFrame, meta, task):

    data_dict = get_gazeXY(df)
    hit_dict = get_hit(df)
    cnt_x, cnt_y = meta.get_cnt()
    T = 84
    if task == 'task 0.7':
        tick = 0.7/T
    elif task == 'task 0.5':
        tick = 0.5/T
    
    SLOPE_THRESHOLD = -30
    # print(f"total row count (=T): {T}" )
    # print(f"target poistion: {cnt_x}, {cnt_y}")
    # gaze_plot(data_dict, cnt_x, cnt_y, f'data/plot/{participant}_{session}_{task}_{iter}_{hit}', bbox = (bbx_x1, bbx_y1, bbx_x2, bbx_y2))
    percevie_idx = get_perceive_idx(data_dict, hit_dict, cnt_x, cnt_y, T, SLOPE_THRESHOLD, plot=False, save_path=None)
    if percevie_idx > 0:
        return percevie_idx*tick
    else:
        return T*tick

def get_YG_data(x_data, y_data, task):
    # if task == 'task 0.7':
        # frame = 84
    # elif task == 'task 0.5':
        # frame = 60
    frame = 84
    xy_dict = {}
    for i in range(frame):
        xy_dict[f'new_x{i+1}'] = x_data[i]
        xy_dict[f'new_y{i+1}'] = y_data[i]
    return xy_dict


if __name__ == '__main__':

    """
    # example_path = 'C:\\Users\\scilab\\IdentiGaze\\data\\AOI_HitScoring\\IdentiGaze_Processed Data_ver2\\chungha\\2023-03-20_14_11_59_task1\\1_hit.csv'
    # log_path = 'C:\\Users\\scilab\\IdentiGaze\\data\\AOI_HitScoring\\IdentiGaze_data\\P8_chungha\\1 session\\task 0.5'
    # log_json = feature_load(log_path)
    # iter = 1
    # task = 'task 0.5'
    # data_df = pd.read_csv(example_path, index_col=0)
    # meta = MetaAnalysis(log_json[f'{iter}'])
    # task_target = meta.get_task()
    # level = meta.get_level()
    # cnt_x , cnt_y = meta.get_cnt()
    # bbx_x1, bbx_y1, bbx_x2, bbx_y2 = meta.get_real_bbx()
    # bc_stimuli = slice_stimuli(data_df, task)
    # velocity_data = velocity(bc_stimuli, task)
    # bc_stimuli.to_csv("bc_stimuli_example.csv")

    """
    participant_dict = {'chungha': '8', 'dongik': '7', 'eunhye': '1', 'In-Taek': '5', 'jooyeong': '13', 'juchanseo': '3', 'junryeol': '11', 
                        'juyeon': '4', 'myounghun': '9', 'songmin': '10', 'sooyeon': '6', 'woojinkang': '2', 'yeogyeong': '12'}
    
    processed_datadir_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data_ver2'
    logdir_path = 'data/AOI_HitScoring/IdentiGaze_Data'
    example_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data/chungha'
    # whole_dataframe_task1 = pd.DataFrame()
    # whole_dataframe_task2 = pd.DataFrame()
    whole_dataframe = pd.DataFrame()
    for participant in participant_dict:
        p_tasklist = sorted(os.listdir(os.path.join(processed_datadir_path, participant)))
        # p_tasklist.remove('.DS_Store')
        print(participant)
        for idx, foldername in tqdm(enumerate(p_tasklist)):
            session, task = decide_session_and_task(idx, foldername)
            task = 'task 0.7'
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
                fft = first_fixation_time(bc_stimuli, meta, task)
                duration_data = fixation_duration(bc_stimuli, meta, task)
                total_duration_average = duration_data[0]
                total_duration_max = duration_data[1]
                total_duration_min = duration_data[2]
                target_duration_average = duration_data[3]
                target_duration_max = duration_data[4]
                target_duration_min = duration_data[5]
                nontarget_duration_average = duration_data[6]
                nontarget_duration_max = duration_data[7]
                nontarget_duration_min = duration_data[8]
                rt = reaction_time(bc_stimuli, task)
                velocity_data = velocity(bc_stimuli, task)
                total_velocity_average = velocity_data[0]
                total_velocity_max = velocity_data[1]
                total_velocity_min = velocity_data[2]
                hit_velocity_average = velocity_data[3]
                hit_velocity_max = velocity_data[4]
                hit_velocity_min = velocity_data[5]
                nohit_velocity_average = velocity_data[6]
                nohit_velocity_max = velocity_data[7]
                nohit_velocity_min = velocity_data[8]
                onhit_velocity_average = velocity_data[9]
                onhit_velocity_max = velocity_data[10]
                onhit_velocity_min = velocity_data[11]
                mfcc1 = velocity_data[12]
                mfcc2 = velocity_data[13]
                mfcc3 = velocity_data[14]
                mfcc4 = velocity_data[15]
                mfcc5 = velocity_data[16]
                mfcc6 = velocity_data[17]
                mfcc7 = velocity_data[18]
                mfcc8 = velocity_data[19]
                mfcc9 = velocity_data[20]
                mfcc10 = velocity_data[21]
                mfcc11 = velocity_data[22]
                mfcc12 = velocity_data[23]

                pupil_data = pupil(bc_stimuli)
                pupil_average = pupil_data[0]
                pupil_max = pupil_data[1]
                pupil_min = pupil_data[2]
                # print(onhit_velocity_average, onhit_velocity_max, onhit_velocity_min)
                percevie_time = jspirit_func(bc_stimuli, meta, task)
                # print(percevie_time)
                # print(velocity_data)
                # if sum(velocity_data)==0:
                #     print(bc_stimuli[['Gaze point X [DACS px]', 'Gaze point Y [DACS px]']])
                # print(rt)
                # print(fft, task)
                # first_fixation_time(bc_stimuli, meta, task)
                # data_dict = get_gazeXY(bc_stimuli)
                x_yg, y_yg = rotated_path(bc_stimuli, meta)
                data_dict = get_YG_data(x_yg, y_yg, task)
                data_dict.update(get_gazeXY(bc_stimuli))
                data_dict['gaze_hit'] = hit
                data_dict['first_visit_time'] = fft
                data_dict['total_duration_average'] = total_duration_average
                data_dict['total_duration_max'] = total_duration_max
                data_dict['total_duration_min'] = total_duration_min
                data_dict['target_duration_average'] = target_duration_average
                data_dict['target_duration_max'] = target_duration_max
                data_dict['target_duration_min'] = target_duration_min
                data_dict['nontarget_duration_average'] = nontarget_duration_average
                data_dict['nontarget_duration_max'] = nontarget_duration_max
                data_dict['nontarget_duration_min'] = nontarget_duration_min
                data_dict['reaction_time'] = rt
                data_dict['total_velocity_average'] = total_velocity_average
                data_dict['total_velocity_max'] = total_velocity_max
                data_dict['total_velocity_min'] = total_velocity_min
                data_dict['hit_velocity_average'] = hit_velocity_average
                data_dict['hit_velocity_max'] = hit_velocity_max
                data_dict['hit_velocity_min'] = hit_velocity_min
                data_dict['nohit_velocity_average'] = nohit_velocity_average
                data_dict['nohit_velocity_max'] = nohit_velocity_max
                data_dict['nohit_velocity_min'] = nohit_velocity_min
                data_dict['onhit_velocity_average'] = onhit_velocity_average
                data_dict['onhit_velocity_max'] = onhit_velocity_max
                data_dict['onhit_velocity_min'] = onhit_velocity_min
                data_dict['mfcc1'] = mfcc1
                data_dict['mfcc2'] = mfcc2
                data_dict['mfcc3'] = mfcc3
                data_dict['mfcc4'] = mfcc4
                data_dict['mfcc5'] = mfcc5
                data_dict['mfcc6'] = mfcc6
                data_dict['mfcc7'] = mfcc7
                data_dict['mfcc8'] = mfcc8
                data_dict['mfcc9'] = mfcc9
                data_dict['mfcc10'] = mfcc10
                data_dict['mfcc11'] = mfcc11
                data_dict['mfcc12'] = mfcc12
                data_dict['pupil_average'] = pupil_average
                data_dict['pupil_max'] = pupil_max
                data_dict['pupil_min'] = pupil_min
                data_dict['perceive_time'] = percevie_time
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
                whole_dataframe = pd.concat([whole_dataframe, this_df])
                # if task == 'task 0.7':
                #     whole_dataframe_task2 = pd.concat([whole_dataframe_task2, this_df])
                # elif task == 'task 0.5':
                #     whole_dataframe_task1 = pd.concat([whole_dataframe_task1, this_df])
    
    whole_dataframe.to_csv('data/BlueMediumRarePupilMfcc_total.csv', index=False)
    # whole_dataframe_task2.to_csv('data/BlueMediumRarePupilMfcc_task2.csv', index=False)

    # """
                
            

