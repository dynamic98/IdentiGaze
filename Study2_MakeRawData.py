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


def pupil(df: pd.DataFrame):
    pupilDiameter = 'Pupil diameter filtered'
    pupilData = df[pupilDiameter]
    pupilData = pupilData.fillna(method='bfill')
    pupilData = pupilData.fillna(method='ffill')
    pupilData = pupilData.to_list()
    pupilStatistic = get_list_statistic(pupilData)
    return pupilStatistic

def get_gazeXY(df: pd.DataFrame):
    x = 'Gaze point X'
    y = 'Gaze point Y'
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

def reaction_time(df: pd.DataFrame):
    max_reactiontime = 0.7
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

def velocity(df:pd.DataFrame):
    total_size = df.index.size
    tick = 0.7/total_size

    gx = 'Gaze point X'
    gy = 'Gaze point Y'
    gxlist_1 = df[gx].to_list()
    gylist_1 = df[gy].to_list()

    gxlist_2 = gxlist_1[1:]
    gylist_2 = gylist_1[1:]
    gxlist_1 = gxlist_1[:-1]
    gylist_1 = gylist_1[:-1]
    total_velocity = []
    for j in range(len(gxlist_1)):
        p = [gxlist_1[j], gylist_1[j]]
        q = [gxlist_2[j], gylist_2[j]]
        this_distance = math.dist(p,q)
        this_velocity = this_distance/tick
        total_velocity.append(this_velocity)
    mfcc_data = mfcc(total_velocity)
   
    total_velocity_statistic = get_list_statistic(total_velocity)
    velocity_data = extend_list(total_velocity_statistic, mfcc_data)

    # plt.bar(list(range(len(gxlist_1))), total_velocity)
    # plt.show()
    return velocity_data

def hammingwindow(array):
    array_length = len(array)
    frames = array*np.array([0.54-0.46*np.cos((2*np.pi*n)/(array_length -1)) for n in range(array_length)])
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
    # results = dct(filter_banks, type=2, axis=1, norm='ortho')
    return results


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



if __name__ == '__main__':
    participant_dict = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,
                        11:10,12:11,13:12,14:13,15:14,17:15,18:16,
                        19:17,20:18,21:19,22:20,23:21,24:22,25:23,
                        26:24,27:25,28:26,29:27,30:28,31:29,32:30,
                        33:31,34:32,35:33}
    
    processed_datadir_path = 'data/data_processed_Study2'
    logdir_path = 'data/madeSet'
    task = 'different'
    blockName = "Block3"

    # whole_dataframe_task1 = pd.DataFrame()
    # whole_dataframe_task2 = pd.DataFrame()
    whole_dataframe = pd.DataFrame()
    for participant in participant_dict:
        print("======")
        print(participant)
        for session in range(1,6):
            feature_log_path = os.path.join(logdir_path, f'{participant}',f"session{session}", f"{task}_set.json")
            with open(os.path.join(feature_log_path), 'r') as f:
                log_json = json.load(f)
            stimuliLength = len(log_json)
            halfLength = stimuliLength//2
            if task == 'different':
                stimuliStrList = ["A", "B"]
            elif task == 'similar':
                stimuliStrList = ["C", "D"]
            for stimuliIndexNum in range(stimuliLength):
                if stimuliIndexNum < halfLength:
                    stimuliIndex = stimuliIndexNum
                    stimuliStr = stimuliStrList[0]
                else:
                    stimuliIndex = stimuliIndexNum%halfLength
                    stimuliStr = stimuliStrList[1]

                levelIndex = log_json[str(stimuliIndexNum)]["level_index"]
                target_list = log_json[str(stimuliIndexNum)]["target_list"]
                targetPath = os.path.join(processed_datadir_path, f"{participant}", f"{session}", f"{stimuliStr}", f"{stimuliIndex}_{blockName}.tsv")
                gazeDataFrame = pd.read_csv(targetPath, sep="\t").iloc[1:]

                rt = reaction_time(gazeDataFrame)
                velocity_data = velocity(gazeDataFrame)
                total_velocity_average = velocity_data[0]
                total_velocity_max = velocity_data[1]
                total_velocity_min = velocity_data[2]
                mfcc1 = velocity_data[3]
                mfcc2 = velocity_data[4]
                mfcc3 = velocity_data[5]
                mfcc4 = velocity_data[6]
                mfcc5 = velocity_data[7]
                mfcc6 = velocity_data[8]
                mfcc7 = velocity_data[9]
                mfcc8 = velocity_data[10]
                mfcc9 = velocity_data[11]
                mfcc10 = velocity_data[12]
                mfcc11 = velocity_data[13]
                mfcc12 = velocity_data[14]

                pupil_data = pupil(gazeDataFrame)
                pupil_average = pupil_data[0]
                pupil_max = pupil_data[1]
                pupil_min = pupil_data[2]

                data_dict = get_gazeXY(gazeDataFrame)
                data_dict['reaction_time'] = rt
                data_dict['total_velocity_average'] = total_velocity_average
                data_dict['total_velocity_max'] = total_velocity_max
                data_dict['total_velocity_min'] = total_velocity_min
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
                data_dict['participant'] = int(participant_dict[participant])
                data_dict['level_index'] = levelIndex
                data_dict['target_list_1'] = target_list[0]
                data_dict['target_list_2'] = target_list[1]
                data_dict['target_list_3'] = target_list[2]
                data_dict['target_list_4'] = target_list[3]
                this_df = pd.DataFrame(data_dict, index=[0])
                whole_dataframe = pd.concat([whole_dataframe, this_df])

    whole_dataframe.to_csv('data/BlueRareStudy2Entire_different_interpolated.csv', index=False)

