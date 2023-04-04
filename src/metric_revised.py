import math
import pandas as pd
import numpy as np

""" column name constant """
PUPIL_DIAMETER_LEFT = 'Pupil diameter left [mm]'
PUPIL_DIAMETER_RIGHT = 'Pupil diameter right [mm]'
PUPIL_VALIDITY_LEFT = 'Validity left'
PUPIL_VALIDITY_RIGHT = 'Validity right'
FIXATION_POINT_X = 'Fixation point X [DACS px]'
FIXATION_POINT_Y = 'Fixation point Y [DACS px]'

TIMESTAMP = 'Recording timestamp'
EYE_MOVEMENT_TYPE = 'Eye movement type'
EYE_MOVEMENT_TYPE_INDEX = 'Eye movement type index'
GAZE_EVENT_DURATION = 'Gaze event duration [ms]'

EYE_POSITION_LEFT_X = 'Eye position left X [DACS mm]'
EYE_POSITION_LEFT_Y = 'Eye position left Y [DACS mm]'
EYE_POSITION_LEFT_Z = 'Eye position left Z [DACS mm]'
EYE_POSITION_RIGHT_X = 'Eye position right X [DACS mm]'
EYE_POSITION_RIGHT_Y = 'Eye position right Y [DACS mm]'
EYE_POSITION_RIGHT_Z = 'Eye position right Z [DACS mm]'


def Avarage_Pupil_Diameter(df):
    avg_left = df[PUPIL_DIAMETER_LEFT].mean(axis=0, skipna=True)
    avg_right = df[PUPIL_DIAMETER_RIGHT].mean(axis=0, skipna=True)
    return avg_left, avg_right

def Pupil_Diameter_per_block(df, sample_size):
    pd_left_list = []
    pd_right_list = []
    row_count = len(df)
    blocks = int(row_count / sample_size)

    # per block of sample size
    for block in range(0, blocks):
        b_start = block * sample_size
        b_end = b_start + sample_size
        df_sampling = df.iloc[b_start:b_end].copy()
        df_sampling[PUPIL_DIAMETER_LEFT].fillna(value=0, inplace=True)
        df_sampling[PUPIL_DIAMETER_RIGHT].fillna(value=0, inplace=True)
        if len(df_sampling) != 0:
            pd_left = df_sampling[PUPIL_DIAMETER_LEFT].mean(axis=0, skipna=True)
            pd_right = df_sampling[PUPIL_DIAMETER_RIGHT].mean(axis=0, skipna=True)
        else:
            pd_left = 0
            pd_right = 0
        pd_left_list.append(round(pd_left, 2))
        pd_right_list.append(round(pd_right, 2))

    # the rest of the rows
    b_start = blocks * sample_size
    df_sampling = df.iloc[b_start:].copy()
    df_sampling[PUPIL_DIAMETER_LEFT].fillna(value=0, inplace=True)
    df_sampling[PUPIL_DIAMETER_RIGHT].fillna(value=0, inplace=True)
    if len(df_sampling) != 0:
        pd_left = df_sampling[PUPIL_DIAMETER_LEFT].mean(axis=0, skipna=True)
        pd_right = df_sampling[PUPIL_DIAMETER_RIGHT].mean(axis=0, skipna=True)
    else:
        pd_left = 0
        pd_right = 0
    pd_left_list.append(round(pd_left, 2))
    pd_right_list.append(round(pd_right, 2))

    assert (len(pd_left_list) == len(pd_right_list))

    return pd_left_list, pd_right_list, blocks



def Fixation_Count(df):
    df_fp = df[df[EYE_MOVEMENT_TYPE].isin(['Fixation'])]
    if len(df_fp) != 0:
        first_index, last_index = (df_fp[EYE_MOVEMENT_TYPE_INDEX].iloc[[0, -1]].values)
        fixation_count = last_index - first_index + 1       # first_index <= x <= last_index : the number of x
    else:
        fixation_count = 0
    return fixation_count

def Saccade_Count(df):
    df_saccade = df[df[EYE_MOVEMENT_TYPE].isin(['Saccade'])]
    if len(df_saccade) != 0:
        first_index, last_index = (df_saccade[EYE_MOVEMENT_TYPE_INDEX].iloc[[0, -1]].values)
        saccade_count = last_index - first_index + 1        # first_index <= x <= last_index : the number of x
    else:
        saccade_count = 0
    return saccade_count


def Stimuli_Duration(df):
    """ get stimuli duration (sec) """
    df_ts = df[TIMESTAMP]       # timestamp column
    start_ts, end_ts = df_ts.iloc[[0,-1]].values    # start timestamp, end timestamp
    video_duration = end_ts - start_ts          # microsec
    video_duration = video_duration / 1000000   # sec
    return video_duration

def Fixation_Rate(df):
    fixation_count = Fixation_Count(df)
    duration = Stimuli_Duration(df)           # sec
    fixationRate = fixation_count/duration
    return fixationRate

def Saccade_Rate(df):
    saccade_count = Saccade_Count(df)
    duration = Stimuli_Duration(df)           # sec
    saccadeRate = saccade_count/duration
    return saccadeRate

def Saccade_Fixation_Ratio(saccade, fixation):
    return saccade/fixation

def fixation_count_per_block2(df, sample_size):
    fc_list = []
    row_count = len(df)
    blocks = int(row_count / sample_size)

    # per block of sample size
    for block in range(0, blocks):
        b_start = block * sample_size
        b_end = b_start + sample_size
        df_sampling = df[b_start:b_end]         
        fixation_count = Fixation_Count(df_sampling)
        fc_list.append(fixation_count)

    # the rest of the rows
    b_start = blocks * sample_size
    df_sampling = df[b_start:] 
    fixation_count = Fixation_Count(df_sampling)
    fc_list.append(fixation_count)

    return fc_list, blocks

def fixation_count_per_block(df, sample_size):
    fc_list = []
    row_count = len(df)
    blocks = int(row_count / sample_size)

    # per block of sample size
    for block in range(0, blocks):
        b_start = block * sample_size
        b_end = b_start + sample_size
        df_sampling = df.iloc[b_start:b_end]         
        fixation_count = Fixation_Count(df_sampling)
        fc_list.append(fixation_count)

    # the rest of the rows
    b_start = blocks * sample_size
    df_sampling = df.iloc[b_start:]
    fixation_count = Fixation_Count(df_sampling)
    fc_list.append(fixation_count)

    return fc_list, blocks

def saccade_count_per_block(df, sample_size):
    sc_list = []
    row_count = len(df)
    blocks = int(row_count / sample_size)

    # per block of sample size
    for block in range(0, blocks):
        b_start = block * sample_size
        b_end = b_start + sample_size
        df_sampling = df.iloc[b_start:b_end] 
        saccade_count = Saccade_Count(df_sampling)
        sc_list.append(saccade_count)

    # the rest of the rows
    b_start = blocks * sample_size
    df_sampling = df.iloc[b_start:]  
    fixation_count = Fixation_Count(df_sampling)
    sc_list.append(fixation_count)

    return sc_list, blocks

def Avg_Peak_Velocity_Saccade(mf):
    return mf["Average_peak_velocity_of_saccades"].values[0]

def Min_Peak_Velocity_Saccade(mf):
    return mf["Minimum_peak_velocity_of_saccades"].values[0]

def Max_Peak_Velocity_Saccade(mf):
    return mf["Maximum_peak_velocity_of_saccades"].values[0]

def Avg_Amplitude_Saccade(mf):
    return mf["Average_amplitude_of_saccades"].values[0]

def Min_Amplitude_Saccade(mf):
    return mf["Minimum_amplitude_of_saccades"].values[0]

def Max_Amplitude_Saccade(mf):
    return mf["Maximum_amplitude_of_saccades"].values[0]

def Avg_Fixation_Duration(mf):
    return mf["Average_duration_of_fixations"].values[0]

def Fixation_Duration(df):
    df = df.drop_duplicates([EYE_MOVEMENT_TYPE, GAZE_EVENT_DURATION, EYE_MOVEMENT_TYPE_INDEX], keep='first')
    df = df.reset_index()
    df_fixation = df[df[EYE_MOVEMENT_TYPE].isin(['Fixation'])]
    fd_list = df_fixation[GAZE_EVENT_DURATION].tolist()

    return fd_list

def Saccade_Duration(df):
    df = df.drop_duplicates([EYE_MOVEMENT_TYPE, GAZE_EVENT_DURATION, EYE_MOVEMENT_TYPE_INDEX], keep='first')
    df = df.reset_index()
    df_saccade = df[df[EYE_MOVEMENT_TYPE].isin(['Saccade'])]
    sd_list = df_saccade[GAZE_EVENT_DURATION].tolist()

    return sd_list

def Saccade_Velocity_Amplitude(df):
    df = df.drop_duplicates([EYE_MOVEMENT_TYPE, GAZE_EVENT_DURATION, EYE_MOVEMENT_TYPE_INDEX], keep='first')
    df = df.reset_index()
    df_saccade = df[df[EYE_MOVEMENT_TYPE].isin(['Saccade'])]
    saccade_indexs = df_saccade.index

    process_count = 1
    sa_list = []
    sv_list = []
    for i in saccade_indexs:
        try:
            if df.iloc[i-1][EYE_MOVEMENT_TYPE] == 'Fixation' and df.iloc[i+1][EYE_MOVEMENT_TYPE] == 'Fixation' \
                    and not np.isnan(df.iloc[i-1][EYE_POSITION_LEFT_X]):

                x1, y1 = df.iloc[i-1][FIXATION_POINT_X], df.iloc[i-1][FIXATION_POINT_Y]
                x2, y2 = df.iloc[i+1][FIXATION_POINT_X], df.iloc[i+1][FIXATION_POINT_Y]

                l_x3, l_y3, l_z3 = df.iloc[i - 1][EYE_POSITION_LEFT_X], df.iloc[i - 1][EYE_POSITION_LEFT_Y], df.iloc[i - 1][EYE_POSITION_LEFT_Z]
                r_x3, r_y3, r_z3 = df.iloc[i - 1][EYE_POSITION_RIGHT_X], df.iloc[i - 1][EYE_POSITION_RIGHT_Y], df.iloc[i - 1][EYE_POSITION_RIGHT_Z]
                x3, y3, z3 = max([l_x3, r_x3]), max([l_y3, r_y3]), max([l_z3, r_z3])

                a = np.array([x1, y1, 0]) - np.array([x3, y3, z3])
                b = np.array([x2, y2, 0]) - np.array([x3, y3, z3])
                degree = calculate_degree(a,b)

                duration = df.iloc[i][GAZE_EVENT_DURATION] / 1000       # second
                velocity = degree / duration
                sv_list.append(velocity)
                sa_list.append(degree)

                process_count += 1
            else:
                continue
        except IndexError:
            # last index
            break

    # print(process_count)
    return sv_list, sa_list

def calculate_degree(a, b):
    def dist(v):
        return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    distA = dist(a)
    distB = dist(b)

    # theta
    ip = np.dot(a, b)
    ip2 = distA * distB
    cost = round(ip / ip2,4)
    x = math.acos(cost)
    deg_x = math.degrees(x)
    # print(deg_x)

    # 180 - theta
    ip = np.dot(a, -b)
    ip2 = distA * distB
    cost = round(ip / ip2,4)
    x2 = math.acos(cost)
    deg_x2 = math.degrees(x2)
    # print(deg_x2)

    return min([deg_x, deg_x2])