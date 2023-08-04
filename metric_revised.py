import math
import pandas as pd
import numpy as np
from scipy.fftpack import dct


""" column name constant """
PUPIL_DIAMETER_LEFT = 'Pupil diameter left'
PUPIL_DIAMETER_RIGHT = 'Pupil diameter right'
PUPIL_VALIDITY_LEFT = 'Validity left'
PUPIL_VALIDITY_RIGHT = 'Validity right'
FIXATION_POINT_X = 'Fixation point X'
FIXATION_POINT_Y = 'Fixation point Y'

TIMESTAMP = 'Recording timestamp'
EYE_MOVEMENT_TYPE = 'Eye movement type'
EYE_MOVEMENT_TYPE_INDEX = 'Eye movement type index'
GAZE_EVENT_DURATION = 'Gaze event duration'

EYE_POSITION_LEFT_X = 'Eye position left X (DACSmm)'
EYE_POSITION_LEFT_Y = 'Eye position left Y (DACSmm)'
EYE_POSITION_LEFT_Z = 'Eye position left Z (DACSmm)'
EYE_POSITION_RIGHT_X = 'Eye position right X (DACSmm)'
EYE_POSITION_RIGHT_Y = 'Eye position right Y (DACSmm)'
EYE_POSITION_RIGHT_Z = 'Eye position right Z (DACSmm)'


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



def get_Fixation(df: pd.DataFrame):
    return df[df[EYE_MOVEMENT_TYPE].isin(['Fixation'])]

def get_Saccade(df: pd.DataFrame):
    return df[df[EYE_MOVEMENT_TYPE].isin(['Saccade'])]

def get_path_length(df: pd.DataFrame):
    x_list, y_list = get_gazeXY(df)

    dist_list = []
    for i in range(0, len(x_list)-1):
        x1 = x_list[i]
        y1 = y_list[i]
        x2 = x_list[i+1]
        y2 = y_list[i+1]
        dist = np.sqrt((x2 - x1)**2 + (y2-y1)**2)
        dist_list.append(dist)

    return np.sum(dist_list)


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

def angular(df:pd.DataFrame):

    gx = 'Gaze point X'
    gy = 'Gaze point Y'
    x_list = df[gx].to_list()
    y_list = df[gy].to_list()

    angle_changes = []
    for i in range(1, len(x_list) -1):
        dx2 = x_list[i+1] - x_list[i]
        dy2 = y_list[i+1] - y_list[i]
        angle_now = np.arctan2(dy2, dx2)
        
        if len(angle_changes) == 0:
            angle_changes.append(angle_now)
        else:
            angle_past = angle_changes[-1]
            angle_diff = angle_now - angle_past
            angle_diff = np.abs((angle_diff + np.pi) % (2*np.pi) - np.pi)
            angle_changes.append(angle_now)

    return angle_changes


def pupil(df: pd.DataFrame):
    pupilDiameter = 'Pupil diameter filtered'
    pupilData = df[pupilDiameter]
    pupilData = pupilData.fillna(method='bfill')
    pupilData = pupilData.fillna(method='ffill')
    pupilData = pupilData.to_list()
    pupilStatistic = get_list_statistic(pupilData)
    return pupilStatistic

def pupilLeft(df: pd.DataFrame):
    pupilDiameter = PUPIL_DIAMETER_LEFT
    pupilData = df[pupilDiameter]
    pupilData = pupilData.fillna(method='bfill')
    pupilData = pupilData.fillna(method='ffill')
    pupilData = pupilData.to_list()
    pupilStatistic = get_list_statistic(pupilData)
    return pupilStatistic

def pupilRight(df: pd.DataFrame):
    pupilDiameter = PUPIL_DIAMETER_RIGHT
    pupilData = df[pupilDiameter]
    pupilData = pupilData.fillna(method='bfill')
    pupilData = pupilData.fillna(method='ffill')
    pupilData = pupilData.to_list()
    pupilStatistic = get_list_statistic(pupilData)
    return pupilStatistic

def extend_list(*arg):
    result = []
    for i in arg:
        result.extend(i)
    return result

def get_list_statistic(data_list):
    if len(data_list)>0:
        average_value = np.mean(data_list)
        max_value = max(data_list)
        min_value = min(data_list)
        std_value = np.std(data_list)
    else:
        average_value = 0
        max_value = 0
        min_value = 0
        std_value = 0
    return [average_value, max_value, min_value, std_value]


def get_gazeXY(df: pd.DataFrame):
    x = 'Gaze point X'
    y = 'Gaze point Y'
    x_data = df[x]
    x_data = x_data.fillna(method='bfill')
    x_data = x_data.fillna(method='ffill')
    x_data = x_data.to_list()
    
    y_data = df[y]
    y_data = y_data.fillna(method='bfill')
    y_data = y_data.fillna(method='ffill')
    y_data = y_data.to_list()
    # xy_dict = {}
    # for i in range(frame):
    #     xy_dict[f'x{i+1}'] = x_data[i]
    #     xy_dict[f'y{i+1}'] = y_data[i]
    return x_data, y_data

def get_rawgazeXY(df: pd.DataFrame):
    x = 'Gaze point X'
    y = 'Gaze point Y'
    x_data = df[x]
    x_data = x_data.fillna(method='bfill')
    x_data = x_data.fillna(method='ffill')
    x_data = x_data.to_list()
    
    y_data = df[y]
    y_data = y_data.fillna(method='bfill')
    y_data = y_data.fillna(method='ffill')
    y_data = y_data.to_list()
    xy_dict = {}

    for i in range(84):
        xy_dict[f'x{i+1}'] = x_data[i]
        xy_dict[f'y{i+1}'] = y_data[i]
    return xy_dict

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


def fixation_dispersion(df: pd.DataFrame):
    f_df = get_Fixation(df)
    fixation_indices = f_df[EYE_MOVEMENT_TYPE_INDEX].unique()
    
    dispersion_list = []
    for fidx in fixation_indices:
        fixation_rows = f_df[f_df[EYE_MOVEMENT_TYPE_INDEX] == fidx]
        x_list, y_list = get_gazeXY(fixation_rows)
        _, x_max, x_min,_ = get_list_statistic(x_list)
        _, y_max, y_min,_ = get_list_statistic(y_list)
        
        dispersion = (x_max - x_min) + (y_max - y_min)
        dispersion_list.append(dispersion)
    
    return get_list_statistic(dispersion_list)

def saccade_dispersion(df: pd.DataFrame):
    f_df = get_Saccade(df)
    saccade_indices = f_df[EYE_MOVEMENT_TYPE_INDEX].unique()
    
    dispersion_list = []
    for sidx in saccade_indices:
        saccade_rows = f_df[f_df[EYE_MOVEMENT_TYPE_INDEX] == sidx]
        x_list, y_list = get_gazeXY(saccade_rows)
        _, x_max, x_min, _ = get_list_statistic(x_list)
        _, y_max, y_min, _ = get_list_statistic(y_list)
        
        dispersion = (x_max - x_min) + (y_max - y_min)
        dispersion_list.append(dispersion)
    
    return get_list_statistic(dispersion_list)



def skewness(df: pd.DataFrame):
    """
    skewness (왜도): 분포의 비대칭도.

    - 정규분포 = 왜도 0
    - 왼쪽으로 치우침 = 왜도 > 0
    - 오른쪽으로 치우침 = 왜도 < 0
   
    x_skew < 0 : 시선이 오른쪽으로 치우침
    y_skew < 0 : 시선이 아래로 치우침
    """

    from scipy.stats import skew
    
    x_list, y_list = get_gazeXY(df)
    x_skew = skew(x_list)
    y_skew = skew(y_list)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # print(x_skew, y_skew)
    # sns.distplot(x_list)
    # plt.show()
    # sns.distplot(y_list)
    # plt.show()

    return x_skew, y_skew

def kurtosis(df: pd.DataFrame):
    """
    Kurtosis (첨도): 확률분포의 뾰족한 정도.
    
    - 정규분포 = 첨도 0(Pearson 첨도 = 3)
    - 위로 뾰족함 = 첨도 > 0(Pearson 첨도 >3) : 시선의 분포가 밀집해있다
    - 아래로 뾰족함 = 첨도 < 0 (Pearson 첨도 < 3) : 시선의 분포가 퍼져있다
    
    """

    from scipy.stats import kurtosis
    
    
    x_list, y_list = get_gazeXY(df)
    x_kurtosis = kurtosis(x_list) 
    y_kurtosis = kurtosis(y_list)

    import matplotlib.pyplot as plt
    import seaborn as sns
    print(x_kurtosis, y_kurtosis)
    sns.distplot(x_list)
    plt.show()
    sns.distplot(y_list)
    plt.show()

    return x_kurtosis, y_kurtosis