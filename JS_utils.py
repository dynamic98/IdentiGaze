from metric_revised import *
import pandas as pd

def extract_all_features(gazeDataFrame: pd.DataFrame):

    # Calculate gaze feature
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


    fixation_count = Fixation_Count(gazeDataFrame)
    saccade_count = Saccade_Count(gazeDataFrame)

    fd_list = Fixation_Duration(gazeDataFrame)
    sd_list = Saccade_Duration(gazeDataFrame)
    mean_fixation_duration = np.mean(fd_list)
    mean_saccade_duration = np.mean(sd_list)

    sv_list, sa_list = Saccade_Velocity_Amplitude(gazeDataFrame)
    mean_saccade_velocity = np.mean(sv_list)
    mean_saccade_amplitude = np.mean(sa_list)

    path_length = get_path_length(gazeDataFrame)
    average_velocity = path_length / 0.7
    
    fixation_dispersion_list = fixation_dispersion(gazeDataFrame)
    saccade_dispersion_list = saccade_dispersion(gazeDataFrame)
    avg_fixation_dispersion = np.mean(fixation_dispersion_list)
    avg_saccade_dispersion = np.mean(saccade_dispersion_list)

    # Make dictionary including all gaze features
    feature_dict = dict()

    feature_dict['reaction_time'] = rt
    feature_dict['total_velocity_average'] = total_velocity_average
    feature_dict['total_velocity_max'] = total_velocity_max
    feature_dict['total_velocity_min'] = total_velocity_min
    feature_dict['mfcc1'] = mfcc1
    feature_dict['mfcc2'] = mfcc2
    feature_dict['mfcc3'] = mfcc3
    feature_dict['mfcc4'] = mfcc4
    feature_dict['mfcc5'] = mfcc5
    feature_dict['mfcc6'] = mfcc6
    feature_dict['mfcc7'] = mfcc7
    feature_dict['mfcc8'] = mfcc8
    feature_dict['mfcc9'] = mfcc9
    feature_dict['mfcc10'] = mfcc10
    feature_dict['mfcc11'] = mfcc11
    feature_dict['mfcc12'] = mfcc12
    feature_dict['pupil_average'] = pupil_average
    feature_dict['pupil_max'] = pupil_max
    feature_dict['pupil_min'] = pupil_min
    feature_dict['fixation_count'] = fixation_count
    feature_dict['saccade_count'] = saccade_count
    feature_dict['fixation_duration_avg'] = mean_fixation_duration
    feature_dict['saccade_duration_avg'] = mean_saccade_duration
    feature_dict['saccade_velocity_avg'] = mean_saccade_velocity
    feature_dict['saccade_amplitude_avg'] = mean_saccade_amplitude
    feature_dict['path_length'] = path_length
    feature_dict['average_velocity'] = average_velocity
    feature_dict['avg_fixation_dispersion'] = avg_fixation_dispersion
    feature_dict['avg_saccade_dispersion'] = avg_saccade_dispersion
    
    return feature_dict