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
from utils import feature_processing
from metric_revised import *
from DataAnalysis_util import get_fixationXY, gaze_entropy



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

    whole_dataframe = pd.DataFrame()
    raw_gaze_dataframe = pd.DataFrame()
    eye_movement_dataframe = pd.DataFrame()
    fixation_dataframe = pd.DataFrame()
    saccade_dataframe = pd.DataFrame()
    MFCC_dataframe = pd.DataFrame()
    pupil_dataframe = pd.DataFrame()


    for participant in tqdm(participant_dict):
        # print("======")
        # print(participant)
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

                # GroundTruth
                participant_data = int(participant_dict[participant])

                # Raw Gaze
                raw_gaze = get_rawgazeXY(gazeDataFrame)

                # Gaze
                gaze_motion = get_path_length(gazeDataFrame)
                velocity_data = velocity(gazeDataFrame)
                gaze_velocity_avg = velocity_data[0]
                gaze_velocity_max = velocity_data[1]
                gaze_velocity_min = velocity_data[2]
                gaze_velocity_std = velocity_data[3]
                angular_data = angular(gazeDataFrame)
                gaze_rotation_avg, gaze_rotation_max, gaze_rotation_min, gaze_rotation_std = get_list_statistic(angular_data)
                Gaze_dict = {}
                Gaze_dict['gaze_motion'] = gaze_motion
                Gaze_dict['gaze_velocity_avg'] = gaze_velocity_avg
                Gaze_dict['gaze_velocity_max'] = gaze_velocity_max
                Gaze_dict['gaze_velocity_min'] = gaze_velocity_min
                Gaze_dict['gaze_velocity_std'] = gaze_velocity_std
                Gaze_dict['gaze_rotation_avg'] = gaze_rotation_avg
                Gaze_dict['gaze_rotation_max'] = gaze_rotation_max
                Gaze_dict['gaze_rotation_min'] = gaze_rotation_min
                Gaze_dict['gaze_rotation_std'] = gaze_rotation_std

                # Fixation
                rt = reaction_time(gazeDataFrame)
                fixation_duration_data = Fixation_Duration(gazeDataFrame)
                fixation_duration_avg, fixation_duration_max, fixation_duration_min, fixation_duration_std = get_list_statistic(fixation_duration_data)
                fixation_dispersion_data = fixation_dispersion(gazeDataFrame)
                fixation_dispersion_avg = fixation_dispersion_data[0]
                fixation_dispersion_max = fixation_dispersion_data[1]
                fixation_dispersion_min = fixation_dispersion_data[2]
                fixation_dispersion_std = fixation_dispersion_data[3]
                fixation_count = Fixation_Count(gazeDataFrame)
                Fixation_dict = {}
                Fixation_dict['reaction_time'] = rt
                Fixation_dict['fixation_duration_avg'] = fixation_duration_avg
                Fixation_dict['fixation_duration_max'] = fixation_duration_max
                Fixation_dict['fixation_duration_min'] = fixation_duration_min
                Fixation_dict['fixation_duration_std'] = fixation_duration_std
                Fixation_dict['fixation_dispersion_avg'] = fixation_dispersion_avg
                Fixation_dict['fixation_dispersion_max'] = fixation_dispersion_max
                Fixation_dict['fixation_dispersion_min'] = fixation_dispersion_min
                Fixation_dict['fixation_dispersion_std'] = fixation_dispersion_std
                Fixation_dict['fixation_count'] = fixation_count

                # Saccade
                saccade_duration_data = Saccade_Duration(gazeDataFrame)
                saccade_duration_avg, saccade_duration_max, saccade_duration_min, saccade_duration_std = get_list_statistic(saccade_duration_data)
                saccade_velocity_data, saccade_amplitude_data = Saccade_Velocity_Amplitude(gazeDataFrame)
                saccade_velocity_avg, saccade_velocity_max, saccade_velocity_min, saccade_velocity_std = get_list_statistic(saccade_velocity_data)
                saccade_amplitude_avg, saccade_amplitude_max, saccade_amplitude_min, saccade_amplitude_std = get_list_statistic(saccade_amplitude_data)
                saccade_dispersion_data = saccade_dispersion(gazeDataFrame)
                saccade_dispersion_avg = saccade_dispersion_data[0]
                saccade_dispersion_max = saccade_dispersion_data[1]
                saccade_dispersion_min = saccade_dispersion_data[2]
                saccade_dispersion_std = saccade_dispersion_data[3]
                saccade_count = Saccade_Count(gazeDataFrame)
                Saccade_dict = {}
                Saccade_dict['saccade_duration_avg'] = saccade_duration_avg
                Saccade_dict['saccade_duration_max'] = saccade_duration_max
                Saccade_dict['saccade_duration_min'] = saccade_duration_min
                Saccade_dict['saccade_duration_std'] = saccade_duration_std
                Saccade_dict['saccade_velocity_avg'] = saccade_velocity_avg
                Saccade_dict['saccade_velocity_max'] = saccade_velocity_max
                Saccade_dict['saccade_velocity_min'] = saccade_velocity_min
                Saccade_dict['saccade_velocity_std'] = saccade_velocity_std
                Saccade_dict['saccade_amplitude_avg'] = saccade_amplitude_avg
                Saccade_dict['saccade_amplitude_max'] = saccade_amplitude_max
                Saccade_dict['saccade_amplitude_min'] = saccade_amplitude_min
                Saccade_dict['saccade_amplitude_std'] = saccade_amplitude_std
                Saccade_dict['saccade_dispersion_avg'] = saccade_dispersion_avg
                Saccade_dict['saccade_dispersion_max'] = saccade_dispersion_max
                Saccade_dict['saccade_dispersion_min'] = saccade_dispersion_min
                Saccade_dict['saccade_dispersion_std'] = saccade_dispersion_std
                Saccade_dict['saccade_count'] = saccade_count


                # MFCC
                mfcc1 = velocity_data[4]
                mfcc2 = velocity_data[5]
                mfcc3 = velocity_data[6]
                mfcc4 = velocity_data[7]
                mfcc5 = velocity_data[8]
                mfcc6 = velocity_data[9]
                mfcc7 = velocity_data[10]
                mfcc8 = velocity_data[11]
                mfcc9 = velocity_data[12]
                mfcc10 = velocity_data[13]
                mfcc11 = velocity_data[14]
                mfcc12 = velocity_data[15]
                MFCC_dict = {}
                MFCC_dict['mfcc1'] = mfcc1
                MFCC_dict['mfcc2'] = mfcc2
                MFCC_dict['mfcc3'] = mfcc3
                MFCC_dict['mfcc4'] = mfcc4
                MFCC_dict['mfcc5'] = mfcc5
                MFCC_dict['mfcc6'] = mfcc6
                MFCC_dict['mfcc7'] = mfcc7
                MFCC_dict['mfcc8'] = mfcc8
                MFCC_dict['mfcc9'] = mfcc9
                MFCC_dict['mfcc10'] = mfcc10
                MFCC_dict['mfcc11'] = mfcc11
                MFCC_dict['mfcc12'] = mfcc12

                # Pupil
                left_diameter_data = pupilLeft(gazeDataFrame)
                left_diameter_avg = left_diameter_data[0]
                left_diameter_max = left_diameter_data[1]
                left_diameter_min = left_diameter_data[2]
                left_diameter_std = left_diameter_data[3]
                right_diameter_data = pupilRight(gazeDataFrame)
                right_diameter_avg = right_diameter_data[0]
                right_diameter_max = right_diameter_data[1]
                right_diameter_min = right_diameter_data[2]
                right_diameter_std = right_diameter_data[3]
                together_diameter_data = pupil(gazeDataFrame)
                together_diameter_avg = together_diameter_data[0]
                together_diameter_max = together_diameter_data[1]
                together_diameter_min = together_diameter_data[2]
                together_diameter_std = together_diameter_data[3]
                Pupil_dict = {}
                Pupil_dict['pupil_left_avg'] = left_diameter_avg
                Pupil_dict['pupil_left_max'] = left_diameter_max
                Pupil_dict['pupil_left_min'] = left_diameter_min
                Pupil_dict['pupil_left_std'] = left_diameter_std                
                Pupil_dict['pupil_right_avg'] = right_diameter_avg
                Pupil_dict['pupil_right_max'] = right_diameter_max
                Pupil_dict['pupil_right_min'] = right_diameter_min
                Pupil_dict['pupil_right_std'] = right_diameter_std   
                Pupil_dict['pupil_together_avg'] = together_diameter_avg
                Pupil_dict['pupil_together_max'] = together_diameter_max
                Pupil_dict['pupil_together_min'] = together_diameter_min
                Pupil_dict['pupil_together_std'] = together_diameter_std   

                # All
                all_dict = {}
                all_dict.update(raw_gaze)
                all_dict.update(Gaze_dict)
                all_dict.update(Fixation_dict)
                all_dict.update(Saccade_dict)
                all_dict.update(MFCC_dict)
                all_dict.update(Pupil_dict)

                # Make Dataframe
                raw_gaze['participant'] = participant_data
                this_raw_gaze_df = pd.DataFrame(raw_gaze, index=[0])
                raw_gaze_dataframe = pd.concat([raw_gaze_dataframe, this_raw_gaze_df])

                Gaze_dict['participant'] = participant_data
                this_Gaze_df = pd.DataFrame(Gaze_dict, index=[0])
                eye_movement_dataframe = pd.concat([eye_movement_dataframe, this_Gaze_df])

                Fixation_dict['participant'] = participant_data
                this_Fixation_df = pd.DataFrame(Fixation_dict, index=[0])
                fixation_dataframe = pd.concat([fixation_dataframe, this_Fixation_df])

                Saccade_dict['participant'] = participant_data
                this_Saccade_df = pd.DataFrame(Saccade_dict, index=[0])
                saccade_dataframe = pd.concat([saccade_dataframe, this_Saccade_df])

                MFCC_dict['participant'] = participant_data
                this_MFCC_df = pd.DataFrame(MFCC_dict, index=[0])
                MFCC_dataframe = pd.concat([MFCC_dataframe, this_MFCC_df])

                Pupil_dict['participant'] = participant_data
                this_Pupil_df = pd.DataFrame(Pupil_dict, index=[0])
                pupil_dataframe = pd.concat([pupil_dataframe, this_Pupil_df])
                
                all_dict['participant'] = participant_data
                this_all_df = pd.DataFrame(all_dict, index=[0])
                whole_dataframe = pd.concat([whole_dataframe, this_all_df])
    
    whole_dataframe.to_csv('data/Result1/Different_All.csv', index=False)
    raw_gaze_dataframe.to_csv('data/Result1/Different_RawGaze.csv', index=False)
    eye_movement_dataframe.to_csv('data/Result1/Different_EyeMovement.csv', index=False)
    fixation_dataframe.to_csv('data/Result1/Different_Fixation.csv', index=False)
    saccade_dataframe.to_csv('data/Result1/Different_Saccade.csv', index=False)
    MFCC_dataframe.to_csv('data/Result1/Different_MFCC.csv', index=False)
    pupil_dataframe.to_csv('data/Result1/Different_Pupil.csv', index=False)







