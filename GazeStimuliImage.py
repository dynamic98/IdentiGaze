import matplotlib.pyplot as plt
import json
import cv2
from ML_MakeRawdata import feature_load
from preattentive_object import PreattentiveObject
from ML_Analysis_rawdata import *
from ML_MakeRawdata import *
from SingleGazeAnalysis import MetaAnalysis
from gazeheatplot import draw_heatmap

def re_generate_stimuli(preattentive_object, example):
    myMeta = MetaAnalysis(example)
    task = myMeta.get_task()
    if task == 'size':
        bg, _  = preattentive_object.stimuli_size(False, **example)
    elif task == 'shape':
        bg, _ = preattentive_object.stimuli_shape(False, **example)
    elif task == 'hue':
        bg, _ = preattentive_object.stimuli_hue(False, **example)
    elif task == 'brightness':
        bg, _ = preattentive_object.stimuli_brightness(False, **example)
    elif task == 'orientation':
        bg, _ = preattentive_object.stimuli_orientation(False, **example)
    rgb_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    return rgb_bg


if __name__=='__main__':
    # myProto = ProtoTypeTask()
    preattentive_object = PreattentiveObject(1920, 1080, 'black')
    # data = 'data/AOI_HitScoring/IdentiGaze_data/P1_eunhye/1 session/task 0.5/'
    # log_json = feature_load(data)
    # example = log_json["1"]
    # for iter in log_json:
    #     stimuli_log = log_json[iter]
    #     bg = re_generate_stimuli(stimuli_log)

    #     plt.imshow(bg)
    #     plt.show()

    participant_dict = {'chungha': '8', 'dongik': '7', 'eunhye': '1', 'In-Taek': '5', 'jooyeong': '13', 'juchanseo': '3', 'junryeol': '11', 
                        'juyeon': '4', 'myounghun': '9', 'songmin': '10', 'sooyeon': '6', 'woojinkang': '2', 'yeogyeong': '12'}
    processed_datadir_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data'
    logdir_path = 'data/AOI_HitScoring/IdentiGaze_Data'
    example_path = 'data/AOI_HitScoring/IdentiGaze_Processed Data/chungha'
    whole_dataframe_task1 = pd.DataFrame()
    whole_dataframe_task2 = pd.DataFrame()
    for participant in participant_dict:
        p_tasklist = sorted(os.listdir(os.path.join(processed_datadir_path, participant)))
        p_tasklist.remove('.DS_Store')
        for idx, foldername in enumerate(p_tasklist):
            session, task = decide_session_and_task(idx, foldername)
            feature_log_path = os.path.join(logdir_path, f'P{participant_dict[participant]}_{participant}',session, task)
            log_json = feature_load(feature_log_path)
            for iter in range(1,101):
                dataname = f'{iter}_hit.csv'
                data_df = pd.read_csv(os.path.join(processed_datadir_path, participant, foldername, dataname), index_col=0)
                bc_stimuli = slice_stimuli(data_df, task)
                gaze_list = get_gazeXY_for_heatmap(bc_stimuli)
                log_stimuli = log_json[f'{iter}']
                rgb_bg = re_generate_stimuli(preattentive_object, log_stimuli) # colorscale of this image is BGR (Blue:0, Green:1, Red: 2)
                # bg_black = np.zeros_like(rgb_bg)
                plt.imshow(rgb_bg)
                plt.show()
                # draw_heatmap(gaze_list, (1920, 1080), alpha=0.5, savefilename=None, imagefile=None, gaussianwh=200, gaussiansd=None, image=rgb_bg)

