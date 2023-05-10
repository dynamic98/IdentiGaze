import os
from turtle import back
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from gazeheatplot import gaussian
from GazeStimuliImage import re_generate_stimuli as regesti
from preattentive_object import PreattentiveObject as preobj


def draw_heatmap(gazepoints, background:np.array, gaussianwh=200, gaussiansd=None):
    h,w = background.shape
    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = (int(h + 2 * strt), int(w + 2 * strt))
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = int(strt + gazepoints[i][0] - int(gwh / 2))
        y = int(strt + gazepoints[i][1] - int(gwh / 2))
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < w) or (not 0 < y < h):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif w < x:
                hadj[1] = gwh - int(x - w)
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif h < y:
                vadj[1] = gwh - int(y - h)
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:h + strt, strt:w + strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = 0
    return np.add(background, heatmap)


class LoadBlueRawData:
    def __init__(self, datapath:str) -> None:
        self.datapath = datapath
        data = pd.read_csv(self.datapath)
        data = data.dropna(axis=0)
        data = data.reset_index()
        self.data = data
        self.gazeXY_columns()

    def stimuli(self, index:int):
        # return RGB colorscale stimuli
        meta = self.take_meta(index)
        pre_obj = preobj(1920,1080,'black')
        rgb_stimuli = regesti(pre_obj, meta)
        return rgb_stimuli

    def take_meta(self, index:int):
        this_data = self.data.iloc[index]
        task = this_data.loc['task']
        if task != 'orientation':
            shape_target = int(this_data.loc['shape_target'])
            shape_distractor = int(this_data.loc['shape_distractor'])
            target_orientation = None
            distractor_orientation = None
        else:
            shape_target = 'orientation'
            shape_distractor = 'orientation'
            target_orientation = int(this_data.loc['target_orientation'])
            distractor_orientation = int(this_data.loc['distractor_orientation'])
        set_size = int(this_data.loc['set_size'])
        target_cnt = [this_data.loc['cnt_x'],this_data.loc['cnt_y']]
        target_size = int(this_data.loc['target_size'])
        distractor_size = int(this_data.loc['distractor_size'])
        target_color = [int(this_data.loc['target_color_b']),int(this_data.loc['target_color_g']),int(this_data.loc['target_color_r'])]
        distractor_color = [int(this_data.loc['distractor_color_b']),int(this_data.loc['distractor_color_g']),int(this_data.loc['distractor_color_r'])]

        meta_dict = {'task':task, 'shape_target':shape_target, 'shape_distractor':shape_distractor,'set_size':set_size, 'target_cnt':target_cnt,
                     'target_size':target_size, 'distractor_size':distractor_size, 'target_color':target_color, 'distractor_color':distractor_color,
                     'target_orientation':target_orientation, 'distractor_orientation':distractor_orientation}
        return meta_dict
    
    def take_participant(self, index:int):
        this_data = self.data.iloc[index]
        participant = int(this_data.loc['participant'])
        return participant

    def take_gaze(self, index:int):
        this_data = self.data.iloc[index]
        gazeX_list = this_data.loc[self.gazeX].to_list()
        gazeY_list = this_data.loc[self.gazeY].to_list()
        gaze_data = [[gazeX_list[i], gazeY_list[i], 1] for i in range(len(gazeX_list))]
        return gaze_data

    def gazeXY_columns(self):
        columns = self.data.columns
        gazeX = [i for i in columns if i.startswith("x")]
        gazeY = [i for i in columns if i.startswith("y")]
        self.gazeX = gazeX
        self.gazeY = gazeY

    def get_data(self):
        return self.data

    def get_indexamount(self):
        return self.data.index.size


if __name__=='__main__':
    for task, datapath in enumerate(['data/blue_rawdata_task1.csv', 'data/blue_rawdata_task2.csv']):
    # datapath = 'data/blue_rawdata_task1.csv'
    # datapath = 'data/blue_rawdata_task2.csv'
        myraw = LoadBlueRawData(datapath)
        bg = np.zeros((1080,1920))
        participant_heatmaps = {1:bg.copy(), 2:bg.copy(), 3:bg.copy(), 4:bg.copy(), 5:bg.copy(), 6:bg.copy(), 7:bg.copy(),
                                8:bg.copy(), 9:bg.copy(), 10:bg.copy(), 11:bg.copy(), 12:bg.copy(), 13:bg.copy()}
        for i in tqdm(range(myraw.get_indexamount())):
            gaze_list = myraw.take_gaze(i)
            participant = myraw.take_participant(i)
            participant_heatmaps[participant] = draw_heatmap(gaze_list,participant_heatmaps[participant])
            # plt.imshow(participant_heatmaps[participant])
            # plt.show()
        
        for p in participant_heatmaps:
            plt.imshow(participant_heatmaps[p], cmap='jet')
            plt.savefig(f'feature_analysis/participant_{p}_task{task+1}.png')
            plt.close()



