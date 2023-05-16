import os
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

def draw_imagemap(stimuli, background:np.array):
    rgb_image = stimuli
    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    image_float = grayscale_image.astype(np.float64)
    return np.add(background,image_float)

def minmaxscale_image(array):
    min_value = np.amin(array)
    max_value = np.amax(array)
    rescaled = (array - min_value)/(max_value-min_value)
    image = (rescaled*255).astype(np.uint8)
    return image

class LoadBlueRawData:
    def __init__(self, datapath:str) -> None:
        self.datapath = datapath
        data = pd.read_csv(self.datapath)
        data = data.reset_index()
        data = data.dropna(axis=0)
        # data = data.reset_index()
        self.data = data
        self.preobj = preobj(1920,1080,'black')
        self.gazeXY_columns()

    def stimuli(self, index:int):
        # return RGB colorscale stimuli
        meta = self.take_meta(index)
        rgb_stimuli = regesti(self.preobj, meta)
        return rgb_stimuli

    def take_meta(self, index:int):
        this_data = self.data.loc[index]
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
    
    def take_targetnum(self, index:int):
        this_meta = self.take_meta(index)
        self.preobj.set_set_size(this_meta['set_size'])
        grid_list = self.preobj.calc_grid(this_meta['distractor_size'])
        target_num = self.preobj.take_targetnum(grid_list, this_meta['target_cnt'])
        return target_num
    
    def take_hit_bool(self, index:int):
        this_data = self.data.loc[index]
        hit = int(this_data.loc['gaze_hit'])
        return True if hit == 1 else False

    def take_specific(self, index:int, *args):
        this_data = self.data.loc[index]
        specific_data = []
        for arg in args:
            specific_data.append(this_data.loc[f'{arg}'])
        return specific_data

    def take_participant(self, index:int):
        this_data = self.data.loc[index]
        participant = int(this_data.loc['participant'])
        return participant

    def take_gaze(self, index:int):
        this_data = self.data.loc[index]
        gazeX_list = this_data.loc[self.gazeX].to_list()
        gazeY_list = this_data.loc[self.gazeY].to_list()
        gaze_data = [[gazeX_list[i], gazeY_list[i], 1] for i in range(len(gazeX_list))]
        return gaze_data
    
    def take_indexlist(self):
        self.data.index.value

    def gazeXY_columns(self):
        columns = self.data.columns
        gazeX = [i for i in columns if i.startswith("x")]
        gazeY = [i for i in columns if i.startswith("y")]
        self.gazeX = gazeX
        self.gazeY = gazeY

    def get_indexlist(self):
        return self.data.index.to_list()

    def get_indexamount(self):
        return self.data.index.size



if __name__=='__main__':
    stimuli_list = ['shape','size','hue','brightness','orientation']
    bg = np.zeros((1080,1920))
    array_dict = {'shape':bg.copy(), 'size':bg.copy(), 'hue':bg.copy(), 'brightness':bg.copy(), 'orientation':bg.copy()}
    for task, datapath in enumerate(['data/blue_rawdata_task1.csv', 'data/blue_rawdata_task2.csv']):
    # datapath = 'data/blue_rawdata_task1.csv'
    # datapath = 'data/blue_rawdata_task2.csv'
        avaliable_location = [[] for i in range(77)]
        myraw = LoadBlueRawData(datapath)
        participant_heatmaps = {1:array_dict.copy(), 2:array_dict.copy(), 3:array_dict.copy(), 4:array_dict.copy(), 5:array_dict.copy(), 6:array_dict.copy(), 7:array_dict.copy(),
                                8:array_dict.copy(), 9:array_dict.copy(), 10:array_dict.copy(), 11:array_dict.copy(), 12:array_dict.copy(), 13:array_dict.copy()}
        participant_imagemaps = {1:array_dict.copy(), 2:array_dict.copy(), 3:array_dict.copy(), 4:array_dict.copy(), 5:array_dict.copy(), 6:array_dict.copy(), 7:array_dict.copy(),
                                8:array_dict.copy(), 9:array_dict.copy(), 10:array_dict.copy(), 11:array_dict.copy(), 12:array_dict.copy(), 13:array_dict.copy()}
        # myraw.take_specific()
        # for i in tqdm(range(myraw.get_indexamount())):
        # # for i in tqdm(range(100)):
        #     gaze_list = myraw.take_gaze(i)
        #     participant = myraw.take_participant(i)
        #     meta = myraw.take_meta(i)
        #     set_size_num = meta['set_size']-4
        #     target_num = myraw.take_targetnum(i)
        #     if set_size_num == 0:
        #         avaliable_location_num = target_num
        #     elif set_size_num == 1:
        #         avaliable_location_num = 16+target_num
        #     elif set_size_num == 2:
        #         avaliable_location_num = 41+target_num
        #     avaliable_location[avaliable_location_num].append(i)
        #     # stimuli = meta['task']
        #     # participant_heatmaps[participant][stimuli] = draw_heatmap(gaze_list,participant_heatmaps[participant][stimuli])
        #     # participant_imagemaps[participant][stimuli] = draw_imagemap(myraw.stimuli(i),participant_imagemaps[participant][stimuli])

        #     # plt.imshow(participant_heatmaps[participant])
        #     # plt.show()
        # with open(os.path.join('feature_analysis',f'location_task{task+1}.txt'), 'w') as p:
        #     for line in avaliable_location:
        #         for element in line:
        #             p.write(f"{element}\t")
        #         p.write("\n")


        """
        for p in participant_heatmaps:
            if os.path.exists(f'feature_analysis/participant_{p}'):
                pass
            else:
                os.makedirs(f'feature_analysis/participant_{p}')
            
            for stimuli in stimuli_list:
                if os.path.exists(f'feature_analysis/participant_{p}/{stimuli}'):
                    pass
                else:
                    os.makedirs(f'feature_analysis/participant_{p}/{stimuli}')

                plt.imshow(participant_heatmaps[p][stimuli], cmap='jet')
                plt.savefig(f'feature_analysis/participant_{p}/{stimuli}/task{task+1}_heatmap.png')
                plt.close()
                image_from_imagemap = minmaxscale_image(participant_imagemaps[p][stimuli])
                cv2.imwrite(f'feature_analysis/participant_{p}/{stimuli}/task{task+1}_imagemap.png',image_from_imagemap)
                # plt.savefig(f'feature_analysis/participant_{p}/{stimuli}/task{task+1}_imagemap.png')
                # plt.close()
        """


