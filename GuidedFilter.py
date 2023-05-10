import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from gazeheatplot import gaussian, draw_display
from GazeStimuliImage import re_generate_stimuli
from preattentive_object import PreattentiveObject



def hue_heatmap(color:str, img:np.array, alpha=0.5, gaussianwh=200, gaussiansd=None):
    color = color.lower()
    if color == 'r' or color == 'red':
        chl = 0
    elif color == 'g' or color == 'green':
        chl = 1
    elif color == 'b' or color == 'blue':
        chl = 2
    else:
        raise Exception("color should be one of red, green and blue.")
    chl_image = img[:,:,chl]
    h,w = chl_image.shape
    small_image = cv2.resize(chl_image, (200,200), cv2.INTER_AREA)
    normalized_image = small_image.astype(np.float128)/255
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    fig, ax = draw_display((w,h), imagefile=None, image=img)
    heatmapsize = (int(h + 2 * strt), int(w + 2 * strt))
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(200):
        for j in range(200):
            # get x and y coordinates
            x = int(strt + (i/200*w) - int(gwh / 2))
            y = int(strt + (j/200*h) - int(gwh / 2))
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
                    heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * normalized_image[j,i]
                except:
                    # fixation was probably outside of display
                    pass
            else:
                # add Gaussian to the current heatmap
                heatmap[y:y + gwh, x:x + gwh] += gaus * normalized_image[j,i]
    # resize heatmap
    heatmap = heatmap[strt:h + strt, strt:w + strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = np.NaN
    print(heatmap)
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)
    ax.invert_yaxis()
    plt.show()



if __name__=='__main__':
    preattentive_object = PreattentiveObject(1920, 1080, 'black')
    log_example = {"task": "size", "shape_target": 1, "shape_distractor": 1, "set_size": 6, "target_cnt": [1245, 483], "target_size": 60, 
                   "distractor_size": 30, "target_color": [82, 127, 62], "distractor_color": [82, 127, 62], "target_orientation": None, 
                   "distractor_orientation": None, "features": "features/eunhye_1 session_task 0.5.csv", "area": [1145, 383, 1345, 583]}
    rgb_bg = re_generate_stimuli(preattentive_object, log_example)
    hue_heatmap(color='r', img=rgb_bg)


