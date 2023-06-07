import os
import numpy as np
import pandas as pd
import cv2
from GazeStimuliImage import re_generate_stimuli as regesti
from preattentive_object import PreattentiveObject as preobj
from ML_Analysis_OptimalStimuli import LoadSelectiveData

def resize_img(img):
    img = img[200:880,620:1300]
    resize_img = cv2.resize(img, dsize=(200,200), interpolation=cv2.INTER_AREA )
    img_arr = np.array(resize_img, dtype=np.float_)
    return img_arr


if __name__ == '__main__':
    this_preobj = preobj(1920,1080,'black')
    path = 'data/blue_medium_data_task1.csv'
    myData = LoadSelectiveData(path)
    myData.set_domain_except('cnt_x','cnt_y','Task Encoding', 'Similarity Encoding')
    total_df_x = myData.take_x()
    total_df_y = myData.take_y()
    total_df = pd.concat([total_df_x, total_df_y], axis=1)
    total_df.to_csv('data/DL_data_task1.csv')

    new_df_array = myData.get_data().apply(lambda x: resize_img(regesti(this_preobj, myData.take_meta(x.name))), axis=1)
    new_df_array = np.array(new_df_array.to_list())
    np.savetxt("data/DL_Image_task1.csv", new_df_array, delimiter=',')


    