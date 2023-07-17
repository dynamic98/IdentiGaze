import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from preattentive_second import PreattentiveObjectSecond

class Study2Analysis:
    def __init__(self, participant, session, stimuli) -> None:
        self.participant = participant
        self.session = session
        self.stimuli = stimuli
        self.preattentive_second = PreattentiveObjectSecond(1920, 1080, 'black')
        self.takeJSON()
    
    def takeJSON(self):
        if self.stimuli == "A":
            self.jsonName = "different_set.json"
            self.indexList = list(range(0,128))
        elif self.stimuli == "B":
            self.jsonName = "different_set.json"
            self.indexList = list(range(128,256))
        elif self.stimuli == "C":
            self.jsonName = "similar_set.json"
            self.indexList = list(range(0,98))
        elif self.stimuli == "D":
            self.jsonName = "similar_set.json"
            self.indexList = list(range(98,196))
        path = os.path.join('data/madeSet', f"{self.participant}", f"session{self.session}", f"{self.jsonName}")
        with open(path, "r") as f:
            self.indexData = json.load(f)

    def takeBg(self, indexNum):
        """
        indexNum 번째 stimuli image를 가져옴

        Parameters
        ----------
        indexNum : int
            A,B stimuli의 경우 0~128, C,D stimuli의 경우 0~98

        Returns
        -------
        rgb_bg : ndarray
            stimuli의 rgb image

        Examples
        --------
        >>> takeBg(1)
        """

        index = self.indexList[indexNum]
        if self.stimuli == "A" or self.stimuli == "B":
            levelIndex = self.indexData[str(index)]['level_index']
            targetList = self.indexData[str(index)]['target_list']
            bg = self.preattentive_second.stimuli_shape(targetList, levelIndex)

        elif self.stimuli == "C" or self.stimuli == "D":
            indexData = self.indexData[str(index)]['index_data']
            targetList = self.indexData[str(index)]['target_list']
            visualComponent = self.indexData[str(index)]['stimuli']
            bg = self.preattentive_second.stimuli_similar(visualComponent, targetList, indexData)
            
        # bg는 opencv를 통해 만든 이미지이기 때문에 BGR 컬러스케일을 가짐. matplotlib pyplot에서 imshow를 하기 위해 RGB 컬러스케일로 바꿔줌
        rgb_bg = np.dstack((bg[:,:,2], bg[:,:,1], bg[:,:,0]))
        return rgb_bg
    
    def takeGaze(self, indexNum, blockName):
        """
        indexNum 번째 blockName의 raw gaze dataframe을 가져옴

        Parameters
        ----------
        indexNum : int
            A,B stimuli의 경우 0~128, C,D stimuli의 경우 0~98
        blockName : str
            Block1 > 십자가, Block2 > 암전, Block3 > stimuli, Block4 > 암전

        Returns
        -------
        gazeDataFrame : pd.dataframe
            토비에서 갓 추출된 따끈따끈한 해당 블록의 raw data

        Examples
        --------
        >>> takeGaze(1, "Block3")
        """
        savepath = "data/data_processed_Study2"
        targetPath = os.path.join(savepath, f"{self.participant}", f"{self.session}", f"{self.stimuli}", f"{indexNum}_{blockName}.tsv")
        gazeDataFrame = pd.read_csv(targetPath, sep="\t")
        
        return gazeDataFrame

def get_gazeXY(df: pd.DataFrame):
    """
    데이터프레임에서 raw gaze xy 좌표 리스트를 추출함

    Parameters
    ----------
    df : DataFrame
        토비에서 갓 추출된 따끈따끈한 해당 블록의 raw data

    Returns
    -------
    x_data : list
        raw gaze의 x 좌표 리스트
    y_data : list
        raw gaze의 y 좌표 리스트

    Examples
    --------
    >>> get_gazeXY(myStudy2Analysis.takeGaze(1, "Block3"))
    """
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
    # for i in range(len(x_data)):
    #     xy_dict[f'x{i+1}'] = x_data[i]
    #     xy_dict[f'y{i+1}'] = y_data[i]
    return x_data, y_data


def gaze_plot(x_list, y_list, bg=np.array([0])):
    """
    gaze를 시각화함. stimuli 위에 덧그릴 수 있음

    Parameters
    ----------
    x_list : list
        gaze의 x 좌표 리스트
    y_list : list
        gaze의 y 좌표 리스트
    bg : ndarray
        stimuli 위에 덧그리고 싶으면 해당 rgb_bg를 imput으로 넣어주셈

    Returns
    -------
    void

    Examples
    --------
    >>> gaze_plot(x_list, y_list, bg)
    """
    if bg.any() > 0:
        plt.imshow(bg)
        textcolor = 'white'
    else:
        plt.xlim((0, 1920))
        plt.ylim((0, 1080))
        textcolor = 'black'

    T = len(x_list)
    # plt.axhline(1080/2,0,1, color='lightgray', linestyle='--', linewidth=1)
    # plt.axvline(1920/2,0,1, color='lightgray', linestyle='--', linewidth=1)
    plt.scatter(x_list,y_list,c=range(1,T+1), linewidth = 2, marker='o', alpha=0.5, cmap="jet", label='gaze point')
    plt.annotate('1', (x_list[0], y_list[0]), color=textcolor)
    plt.annotate(str(T), (x_list[-1], y_list[-1]), color=textcolor)
    plt.show()




if __name__ == "__main__":
    participant = 7
    session = 3
    stimuli = "C"

    # 이상한놈 2_3_C_30_Block1

    AnalysisExample = Study2Analysis(participant, session, stimuli)
    
    for stimuliNum in range(98):
        # stimuliNum = 1
        print(stimuliNum)
        dataFrame = AnalysisExample.takeGaze(stimuliNum, "Block3")
        # print(dataFrame.columns.to_list())
        bg = AnalysisExample.takeBg(stimuliNum)
        x_list, y_list = get_gazeXY(dataFrame)
        gaze_plot(x_list, y_list, bg)

    # plt.show()

