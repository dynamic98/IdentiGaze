from math import dist
import os
import numpy as np
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from preattentive_second import PreattentiveObjectSecond

class Study2AnalysisIndividual:
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


class Study2AnalysisStimuli:
    def __init__(self, stimuli) -> None:
        """
        동일 stimuli를 각 participant마다 어떻게 다르게 봤는지 분석하기 위한 class

        Parameters
        ----------
        stimuli : str
            'different' or 'similar'를 input으로 받음

        Returns
        -------
        void

        Examples
        --------
        >>> Study2AnalysisStimuli("different")
        """
        self.stimuli = stimuli
        self.takeSetDictionary()
        self.preattentive_second = PreattentiveObjectSecond(1920, 1080, 'black')

        # print(self.setDictionary)
    
    def takeSetDictionary(self):
        # SetDictionary 데이터 불러옴. 이 데이터엔 동일 stimuli를 각 participant가 언제 봤는지 기록되어 있음
        if self.stimuli == 'different':
            with open('SetDictionary_Different.json', 'r') as f:
                self.setDictionary = json.load(f)
        elif self.stimuli == 'similar':
            with open('SetDictionary_Similar.json', 'r') as f:
                self.setDictionary = json.load(f)
        else:
            raise Exception("Stimuli should be 'different' or 'similar")

    def takeBg(self, stimuliNum):
        """
        stimuliNum 번째 stimuli image를 가져옴

        Parameters
        ----------
        stimuliNum : int
            different stimuli의 경우 0~1279, similar stimuli의 경우 0~979

        Returns
        -------
        rgb_bg : ndarray
            stimuli의 rgb image

        Examples
        --------
        >>> takeBg(1)
        """
        if self.stimuli == 'different':
            session = stimuliNum//256+1
            stimuliIndexNum = stimuliNum%256
            path = os.path.join('data/madeSet', "1", f"session{session}", f"{self.stimuli}_set.json")
            with open(path, "r") as f:
                indexData = json.load(f)
            levelIndex = indexData[str(stimuliIndexNum)]['level_index']
            targetList = indexData[str(stimuliIndexNum)]['target_list']
            bg = self.preattentive_second.stimuli_shape(targetList, levelIndex)

        elif self.stimuli == 'similar':
            session = stimuliNum//196+1
            stimuliIndexNum = stimuliNum%196
            path = os.path.join('data/madeSet', "1", f"session{session}", f"{self.stimuli}_set.json")
            with open(path, "r") as f:
                indexData = json.load(f)
            indexLevel = indexData[str(stimuliIndexNum)]['index_data']
            targetList = indexData[str(stimuliIndexNum)]['target_list']
            visualComponent = indexData[str(stimuliIndexNum)]['stimuli']
            bg = self.preattentive_second.stimuli_similar(visualComponent, targetList, indexLevel)
            
        # bg는 opencv를 통해 만든 이미지이기 때문에 BGR 컬러스케일을 가짐. matplotlib pyplot에서 imshow를 하기 위해 RGB 컬러스케일로 바꿔줌
        rgb_bg = np.dstack((bg[:,:,2], bg[:,:,1], bg[:,:,0]))
        return rgb_bg

    def takeGaze(self, stimuliNum, participant, blockName, overlap=0):
        """
         participant의 StimuliNum에서 blockName 동안의 raw gaze dataframe을 가져옴

        Parameters
        ----------
        stimuliNum : int
            different stimuli의 경우 0~1279, similar stimuli의 경우 0~979
        participant : int
            1~40 사이의 participant number
        blockName : str
            Block1 > 십자가, Block2 > 암전, Block3 > stimuli, Block4 > 암전
        overlap: int
            몇몇 stimuli는 중복되어서 한 사람에게 여러번 보여준 경우가 있음. 이때 몇번째를 쓸거냐의 문제.
            different_set의 경우 (919,1183) 번이 서로 겹침
            similar_set의 경우 (75,851),(135,509),(158,602),(193,335),(212,679),(297,843),
            (394,725),(531,963),(534,886),(541,823),(575,740),(662,974),(697,816) 번이 겹침
            세번 겹친 경우는 없어서 0 또는 1의 숫자를 넣으면 됨

        Returns
        -------
        gazeDataFrame : pd.dataframe
            토비에서 갓 추출된 따끈따끈한 해당 블록의 raw data

        Examples
        --------
        >>> takeGaze(919, 1, "Block3", 1)
        """
        savepath = "data/data_processed_Study2"
        stimuliSet = self.setDictionary[str(stimuliNum)][str(participant)]
        if len(stimuliSet) == 1:
            targetStimuli = stimuliSet[0]
        else:
            print(f"해당 stimuli는 overlap되었음. 중복 개수는 {len(stimuliSet)}. 현재 {overlap}번째 stimuli를 가져옴")
            targetStimuli = stimuliSet[overlap]
        
        session, stimuliIndexNum = targetStimuli.split("_")
        if self.stimuli == 'different':
            # stimuliIndexNum은 0~256 사이의 숫자. 이걸 A B + 0~128로 바꿔줘야 raw gaze path를 구할 수 있음
            stimuliStrNum = int(stimuliIndexNum)//128
            indexNum = int(stimuliIndexNum)%128
            if stimuliStrNum == 0:
                stimuliStr = 'A'
            elif stimuliStrNum == 1:
                stimuliStr = 'B'
        elif self.stimuli == 'similar':
            # stimuliIndexNum은 0~196 사이의 숫자. 이걸 A B + 0~98로 바꿔줘야 raw gaze path를 구할 수 있음
            stimuliStrNum = int(stimuliIndexNum)//98
            indexNum = int(stimuliIndexNum)%98
            if stimuliStrNum == 0:
                stimuliStr = 'A'
            elif stimuliStrNum == 1:
                stimuliStr = 'B'

        targetPath = os.path.join(savepath, f"{participant}", f"{session}", f"{stimuliStr}", f"{indexNum}_{blockName}.tsv")
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

def gaze_angular(x_list, y_list):

    angle_changes = []
    print("=================")
    fixations = [0]
    for i in range(1, len(x_list) -1):
        dx2 = x_list[i+1] - x_list[i]
        dy2 = y_list[i+1] - y_list[i]
        angle_now = np.arctan2(dy2, dx2)
        
        velocity = np.sqrt(dx2**2+dy2**2)
        if velocity > 23:
            if len(angle_changes) == 0:
                angle_changes.append(angle_now)
            else:
                angle_past = angle_changes[-1]
                # angle_changes.append(angle_change)
                angle_diff = angle_now - angle_past
                angle_diff = np.abs((angle_diff + np.pi) % (2*np.pi) - np.pi)
                angle_changes.append(angle_now)

                if angle_diff > 0.9:
                    print("first", fixations[0])
                    print("last", i)
                    print(angle_diff)
                    print("======")
            fixations = [i]
        else:
            # fixation
            fixations.append(i)

    return None


def gaze_entropy(x_list, y_list):

    innerRadius = int(np.sqrt(370**2/5))
    mask = np.zeros((1080,1920,3), dtype=np.uint8)
    mask = cv2.ellipse(mask, (960,540), (370,370), angle=0, startAngle=45, endAngle=135, color=(1,0,0), thickness=-1)
    mask = cv2.ellipse(mask, (960,540), (370,370), angle=0, startAngle=135, endAngle=225, color=(2,0,0), thickness=-1)
    mask = cv2.ellipse(mask, (960,540), (370,370), angle=0, startAngle=225, endAngle=315, color=(3,0,0), thickness=-1)
    mask = cv2.ellipse(mask, (960,540), (370,370), angle=0, startAngle=315, endAngle=405, color=(4,0,0), thickness=-1)
    mask = cv2.circle(mask, (960,540), innerRadius, color=(5,0,0), thickness=-1)

    transitionMatrix = np.zeros((6,6), dtype=np.float)
    stationaryVector = np.zeros((6,), dtype=np.float)

    for i in range(len(x_list)):
        gazeX = x_list[i]
        gazeY = y_list[i]
        print(mask[gazeX, gazeY, 1])

    return None


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
    for i in range(T):
        plt.annotate(f'{i}', (x_list[i], y_list[i]), color=textcolor)
    # plt.annotate('1', (x_list[0], y_list[0]), color=textcolor)
    # plt.annotate(str(T), (x_list[-1], y_list[-1]), color=textcolor)
    plt.show()

def takeLevel_different(stimuliNum:int):
    """
    different stimuli에 해당하는 level을 불러옴

    Parameters
    ----------
    stimuliNum : int
        0~1279

    Returns
    -------
    data[str(stimuliNum)] : dict
        "level"은 각각 shape, size, hue, brightness\n
        "target_list"는 이 visual component가 동글뱅이 속 어디에 위치하는지 특정\n
        아 근데 코드 짤때 뇌를 비웠었나 shape, hue, brightness, size 순임 \n
        hue >>> level에서 세번째 놈의 단계고, target_list에서 두번째 놈의 위치에 배치됨

    Examples
    --------
    >>> takeLevel_different(109)
    {"level": [4, 50, 1, 0], "target_list": [6, 13, 15, 4]}
    """
    path = "LevelDictionary_Different.json"
    with open(path, 'r') as f:
        data = json.load(f)
    return data[str(stimuliNum)]

def takeLevel_similar(stimuliNum):
    """
    similar stimuli에 해당하는 level을 불러옴

    Parameters
    ----------
    stimuliNum : int
        0~979

    Returns
    -------
    data[str(stimuliNum)] : dict
        "visual_component"는 shape, size, hue, brightness 중 하나. 어떤 놈인지 정해줌\n
        "level"은 각각 그 놈들의 단계. takeLevel_different의 level이랑 다름(중요).\n
        shape 일때 0>>>삼각형, 1>>>사각형, 2>>>오각형, 3>>>육각형\n
        size 일때 0>>>30, 1>>>40, 2>>>50, 3>>>60\n
        hue, brightness는 takeLevel_different랑 똑같음\n
        "target_list"는 이 visual component가 동글뱅이 속 어디에 위치하는지 특정\n
        이거는 코드 짤때 뇌를 제대로 쓴것 같음. 순서대로 위치함

    Examples
    --------
    >>> takeLevel_similar(12)
    {"visual_component": "hue", "level": [0, 3], "target_list": [7, 15]},
    """
    path = "LevelDictionary_Similar.json"
    with open(path, 'r') as f:
        data = json.load(f)
    return data[str(stimuliNum)]




if __name__ == "__main__":

    # 이건 Study2AnalysisIndividual 쓰는 예시
    """
    participant = 7
    session = 3
    stimuli = "C"

    # 이상한놈 2_3_C_30_Block1

    AnalysisExample = Study2AnalysisIndividual(participant, session, stimuli)
    
    for stimuliNum in range(98):
        # stimuliNum = 1
        print(stimuliNum)
        dataFrame = AnalysisExample.takeGaze(stimuliNum, "Block3")
        # print(dataFrame.columns.to_list())
        bg = AnalysisExample.takeBg(stimuliNum)
        x_list, y_list = get_gazeXY(dataFrame)
        gaze_plot(x_list, y_list, bg)
    """

    """
    # 이건 Study2AnalysisStimuli 쓰는 예시
    AnalysisExample = Study2AnalysisStimuli("different")
    # stimuliIndexNumList = [212, 287, 673, 898, 1037]
    # stimuliIndexNumList = [96,369,679,951,1107]
    # stimuliIndexNumList = [193, 335, 536, 865, 1101]
    print(takeLevel_different(300))
    stimuliIndexNumList = [300] # {"level": [1, 30, 0, 0], "target_list": [3, 6, 13, 1]} shape 3 size 1 hue 6 brightness 13
    #  "193": {"level": [4, 60, 0, 3], "target_list": [0, 8, 11, 5]}    shape: 0, size: 5, hue: 8, brightness: 11
    for participant in [2,3,5,7,8,9,10,13,18,19,20,22,29]:
        print(participant)
        for stimuliIndexNum in stimuliIndexNumList:
            dataFrame = AnalysisExample.takeGaze(stimuliIndexNum, participant, "Block3")
            bg = AnalysisExample.takeBg(stimuliIndexNum)
            # x_list, y_list = get_gazeXY(dataFrame)
            # gaze_plot(x_list, y_list, bg)
            plt.imshow(bg)
            plt.show()

    """

    oneList = ['6', '13', '22', '25', '44', '50', '54', '60', '75', '128', '130', '136', '158', '162', '169', '193', '196', '197', 
                '203', '212', '222', '232', '242', '243', '282', '297', '305', '330', '331', '335', '337', '375', '394', '395', '400', 
                '411', '423', '450', '458', '468', '484', '499', '506', '531', '534', '537', '541', '575', '588', '597', '599', '602', 
                '606', '612', '615', '621', '662', '679', '697', '722', '725', '737', '740', '771', '804', '816', '818', '823', '843', 
                '851', '880', '886', '889', '895', '944', '947', '949', '953', '963', '974']
    AnalysisExample = Study2AnalysisStimuli("similar")
    for i in oneList:
        pList = list(range(1,36))
        pList.remove(16)
        print(i)
        for participant in pList:
            i = int(i)
            dataFrame = AnalysisExample.takeGaze(i, participant, "Block3")
            bg = AnalysisExample.takeBg(i)
            x_list, y_list = get_gazeXY(dataFrame)
            # gaze_angular(x_list, y_list)
            print(participant)
            gaze_plot(x_list, y_list, bg)

    # print(takeLevel_similar(180))
    # for participant in [2,3,5,7,8,9,10,13,18,19,20,22,29]:
    #     dataFrame = AnalysisExample.takeGaze(stimuliIndexNum, participant, "Block3")
    #     bg = AnalysisExample.takeBg(stimuliIndexNum)
    #     x_list, y_list = get_gazeXY(dataFrame)
    #     gaze_plot(x_list, y_list, bg)


