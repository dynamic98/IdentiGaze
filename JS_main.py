from DataAnalysis_util import *
from JS_utils import *
from metric_revised import *
import seaborn as sns

# 이건 Study2AnalysisIndividual 쓰는 예시

participant = 7
session = 3
stimuli = "C"

# 이상한놈 2_3_C_30_Block1

AnalysisExample = Study2AnalysisIndividual(participant, session, stimuli)

for stimuliNum in range(98):
    dataFrame = AnalysisExample.takeGaze(stimuliNum, "Block3")

    print(dataFrame.columns.to_list())
    bg = AnalysisExample.takeBg(stimuliNum)
    x_list, y_list = get_gazeXY(dataFrame)

    assert len(x_list) == len(y_list)

    feature_dict = extract_all_features(dataFrame)
    # x_skew, y_skew = skewness(dataFrame)
    x_kurtosis, y_kurtosis = kurtosis(dataFrame)
    
    gaze_plot(x_list, y_list, bg)

    

"""
# 이건 Study2AnalysisStimuli 쓰는 예시
AnalysisExample = Study2AnalysisStimuli("different")
stimuliIndexNum = 919
for participant in [2,3,5,7,8,9,10,13,18,19,20,22,29]:
    for overlap in range(2):
        dataFrame = AnalysisExample.takeGaze(stimuliIndexNum, participant, "Block3", overlap)
        bg = AnalysisExample.takeBg(stimuliIndexNum)
        x_list, y_list = get_gazeXY(dataFrame)
        gaze_plot(x_list, y_list, bg)

"""
print(takeLevel_different(180))
print(takeLevel_similar(180))
# for participant in [2,3,5,7,8,9,10,13,18,19,20,22,29]:
#     dataFrame = AnalysisExample.takeGaze(stimuliIndexNum, participant, "Block3")
#     bg = AnalysisExample.takeBg(stimuliIndexNum)
#     x_list, y_list = get_gazeXY(dataFrame)
#     gaze_plot(x_list, y_list, bg)

