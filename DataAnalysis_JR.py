import json
from DataAnalysis_util import *


levelDictionary_similar_path = 'LevelDictionary_Similar.json'
with open(levelDictionary_similar_path, 'r') as f:
    levelDictionary_similar = json.load(f)

def findVC_from_Similar(visual_component:str, level:int):
    matchedStimuliList = []
    for i in levelDictionary_similar:
        stimuliInfo = levelDictionary_similar[i]
        if stimuliInfo['visual_component'] == visual_component and level in stimuliInfo['level']:
            matchedStimuliList.append(int(i))
    return matchedStimuliList

def targetDistance(stimuliIndex, visual_component, level, gazeX, gazeY):
    pass


# oneList = ['6', '13', '22', '25', '44', '50', '54', '60', '75', '128', '130', '136', '158', '162', '169', '193', '196', '197', 
#             '203', '212', '222', '232', '242', '243', '282', '297', '305', '330', '331', '335', '337', '375', '394', '395', '400', 
#             '411', '423', '450', '458', '468', '484', '499', '506', '531', '534', '537', '541', '575', '588', '597', '599', '602', 
#             '606', '612', '615', '621', '662', '679', '697', '722', '725', '737', '740', '771', '804', '816', '818', '823', '843', 
#             '851', '880', '886', '889', '895', '944', '947', '949', '953', '963', '974']
AnalysisExample = Study2AnalysisStimuli("similar")
targetVC = 'hue'
targetLevel = 1
targetList = findVC_from_Similar(targetVC, targetLevel)
print(len(targetList))
for i in targetList:
    # pList = list(range(1,36))
    # pList.remove(16)
    print(i)
    for participant in [1,2]:
#         i = int(i)
        grid_list = AnalysisExample.preattentive_second.calc_grid()
        print(grid_list[0])
        dataFrame = AnalysisExample.takeGaze(i, participant, "Block3")
        bg = AnalysisExample.takeBg(i)
        x_list, y_list = get_gazeXY(dataFrame)
        # gaze_angular(x_list, y_list)
        # print(participant)
        # fixationX, fixationY = get_fixationXY(dataFrame)
        # Hs, Ht = gaze_entropy(fixationX, fixationY)
        # print(Hs, Ht)
        # gaze_plot(x_list, y_list, bg)
        plt.imshow(bg)
        plt.show()