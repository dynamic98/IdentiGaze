import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from DataAnalysis_util import *



if __name__=='__main__':
    # oneList = ['6', '13', '22', '25', '44', '50', '54', '60', '75', '128', '130', '136', '158', '162', '169', '193', '196', '197', 
    #             '203', '212', '222', '232', '242', '243', '282', '297', '305', '330', '331', '335', '337', '375', '394', '395', '400', 
    #             '411', '423', '450', '458', '468', '484', '499', '506', '531', '534', '537', '541', '575', '588', '597', '599', '602', 
    #             '606', '612', '615', '621', '662', '679', '697', '722', '725', '737', '740', '771', '804', '816', '818', '823', '843', 
    #             '851', '880', '886', '889', '895', '944', '947', '949', '953', '963', '974']
    AnalysisExample = Study2AnalysisStimuli("similar")
    vcList = ['shape','size','hue','brightness']
    levelList = [0,1,2,3]
    for targetVC in vcList:
        for targetLevel in levelList:
            # targetVC = 'hue'
            # targetLevel = 1
            targetList = AnalysisExample.findVC_from_Similar(targetVC, targetLevel)
            pList = list(range(1,36))
            pList.remove(16)

            df = pd.DataFrame({'participant':[], 'stimuli_index':[], 'minimumDistance':[]})
            for participant in pList:
                for i in targetList:
                    dataFrame = AnalysisExample.takeGaze(i, participant, "Block3")
                    bg = AnalysisExample.takeBg(i)
                    x_list, y_list = get_gazeXY(dataFrame)

                    closestDistance = AnalysisExample.targetDistance(i, targetVC, targetLevel, x_list, y_list)
                    df2 = pd.DataFrame([{'participant':participant, 'stimuli_index':i, 'minimumDistance':min(closestDistance)}])
                    df = pd.concat([df, df2], axis=0, ignore_index=True)

                    # plt.figure(1) 
                    # sns.lineplot(closestDistance)
                    # plt.ylim(0,1000)
                    # plt.title(f"Participant {participant}, Stimuli {i}")
                    # plt.show()
                    
                    # gaze_angular(x_list, y_list)
                    # print(participant)
                    # fixationX, fixationY = get_fixationXY(dataFrame)
                    # Hs, Ht = gaze_entropy(fixationX, fixationY)
                    # print(Hs, Ht)
                    # plt.figure(2)
                    # gaze_plot(x_list, y_list, bg)
                    # plt.imshow(bg)
                    # plt.show()
                    # plt.show()
            # sns.histplot(data=df, x='minimumDistance', y='participant', element='step', fill=False, legend=False)
            sns.histplot(data=df, x='minimumDistance', y='participant', discrete = (False, True), legend=False, cbar=True)
            plt.xlim(0,600)
            plt.title(f"Histogram for Minimum Distance of {targetVC}, level {targetLevel}")
            plt.savefig(f"ml-results/AnalysisByStimuli/MinimumDistance_{targetVC}_{targetLevel}.png")
            plt.close()