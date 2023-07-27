import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class SingleStimuliData:
    def __init__(self, datapath, filename, savepath) -> None:
        self.dataname = filename
        self.datapath = datapath
        df = pd.read_csv(os.path.join(self.datapath, filename), sep='\t', low_memory=False)
        df['Gaze point X'] = df['Gaze point X'].interpolate()
        df['Gaze point Y'] = df['Gaze point Y'].interpolate()

        self.df = df
        self.startEvent = df[df["Event value"]=='z'].index.to_list()[-1]
        self.savepath = savepath
        self.getDataInfo()
        # self.checkDataInfo()


    def getDataInfo(self):
        participantNum, sessionNum, stimuliType = checkDirExist(self.dataname, self.savepath)
        self.participantNum = participantNum
        self.sessionNum = sessionNum
        self.stimuliType = stimuliType
        # print(participantNum, sessionNum, stimuliType)

    def checkDataInfo(self):
        df = self.df
        beforeStart = df.iloc[:self.startEvent+1]

        event = beforeStart[beforeStart['Event']=='KeyboardEvent']
        eventList = event['Event value'].to_list()

        while 'RightShift' in eventList: eventList.remove('RightShift')
        while 'LeftShift' in eventList: eventList.remove('LeftShift')
        while '[Shift] + RightShift' in eventList: eventList.remove('[Shift] + RightShift')
        while '[Shift] + LeftShift' in eventList: eventList.remove('[Shift] + LeftShift')
        while 'LeftWindowsKey' in eventList: eventList.remove('LeftWindowsKey')
        while 'Tab' in eventList: eventList.remove('Tab')

        # 프로세서 에러때매 두번씩 들어간거 제거
        for i in range(len(eventList)-1):
            if eventList[i]=='Return' and eventList[i+1]=='Return':
                eventList = [eventList[i*2] for i in range(len(eventList)//2)]
                break
        
        returnList = find_indices(eventList, 'Return')
        if len(returnList)<3:
            return '', '', '', eventList
        else:
            participantIndex = returnList[-3]
            sessionIndex = returnList[-2]
            stimuliIndex = returnList[-1]

            if participantIndex > 1:
                participant = eventList[participantIndex-2:participantIndex]
            else:
                participant = eventList[:participantIndex]
            participant = ''.join(participant)
            session = eventList[sessionIndex-1]
            stimuli = eventList[stimuliIndex-1]

            if stimuli == '[Shift] + A':
                stimuli = 'A'
            elif stimuli == '[Shift] + B':
                stimuli = 'B'
            elif stimuli == '[Shift] + C':
                stimuli = 'C'
            elif stimuli == '[Shift] + D':
                stimuli = 'D'
                
            return participant, session, stimuli, eventList


    def saveDataFrame(self):
        totalList = self.get_stimuli_list()
        targetSavePath = os.path.join(self.savepath, self.participantNum, self.sessionNum, self.stimuliType)
        for n, i in enumerate(totalList):
            d1, a, b, c, d2 = i
            firstBlock = self.df.iloc[d1:a]
            secondBlock = self.df.iloc[a:b]
            thirdBlock = self.df.iloc[b:c]
            fourthBlock = self.df.iloc[c:d2]
            firstBlock.to_csv(os.path.join(targetSavePath, f"{n}_Block1.tsv"), sep='\t', index=False)
            secondBlock.to_csv(os.path.join(targetSavePath, f"{n}_Block2.tsv"), sep='\t', index=False)
            thirdBlock.to_csv(os.path.join(targetSavePath, f"{n}_Block3.tsv"), sep='\t', index=False)
            fourthBlock.to_csv(os.path.join(targetSavePath, f"{n}_Block4.tsv"), sep='\t', index=False)

    def get_df(self):
        return self.df

    def analysis_df(self, abc_list):
        # print(len(abc_list))
        print(self.dataname, self.participant, self.date, self.time)
        print(self.get_df().index)
        for i in range(100):
            start, end = abc_list[i]
            df = self.get_df().loc[start:end+1]
            if len(df.index.to_list())==0:
                print(start, end)


    def get_stimuli_list(self):

        a_list = self.slice_stimuli('a')
        b_list = self.slice_stimuli('b')
        c_list = self.slice_stimuli('c')
        d_list = self.slice_stimuli('d')
        d_list.append(c_list[-1]+40)

        totalList = []
        for i in range(len(a_list)):
            if i == 0:
                totalList.append([self.startEvent, a_list[i], b_list[i], c_list[i], d_list[i]])
            else:
                totalList.append([d_list[i-1], a_list[i], b_list[i], c_list[i], d_list[i]])

        return totalList

    def slice_stimuli(self, keyboardevent):
        df = self.df.iloc[self.startEvent:]
        df_event = df[df['Event']=='KeyboardEvent'].copy()
        df_event = df_event[df_event['Event value']==keyboardevent]
        event_list = df_event.index.to_list()
        stimuli_list = []
        for event in event_list:
            if len(stimuli_list)>0 and (event == stimuli_list[-1]+1 or event == stimuli_list[-1]+2 or event == stimuli_list[-1]+3):
                del stimuli_list[-1]
                stimuli_list.append(event)
            else:
                stimuli_list.append(event)
        return stimuli_list
        # print(df_event)

    def task_start_time(self, df):
        starttime = df[df['Event value'].isin(['1','2'])].index.to_list()
        if len(starttime)==1:
            return starttime[0]
        else:
            # print(f"Caution! {self.dataname} has a multiple task name. please check it.")
            return starttime[-1]

def checkDirExist(filename:str, savepath:str):
    fileInfo = filename.split(' ')[1].split('.')[0].split('_')
    participantNum = fileInfo[0][1:]
    sessionNum = fileInfo[1][7:]
    stimuliType = fileInfo[2]

    participantPath = os.path.join(savepath, participantNum)
    sessionPath = os.path.join(participantPath, sessionNum)
    stimuliPath = os.path.join(sessionPath, stimuliType)

    if not os.path.exists(participantPath):
        os.makedirs(participantPath)
    if not os.path.exists(sessionPath):
        os.makedirs(sessionPath)
    if not os.path.exists(stimuliPath):
        os.makedirs(stimuliPath)

    return participantNum, sessionNum, stimuliType


def find_indices(input_list, element):
    return [i for i, x in enumerate(input_list) if x == element]




if __name__ == "__main__":
    exportDataPath = "data/Identigaze_study2_entire data/Data Export - IdentiGaze-Study2"
    saveDataPath = "data/data_processed_Study2"

    # problem = 'IdentiGaze-Study2 P7_Session5_B.tsv'
    # problemDataframe = SingleStimuliData(exportDataPath, problem, saveDataPath)
    # problemDataframe.get_stimuli_list()

    for i in tqdm(os.listdir(exportDataPath)):
        # checkDirExist(i, saveDataPath)
        thisDataframe = SingleStimuliData(exportDataPath, i, saveDataPath)
        participant, session, stimuli, eventList = thisDataframe.checkDataInfo()
        name_p = thisDataframe.participantNum
        name_session = thisDataframe.sessionNum
        name_stimuli = thisDataframe.stimuliType
        if (participant != name_p) or (session != name_session) or (stimuli != name_stimuli):
            print("******************************")
            print(thisDataframe.dataname)
            print(eventList)
            print(name_p, name_session, name_stimuli)
            print("******************************")