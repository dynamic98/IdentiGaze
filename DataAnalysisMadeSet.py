import os
import json
import numpy as np

def differentSetDictionary(save=False):
    path = "data/madeSet"
    setIndex = sessionSet('different')
    originParticipant = "1"
    originSetData = loadDifferentSet(path, originParticipant)
    totalSet = {k:{v:[] for v in range(1,41)} for k in range(1280)}

    for participant in range(1,41):
        targetData = loadDifferentSet(path, str(participant))
        for i in range(1280):
            matchStimuliList = retriveKeyByValue(originSetData, targetData[i])
            for matchStimuli in matchStimuliList:
                thisSetIndex = setIndex[i]
                if thisSetIndex not in totalSet[matchStimuli][participant]:
                    totalSet[matchStimuli][participant].append(thisSetIndex)
    if save:
        with open("SetDictionary_Different.json", 'w') as p:
            json.dump(totalSet, p)
            
def loadDifferentSet(path, participant):
    totalData = {}
    for sessionNum in range(5):
        session = f"session{sessionNum+1}"
        sessionSetPath = os.path.join(path, participant, session, 'different_set.json')
        with open(sessionSetPath, 'r') as f:
            jsondata = json.load(f)
        reorganizedData = {int(i)+256*sessionNum:jsondata[i] for i in jsondata}
        totalData.update(reorganizedData)
    return totalData


def similarSetDictionary(save=False):
    path = "data/madeSet"
    setIndex = sessionSet('similar')
    originParticipant = "1"
    originSetData = loadSimilarSet(path, originParticipant)

    totalSet = {k:{v:[] for v in range(1,41)} for k in range(980)}


    for participant in range(1,41):
        targetData = loadSimilarSet(path, str(participant))
        for i in range(980):
            matchStimuliList = retriveKeyByValue(originSetData, targetData[i])
            for matchStimuli in matchStimuliList:
                thisSetIndex = setIndex[i]
                if thisSetIndex not in totalSet[matchStimuli][participant]:
                    totalSet[matchStimuli][participant].append(thisSetIndex)
    if save:
        with open("SetDictionary_Similar.json", 'w') as p:
            json.dump(totalSet, p)

def loadSimilarSet(path, participant):
    totalData = {}
    for sessionNum in range(5):
        session = f"session{sessionNum+1}"
        sessionSetPath = os.path.join(path, participant, session, 'similar_set.json')
        with open(sessionSetPath, 'r') as f:
            jsondata = json.load(f)
        reorganizedData = {int(i)+196*sessionNum:jsondata[i] for i in jsondata}
        totalData.update(reorganizedData)
    return totalData


def retriveKeyByValue(data, value):
    keys = [k for k, v in data.items() if v == value]
    return keys

def sessionSet(stimuliType):
    setdict = {}
    if stimuliType == 'different':
        for i in range(1280):
            setdict[i] = f"{i//256+1}_{i%256}"
    elif stimuliType == 'similar':
        for i in range(980):
            setdict[i] = f"{i//196+1}_{i%196}"
    return setdict


if __name__ == "__main__":
    pass
    # differentSetDictionary(save=True)
    # similarSetDictionary(save=True)

    # data = loadDifferentSet("data/madeSet", str(1))
    # for i in range(1280):
    #     retriveKeyByValue(data, data[i])

    #  different >>> {"0": {"level_index": 174, "target_list": [6, 5, 1, 10]}
    #  similiar >>> {"0": {"stimuli": "brightness", "index_data": [1, 1, 1, 1], "target_list": [9, 10, 14, 5]}
