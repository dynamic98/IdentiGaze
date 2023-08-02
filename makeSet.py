import os
import random
import json


def takeTargetList():
    # Take Target List excluding the nearby elements
    numList = list(range(16))
    targetList = []
    for _ in range(4):
        thisRandom = random.choice(numList)
        targetList.append(thisRandom)
        numList.remove(thisRandom)
        thisRandomLeft = thisRandom - 1
        thisRandomRight = thisRandom + 1
        if thisRandomLeft == -1:
            thisRandomLeft = 15
        if thisRandomRight == 16:
            thisRandomRight = 0
        if thisRandomLeft in numList:
            numList.remove(thisRandomLeft)
        if thisRandomRight in numList:
            numList.remove(thisRandomRight)
    return targetList


if __name__ == "__main__":

    dictList1 = []
    dictList2 = []
    dictList3 = []
    dictList4 = []
    dictList5 = []

    for levelIndex in range(256):
        targetList = takeTargetList()
        dictList1.append({'level_index': levelIndex, 'target_list':targetList})

    for levelIndex in range(256):
        targetList = takeTargetList()
        dictList2.append({'level_index': levelIndex, 'target_list':targetList})

    for levelIndex in range(256):
        targetList = takeTargetList()
        dictList3.append({'level_index': levelIndex, 'target_list':targetList})

    for levelIndex in range(256):
        targetList = takeTargetList()
        dictList4.append({'level_index': levelIndex, 'target_list':targetList})

    for levelIndex in range(256):
        targetList = takeTargetList()
        dictList5.append({'level_index': levelIndex, 'target_list':targetList})

    totalDictList = [dictList1, dictList2, dictList3, dictList4, dictList5]

    for i in range(1,31):
        os.makedirs(os.path.join(os.getcwd(), 'madeSet', f'{i}'))
        dictIndex = list(range(5))
        random.shuffle(dictIndex)
        for n, j in enumerate(dictIndex):
            os.makedirs(os.path.join(os.getcwd(), 'madeSet', f'{i}', f'session{n+1}'))
            dictList = totalDictList[j].copy()
            random.shuffle(dictList)
            jsonDict = {}
            for index, thisDict in enumerate(dictList):
                jsonDict[index] = thisDict
            with open(os.path.join(os.getcwd(), 'madeSet', f'{i}', f'session{n+1}','different_set.json'), 'w') as f:
                json.dump(jsonDict, f)