import os
import random
import json


def takeTargetList(length):
    # Take Target List excluding the nearby elements
    numList = list(range(16))
    targetList = []
    for _ in range(length):
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
    with open('everyMatch.txt', 'r') as f:
        data = f.read()
    data = data.split('\n')
    dataList = []
    for i in data:
        if ',' in i:
            thisdata = i.split(',')
            thisdata = list(map(int, thisdata))
            dataList.append(thisdata)
        else:
            dataList.append([int(i)])
    totalData = []
    for _ in range(4):
        totalData.extend(dataList)

    dictList1 = []
    dictList2 = []
    dictList3 = []
    dictList4 = []
    dictList5 = []

    for levelIndex in range(196):
        this_data = totalData[levelIndex]
        targetList = takeTargetList(len(this_data))
        if levelIndex<49:
            stimuli = 'shape'
        elif 49<=levelIndex<98:
            stimuli = 'size'
        elif 98<=levelIndex<147:
            stimuli = 'brightness'
        elif 147<=levelIndex<196:
            stimuli = 'hue'
        dictList1.append({'stimuli': stimuli, 'index_data':this_data, 'target_list': targetList})

    for levelIndex in range(196):
        this_data = totalData[levelIndex]
        targetList = takeTargetList(len(this_data))
        if levelIndex<49:
            stimuli = 'shape'
        elif 49<=levelIndex<98:
            stimuli = 'size'
        elif 98<=levelIndex<147:
            stimuli = 'brightness'
        elif 147<=levelIndex<196:
            stimuli = 'hue'
        dictList2.append({'stimuli': stimuli, 'index_data':this_data, 'target_list': targetList})

    for levelIndex in range(196):
        this_data = totalData[levelIndex]
        targetList = takeTargetList(len(this_data))
        if levelIndex<49:
            stimuli = 'shape'
        elif 49<=levelIndex<98:
            stimuli = 'size'
        elif 98<=levelIndex<147:
            stimuli = 'brightness'
        elif 147<=levelIndex<196:
            stimuli = 'hue'
        dictList3.append({'stimuli': stimuli, 'index_data':this_data, 'target_list': targetList})

    for levelIndex in range(196):
        this_data = totalData[levelIndex]
        targetList = takeTargetList(len(this_data))
        if levelIndex<49:
            stimuli = 'shape'
        elif 49<=levelIndex<98:
            stimuli = 'size'
        elif 98<=levelIndex<147:
            stimuli = 'brightness'
        elif 147<=levelIndex<196:
            stimuli = 'hue'
        dictList4.append({'stimuli': stimuli, 'index_data':this_data, 'target_list': targetList})

    for levelIndex in range(196):
        this_data = totalData[levelIndex]
        targetList = takeTargetList(len(this_data))
        if levelIndex<49:
            stimuli = 'shape'
        elif 49<=levelIndex<98:
            stimuli = 'size'
        elif 98<=levelIndex<147:
            stimuli = 'brightness'
        elif 147<=levelIndex<196:
            stimuli = 'hue'
        dictList5.append({'stimuli': stimuli, 'index_data':this_data, 'target_list': targetList})


    totalDictList = [dictList1, dictList2, dictList3, dictList4, dictList5]

    for i in range(1,31):
        # os.makedirs(os.path.join(os.getcwd(), 'madeSet', f'{i}'))
        dictIndex = list(range(5))
        random.shuffle(dictIndex)
        for n, j in enumerate(dictIndex):
            # os.makedirs(os.path.join(os.getcwd(), 'madeSet', f'{i}', f'session{n+1}'))
            dictList = totalDictList[j].copy()
            random.shuffle(dictList)
            jsonDict = {}
            for index, thisDict in enumerate(dictList):
                jsonDict[index] = thisDict
            with open(os.path.join(os.getcwd(), 'madeSet', f'{i}', f'session{n+1}','similar_set.json'), 'w') as f:
                json.dump(jsonDict, f)