import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re
import cv2


def task_encoding(task):
    task_dict = {'shape':1, 'size':2, 'orientation':3, 'hue':4, 'brightness':5}
    encoded_task = task_dict[task]
    return encoded_task

def similarity_encoding(similarity):
    if '#' in similarity:
        similarity = similarity.split('#')[1]
    similarity = list(map(int,similarity.split('-')))
    difference = abs(similarity[0]-similarity[1])
    return difference

def normalize_cm(cm: np.array):
    new_cm = np.zeros_like(cm, dtype=np.float_)
    cm_sum_list = cm.sum(axis=1)
    cm_length = len(cm_sum_list)
    for i in range(cm_length):
        for j in range(cm_length):
            new_cm[i,j] = (float(cm[i,j])/float(cm_sum_list[i]))
    return new_cm

def average(datalist):
    data = datalist.strip("[]")
    if len(data)!= 0:
        data = list(map(float, data.split(",")))
        return sum(data)/len(data)
    else:
        return 0

def stack_ydata_from_same(y_data, stack):
    # 일단 들어온 y_data를 label dict에 할당시켜줌
    # stack으로 묶을 때 label이 같은 것끼리 묶어야 하기 때문임

    label_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        label_dict[label].append(index)
    
    stack_y = []
    stack_index = [[] for _ in range(stack)]

    #
    for label in label_dict:
        index_list = label_dict[label]
        stack_size = len(index_list)//stack
        for i in range(stack):
            stack_index[i].extend(index_list[i*stack_size:(i+1)*stack_size])
        stack_y.extend([label for _ in range(stack_size)])

    stack_index = np.array(stack_index)
    return stack_index, stack_y

def stack_ydata_from_same_combinations(y_data, stack):
    label_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        label_dict[label].append(index)
    stack_y = []
    for n, label in enumerate(label_dict):
        index_list = label_dict[label]
        stack_combinations = itertools.combinations(index_list, stack)
        stack_array = np.array(list(stack_combinations)).T
        _, stack_size = stack_array.shape
        stack_y.extend([label for _ in range(stack_size)])
        if n == 0:
            stack_index = stack_array
        else:
            stack_index = np.concatenate((stack_index, stack_array),axis=1)
    return stack_index, stack_y

def stack_ydata_from_each_combinations(y_data, index1, index2, index3):
    # stack = 3
    label1_dict = {k:[] for k in list(set(y_data))}
    label2_dict = {k:[] for k in list(set(y_data))}
    label3_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        if index in index1:
            label1_dict[label].append(index)

        elif index in index2:
            label2_dict[label].append(index)
        
        elif index in index3:
            label3_dict[label].append(index)

    stack_y = []
    for n in range(13):
        label = n
        index_list = [label1_dict[label], label2_dict[label], label3_dict[label]]
        stack_array = cartesian(index_list).T
        _, stack_size = stack_array.shape
        stack_y.extend([label for _ in range(stack_size)])
        if n == 0:
            stack_index = stack_array
        else:
            stack_index = np.concatenate((stack_index, stack_array),axis=1)
    return stack_index, stack_y

def stack_ydata_from_anything(y_data, index1, index2, index3):
    # stack = 3
    label1_dict = {k:[] for k in list(set(y_data))}
    label2_dict = {k:[] for k in list(set(y_data))}
    label3_dict = {k:[] for k in list(set(y_data))}
    for index,label in enumerate(y_data):
        if index in index1:
            label1_dict[label].append(index)

        if index in index2:
            label2_dict[label].append(index)
        
        if index in index3:
            label3_dict[label].append(index)

    firstOne = label1_dict[0][0]
    secondOne = label2_dict[0][0]
    thirdOne = label3_dict[0][0]
    stack_y = []

    if (firstOne == secondOne) & (firstOne != thirdOne):
        for n in range(13):
            stackCombination = np.array(list(itertools.combinations(label1_dict[n], 2)))
            stackCombination = cartesian2Stack([stackCombination, label3_dict[n]]).T
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    elif (firstOne == thirdOne) & (firstOne != secondOne):
        for n in range(13):
            stackCombination = np.array(list(itertools.combinations(label1_dict[n], 2)))
            stackCombination = cartesian2Stack([stackCombination, label2_dict[n]]).T
            stackCombination = np.vstack([stackCombination[0], stackCombination[2], stackCombination[1]])
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    elif (thirdOne == secondOne) & (thirdOne != firstOne):
        for n in range(13):
            stackCombination = np.array(list(itertools.combinations(label3_dict[n], 2)))
            stackCombination = cartesian2Stack([stackCombination, label1_dict[n]]).T
            stackCombination = np.vstack([stackCombination[2], stackCombination[1], stackCombination[0]])
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    elif (firstOne != secondOne) & (firstOne != secondOne):
        for n in range(13):
            stackCombination = cartesian([label1_dict[n], label2_dict[n], label3_dict[n]]).T
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    elif (firstOne == secondOne == thirdOne):
        for n in range(13):
            stackCombination = np.array(list(itertools.combinations(label1_dict[n], 3))).T
            _, stack_size = stackCombination.shape
            stack_y.extend([n for _ in range(stack_size)])
            if n == 0:
                stack_index = stackCombination
            else:
                stack_index = np.concatenate((stack_index, stackCombination),axis=1)
    return stack_index, stack_y


def latefusion(clf: RandomForestClassifier, x_data, stack_index:np.array, stack_y):
    stack, total_stack_size = stack_index.shape
    plus_predict = []
    multiply_predict = []
    max_predict = []
    probabilities = clf.predict_proba(x_data)
    yLabel = list(set(stack_y))

    for i in range(total_stack_size):
        # probabilities = clf.predict_proba(x_data[stack_index[:,i]])
        plus = np.zeros((1,13))
        multiply = np.ones((1,13))
        maximum = [0 for _ in range(13)]
        for j in range(stack):
            this_probabilities = probabilities[stack_index[j,i]]
            plus = plus + np.array(this_probabilities)
            multiply = multiply * np.array(this_probabilities)
            maximum = list(map(max, maximum, this_probabilities))
        plus_predict.append(yLabel[np.argmax(plus)])
        multiply_predict.append(yLabel[np.argmax(multiply)])
        max_predict.append(yLabel[np.argmax(maximum)])

    cm_plus = confusion_matrix(stack_y, plus_predict)
    cm_multiply = confusion_matrix(stack_y, multiply_predict)
    cm_max = confusion_matrix(stack_y, max_predict)

    acc_plus = accuracy_score(stack_y, plus_predict)
    acc_multiply = accuracy_score(stack_y, multiply_predict)
    acc_max = accuracy_score(stack_y, max_predict)

    precision_plus = precision_score(stack_y, plus_predict, average='macro')
    precision_multiply = precision_score(stack_y, multiply_predict, average='macro')
    precision_max = precision_score(stack_y, max_predict, average='macro')

    f1_plus = f1_score(stack_y, plus_predict, average='macro')
    f1_multiply = f1_score(stack_y, multiply_predict, average='macro')
    f1_max = f1_score(stack_y, max_predict, average='macro')

    results = {
        'cm_plus':cm_plus, 'cm_multiply':cm_multiply, 'cm_max':cm_max,
        'acc_plus':acc_plus, 'acc_multiply':acc_multiply, 'acc_max':acc_max,
        'precision_plus':precision_plus, 'precision_multiply':precision_multiply, 'precision_max':precision_max,
        'f1_plus':f1_plus, 'f1_multiply':f1_multiply, 'f1_max':f1_max
    }

    return results

def latefusionVerification(clf: RandomForestClassifier, x_data, stack_index:np.array, stack_y, targetY):
    stack, total_stack_size = stack_index.shape
    multiply_predict = []
    probabilities = clf.predict_proba(x_data)
    yLabel = list(set(stack_y))

    for i in range(total_stack_size):
        # probabilities = clf.predict_proba(x_data[stack_index[:,i]])
        multiply = np.ones((1,13))
        for j in range(stack):
            this_probabilities = probabilities[stack_index[j,i]]
            multiply = multiply * np.array(this_probabilities)
        print(multiply[0][targetY], targetY, np.argmax(multiply))

    #     multiply_predict.append(yLabel[np.argmax(multiply)])
    # cm_multiply = confusion_matrix(stack_y, multiply_predict)
    # acc_multiply = accuracy_score(stack_y, multiply_predict)
    # precision_multiply = precision_score(stack_y, multiply_predict, average='macro')
    # f1_multiply = f1_score(stack_y, multiply_predict, average='macro')
    # results = {
    #    'cm_multiply':cm_multiply, 'acc_multiply':acc_multiply,
    #    'precision_multiply':precision_multiply, 'f1_multiply':f1_multiply
    # }

    # return results


def visualize_cm(cm, clf_name:str, title:str, path='', iv=False):
    length_cm = cm.shape[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(list(range(1,length_cm+1))))
    disp.plot(include_values=iv)
    plt.title(f'{title}_{clf_name}')
    # plt.show()
    if len(path)==0:
        plt.show()
    else:
        plt.savefig(os.path.join('ml-results','latefusion','from cm', path, f'{title}_{clf_name}.png'))
        plt.close()


def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def cartesian2Stack(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.shape[0] for x in arrays])
    if out is None:
        out = np.zeros([n, 3], dtype=dtype)

    m = int(n / arrays[0].shape[0])
    # print(np.repeat(arrays[0], m, axis=0))
    out[:,:2] = np.repeat(arrays[0], m, axis=0)
    for j in range(0, arrays[0].shape[0]):
        out[j*m:(j+1)*m, 2] = arrays[1]

    return out


def bracket2array(arraylike_string:str):
    result = []
    line_slice = arraylike_string.split(']\n')
    for i in line_slice:
        result.append([*map(float, re.findall('-?[\d]+\.?[\d]*', i))])
    return np.array(result)

def convert2binaryCM(cm:np.array, participant):
    array_line = participant-1
    tp = cm[array_line, array_line]
    fp = cm[:,array_line].sum() - tp
    fn = cm[array_line,:].sum() - tp
    tn = cm.sum() - (tp+fp+fn)
    out = np.array([[tp, fp],[fn,tn]])

    return out

def resize_img(img):
    img = img[200:880,620:1300]
    resize_img = cv2.resize(img, dsize=(200,200), interpolation=cv2.INTER_AREA )
    img_arr = np.array(resize_img,dtype=np.float_)
    return img_arr

def BiometricEvaluation(confusionMatrix, targetLabel, metric:str):
    totalValue = np.sum(confusionMatrix)
    tp = confusionMatrix[targetLabel, targetLabel]
    fn = confusionMatrix[targetLabel,:].sum() - tp
    fp = confusionMatrix[:,targetLabel].sum() - tp
    tn = totalValue - (tp+fn+fp)
    if metric == 'FAR':
        return fp/(fp+tn)
    elif metric == 'FRR':
        return fn/(fn+tp)

def accuracyMeasurementForVerification(confusionMatrix, targetLabel):
    totalValue = np.sum(confusionMatrix)
    tp = confusionMatrix[targetLabel, targetLabel]
    fn = confusionMatrix[targetLabel,:].sum() - tp
    fp = confusionMatrix[:,targetLabel].sum() - tp
    tn = totalValue - (tp+fn+fp)
    return (tp+tn)/totalValue

def accuracyMeasurement(confusionMatrix):
    labelAmount,_ = confusionMatrix.shape
    diagonalValue = 0
    totalValue = np.sum(confusionMatrix)
    for i in range(labelAmount):
        diagonalValue += confusionMatrix[i,i]
    return diagonalValue/totalValue



if __name__ == '__main__':
    pass