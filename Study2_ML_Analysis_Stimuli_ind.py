import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import json
from Study2_ML_util import *
import gc

class LoadSelectiveData:
    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        data = data.dropna(axis=0)
        data = data.sample(frac=1, random_state=5).reset_index(drop=True)

        self.data = data
        self.exceptlist = ['participant']

    def set_domain_except(self, *args):
        self.exceptlist = self.exceptlist + list(args)

    def take_x(self, data=pd.DataFrame()):
        if data.empty:
            x_data = self.data.loc[:,~self.data.columns.isin(self.exceptlist)]
        else:
            x_data = data.loc[:,~self.data.columns.isin(self.exceptlist)]
        return x_data

    def take_individual(self, individual, data=pd.DataFrame()):
        if data.empty:
            y_data = self.data['participant'].apply(lambda x: 1 if x == individual else 0)

        else:
            y_data = data['participant'].apply(lambda x: 1 if x == individual else 0)

        return y_data

    def take_y(self, data=pd.DataFrame()):
        if data.empty:
            y_data = self.data['participant']
        else:
            y_data = data['participant']
        return y_data
    
    def get_data(self):
        return self.data

    def split_data(self, train_data_ratio, valid_data_ratio, test_data_ratio):
        if train_data_ratio+valid_data_ratio+test_data_ratio != 1:
            raise Exception("train + valid + test should be equal 1.")
        
        kf = StratifiedKFold(n_splits=10)
        x_data = self.take_x().to_numpy()
        y_data = self.take_y().to_numpy()
        index_blocks = []

        for _, test in kf.split(x_data, y_data):
            index_blocks.append(test.tolist())

        train_data_block = []
        valid_data_block = []
        test_data_block = []

        train_data_index = list(range(0, int(train_data_ratio*10)))
        valid_data_index = list(range(int(train_data_ratio*10), int((train_data_ratio+valid_data_ratio)*10)))
        test_data_index = list(range(int((train_data_ratio+valid_data_ratio)*10), 10))

        for i in range(10):
            if i in train_data_index:
                train_data_block.extend(index_blocks[i])
            elif i in valid_data_index:
                valid_data_block.extend(index_blocks[i])
            elif i in test_data_index:
                test_data_block.extend(index_blocks[i])

        return train_data_block, valid_data_block, test_data_block

    def ml_train(self, individual, train_data_block):
        x_train_data = self.take_x().to_numpy()[train_data_block]
        y_train_data = self.take_individual(individual).to_numpy()[train_data_block]
        # warnings.filterwarnings('ignore')
        rf_model = RandomForestClassifier(random_state=0)  # Random Forest
        rf_model.fit(x_train_data, y_train_data)
        return rf_model

    def ml_test_strategy(self, model:RandomForestClassifier, strategy_dict, test_data_block):
        total_x = self.take_x().to_numpy()[test_data_block]
        total_y = self.take_y().to_numpy()[test_data_block]
        totalResults = {}
        for participant in range(13):
            this_data = strategy_dict[participant]
            stack_index, stack_y = stack_ydata_from_anything(total_y, this_data[0], this_data[1], this_data[2])
            results = latefusion(model, total_x, stack_index, stack_y)
            totalResults[f"participant_{participant}"] = results
        return totalResults

    def ml_test_strategy_verification(self, model:RandomForestClassifier, strategy_dict, test_data_block):
        total_x = self.take_x().to_numpy()[test_data_block]
        total_y = self.take_y().to_numpy()[test_data_block]
        totalResults = {}
        for participant in range(13):
            this_data = strategy_dict[participant]
            stack_index, stack_y = stack_ydata_from_anything(total_y, this_data[0], this_data[1], this_data[2])
            multiply_predict, multiply_predict_prob = latefusionVerification(model, total_x, stack_index, stack_y, participant)
            # totalResults[f"participant_{participant}"] = results
        # return totalResults


    def ml_test_baseline(self, individual, model:RandomForestClassifier, test_data_block):
        total_x = self.take_x().to_numpy()[test_data_block]
        total_y = self.take_individual(individual).to_numpy()[test_data_block]
        stack_index, stack_y = stack_ydata_from_same_combinations(total_y, 3)
        results = latefusion(model, total_x, stack_index, stack_y)
        return results

    
    def get_indexlist(self):
        return self.data.index.to_list()

if __name__ == '__main__':
    # path = "data/BlueRareStudy2Entire_different_interpolated.csv"
    # path = "data/BlueRareStudy2_different_jyjrGE.csv"
    path = "data/BlueRareStudy2_different_ReactionTime_change.csv"

    thisData = LoadSelectiveData(path)
    # thisData.set_domain_except('Hs','Ht')
    # thisData.set_domain_except('mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12', 'Hs', 'Ht')
    # thisData.set_domain_except('reaction_time','fixation_count','saccade_count','fixation_duration_avg','saccade_duration_avg','pupil_left','pupil_right','pupil_avg','saccade_velocity_avg','saccade_amplitude_avg','pupil_min','pupil_max','path_length','total_velocity_average','total_velocity_max','total_velocity_min','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','Hs','Ht')
    # thisData.set_domain_except('reaction_time','fixation_count','saccade_count','fixation_duration_avg','saccade_duration_avg','pupil_left','pupil_right','pupil_avg','saccade_velocity_avg','saccade_amplitude_avg','pupil_min','pupil_max','path_length','total_velocity_average','total_velocity_max','total_velocity_min','mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','level_index','target_list_1','target_list_2','target_list_3','target_list_4','Hs','Ht')
    # thisData.set_domain_except('reaction_time','fixation_count','saccade_count','fixation_duration_avg','saccade_duration_avg','pupil_left','pupil_right','pupil_avg','saccade_velocity_avg','saccade_amplitude_avg','pupil_min','pupil_max','path_length','total_velocity_average','total_velocity_max','total_velocity_min','level_index','target_list_1','target_list_2','target_list_3','target_list_4','Hs', 'Ht')
    dataFrame = thisData.get_data()
    # for i in range(13):
    #     print(i)
    #     print(dataFrame[dataFrame['participant']==i].describe())

    train_data_ratio = 0.9
    valid_data_ratio = 0
    test_data_ratio = 0.1


    trb, vdb, teb = thisData.split_data(train_data_ratio, valid_data_ratio, test_data_ratio)
    # print(thisData.take_y()[teb].to_list())
    for i in range(34):
        print("====")
        print(i)
        model = thisData.ml_train(i, trb)
        print("train Done")

        # print(set(y_test_data.tolist()))
        print("normal accuracy")
        testX = thisData.take_x().to_numpy()[teb]
        testY = thisData.take_individual(i).to_numpy()[teb]
        predY = model.predict(testX)
        monoCM = confusion_matrix(testY, predY)
        print(monoCM)
        # visualize_cm(monoCM, "RF", "Study2_MONO_jr")
        print(model.score(testX, testY))
        result = thisData.ml_test_baseline(i, model, teb)
        confusionMatrix = result["cm_multiply"]
        print(confusionMatrix)
        print(accuracyMeasurement(confusionMatrix))
    # # visualize_cm(confusionMatrix, "RF", "study2", path='Study2EntireCM')

    # # """

    # result = thisData.ml_test_baseline(model, teb)
    # confusionMatrix = result["cm_multiply"]
    # print(confusionMatrix)
    # print(accuracyMeasurement(confusionMatrix))
    # # visualize_cm(confusionMatrix, "RF", "study2", path='Study2EntireCM')

    # totalFRR = []
    # totalFAR = []
    # for p in range(34):
    #     print(accuracyMeasurementForVerification(confusionMatrix, p))
    #     far = BiometricEvaluation(confusionMatrix, p, 'FAR')
    #     frr = BiometricEvaluation(confusionMatrix, p, 'FRR')
    #     print(far, frr)
    #     totalFAR.append(far)
    #     totalFRR.append(frr)
    #     print("=======")
    # print((sum(totalFRR)/34), (sum(totalFAR)/34))

    # """
    # confusionMatrixDict, jspiritSortedDict = thisData.ml_validate(model, vdb)
    # strategy1 = thisData.stimuli_strategy1(jspiritSortedDict, teb)
    # results1 = thisData.ml_test_strategy(model, strategy1, teb)
    # print("result 1")
    # for i, participant in enumerate(results1):
    #     print("====================")
    #     print(participant)
    #     thisConfusionMatrix = results1[participant]["cm_multiply"]
    #     print(f"Accuracy for verification: {accuracyMeasurementForVerification(thisConfusionMatrix, i)}")
    #     print(f"FAR: {BiometricEvaluation(thisConfusionMatrix, i, 'FAR')}")
    #     print(f"FRR: {BiometricEvaluation(thisConfusionMatrix, i, 'FRR')}")
    # strategy2 = thisData.stimuli_strategy2(confusionMatrixDict, teb)
    # results2 = thisData.ml_test_strategy(model, strategy2, teb)
    # print("result 2")
    # for i, participant in enumerate(results2):
    #     print("====================")
    #     print(participant)
    #     thisConfusionMatrix = results2[participant]["cm_multiply"]
    #     print(f"Accuracy for verification: {accuracyMeasurementForVerification(thisConfusionMatrix, i)}")
    #     print(f"FAR: {BiometricEvaluation(thisConfusionMatrix, i, 'FAR')}")
    #     print(f"FRR: {BiometricEvaluation(thisConfusionMatrix, i, 'FRR')}")
