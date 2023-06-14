import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import json
from ML_util import *
import gc

class LoadSelectiveData:
    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        data['Task Encoding'] = data.apply(lambda x: task_encoding(x['task']), axis=1)
        data['Similarity Encoding'] = data.apply(lambda x: similarity_encoding(x['level']), axis=1)
        data = data.dropna(axis=0)
        data = data.sample(frac=1, random_state=30).reset_index(drop=True)

        self.data = data
        self.exceptlist = ['bbx_x1','bbx_x2','bbx_y1','bbx_y2','task','level','participant','shape_target','shape_distractor',
                           'set_size','target_size','distractor_size','target_color_b','target_color_g','target_color_r',
                           'distractor_color_b','distractor_color_g','distractor_color_r','target_orientation','distractor_orientation', 'index']

    def set_domain_except(self, *args):
        self.exceptlist = self.exceptlist + list(args)

    def take_x(self, data=pd.DataFrame()):
        if data.empty:
            x_data = self.data.loc[:,~self.data.columns.isin(self.exceptlist)]
        else:
            x_data = data.loc[:,~self.data.columns.isin(self.exceptlist)]
        return x_data

    def take_y(self, data=pd.DataFrame()):
        if data.empty:
            y_data = self.data['participant']
        else:
            y_data = data['participant']
        return y_data.apply(lambda y: y-1)
    
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

    def ml_train(self, train_data_block):
        x_train_data = self.take_x().to_numpy()[train_data_block]
        y_train_data = self.take_y().to_numpy()[train_data_block]
        # warnings.filterwarnings('ignore')
        rf_model = RandomForestClassifier(random_state=0)  # Random Forest
        rf_model.fit(x_train_data, y_train_data)
        return rf_model

    def ml_validate(self, model:RandomForestClassifier, valid_data_block):
        log_dict = {f"true_{i}":{f"pred_{j}":[] for j in range(13)} for i in range(13)}
        x_valid_data = self.take_x().to_numpy()[valid_data_block]
        y_valid_data = self.take_y().to_numpy()[valid_data_block]
        predict_y = model.predict(x_valid_data)
        for i in range(len(valid_data_block)):
            this_predict = predict_y[i]
            this_gt = y_valid_data[i]
            # this_index = self.get_indexlist()[valid_data_block[i]]
            this_index = valid_data_block[i]
            this_meta = self.take_meta(this_index)
            log_dict[f"true_{this_gt}"][f"pred_{this_predict}"].append(this_meta)

        shape_cm = np.zeros((13,13), dtype=np.float64)
        orientation_cm = np.zeros((13,13), dtype=np.float64)
        size_cm = np.zeros((13,13), dtype=np.float64)
        hue_cm = np.zeros((13,13), dtype=np.float64)
        brightness_cm = np.zeros((13,13), dtype=np.float64)

        for i in range(13):
            for j in range(13):
                this_data = log_dict[f"true_{i}"][f"pred_{j}"]
                for meta in this_data:
                    if meta['task'] == 'shape':
                        shape_cm[i,j] += 1
                    elif meta['task'] == 'size':
                        size_cm[i,j] += 1
                    elif meta['task'] == 'brightness':
                        brightness_cm[i,j] += 1
                    elif meta['task'] == 'hue':
                        hue_cm[i,j] += 1
                    elif meta['task'] == 'orientation':
                        orientation_cm[i,j] += 1
        shape_cm = normalize_cm(shape_cm)
        size_cm = normalize_cm(size_cm)
        brightness_cm = normalize_cm(brightness_cm)
        hue_cm = normalize_cm(hue_cm)
        orientation_cm = normalize_cm(orientation_cm)

        cm_identify_dict = {'shape':shape_cm, 'size':size_cm, 'hue':hue_cm, 'brightness':brightness_cm, 'orientation':orientation_cm}
        # with open(f"ml-results/latefusion/from cm/validation_result.json", "w") as f:
        #         json.dump(cm_identify_dict, f, default=str)
        jspirit_dict = {}
        for i in range(13):
            this_dict = {}
            for cm in cm_identify_dict:
                this_dict[cm] = cm_identify_dict[cm][i,i]
            jspirit_dict[i] = this_dict

        jspirit_sorted_dict = {}
        for i in jspirit_dict:
            jspirit_sorted_dict[i] = dict(sorted(jspirit_dict[i].items(), key=lambda item: item[1], reverse=True))
        return cm_identify_dict, jspirit_sorted_dict

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
            results = latefusionVerification(model, total_x, stack_index, stack_y, participant)
            totalResults[f"participant_{participant}"] = results
        return totalResults


    def ml_test_baseline(self, model:RandomForestClassifier, test_data_block):
        total_x = self.take_x().to_numpy()[test_data_block]
        total_y = self.take_y().to_numpy()[test_data_block]
        stack_index, stack_y = stack_ydata_from_same_combinations(total_y, 3)
        results = latefusion(model, total_x, stack_index, stack_y)
        # with open(f"ml-results/latefusion/from cm/strategy_baseline/total_result.json", "w") as f:
        #     json.dump(results, f, default=str)
        # visualize_cm(normalize_cm(np.array(results['cm_multiply'], dtype=np.single)), clf_name='randomforest_stack3', title=participant, path='strategy1')
        return results

    def stimuli_strategy1(self, jspirit_dict:dict, test_data_block):
        # strategy with taking three top contenders (3 serial stimuli)
        test_data = self.data.iloc[test_data_block].reset_index()
        shape_data = test_data.loc[test_data['task']=='shape']
        size_data = test_data.loc[test_data['task']=='size']
        hue_data = test_data.loc[test_data['task']=='hue']
        brightness_data = test_data.loc[test_data['task']=='brightness']
        orientation_data = test_data.loc[test_data['task']=='orientation']

        strategy_dict = {}
        for i in jspirit_dict:
            strategy_dict[i] = []
            participant_dominance = list(jspirit_dict[i].keys())
            for j in range(3):
                if participant_dominance[j] == 'shape':
                    this_data = shape_data.index.to_list()
                elif participant_dominance[j] == 'size':
                    this_data = size_data.index.to_list()
                elif participant_dominance[j] == 'hue':
                    this_data = hue_data.index.to_list()
                elif participant_dominance[j] == 'brightness':
                    this_data = brightness_data.index.to_list()
                elif participant_dominance[j] == 'orientation':
                    this_data = orientation_data.index.to_list()
                strategy_dict[i].append(this_data)
        return strategy_dict

    def stimuli_strategy2(self, confusionMatrixDict:dict, test_data_block):
        # strategy with taking three top contenders (3 serial stimuli)
        test_data = self.data.iloc[test_data_block].reset_index()
        shape_data = test_data.loc[test_data['task']=='shape']
        size_data = test_data.loc[test_data['task']=='size']
        hue_data = test_data.loc[test_data['task']=='hue']
        brightness_data = test_data.loc[test_data['task']=='brightness']
        orientation_data = test_data.loc[test_data['task']=='orientation']


        result=[]
        duplicated_order_vc=[]
        for part_main in range(0, 13):
            
            order_temp=[]
            temp_list=[]
            # print(task_cm_total)
            for vc in ["shape", "size", "orientation", "hue", "brightness"]:
                # print(task_cm_total[vc][part_main])
                for part in range(13):
                    if part_main==part:
                        pass
                    else:
                        # 비교할 participant&vc  list-up
                        temp_list.append([part, vc, confusionMatrixDict[vc][part_main][part]])
            # 본인과 가장 유사한 participant의 가장 잘 구분되는 vc 찾기 
            temp_list=sorted(temp_list, key=lambda x:x[2], reverse=True)
            while (len(temp_list)!=0):
                # print(temp_list)
                similar_participant=temp_list[0][0]
                vc_candidate=list(filter(lambda x:x[0]==similar_participant, temp_list))
                # print(vc_candidate)
                order_temp.append(vc_candidate[-1][1])
                new_list=[x for x in temp_list if (x not in vc_candidate) ]
                temp_list=new_list
                # print(temp_list)
            duplicated_order_vc.append(order_temp)

        # # 가장 유사한 participant 순으로 구분이 용이한 vc 
        # print("\n 중복 vc")
        # for item in duplicated_order_vc:
        #     print(item)

        # 가장 유사한 participant 순으로 구분이 용이한 vc [중복 제거]
        # order_vc=[]
        # for duplicated_item in duplicated_order_vc:
        #     duplicated_set=set(duplicated_item)
        #     order_vc.append(list(duplicated_set))

        # print("\n 중복 제거된 vc")
        # for item in order_vc:
        #     print(item)

        strategy_dict = {}
        for i in range(13):
            strategy_dict[i] = []
            participant_dominance = duplicated_order_vc[i]
            for j in range(3):
                if participant_dominance[j] == 'shape':
                    this_data = shape_data.index.to_list()
                elif participant_dominance[j] == 'size':
                    this_data = size_data.index.to_list()
                elif participant_dominance[j] == 'hue':
                    this_data = hue_data.index.to_list()
                elif participant_dominance[j] == 'brightness':
                    this_data = brightness_data.index.to_list()
                elif participant_dominance[j] == 'orientation':
                    this_data = orientation_data.index.to_list()
                strategy_dict[i].append(this_data)
        return strategy_dict


    def each_stimuli_data(self, test_data_block):
        test_data = self.data.iloc[test_data_block]
        shape_data = test_data.loc[test_data['task']=='shape']
        size_data = test_data.loc[test_data['task']=='size']
        hue_data = test_data.loc[test_data['task']=='hue']
        brightness_data = test_data.loc[test_data['task']=='brightness']
        orientation_data = test_data.loc[test_data['task']=='orientation']
        return {'shape':shape_data, 'size':size_data, 'hue':hue_data, 'brightness':brightness_data, 'orientation':orientation_data}


    def take_meta(self, index:int):
        this_data = self.data.loc[index]
        task = this_data.loc['task']
        if task != 'orientation':
            shape_target = int(this_data.loc['shape_target'])
            shape_distractor = int(this_data.loc['shape_distractor'])
            target_orientation = None
            distractor_orientation = None
        else:
            shape_target = 'orientation'
            shape_distractor = 'orientation'
            target_orientation = int(this_data.loc['target_orientation'])
            distractor_orientation = int(this_data.loc['distractor_orientation'])
        set_size = int(this_data.loc['set_size'])
        target_cnt = [this_data.loc['cnt_x'],this_data.loc['cnt_y']]
        target_size = int(this_data.loc['target_size'])
        distractor_size = int(this_data.loc['distractor_size'])
        target_color = [int(this_data.loc['target_color_b']),int(this_data.loc['target_color_g']),int(this_data.loc['target_color_r'])]
        distractor_color = [int(this_data.loc['distractor_color_b']),int(this_data.loc['distractor_color_g']),int(this_data.loc['distractor_color_r'])]

        meta_dict = {'task':task, 'shape_target':shape_target, 'shape_distractor':shape_distractor,'set_size':set_size, 'target_cnt':target_cnt,
                     'target_size':target_size, 'distractor_size':distractor_size, 'target_color':target_color, 'distractor_color':distractor_color,
                     'target_orientation':target_orientation, 'distractor_orientation':distractor_orientation}
        return meta_dict
    
    def get_indexlist(self):
        return self.data.index.to_list()

if __name__ == '__main__':
    path = "data/BlueMediumRarePupil_task1-1.csv"
    thisData = LoadSelectiveData(path)
    train_data_ratio = 0.6
    valid_data_ratio = 0.2
    test_data_ratio = 0.2

    trb, vdb, teb = thisData.split_data(train_data_ratio, valid_data_ratio, test_data_ratio)
    model = thisData.ml_train(trb)
    print("train Done")
    confusionMatrixDict, jspiritSortedDict = thisData.ml_validate(model, vdb)
    strategy1 = thisData.stimuli_strategy1(jspiritSortedDict, teb)
    results1 = thisData.ml_test_strategy(model, strategy1, teb)
    print("result 1")
    for i, participant in enumerate(results1):
        print("====================")
        print(participant)
        thisConfusionMatrix = results1[participant]["cm_multiply"]
        print(f"Accuracy for verification: {accuracyMeasurementForVerification(thisConfusionMatrix, i)}")
        print(f"FAR: {BiometricEvaluation(thisConfusionMatrix, i, 'FAR')}")
        print(f"FRR: {BiometricEvaluation(thisConfusionMatrix, i, 'FRR')}")
    strategy2 = thisData.stimuli_strategy2(confusionMatrixDict, teb)
    results2 = thisData.ml_test_strategy(model, strategy2, teb)
    print("result 2")
    for i, participant in enumerate(results2):
        print("====================")
        print(participant)
        thisConfusionMatrix = results2[participant]["cm_multiply"]
        print(f"Accuracy for verification: {accuracyMeasurementForVerification(thisConfusionMatrix, i)}")
        print(f"FAR: {BiometricEvaluation(thisConfusionMatrix, i, 'FAR')}")
        print(f"FRR: {BiometricEvaluation(thisConfusionMatrix, i, 'FRR')}")
