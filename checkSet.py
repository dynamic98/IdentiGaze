import os
import pandas as pd
import json

path = os.path.join(os.getcwd(), 'madeSet')
target = 'different_set.json'

for participant in range(1,31):
    totalData = pd.DataFrame()
    for session in range(1,6):
        datapath = os.path.join(path, f"{participant}", f"session{session}", target)
        with open(datapath, 'r') as f:
            json_data = json.load(f)
        data_tolist = [json_data[i] for i in json_data]
        data = pd.DataFrame(data_tolist)
        totalData = pd.concat([totalData, data], axis=0)
    sortedData=totalData.sort_values(by=['level_index'])
    print(sortedData.tail())