'''
Randomly split meters into groups evenly.
The original data should be the real reading of each users 
while each meter are stored into one column with its id as header. 
Each row represent a time stamp.
The grouped data will be stored into "data/groupedData"
'''
import pandas as pd
import numpy as np

def main():
    day = 30
    month = 2

    data = pd.read_csv(str(day) + "_day_" + str(month) + "_num.csv")

    id_list = data.columns.tolist()
    length = len(id_list)

    # The total users should be divisible by the number of groups.
    drop_id = np.random.randint(length, size=length % 10)

    drop_id[::-1].sort()
    for i in drop_id:
        id_list.pop(i)

    # randomly split into 10 groups and output
    id_arr = np.array(id_list)
    np.random.shuffle(id_arr)
    split_id = np.split(id_arr, 10)
    i = 0
    for g_number in split_id:  
        data[g_number].to_csv("data/groupedData/g" + str(i) + ".csv", index=False)
        i += 1

if __name__ == '__main__':
    main()
    