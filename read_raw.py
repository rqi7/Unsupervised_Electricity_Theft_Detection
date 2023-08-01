import pandas as pd
import func

def main():
    print('Data preprocessing...')
    day = 30
    month = 2

    # print("month", month)
    # print("Num days", day)
    # read data, replace your direction of dataset
    dir = "Data/"
    df = pd.read_csv( dir + "File1.txt", sep=" ", header=None)


    # append rest of files into a single dataframe
    new_df = pd.read_csv(dir + "File2.txt", sep=" ", header=None)

    df = df.append(new_df)

    new_df = pd.read_csv(dir + "File3.txt", sep=" ", header=None)

    df = df.append(new_df)

    new_df = pd.read_csv(dir + "File4.txt", sep=" ", header=None)
    df = df.append(new_df)

    new_df = pd.read_csv(dir + "File5.txt", sep=" ", header=None)
    df = df.append(new_df)

    new_df = pd.read_csv(dir + "File6.txt", sep=" ", header=None)
    df = df.append(new_df)

    # filter out non SME data
    with open("list_sme.txt") as f:
        sm_id = f.readlines()
    sm_id = [int(x.strip()) for x in sm_id] 
    df = df.loc[df[0].isin(sm_id)]

    # create empty dataframe with time stamp as row
    data = pd.DataFrame() #index=range(19501, df[0].max() + 1)
    # group data by their id
    group_df = df.groupby(df.columns[0])
    for i, meter in group_df:
        temp_df = pd.DataFrame(data=meter[2].tolist(), columns=[i], index=meter[1])
        temp_df = temp_df.sort_index()
        temp_df['ts'] = temp_df.index
        temp_df.drop_duplicates(inplace=True, subset=['ts'])
        # Filter out the bad data that contain too many consecutive 0
        if func.find_con_0(temp_df[i].tolist(), 5) == False:
            data[i] = temp_df[i]
        



    # filter out N/A
    data = data.fillna(data.mean())

    # Apply three sigma rule of thumb into each readings.
    for id, reading in data.iteritems():
        data[id] = func.tsrt(reading)
    data = data[(month - 1) * day * 48 : month * day * 48]
    data.reset_index(inplace=True, drop=True)
    
    print(data)
    data.to_csv(str(day) + "_day_" + str(month) + "_num.csv")

if __name__ == '__main__':
    main()