'''
This program manipulated the reading data following specific functions.
It also generate the real total power consumption of each group and stored into "data/observed"
Check the paper to get the details of each FDI attack function.
Placed grouped user reading profile into "groupedData" directory.
Generated observed metering data will be placed into "observed" directory.
The tampered data and fradulent user id will be stored into each "FDI#" subdirecroty.
'''
import pandas as pd
import func

def main():
    interval = 48
    days = 30
    for FDI_TYPE in range(1, 8):
        dir = "data/FDI" + str(FDI_TYPE) + "/g"
        for i in range(10):
            g1 = pd.read_csv("data/groupedData/g" + str(i) + ".csv")
            if FDI_TYPE == 1:
                ob = g1.sum(axis=1)
                ob.to_csv("data/observed/g" + str(i) + ".csv", index=False)
            g_number = g1.columns
            if FDI_TYPE == 1:
                t_id, t_df = func.FDI_1(g1, g_number, interval=interval, days=days)
            elif FDI_TYPE == 2:
                t_id, t_df = func.FDI_2(g1, g_number, interval=interval, days=days)
            elif FDI_TYPE == 3:
                t_id, t_df = func.FDI_3(g1, g_number, interval=interval, days=days)
            elif FDI_TYPE == 4:
                t_id, t_df = func.FDI_4(g1, g_number, interval=interval, days=days)
            elif FDI_TYPE == 5:
                t_id, t_df = func.FDI_5(g1, g_number, interval=interval, days=days)
            elif FDI_TYPE == 6:
                t_id, t_df = func.FDI_6(g1, g_number, interval=interval, days=days)
            elif FDI_TYPE == 7:
                t_id, t_df = func.FDI_MIX(g1, g_number, interval=interval, days=days)
            g1 = t_df.combine_first(g1)
            g1.to_csv(dir + str(i) + ".csv", index=False)
            pd.Series(t_id).to_csv(dir + str(i) + "_id.csv", index=False, header=False)

if __name__ == '__main__':
    main()