import pandas as pd
import func

def main():
    index = ['TYPE_1', 'TYPE_2', 'TYPE_3', 'TYPE_4', 'TYPE_5', 'TYPE_6', 'MIX']
    summary = pd.DataFrame()
    g_n = 0
    for FDI_TYPE in range(1, 8):
        result_table = pd.DataFrame()
        input_data = pd.read_csv("data/FDI" + str(FDI_TYPE) + "/reading.csv")
        
        t_id = pd.read_csv("data/FDI" + str(FDI_TYPE) + "/fraudulent_user.csv", header=None)[0].map(str).tolist()
        g_number = list(input_data)
        
        ob = pd.read_csv("data/observed/observer_reading.csv", squeeze=True, header=0)
        
        wavelet_w = func.wavedec_ac(input_data, ob, lvl=4)
        result_table.at[g_n, 'auc_proposed_method'] = func.compute_auc(wavelet_w, set(g_number) - set(t_id), set(t_id))
        result_table.at[g_n, 'map_proposed_method'] = func.MAPatN(wavelet_w, set(t_id), 20)
            
        avg =  result_table.mean()
        result_table.loc['average'] = avg
        summary = summary.append(avg, ignore_index=True)
    
    summary['index'] = index
    summary = summary.set_index('index', drop=True)
    print(summary.round(3))

if __name__ == '__main__':
    main()