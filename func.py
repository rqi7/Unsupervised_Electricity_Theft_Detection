'''
Help funtions
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fcmeans import FCM
from tsfresh.feature_extraction import feature_calculators
import pywt
import warnings
import random

# suppress UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

def compute_auc(df, b_id, f_id):
    # AUC score calculation
    l1_f = df.loc[f_id]
    l1_b = df.loc[b_id]
    l1_f['label'] = 'F'
    l1_b['label'] = 'B'
    df = l1_f.append(l1_b)
    df = df.sort_values('score', ascending=True)
    df.index = np.arange(1, len(df) + 1)
    c_B, c_F = df['label'].value_counts()
    s = 0
    for index, rows in df.iterrows():
        if rows['label'] == 'F':
            s += index
    return (s - 0.5 * c_F * (c_F + 1)) / (c_F * c_B)

def MAPatN(df, f_id, N):
    # map@N score calculation (n=20 in this sample code)
    l1_f = df.loc[f_id]
    df = df.sort_values('score', ascending=False)
    df.index = np.arange(1, len(df) + 1)
    df = df.head(N)
    count = 0
    precision = 0
    for index, rows in df.iterrows():
        if rows['score'] in l1_f.values:
            count += 1
            precision += count / index
    if count == 0:
        return 0
    return precision / count

def wavedec_ac(df, ob, lvl=3, days=30, interval=48):
    # Feature Extraction With Discrete Wavelet Transform (DWT)
    score = pd.DataFrame(index=df.columns)
    df = df.replace(0, 0.0001)
    r_df = df.copy()
    for _id, r in r_df.iteritems():
        r_df[_id] = ob / r
    r_df_n = r_df.copy()
    ml_grid = pd.DataFrame(index=range(days))
    for _id, single_df in r_df_n.iteritems():
        group_wvt = r_df_n[_id].groupby(single_df.index // interval)
        for _day in range(days):
            d1 = group_wvt.get_group(_day)
            coeffs = pywt.wavedec(d1, 'db1', level=lvl)
            cA = coeffs[0]
            for i in range(len(cA)):
                ml_grid.at[_day, str(i)] = cA[i]
        df_by_day = r_df[_id].groupby(single_df.index // interval)
        energy = pd.DataFrame(columns=['score'], index=range(0, len(df.index) // interval))
        for day, reading in df_by_day:
            abs_energy = feature_calculators.abs_energy(reading)
            energy.at[day, 'score'] = abs_energy
        score.at[_id, 'score'] = energy_fcm(ml_grid, energy)
    return score

def energy_fcm(score, energy):
    # FCM clustering
    if len(score.columns) != 1:
        Scalar = MinMaxScaler()
        score = Scalar.fit_transform(score.values)
    clf = FCM(n_clusters=2)
    clf.fit(score)
    aa = pd.DataFrame(clf.soft_predict(score))

    # Compute the anomaly score of each user.
    ee = energy.copy()
    g0m = (aa[0] * ee['score']).sum() / aa[0].sum()
    g1m = (aa[1] * ee['score']).sum() / aa[1].sum()
    gmin = min(g0m, g1m)
    gmax = max(g0m, g1m)
    return (gmax - gmin) / gmax


# FDI1 - MIX are false data injection attack functions
def FDI_1(df, id, interval=48, days=30):
    t_day = days // 2
    ts_length = days * interval
    if isinstance(id, str) is True:
        g_df = df.groupby(df.index // interval)
        n_df = pd.Series(index=range(0, ts_length))
        F_day = random.sample(range(0, days), t_day)
        ls = []
        for day, i in g_df:  
            l = i
            if day in F_day:
                alpha = random.uniform(0.20001, 0.79999)
                l = l * alpha
            ls = ls + list(l)
        n_df = pd.Series(ls)
        F_id = id
    else:
        F_id = random.sample(list(id), 5)
        F = df[F_id]
        columns = list(F)
        g_df = F.groupby(F.index // interval)
        n_df = pd.DataFrame(columns=F_id, index=range(0, ts_length))
        for c in columns:
            ls = []
            F_day = random.sample(range(0, days), t_day)
            for day, i in g_df:  
                l = i[c]
                if day in F_day:
                    alpha = random.uniform(0.20001, 0.79999)
                    l = l * alpha
                ls = ls + list(l)
            temp = pd.DataFrame(ls)
            n_df[c] = temp
    return F_id, n_df

def FDI_2(df, id, interval=48, days=30):
    t_day = days // 2
    ts_length = days * interval
    if isinstance(id, str) is True:
        g_df = df.groupby(df.index // interval)
        n_df = pd.Series(index=range(0, 1440))
        F_day = random.sample(range(0, 30), 15)
        ls = []
        for day, i in g_df:  
            l = i
            if day in F_day:
                gamma = random.uniform(0.00001, np.max(l))
                l[l >= gamma] = gamma
            ls = ls + list(l)
        n_df = pd.Series(ls)
        F_id = id
    else:
        F_id = random.sample(list(id), 5)
        F = df[F_id]
        columns = list(F)
        g_df = F.groupby(F.index // interval)
        n_df = pd.DataFrame(columns=F_id, index=range(0, ts_length))
        for c in columns:
            ls = []
            F_day = random.sample(range(0, days), t_day)
            for day, i in g_df:  
                l = i[c]
                if day in F_day:
                    gamma = random.uniform(0.00001, np.max(l))
                    l[l >= gamma] = gamma
                ls = ls + list(l)
            temp = pd.DataFrame(ls)
            n_df[c] = temp
    return F_id, n_df

def FDI_3(df, id, interval=48, days=30):
    t_day = days // 2
    ts_length = days * interval
    if isinstance(id, str) is True:
        g_df = df.groupby(df.index // interval)
        n_df = pd.Series(index=range(0, 1440))
        F_day = random.sample(range(0, 30), 15)
        ls = []
        for day, i in g_df:  
            l = i
            if day in F_day:
                gamma = random.uniform(0.00001, np.max(l))
                l = l - gamma
                l[l < 0] = 0
            ls = ls + list(l)
        n_df = pd.Series(ls)
        F_id = id
    else:
        F_id = random.sample(list(id), 5)
        F = df[F_id]
        columns = list(F)
        g_df = F.groupby(F.index // interval)
        n_df = pd.DataFrame(columns=F_id, index=range(0, ts_length))
        for c in columns:
            ls = []
            F_day = random.sample(range(0, days), t_day)
            for day, i in g_df:  
                l = i[c]
                if day in F_day:
                    gamma = random.uniform(0.00001, np.max(l))
                    l = l - gamma
                    l[l < 0] = 0
                ls = ls + list(l)
            temp = pd.DataFrame(ls)
            n_df[c] = temp
    return F_id, n_df

def FDI_4(df, id, interval=48, days=30):
    t_day = days // 2
    ts_length = days * interval
    if isinstance(id, str) is True:
        g_df = df.groupby(df.index // interval)
        n_df = pd.Series(index=range(0, 1440))
        F_day = random.sample(range(0, 30), 15)
        ls = []
        for day, i in g_df:  
            l = i
            if day in F_day:
                len_t = random.randint(8, interval)
                s_range = 48 - len_t
                s_t = random.randint(0, s_range)
                e_t = s_t + len_t
                for t in range(len(l)):
                    if t > s_t and t < e_t:
                        l.iloc[[t]] = 0
            ls = ls + list(l)
        n_df = pd.Series(ls)
        F_id = id
    else:
        F_id = random.sample(list(id), 5)
        F = df[F_id]
        columns = list(F)
        g_df = F.groupby(F.index // 48)
        n_df = pd.DataFrame(columns=F_id, index=range(0, ts_length))
        for c in columns:
            ls = []
            F_day = random.sample(range(0, days), t_day)
            for day, i in g_df:  
                l = i[c]
                if day in F_day:
                    len_t = random.randint(8, interval)
                    s_range = 48 - len_t
                    s_t = random.randint(0, s_range)
                    e_t = s_t + len_t
                    for t in range(len(l)):
                        if t > s_t and t < e_t:
                            l.iloc[[t]] = 0
                ls = ls + list(l)
            temp = pd.DataFrame(ls)
            n_df[c] = temp
    return F_id, n_df

def FDI_5(df, id, interval=48, days=30):
    t_day = days // 2
    ts_length = days * interval
    if isinstance(id, str) is True:
        g_df = df.groupby(df.index // 48)
        n_df = pd.Series(index=range(0, 1440))
        F_day = random.sample(range(0, 30), 15)
        ls = []
        for day, i in g_df:  
            l = i
            if day in F_day:
                for t in range(len(l)):
                    l.iloc[[t]] = l.iloc[[t]] * random.uniform(0.20001, 0.79999)
            ls = ls + list(l)
        n_df = pd.Series(ls)
        F_id = id
    else:
        F_id = random.sample(list(id), 5)
        F = df[F_id]
        columns = list(F)
        g_df = F.groupby(F.index // interval)
        n_df = pd.DataFrame(columns=F_id, index=range(0, ts_length))
        for c in columns:
            ls = []
            F_day = random.sample(range(0, days), t_day)
            for day, i in g_df:  
                l = i[c]
                if day in F_day:
                    for t in range(len(l)):
                        l.iloc[[t]] = l.iloc[[t]] * random.uniform(0.20001, 0.79999)                    
                ls = ls + list(l)
            temp = pd.DataFrame(ls)
            n_df[c] = temp
    return F_id, n_df

def FDI_6(df, id, interval=48, days=30):
    t_day = days // 2
    ts_length = days * interval
    if isinstance(id, str) is True:
        g_df = df.groupby(df.index // interval)
        n_df = pd.Series(index=range(0, 1440))
        F_day = random.sample(range(0, 30), 15)
        ls = []
        for day, i in g_df:  
            l = i
            if day in F_day:
                mean_l = np.mean(l)
                for t in range(len(l)):
                    l.iloc[[t]] = mean_l * random.uniform(0.20001, 0.79999)
            ls = ls + list(l)
        n_df = pd.Series(ls)
        F_id = id
    else:
        F_id = random.sample(list(id), 5)
        F = df[F_id]
        columns = list(F)
        g_df = F.groupby(F.index // interval)
        n_df = pd.DataFrame(columns=F_id, index=range(0, ts_length))
        for c in columns:
            ls = []
            F_day = random.sample(range(0, days), t_day)
            for day, i in g_df:  
                l = i[c]
                if day in F_day:
                    mean_l = np.mean(l)
                    for t in range(len(l)):
                        l.iloc[[t]] = mean_l * random.uniform(0.20001, 0.79999)
                ls = ls + list(l)
            temp = pd.DataFrame(ls)
            n_df[c] = temp
    return F_id, n_df

def FDI_MIX(df, id, interval=48, days=30):
    F_id = random.sample(list(id), 5)
    for i in F_id:
        fdi_type = random.randint(1, 6)
        if fdi_type == 1:
            id, df[i] = FDI_1(df[i], i)
        elif fdi_type == 2:
            id, df[i] = FDI_2(df[i], i)
        elif fdi_type == 3:
            id, df[i] = FDI_3(df[i], i)
        elif fdi_type == 4:
            id, df[i] = FDI_4(df[i], i)
        elif fdi_type == 5:
            id, df[i] = FDI_5(df[i], i)
        elif fdi_type == 6:
            id, df[i] = FDI_6(df[i], i)
    return F_id, df

# Giving a list, find out if the list contain specific number of contiuous 0 values.
def find_con_0(lst, con_len):
    for i in range(len(lst) - con_len + 1):
        sum = 0
        for j in range(con_len):
            sum += lst[j + i]
        if sum < 0.00001:
            return True
    return False

# Three sigma rule of thumb
def tsrt(df):
    sigma = df.std()
    mean = df.mean()
    ndf = df.copy()
    for i in range(1, len(df) - 1):
        if df.iloc[i] > 3 * sigma + mean:
            ndf.iloc[i] = (df.iloc[i-1] + df.iloc[i+1]) / 2
    return ndf