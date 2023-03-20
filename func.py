import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fcmeans import FCM
from tsfresh.feature_extraction import feature_calculators
import pywt
import warnings

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

def wavedec_ac(df, ob, lvl=3, days=30):
    # Feature Extraction With Discrete Wavelet Transform (DWT)
    score = pd.DataFrame(index=df.columns)
    df = df.replace(0, 0.0001)
    r_df = df.copy()
    for _id, r in r_df.iteritems():
        r_df[_id] = ob / r
    r_df_n = r_df.copy()
    ml_grid = pd.DataFrame(index=range(days))
    for _id, single_df in r_df_n.iteritems():
        group_wvt = r_df_n[_id].groupby(single_df.index // 48)
        for _day in range(days):
            d1 = group_wvt.get_group(_day)
            coeffs = pywt.wavedec(d1, 'db1', level=lvl)
            cA = coeffs[0]
            for i in range(len(cA)):
                ml_grid.at[_day, str(i)] = cA[i]
        df_by_day = r_df[_id].groupby(single_df.index // 48)
        energy = pd.DataFrame(columns=['score'], index=range(0, len(df.index) // 48))
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

