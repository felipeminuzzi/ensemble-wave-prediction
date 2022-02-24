from importlib.resources import path
import numpy as np
import pandas as pd
import xarray as xr
import glob
import warnings
import pickle
from pyod.models.knn import KNN
from scipy import stats
from sklearn import preprocessing
from src.config.config import Config

warnings.filterwarnings('ignore')
normalize     = preprocessing.MinMaxScaler((-1,1))

def create_df_error(ens, boia, flag):
    config    = Config()
    cols      = config.target
    cols_feat = config.features
    col_error = config.var_to_error

    ensembles = pd.read_csv(ens, encoding='utf-8', sep=';', decimal=',').drop('Unnamed: 0', axis=1)
    df_boia   = pd.read_csv(boia)

    df_boia[df_boia[cols[0]] == -9999] = np.nan
    df_boia   = df_boia[['Datetime',cols[0]]]
    
    df_boia['Datetime'] = pd.to_datetime(df_boia['Datetime']) 
    ensembles['time']   = pd.to_datetime(ensembles['time'])
    df_boia.set_index('Datetime', inplace=True) 
    ensembles.set_index('time', inplace=True)

    ensembles, df_boia  = correct_data(ensembles,df_boia)
    full_df   = df_boia.join(ensembles).fillna(method='bfill')
    
    if flag:
        full_df['target'] = full_df[col_error] - full_df[cols[0]]
        full_df['negative'] = full_df['target'].apply(lambda x: 1 if x > 0 else -1) 
        full_df['target'] = full_df['target'].apply(abs) 
    else:
        full_df['target'] = full_df[cols[0]]

    df_target= full_df[['target','negative']]
    feat_cols= [f'membro_{i}' for i in range(1,31)]
    df_feat  = full_df[feat_cols].copy()
    
    return df_feat, df_target

def create_df_error_era(df, boia, flag):
    config              = Config()
    cols                = config.target
    col_error           = config.var_to_error
    cols_feat           = ['Hs','10m-direc','10m-speed']

    df_boia             = pd.read_csv(boia)
    df_boia[df_boia[cols[0]] == -9999] = np.nan
    df_boia             = df_boia[['Datetime',cols[0]]]
    df_boia['Datetime'] = pd.to_datetime(df_boia['Datetime']) 
    df_boia.set_index('Datetime', inplace=True)

    ensembles, df_boia  = correct_data(df,df_boia)   
    cols_to_use         = [f'Hs-{i}' for i in range(10)]
    to_mean             = ensembles[cols_to_use]
    to_mean['mean']     = to_mean.mean(axis=1)
    full_df             = df_boia.join(to_mean)

    if flag:
        full_df['target'] = full_df[col_error] - full_df[cols[0]]
        full_df['negative'] = full_df['target'].apply(lambda x: 1 if x > 0 else -1) 
        full_df['target'] = full_df['target'].apply(abs) 
    else:
        full_df['target'] = full_df[cols[0]]

    df_target= full_df[['target','negative']]
    df_feat  = ensembles.copy()

    return df_feat, df_target
    
def correct_data(df1,df2):
    if df1.index.min() <= df2.index.min():
        inicio= df2.index.min()
    else:
        inicio= df1.index.min()

    if df1.index.max() <= df2.index.max():
        final = df1.index.max()
    else:
        final = df2.index.max()
    
    df_1 = df1.loc[(df1.index <= final) & (df1.index >= inicio)]
    df_2 = df2.loc[(df2.index <= final) & (df2.index >= inicio)]

    return df_1, df_2

def multi_target_setup(path):
    files = glob.glob(path+'*')
    for f in files:
        if f.split('/')[-1][-3:] == 'pkl':
            with open(f, 'rb') as handle:
                first_predict_date  = pickle.load(handle)
        elif f.split('/')[-1][:-4].split('_')[-1] == 'features':
            df_feat = pd.read_csv(f, encoding='utf-8', sep=';', decimal=',').drop('Unnamed: 0', axis=1)
        else:
            df_target = pd.read_csv(f, encoding='utf-8', sep=';', decimal=',').drop('Unnamed: 0', axis=1)

    pred_dates = pd.date_range(start=first_predict_date, periods=df_feat.shape[1], freq='3H')
    return df_target, df_feat, pred_dates

def create_df(pth):
    config    = Config()
    cols      = config.target
    cols_feat = config.features

    df        = pd.read_csv(pth)
    df['Datetime'] = pd.to_datetime(df['Datetime']) 
    df.set_index('Datetime', inplace=True) 
    df_target = df[cols] 
    df_feat   = df[cols_feat]

    return df_feat, df_target

def normalize_data(df):
    dates    = df.index
    x        = df.values
    x_scaled = normalize.fit_transform(x)
    df_back  = pd.DataFrame(x_scaled)
    df_back.index = dates
    return df_back

def get_era5_data(dataset, lat, lon):
    rea         = xr.open_dataset(dataset)
    ens_member  = rea.number.values.tolist()
    rg_lat      = lat
    rg_long     = lon
    latlon      = rea.sel(latitude=rg_lat, longitude=rg_long, method='nearest')
    teste       = {'time':pd.to_datetime(rea.time.values)}
    for member in ens_member:
        teste[f'Hs-{member}']        = latlon.swh.values[:,member]
        teste[f'10m-direc-{member}'] = latlon.dwi.values[:,member]
        teste[f'10m-speed-{member}'] = latlon.wind.values[:,member]
        teste[f'Period-{member}']    = latlon.pp1d.values[:,member]
    
    rg_rea      = pd.DataFrame(teste)    
    rg_rea      = rg_rea.sort_values(by='time', ignore_index=True)
    rg_rea      = rg_rea.set_index('time')
    
    return rg_rea

def split_sequence(sequence, sequence2, n_steps_in, lead_time, flag):
    X, y = [], []
    m = len(sequence) - lead_time
    for i in range(m):
        ls         = [0]*lead_time 
        end_ix     = i + n_steps_in
        out_end_ix = end_ix + lead_time
        if flag:
            if out_end_ix > m:
                break
            seq_x, seq_y = sequence[i:end_ix+lead_time, 1:], sequence2[end_ix+lead_time]
            seq_x2 = sequence[i:end_ix, 0].tolist()
            seq_x2 = seq_x2 + ls 
            seq_x2 = np.array(seq_x2)
            seq_x2 = np.concatenate((seq_x2.reshape(-1,1), seq_x), axis=1)
            seq_x  = seq_x2
        else:
            if end_ix > m:
                break
            seq_x, seq_y = sequence[i:end_ix, 0:], sequence2[end_ix+lead_time-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y) 

def prepare_data_lstm(X,y, flag, num_features):
    dim_1                        = X.shape[0]
    dim_2                        = X.shape[1]
    dim_y                        = y.shape[0]
    if flag:
        dim_3                     = y.shape[1]
    X                            = X.flatten()
    y                            = y.flatten()
    #X                            = min_max_scalle_x.fit_transform(X.reshape(-1, 1))
    #y                            = min_max_scalle_y.fit_transform(y.reshape(-1, 1))
    X                            = X.reshape((dim_1, dim_2, num_features))
    if flag:
        y                            = y.reshape((dim_y, dim_3, 1))
    else:
        y                            = y.reshape((dim_y, 1, 1))
    return X,y

def create_train_test(df, df2, npredict, lead):
    if (lead==6):
        count = 0
    elif (lead==12):
        count = 6
    elif (lead==18):
        count = 12
    else:
        count = 18
    
    datas                    = []
    for col in df.columns:
        inputs_1             = df[col][:-(npredict+count)].values
        inputs_1             = inputs_1.reshape((len(inputs_1), 1))
        datas.append(inputs_1)
    
    inputs                   = np.hstack(datas)
    target                   = df2[:-(npredict+count)]

    predict_data = []
    for col in df.columns:
        inputs_2             = df[col][-(npredict+count):].values
        inputs_2             = inputs_2.reshape((len(inputs_2), 1))
        predict_data.append(inputs_2)

    x_input                  = np.hstack(predict_data)
    target_predict           = df2[-(npredict+count):]

    return inputs, target, x_input, target_predict

def inverse_transform(x, flag):
    if flag:
        inv                  = normalize.inverse_transform(x.reshape(-1, 1))
    else:
        inv                  = normalize.inverse_transform(x.reshape(-1, 1))

    return inv

def convert_cnn_lstm(data):
    n1                       = data.shape[0]
    n2                       = data.shape[1]
    n3                       = data.shape[2]
    new_data                 = data.reshape(n1,1,n2,n3)
    return new_data

def knn_filter(buoy_filepath, cols):    

    wave                  = pd.read_csv(buoy_filepath, sep=',')
    wave[wave[cols[0]] == -9999] = np.nan
    wave[wave[cols[1]] == -9999] = np.nan
    wave                  = wave.dropna()
    raw                   = pd.read_csv(buoy_filepath, sep=',')
    raw[raw[cols[0]] == -9999] = np.nan
    raw[raw[cols[1]] == -9999] = np.nan
    raw                   = raw.dropna()
    minmax                = preprocessing.MinMaxScaler(feature_range=(0, 1))
    wave[cols]            = minmax.fit_transform(wave[cols])
    X1                    = wave[cols[1]].values.reshape(-1, 1)
    X2                    = wave[cols[0]].values.reshape(-1, 1)
    X                     = np.concatenate((X1, X2), axis=1)
    outliers_fraction     = 0.01
    classifiers           = {"K Nearest Neighbors (KNN)": KNN(contamination=outliers_fraction)}
    xx , yy               = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    outliers              = {}

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X)
        scores_pred       = clf.decision_function(X) * -1
        y_pred            = clf.predict(X)
        n_inliers         = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers        = np.count_nonzero(y_pred == 1)
        df                = wave.copy()
        df['outlier']     = y_pred.tolist()
        outliers[clf_name]= df.loc[df['outlier'] == 1]
        IN1               =  np.array(df[cols[1]][df['outlier'] == 0]).reshape(-1,1)
        IN2               =  np.array(df[cols[0]][df['outlier'] == 0]).reshape(-1,1)
        OUT1              =  df[cols[1]][df['outlier'] == 1].values.reshape(-1,1)
        OUT2              =  df[cols[0]][df['outlier'] == 1].values.reshape(-1,1)
        threshold         = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
        Z                 = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z                 = Z.reshape(xx.shape)
        raw['outlier']    = df['outlier']
        filtered          = raw[raw['outlier'] != 1]
        
        return filtered.reset_index(drop=True)
