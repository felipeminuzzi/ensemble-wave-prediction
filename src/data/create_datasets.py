import numpy as np
import pandas as pd
import glob
import pickle
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from src.features import features as feat 

def create_new(folders, typ, buoy, dest, lag):
    df_final = pd.DataFrame()
    df_boia  = pd.read_csv(buoy)
    df_boia['Datetime'] = pd.to_datetime(df_boia['Datetime'])
    dict_features   = {}
    dict_target     = {}
    list_dates      = []

    max_number_pred = lag*8
    for j in tqdm(range(0,max_number_pred), desc='Processing dataset...'):
        lista_cols_feat = []
        lista_cols_tgt  = []
        for fold in folders:
            proces_folders = glob.glob(fold+typ+'/*')
            proces_folders.sort()    
            for file in proces_folders:
                df    = pd.read_csv(file, encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1)
                value = df[['time','deterministic']].iloc[j]['deterministic']
                hour  = df[['time','deterministic']].iloc[j]['time']
                if value == 0 or np.isnan(value) == True:
                    value = lista_cols_feat[-1]
                try:
                    real = df_boia.loc[df_boia['Datetime'] == pd.to_datetime(hour)]['Wvht'].values[0]
                except:
                    real = real
                erro  = real-value
                lista_cols_feat.append(value)
                lista_cols_tgt.append(erro)
        dict_features[f'feat_{j}'] = lista_cols_feat
        dict_target[f'tgt_{j}']    = lista_cols_tgt
        list_dates.append(hour)
        if j == 0:
            first_hour_predict = pd.to_datetime(hour)

    df_final_target   =  pd.DataFrame(dict_target)
    df_final_features =  pd.DataFrame(dict_features)

    df_final_target[0:-1].to_csv(f'{dest}noaa_data_target.csv', encoding='utf-8', sep=';', decimal=',')
    df_final_features[0:-1].to_csv(f'{dest}noaa_data_features.csv', encoding='utf-8', sep=';', decimal=',')

    save_name         = f'{dest}first_hour_predict.pkl'
    with open(save_name, 'wb') as fp:
        pickle.dump(first_hour_predict, fp) 

def create_reanalysis(folders, typ, dest):
    df_reanalise = pd.DataFrame()
    for fold in tqdm(folders, desc='Processing dataset...'):
        proces_folders = glob.glob(fold+typ+'/*')
        proces_folders.sort()
        for file in proces_folders:
            df = pd.read_csv(file, encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1)
            df.set_index('time', inplace=True) 
            df = df.loc[df.index == df.index.min()]
            df_reanalise = df_reanalise.append(df)

    df_reanalise.reset_index(inplace=True)
    df_reanalise = df_reanalise.loc[(df_reanalise != 0).any(axis=1)]
    df_reanalise = df_reanalise.fillna(method='ffill')
    df_reanalise = df_reanalise[df_reanalise['time'] != 0].reset_index(drop=True)
    time = df_reanalise['time'].max()[0:10]
    df_reanalise.to_csv(f'{dest}noaa_data_reanalise_until_{time}_noaa.csv', encoding='utf-8', sep=';', decimal=',')

def create_pred_rea(folders, typ, dest):
    df_reanalise_2 = pd.DataFrame()
    for fold in tqdm(folders, desc='Processing dataset...'):
        proces_folders = glob.glob(fold+typ+'/*')
        proces_folders.sort()
        for file in proces_folders:
            df = pd.read_csv(file, encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1)
            df.set_index('time', inplace=True) 
            df = df.iloc[0:2,:]
            df_reanalise_2 = df_reanalise_2.append(df)

    inicio = pd.to_datetime(df_reanalise_2.index[0])
    final  = pd.to_datetime(df_reanalise_2.index[-1])
    aux    = pd.DataFrame(pd.date_range(inicio, final, freq='3H'), columns=['time'])

    df_reanalise_2.index = pd.DatetimeIndex(df_reanalise_2.index)
    df_reanalise_2 = df_reanalise_2[df_reanalise_2.index >= pd.to_datetime(2021)]
    df_reanalise_2 = aux.join(df_reanalise_2)
    df_reanalise_2 = df_reanalise_2.fillna(method='ffill')
    df_reanalise_2.reset_index(inplace=True)

    time = str(df_reanalise_2['time'].max())[0:10]
    df_reanalise_2.to_csv(f'{dest}noaa_data_1reanalise_1predict_until_{time}_noaa.csv', encoding='utf-8', sep=';', decimal=',')

def create_lagged(folders, typ, lag, dest):
    df_lags = pd.DataFrame()
    for fold in tqdm(folders, desc='Processing dataset...'):
        proces_folders = glob.glob(fold+typ+'/*')
        proces_folders.sort()
        for file in proces_folders:
            df = pd.read_csv(file, encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1)
            df.set_index('time', inplace=True)
            df = df.loc[df.index == str(pd.to_datetime(df.index.min()) + dt.timedelta(hours=lag))]
            df_lags = df_lags.append(df)

    df_lags.reset_index(inplace=True)
    df_lags = df_lags.loc[(df_lags != 0).any(axis=1)]
    df_lags = df_lags.fillna(method='ffill')
    df_lags = df_lags[df_lags['time'] != 0].reset_index(drop=True)
    time = df_lags['time'].max()[0:10]
    df_lags.to_csv(f'{dest}noaa_data_lag_{lag}_until_{time}_noaa.csv', encoding='utf-8', sep=';', decimal=',')

#ori:  /home/g1/minuzzi/storage/OneDrive
#dest: ./data/processed
#buoy: '../Machine_learning/ensemble-wave-prediction/data/raw/buoy/buoy_historic_abrolhos.csv'

def dispatch(ori, dest, buoy_path, name, lag):
    ori     = feat.format_path(ori)
    dest    = feat.format_path(dest)
    boia    = buoy_path
    typ     = f'processed_{name}'

    df_boia             = pd.read_csv(boia)
    df_boia['Datetime'] = pd.to_datetime(df_boia['Datetime'])

    min_boia = df_boia['Datetime'].min()
    max_boia = df_boia['Datetime'].max()
    max_boia = max_boia - dt.timedelta(days=lag)

    folders  = glob.glob(f'{ori}/*/')
    folders.sort()

    min_noaa = pd.to_datetime(folders[0].split('/')[-2])
    max_noaa = pd.to_datetime(folders[-1].split('/')[-2])

    min_date = str(max(min_boia,min_noaa))[0:10].split('-')
    min_date = min_date[0]+min_date[1]+min_date[2]
    max_date = str(min(max_boia,max_noaa))[0:10].split('-')
    max_date = max_date[0]+max_date[1]+max_date[2]

    folders  = folders[folders.index(ori+min_date+'/'):folders.index(ori+max_date+'/')]
    aux = folders[0].split('/')[-2]   
    df_noaa_first = pd.read_csv(folders[0]+typ+'/'+f'processed_{aux}_lead_00.csv', encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1) 
    if pd.to_datetime(df_noaa_first['time'][0]) < df_boia['Datetime'][0]:
        folders = folders[1:]

    last_date   = folders[-1].split('/')[-2]
    noaa_result = folders[-1]+typ+'/'+f'processed_{last_date}_lead_00.csv'
    df_noaa     = pd.read_csv(noaa_result, encoding='utf-8', sep=';', decimal=',').drop(['Unnamed: 0'], axis=1)

    pth_noaa    = feat.format_path(f'/home/storage/minuzzi/Machine_learning/ensemble-wave-prediction/data/raw/noaa/{name}/')
    df_noaa.to_csv(f'{pth_noaa}noaa_forecast.csv', encoding='utf-8', sep=';', decimal=',')

    save_name         = f'{dest}boia.pkl'
    with open(save_name, 'wb') as fp:
        pickle.dump(boia, fp) 

    create_new(folders, typ, boia, dest, lag)
    # create_reanalysis(folders, typ, dest)
    # create_pred_rea(folders, typ, dest)
    # create_lagged(folders, typ, lag, dest)

    print('###############################################')
    print('###############################################')
    print('##                                           ##')
    print('##                Finished                   ##')
    print('##                                           ##')
    print('###############################################')
    print('###############################################')      
