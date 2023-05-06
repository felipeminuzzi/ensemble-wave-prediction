
import glob
import pandas as pd
import numpy as np
import time
import pickle
import os

from joblib import Parallel, delayed
from tqdm import tqdm

from src.models.tensorflow import TFlow
from src.config.config import Config
from src.data import format_data as data_format
from src.features import features as feat 

def train_models(models, features, target, dates, forecast, npredict, lead, num_features, dest, error_prediction, pth, flag, conf, future_predict, pth2):
    config          = conf
    val_size        = config.val_size
    ls_mod          = config.machine
    epochs          = config.epochs
    n_calls         = config.n_calls
    n_initial_points= config.n_initial_points
    target_df       = target[['target']]
    
    for reg in models:
        if reg in ls_mod:
            md          = TFlow(reg, features, target_df, dates, forecast, npredict, lead, num_features, epochs, val_size, flag, future_predict)
        elif reg == 'arima':
            md          = ArimaModel(reg, features, target_df, dates, forecast, npredict, lead, num_features, val_size, flag, is_seazonal=False)
        else:
            model_dict  = config.models[reg]
            regressor   = model_dict['reg']
            space       = model_dict['space']
            parameters  = model_dict['hyper_params']

            md          = SklearnClass(regressor, features, target_df, dates, forecast, npredict, lead, num_features, val_size, n_calls, 
                                    space, n_initial_points, parameters, reg, flag)
        result, metric  = md.create_future()
        if error_prediction:
            negatives = target.loc[(target.index >= result['Data'].min()) & (target.index <= result['Data'].max())].copy()
            negatives.reset_index(inplace=True)
            result['predict'] = result['predict']*negatives['negative'] 
            result = correct_result(result, features, reg, pth, config, pth2)
        else:
            mape_buoy = feat.mape(result['label'], result['predict'])
            print(f'MAPE between predicted x buoy for {reg}: {mape_buoy}')
        
        result.to_csv(f'{dest}predictions_{lead}_{reg}.csv')
        #save_metric(dest,lead,reg,metric)

def train_future_models(mod, features, target, dates, forecast, npredict, dest, num_features, conf, name):
    config          = conf
    epochs          = config.epochs
    cols            = conf.target

    md              = TFlow(mod, features, target, dates, forecast, npredict, 0, num_features, epochs, 0, False, None)
    result          = md.create_multi_output()
    result.set_index('Data', inplace=True)

    pth             = glob.glob(f'./data/raw/noaa/{name}/*')[0]    
    df_noaa         = pd.read_csv(pth, encoding='utf-8', sep=';', decimal=',').drop('Unnamed: 0', axis=1)
    df_noaa         = df_noaa[['time','deterministic']].fillna(method='bfill')
    df_noaa['time'] = pd.to_datetime(df_noaa['time'])
    df_noaa         = df_noaa.loc[(df_noaa['time'] >= result.index.min()) & (df_noaa['time'] <= result.index.max())]
    df_noaa.set_index('time',inplace=True)

    with open(f'./data/processed/{name}/boia.pkl', 'rb') as handle:
        boia = pickle.load(handle)

    tgt             = pd.read_csv(boia)
    tgt['Datetime'] = pd.to_datetime(tgt['Datetime'])   
    tgt.set_index('Datetime',inplace=True)
    tgt         = tgt.loc[(tgt.index >=  result.index.min()) & (tgt.index <= result.index.max())]
    tgt         = tgt[cols]

    result          = result.join(tgt)
    result.rename(columns = {cols[0] : 'real'}, inplace=True)

    result['result']= result['predict'] + df_noaa['deterministic']
    result['noaa']  = df_noaa['deterministic']
    result.fillna(method='bfill', inplace=True)
    result.to_csv(f'{dest}predictions_{0}_{mod}.csv')

def correct_result(df, df_feat, reg, boia, conf, path):
    cols        = conf.target
    use_era     = conf.use_era
    col_error   = conf.var_to_error
    
    df_det         = pd.read_csv(path, encoding='utf-8', sep=';', decimal=',')[['time',col_error]] 
    df_det['time'] = pd.to_datetime(df_det['time'])
    df_det         = df_det.loc[df_det['time'].isin(df['Data'].values)]
    
    if use_era:
        aux         = df_feat.loc[(df_feat.index >=  df['Data'].min()) & (df_feat.index <= df['Data'].max())]
        cols_to_use = [f'Hs-{i}' for i in range(10)]
        aux         = aux[cols_to_use]
        aux['mean'] = aux.mean(axis=1)
    else:
        aux         = df_det.copy()
        aux.columns = ['time','mean']
    
    aux.set_index('time',inplace=True)  
    df.set_index('Data', inplace=True)
    df          = df.join(aux['mean'])
    df['result']= df['predict'] + df['mean']

    tgt         = pd.read_csv(boia)
    tgt['Datetime'] = pd.to_datetime(tgt['Datetime'])   
    tgt.set_index('Datetime',inplace=True)
    tgt         = tgt.loc[(tgt.index >=  df.index.min()) & (tgt.index <= df.index.max())]
    tgt         = tgt[cols]
    df['real']  = tgt[cols] 

    mape_mean  = feat.mape(df['mean'], df['result'])
    mape_buoy  = feat.mape(df['real'], df['result'])
    mape_comp  = feat.mape(df['real'], df['mean'])
    print(f'MAPE between predicted x mean for {reg}: {mape_mean}')
    print(f'MAPE between predicted x buoy for {reg}: {mape_buoy}')
    print(f'MAPE between mean x buoy for {reg}: {mape_comp}')
    
    return df

def save_metric(dest,lead,reg, metric):
    path            = f'{dest}metric/'
    save_name       = f'{dest}metric/metric_{lead}_{reg}.pkl'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(save_name, 'wb') as fp:
        pickle.dump(metric, fp) 

def get_lat_lon(pth):
    df              = pd.read_csv(pth)
    lat             = df['Lat'][0]
    lon             = df['Lon'][0]
    return lat, lon

def setup(error_pred, pth, era):
    if error_pred:
        for i in pth:
            if i.split('/')[-1][:-4].split('_')[-1] == 'noaa':
                path_ensemble = i
            elif i.split('/')[-1][:-4].split('_')[-1] == 'boia':
                path_boia = i
            elif i.split('/')[-1][-2:] == 'nc':
                path_era = i
        to_result        = path_boia
        try:
            to_deterministic = path_ensemble
        except:
            to_deterministic = None
        if era:
            lat, lon     = get_lat_lon(path_boia)
            era_df       = data_format.get_era5_data(path_era,lat, lon)
            features,target  = data_format.create_df_error_era(era_df,path_boia,error_pred)
        else:
            features,target  = data_format.create_df_error(path_ensemble,path_boia, error_pred)
    else:
        for i in pth:
            if i.split('/')[-1][:-4].split('_')[-1] == 'boia':
                path_target = i
        to_result        = None
        to_deterministic = None
        features,target  = data_format.create_df(path_target)
    return features, target, to_result, to_deterministic

def dispatch(ori, dest, name):
    config           = Config()
    error_prediction = config.use_error
    use_era          = config.use_era
    spaced_predict   = config.use_spaced 
    future_predict   = config.future
    multi_target     = config.multi_target

    dest             = feat.format_path(dest)
    path             = glob.glob(ori+'*')

    if multi_target:
        target, features, dates     = data_format.multi_target_setup(ori)
        npredict                    = len(dates)
    else:
        features, target, to_result, to_deterministic = setup(error_prediction,path,use_era)
        dates            = features.index[-npredict:]
        npredict         = config.predict
    
    forecast         = config.forecast
    num_features     = features.shape[1]
    
    if future_predict:
        leads        = [0]
        n_jobs       = 1
    else:
        leads        = config.leads
        n_jobs       = config.n_jobs
    
    models           = config.machine

    start            = time.time()
    if multi_target:
        Parallel(n_jobs=n_jobs,backend='multiprocessing')(delayed(train_future_models)(mod, features, target, dates, forecast, npredict, dest, num_features, config, name) for mod in tqdm(models, desc='Ensemble prediction...'))
    else:
        Parallel(n_jobs=n_jobs,backend='multiprocessing')(delayed(train_models)(models,features, target, dates, 
                forecast, npredict, lead, num_features, dest, error_prediction, to_result, spaced_predict, config, future_predict, to_deterministic) for lead in tqdm(leads, desc='Ensemble prediction...'))
    end              = time.time()

    print('Time of execution: ',(end-start)/60, ' minutes.')
    print('###############################################')
    print('###############################################')
    print('##                                           ##')
    print('##      Simulation succesfully finished!     ##')
    print('##                                           ##')
    print('###############################################')
    print('###############################################')

