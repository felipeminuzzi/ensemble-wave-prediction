import numpy as np
import pandas as pd
import tensorflow as tf
import time
import warnings
from sklearn import preprocessing
warnings.filterwarnings('ignore')

min_max_scalle_x = preprocessing.MinMaxScaler((0,1))
min_max_scalle_y = preprocessing.MinMaxScaler((0,1))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def split_sequence(sequence, n_steps_in, lead_time, flag):
    X, y = [], []
    m = len(sequence) - lead_time
    for i in range(m):
        end_ix     = i + n_steps_in
        out_end_ix = end_ix + int(n_steps_in/2)
        if flag == True:
            if out_end_ix > m:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        else:
            if end_ix > m:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix+lead_time-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y) 

def compile_and_fit(model, X, y, patience=20):
    early_stopping           = tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss',
        patience             = patience,
        mode                 = 'min'
    )
    model.compile(
        loss                 = 'mean_absolute_error',
        optimizer            = 'adam',
        metrics              = ['mean_squared_error']
    )
    history                  = model.fit(
        X,
        y,
        epochs               = 200,
        validation_split     = 0.2,
        callbacks            = [early_stopping],
        verbose              = 0
    )
    return history

def prediction(model, data_in, n_steps_in, num_features):
    x_predict = data_in.reshape(1, n_steps_in, num_features)
    predict                      = model.predict(x_predict)
    predict                      = min_max_scalle_x.inverse_transform(predict[0,0].reshape(-1, 1))
    return predict[0,0]

def prepare_data(X,y, flag, num_features):
    dim_1                        = X.shape[0]
    dim_2                        = X.shape[1]
    if flag:
        dim_3                    = y.shape[1]
    X                            = X.flatten()
    y                            = y.flatten()
    X                            = min_max_scalle_x.fit_transform(X.reshape(-1, 1))
    y                            = min_max_scalle_y.fit_transform(y.reshape(-1, 1))
    X                            = X.reshape((dim_1, dim_2, num_features))
    if flag:
        y                        = y.reshape((dim_1, dim_3, num_features))
    else:
        y                        = y.reshape((dim_1, 1, num_features))
    return X,y

def get_model(num_prev, forecast, num_features):
    modelo             = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(forecast, num_features)),
        tf.keras.layers.LSTM(48, activation='tanh', return_sequences=True, input_shape=(forecast, num_features)),
        tf.keras.layers.LSTM(32, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(num_prev, kernel_initializer=tf.initializers.zeros)])
    return modelo

def create_non_lead_future(df, npredict, forecast, num_features, name):
    inputs                    = df[f'Hs_{name}'][:-npredict].values
    X, y                      = split_sequence(inputs, forecast, 0, True)
    X, y                      = prepare_data(X,y, True, num_features)
    
    model                     = get_model(npredict, forecast, num_features)
    history                   = compile_and_fit(model, X, y)

    x_input                   = df[f'Hs_{name}'][:-npredict][-forecast:].values
    x_in                      = min_max_scalle_x.fit_transform(x_input.reshape(-1, 1))
    x_predict                 = x_in.reshape(1, len(x_in), num_features)
    
    predict                   = model.predict(x_predict)
    predict                   = min_max_scalle_x.inverse_transform(predict[0,:].reshape(-1, 1))
    
    result                    = pd.DataFrame()
    result['Data']            = df['time'][-npredict:].values
    result['hs_predict_era5'] = predict[:,0]
    result['hs_era5_real']    = df[f'Hs_{name}'][-npredict:].values

    metric                    = mape(result['hs_era5_real'].values, result['hs_predict_era5'].values)
    print(f'For {name}, future prediction with LSTM has MAPE: {metric.round(2)}')

    return result

def run_model(name, df_results):
    df_era5      = pd.read_csv('./data/raw/era/era5_reanalysis.csv', encoding='utf-8', sep=';', decimal=',')
    df_era5      = df_era5[df_era5['time'] <= df_results['Data'].values[-1]]
    start        = time.time()
    npredict     = df_results.shape[0]
    forecast     = int(npredict*2)
    num_features = 1

    df_pred_era5 = create_non_lead_future(df_era5, npredict, forecast, num_features, name)
    df_pred_era5.to_csv(f'./data/raw/era/lstm_{name}_predictions_future.csv')

    end          = time.time()
    print('Time of execution: ',(end-start)/60, ' minutes.')

    return df_pred_era5
