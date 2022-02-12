import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def erro(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred)/y_true)*100

def create_plots(num,df,x,y):
    plt.figure(num)
    plt.title(f'Predictions with LSTM')
    mapes = mape(df['Hs Ens member'], df['Hs Predict Value']).round(2)
    plt.title(f'MAPE for mean: {mapes}%')
    plt.plot(df['Data'], df['Hs Predict Value'] , '-', label=f'Predicted Hs',color='red')
    plt.plot(df['Data'], df['Hs Ens member'] , '-', label=f'NOAA',color='blue')
    plt.legend()
    plt.xticks(x,y, rotation=30)
    plt.savefig(f'lstm_predicts_{num}.png', bbox_inches='tight')

def create_windowing(df, col_title, lag_size):
    """Given a univariate time series, generates new features from its lagged values.
       Arguments:
                 df - pandas dataframe containing a raw univariate time series
                 col_title - name of the only column of df
                 lag_size - how many lagged series for creating the new features
       Output: dataframe with the original values of the time series along with its lagged values.
               Rows with missing data are simply dropped.
    """

    final_df = None
    for i in range(0, (lag_size + 1)):
        serie = df.shift(i)
        if (i == 0):
            serie.columns = [col_title]
        else:
            serie.columns = [f'lag{i}_{col_title}']
        final_df = pd.concat([serie, final_df], axis=1)
    return final_df

def format_path(path):
    """""Formats the path string in order to avoid conflicts."""

    if path[-1]!='/':
        path = path + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    return path