import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob
import pickle
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (10,6)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def erro(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred)/y_true)*100

def mean_erro(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))

def create_output_graph_fut(files):
    dfs_6       = []

    for f in files:
        df      = pd.read_csv(f).drop(['Unnamed: 0'],axis=1)
        lead    = f.split('/')[-1].split('_')[1]
        model   = f.split('/')[-1].split('_')[2][:-4]
        df.set_index('Data', inplace=True)
        df.columns = ['Hs_real', f'Hs_{model}_{lead}']
        dfs_6.append(df)
 
    test_6      = pd.concat(dfs_6, axis=1)
 
    real_6      = test_6.iloc[:,0].to_frame()
    test_6.drop(['Hs_real'], axis=1, inplace=True)
    test_6['mean'] = test_6.mean(axis=1)
    
    n1          = real_6.shape[0]
    xx          = [[0, int(n1/4), int(n1/2), int(3*n1/4), n1-1]]
    yy          = [[real_6.index[0],real_6.index[int(n1/4)],real_6.index[int(n1/2)],real_6.index[int(3*n1/4)],real_6.index[n1-1]]]
    reais       = [real_6]
    predictions = [test_6]

    ls      = predictions[0].columns.to_list()
    lead    = int(ls[0].split('_')[-1]) 

    plt.figure(1)
    create_plots(lead,reais[0],predictions[0],xx[0],yy[0], True, None)
    plt.figure(1).text(0.5, 0.04, 'Time', ha='center', va='center')
    plt.figure(1).text(0.06, 0.5, 'Wave height - Hs (m)', ha='center', va='center', rotation='vertical')

    ls      = predictions[0].columns.to_list()
    lead    = int(ls[0].split('_')[-1]) 

    plt.figure(2)
    create_plots(lead,reais[0],predictions[0],xx[0],yy[0], False, None)
    plt.figure(2).text(0.5, 0.04, 'Time', ha='center', va='center')
    plt.figure(2).text(0.06, 0.5, 'Wave height - Hs (m)', ha='center', va='center', rotation='vertical')
    plt.show()

def create_output_graph(files,ls_antigos,dict_metrics, average):
    dfs_6       = []
    dfs_12      = []
    dfs_18      = []
    dfs_24      = []

    for f in files:
        df      = pd.read_csv(f).drop(['Unnamed: 0'],axis=1)
        lead    = f.split('/')[-1].split('_')[1]
        model   = f.split('/')[-1].split('_')[2][:-4]
        df.set_index('Data', inplace=True)
        df.columns = ['Hs_real', f'Hs_{model}_{lead}']
        if lead == '6':
            dfs_6.append(df)
        elif lead == '12':
            dfs_12.append(df)
        elif lead == '18':
            dfs_18.append(df)
        else:
            dfs_24.append(df)

    test_6      = pd.concat(dfs_6, axis=1)
    test_12     = pd.concat(dfs_12, axis=1)
    test_18     = pd.concat(dfs_18, axis=1)
    test_24     = pd.concat(dfs_24, axis=1)

    real_6      = test_6.iloc[:,0].to_frame()
    real_12     = test_12.iloc[:,0].to_frame()
    real_18     = test_18.iloc[:,0].to_frame()
    real_24     = test_24.iloc[:,0].to_frame()
    test_6.drop(['Hs_real'], axis=1, inplace=True)
    test_12.drop(['Hs_real'], axis=1, inplace=True)
    test_18.drop(['Hs_real'], axis=1, inplace=True)
    test_24.drop(['Hs_real'], axis=1, inplace=True)
    test_6['mean'] = test_6.mean(axis=1)
    test_12['mean']= test_12.mean(axis=1)
    test_18['mean']= test_18.mean(axis=1)
    test_24['mean']= test_24.mean(axis=1)
    
    n1          = real_6.shape[0]
    n2          = real_12.shape[0]
    n3          = real_18.shape[0]
    n4          = real_24.shape[0]
    xx          = [[0, int(n1/4), int(n1/2), int(3*n1/4), n1-1],
                   [0, int(n2/4), int(n2/2), int(3*n2/4), n2-1],
                   [0, int(n3/4), int(n3/2), int(3*n3/4), n3-1],
                   [0, int(n4/4), int(n4/2), int(3*n4/4), n4-1]]
    yy          = [[real_6.index[0],real_6.index[int(n1/4)],real_6.index[int(n1/2)],real_6.index[int(3*n1/4)],real_6.index[n1-1]],
                   [real_12.index[0],real_12.index[int(n2/4)],real_12.index[int(n2/2)],real_12.index[int(3*n2/4)],real_12.index[n2-1]],
                   [real_18.index[0],real_18.index[int(n3/4)],real_18.index[int(n3/2)],real_18.index[int(3*n3/4)],real_18.index[n3-1]],
                   [real_24.index[0],real_24.index[int(n4/4)],real_24.index[int(n4/2)],real_24.index[int(3*n4/4)],real_24.index[n4-1]]]
    reais       = [real_6,real_12,real_18,real_24]
    predictions = [test_6,test_12,test_18,test_24]
    
    for i in range(4):
        ls      = predictions[i].columns.to_list()
        lead    = int(ls[0].split('_')[-1]) 
        plt.figure(i+1)
        #plt.subplot(2,2,i+1)
        create_plots(lead,reais[i],predictions[i],xx[i],yy[i], True, None)
        plt.figure(i+1).text(0.5, 0.04, 'Time', ha='center', va='center')
        plt.figure(i+1).text(0.06, 0.5, 'Wave height - Hs (m)', ha='center', va='center', rotation='vertical')
    for i in range(4):
        ls      = predictions[i].columns.to_list()
        lead    = int(ls[0].split('_')[-1]) 
        inicio  = predictions[i].index.min()  
        fim     = predictions[i].index.max()
        antigo  = correct_data(ls_antigos, inicio, fim, lead)
        
        plt.figure(5)
        plt.subplot(2,2,i+1)
        create_plots(lead,reais[i],predictions[i],xx[i],yy[i], False, antigo)
        plt.figure(5).text(0.5, 0.04, 'Time', ha='center', va='center')
        plt.figure(5).text(0.06, 0.5, 'Wave height - Hs (m)', ha='center', va='center', rotation='vertical')
    # # creating the weights
    if average: 
        for i in range(4):
            ls      = predictions[i].columns.to_list()
            lead    = int(ls[0].split('_')[-1]) 
            inicio  = predictions[i].index.min()  
            fim     = predictions[i].index.max()
            antigo  = correct_data(ls_antigos, inicio, fim, lead)
            df_metric = create_weighted_average(predictions[i],dict_metrics)
            plt.figure(6)
            plt.subplot(2,2,i+1)
            create_plots_avg(lead,reais[i],predictions[i],xx[i],yy[i], df_metric, antigo)
            plt.figure(6).text(0.5, 0.04, 'Time', ha='center', va='center')
            plt.figure(6).text(0.06, 0.5, 'Wave height - Hs (m)', ha='center', va='center', rotation='vertical')
    plt.show()

def create_weighted_average(df, dct_metric):
    soma = 0
    aux = pd.DataFrame()
    aux['Data'] = df.index
    for j in range(len(df.columns)-1):
        coluna = df.iloc[:,j]
        tar = coluna.name[3:]
        mapes = dct_metric[tar]
        aux[tar] = coluna.values*mapes
        soma += mapes

    aux['soma'] = aux.sum(axis=1)
    aux['weghted_avg'] = aux['soma']/soma
    
    return aux.set_index('Data')

def create_plots(lead, df_true, df_predict,x,y, flag, antigo):
    ls      = df_predict.columns.to_list()
    plt.title(f'Deep learning prediction of Hs - lead: {lead}')
    plt.plot(df_true.index, df_true['Hs_real'] , '--', label=f'Real buoy data',color='blue')
    if flag:
        for col in ls[:-1]:
            mapes = mape(df_true['Hs_real'], df_predict[col]).round(2)
            plt.plot(df_predict.index, df_predict[col] , '-', label=f'Predicted {col} - {mapes}')
    else:
        for col in ls[-1:]:
            mapes = mape(df_true['Hs_real'], df_predict[col]).round(2)
            #mapes2= mape(antigo['Hs_real'], antigo[f'Hs_artigo_antigo_{lead}']).round(2)
            plt.plot(df_predict.index, df_predict[col] , '-', label=f'Predicted {col} - {mapes}')
            #plt.plot(antigo.index, antigo[f'Hs_artigo_antigo_{lead}'], '--', label=f'LSTM-paper result - {mapes2}')          
    plt.legend()
    plt.xticks(x,y, rotation=30)

def create_plots_avg(lead, df_true, df_predict,x,y, df_metric, antigo):
    plt.title(f'Deep learning prediction of Hs - lead: {lead}')
    plt.plot(df_true.index, df_true['Hs_real'] , '--', label=f'Real buoy data',color='blue')
    
    mapes = mape(df_true['Hs_real'], df_predict['mean']).round(2)
    mapes2= mape(antigo['Hs_real'], antigo[f'Hs_artigo_antigo_{lead}']).round(2)
    mapes3= mape(df_true['Hs_real'], df_metric['weghted_avg']).round(2)

    plt.plot(df_predict.index, df_predict['mean'] , '-', label=f'Mean - {mapes}')
    plt.plot(df_metric.index, df_metric['weghted_avg'] , '-', label=f'Weighted avg - {mapes3}')
    plt.plot(antigo.index, antigo[f'Hs_artigo_antigo_{lead}'], '--', label=f'LSTM-paper result - {mapes2}')          
    plt.legend()
    plt.xticks(x,y, rotation=30)

def get_metrics(data):
    dict_metrics     = {}
    for f in data:
        with open(f, 'rb') as handle:
            metrics  = pickle.load(handle)
            model    = f.split('/')[-1].split('_')[-1][:-4] 
            lead     = f.split('/')[-1].split('_')[-2]
            val_mape = metrics['val_mean_absolute_percentage_error'][-1]
            name     = str(model)+'_'+str(lead)
            dict_metrics[name] = val_mape

    return dict_metrics

def organize_old_result(files):
    dfs         = {}
    for f in files:
        df      = pd.read_csv(f).drop(['Unnamed: 0'],axis=1)
        lead    = f.split('/')[-1].split('_')[-2]
        df.set_index('Data', inplace=True)
        df.columns = ['Hs_real', f'Hs_artigo_antigo_{lead}']
        dfs[lead] = df
    
    return dfs

def correct_data(dict_old,inicio,fim, lead):
    df         = dict_old[str(lead)]
    df         = df.loc[df.index >= inicio]
    df         = df.loc[df.index <= fim]

    return df

def correct_output(files,dest):
    for f in files:
        df      = pd.read_csv(f)
        lead    = f.split('/')[-1].split('_')[1]
        model   = f.split('/')[-1].split('_')[2][:-4]
        df = df.drop(['label','predict','mean'], axis=1)
        df = df[['Data', 'real', 'result']]
        df.columns = ['Data', 'Hs_real', f'Hs_{model}'] 
        df.to_csv(f'{dest}predictions_{lead}_{model}.csv')

def create_graph(dest):
    error_prediction = True
    future      = True
    output      = glob.glob(dest+'*.csv')
    if error_prediction:
        correct_output(output,dest)
    res_antigo  = glob.glob('/Users/felipeminuzzi/Documents/OCEANO/Simulations/Machine_Learning/results/predict_with_historic_buoy_without_outliers/buoy_rio_grande/*.csv')
    ls_antigos  = organize_old_result(res_antigo)
    #metric      = glob.glob(dest+'metric/*')
    #dict_metrics= get_metrics(metric)
    if future:
        create_output_graph_fut(output)
    else:
        create_output_graph(output,ls_antigos,None, False)