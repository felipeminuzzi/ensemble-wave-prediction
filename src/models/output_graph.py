import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob
import pickle
from src.features import features as feat 

plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (10,6)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def erro(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred)/y_true)*100

def erro_abs(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred))

def mean_erro(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))

def create_output_graph_fut(files,plot_error):
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

    if plot_error:
        ls      = predictions[0].columns.to_list()
        lead    = int(ls[0].split('_')[-1]) 

        plt.figure(3)
        create_plots_error(lead,reais[0],predictions[0],xx[0],yy[0], True)        
        plt.figure(3).text(0.5, 0.04, 'Time', ha='center', va='center')
        plt.figure(3).text(0.06, 0.5, 'Relative error $\Delta_{\text{rel}}$', ha='center', va='center', rotation='vertical')  

        plt.figure(4)
        create_plots_error(lead,reais[0],predictions[0],xx[0],yy[0], False)
        plt.figure(4).text(0.5, 0.04, 'Time', ha='center', va='center')
        plt.figure(4).text(0.06, 0.5, 'Relative error $\Delta_{\text{rel}}$', ha='center', va='center', rotation='vertical')
    plt.show()

def create_output_graph(files,ls_antigos,dict_metrics, average, plot_error):
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

    if plot_error:
        plt.figure(6)
        for i in range(4):
            ls      = predictions[i].columns.to_list()
            lead    = int(ls[0].split('_')[-1]) 
        
            plt.subplot(2,2,i+1)
            create_plots_error(lead,reais[i],predictions[i],xx[i],yy[i], True)        
        plt.figure(6).text(0.5, 0.04, 'Time', ha='center', va='center')
        plt.figure(6).text(0.06, 0.5, 'Relative error $\Delta_{\text{rel}}$', ha='center', va='center', rotation='vertical')

        plt.figure(7)
        for i in range(4):
            ls      = predictions[i].columns.to_list()
            lead    = int(ls[0].split('_')[-1]) 
            
            plt.subplot(2,2,i+1)
            create_plots_error(lead,reais[i],predictions[i],xx[i],yy[i], False)
        plt.figure(7).text(0.5, 0.04, 'Time', ha='center', va='center')
        plt.figure(7).text(0.06, 0.5, 'Relative error $\Delta_{\text{rel}}$', ha='center', va='center', rotation='vertical')
    plt.show()

def create_multi_graph(files,save_path):
    dfs_6       = []
    j = 1
    for f in files:
        df      = pd.read_csv(f)
        model   = f.split('/')[-1].split('_')[2][:-4]
        df.set_index('Data', inplace=True)
        df.rename(columns = {'predict' : f'erro_{model}', 'label' : f'erro_real_{j}', 
                             'real' : f'Hs_real_{j}', 'result' : f'Hs_{model}', 'noaa' : f'Noaa_{model}'}, inplace=True)
        dfs_6.append(df)
        j += 1

    test_6      = pd.concat(dfs_6, axis=1)
    real_6      = test_6[['erro_real_1','Hs_real_1','Noaa_cnn-lstm']]
    erros_6     = test_6[['erro_cnn-lstm','erro_rnn','erro_dense','erro_cnn','erro_lstm']]
    erros_6['NN mean - this work'] = erros_6.mean(axis=1)
    test_6.drop(['erro_real_1','Hs_real_1','erro_real_2','Hs_real_2','erro_real_3','Hs_real_3','Noaa_cnn-lstm','Noaa_rnn','Noaa_dense','Noaa_cnn','Noaa_lstm',
                 'erro_real_4','Hs_real_4','erro_real_5','Hs_real_5','erro_cnn-lstm','erro_rnn','erro_dense','erro_cnn','erro_lstm' ],axis=1, inplace=True)
    test_6['NN mean - this work'] = test_6.mean(axis=1)
    
    n1          = real_6.shape[0]
    xx          = [[0, int(n1/4), int(n1/2), int(3*n1/4), n1-1]]
    yy          = [[real_6.index[0],real_6.index[int(n1/4)],real_6.index[int(n1/2)],real_6.index[int(3*n1/4)],real_6.index[n1-1]]]
    reais       = [real_6]
    predictions = [test_6]
    erros       = [erros_6]

    ls      = predictions[0].columns.to_list()
    lead = '0'
    
    plt.figure(1)
    create_plots_multi(lead,reais[0],predictions[0],xx[0],yy[0], True, False, 'Hs_real_1')
    plt.figure(1).text(0.06, 0.5, 'Wave height - $H_s$ (m)', ha='center', va='center', rotation='vertical')
    plt.savefig(save_path+'figure_1.png')

    plt.figure(2)
    create_plots_multi(lead,reais[0],predictions[0],xx[0],yy[0], False, False, 'Hs_real_1')
    plt.figure(2).text(0.06, 0.5, 'Wave height - $H_s$ (m)', ha='center', va='center', rotation='vertical')
    plt.savefig(save_path+'figure_2.png')

    plt.figure(3)
    create_plots_multi(lead,reais[0],erros[0],xx[0],yy[0], True, False, 'erro_real_1')
    plt.figure(3).text(0.06, 0.5, 'Absolute error - $\Delta_{abs}$', ha='center', va='center', rotation='vertical')
    plt.savefig(save_path+'figure_3.png')

    plt.figure(4)
    create_plots_multi(lead,reais[0],predictions[0],xx[0],yy[0], True, True, 'Noaa_cnn-lstm')
    plt.figure(4).text(0.06, 0.5, 'Wave height - $H_s$ (m)', ha='center', va='center', rotation='vertical')
    plt.savefig(save_path+'figure_4.png')

    plt.figure(5)
    create_plots_multi(lead,reais[0],predictions[0],xx[0],yy[0], False, True, 'Noaa_cnn-lstm')
    plt.figure(5).text(0.06, 0.5, 'Wave height - $H_s$ (m)', ha='center', va='center', rotation='vertical')
    plt.savefig(save_path+'figure_5.png')

    plt.figure(6)
    create_error_multi(reais[0],predictions[0],xx[0],yy[0], 'Noaa_cnn-lstm')
    plt.figure(6).text(0.06, 0.5, 'Relative error - $\Delta_{rel}$ (%)', ha='center', va='center', rotation='vertical')
    plt.savefig(save_path+'figure_6.png')

    plt.figure(7)
    create_error_abs(reais[0],predictions[0],xx[0],yy[0], 'Noaa_cnn-lstm')
    plt.figure(7).text(0.06, 0.5, 'Absolute error - $\Delta_{abs}$ (m)', ha='center', va='center', rotation='vertical')
    plt.savefig(save_path+'figure_7.png')

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

def create_plots_multi(lead, df_true, df_predict,x,y, flag, antigo, tgt):
    new_name = {'Hs_cnn-lstm' : 'CNN-LSTM', 'Hs_rnn' : 'RNN', 'Hs_cnn' : 'CNN', 'Hs_dense' : 'MLP', 'Hs_lstm' : 'LSTM',
                'erro_cnn-lstm' : 'Error CNN-LSTM', 'erro_rnn' : 'Error RNN', 'erro_dense' : 'Error MLP', 'erro_cnn' : 'Error CNN', 
                'erro_lstm' : 'Error LSTM'}
    df_predict.rename(columns = new_name, inplace=True)

    ls      = df_predict.columns.to_list()
    leg_dict = {'erro_real_1':'Error NOAA x Buoy', 'Hs_real_1':'Buoy - real observed value', 'Noaa_cnn-lstm':'Ensemble numerical model - NOAA'}
    lab = leg_dict[tgt]

    if antigo:
        plt.plot(df_true.index, df_true['Hs_real_1'], '-*', label=f'Buoy - real observed value',color='black')
        mapes = mape(df_true['Hs_real_1'], df_true['Noaa_cnn-lstm']).round(2)
    else:
        mapes = 0
    if lab == 'Ensemble numerical model - NOAA':
        label_name = f'{lab} - {mapes}'
    else:
        label_name = f'{lab}'
    
    plt.plot(df_true.index, df_true[tgt] , '--', label=label_name, color='blue')
    if flag:
        for col in ls[:-1]:
            if antigo:
                mapes = mape(df_true['Hs_real_1'], df_predict[col]).round(2)
                label_name = f'{col} - {mapes}'
            else:
                mapes = mape(df_true[tgt], df_predict[col]).round(2)
                label_name = f'{col}'
            
            plt.plot(df_predict.index, df_predict[col] , '-', label=label_name)
    else:
        for col in ls[-1:]:
            if antigo:
                mapes = mape(df_true['Hs_real_1'], df_predict[col]).round(2)
            else:
                mapes = mape(df_true[tgt], df_predict[col]).round(2)
            plt.plot(df_predict.index, df_predict[col] , '-', label=f'{col} - {mapes}')
    plt.legend()
    plt.xticks(x,y, rotation=15)

def create_error_multi(df_true, df_predict,x,y, tgt):
    ls      = df_predict.columns.to_list()
    leg_dict = {'erro_real_1':'Error NOAA x Buoy', 'Hs_real_1':'Buoy - real observed value', 'Noaa_cnn-lstm':'Ensemble numerical model - NOAA'}
    lab = leg_dict[tgt]
    
    error1 = erro(df_true['Hs_real_1'], df_true[tgt]).round(2)
    plt.plot(df_true.index, error1 , '--', label=f'{lab}', color='black')
    for col in ls[-1:]:
        error2 = erro(df_true['Hs_real_1'], df_predict[col]).round(2)
        plt.plot(df_predict.index, error2 , '-', label=f'{col}')
    plt.legend()
    plt.xticks(x,y, rotation=15)

def create_error_abs(df_true, df_predict,x,y,tgt):
    ls      = df_predict.columns.to_list()
    leg_dict = {'erro_real_1':'Error NOAA x Buoy', 'Hs_real_1':'Buoy - real observed value', 'Noaa_cnn-lstm':'Ensemble numerical model - NOAA'}
    lab = leg_dict[tgt]
    
    error1 = erro_abs(df_true['Hs_real_1'], df_true[tgt]).round(2)
    plt.plot(df_true.index, error1 , '--', label=f'{lab}', color='black')
    for col in ls[-1:]:
        error2 = erro_abs(df_true['Hs_real_1'], df_predict[col]).round(2)
        plt.plot(df_predict.index, error2 , '-', label=f'{col}')
    plt.legend()
    plt.xticks(x,y, rotation=15)

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

def create_plots_error(lead, df_true, df_predict,x,y,flag):
    ls      = df_predict.columns.to_list()
    plt.title(f'Deep learning prediction of Hs - lead: {lead}')
    tick    = ['-','--','.','-.','-*']
    if flag:
        j = 0
        for col in ls[:-1]:
            mapes = mape(df_true['Hs_real'], df_predict[col]).round(2)
            errors = erro(df_true['Hs_real'], df_predict[col])
            m_erro = mean_erro(df_true['Hs_real'], df_predict[col]).round(2)
            plt.plot(df_predict.index, errors ,tick[j],label=f'{col} - {mapes} - {m_erro}',color='black')
            j += 1
    else:
        j = 0
        for col in ls[-1:]:
            mapes = mape(df_true['Hs_real'], df_predict[col]).round(2)
            errors = erro(df_true['Hs_real'], df_predict[col])
            m_erro = mean_erro(df_true['Hs_real'], df_predict[col]).round(2)
            plt.plot(df_predict.index, errors ,tick[j],label=f'Predicted {col} - {mapes} - {m_erro}',color='black')
            j += 1
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
    output           = glob.glob(dest+'*.csv')
    fold_name_report = dest.split('/')[-2]
    save_path        = f'./reports/{fold_name_report}/'
    save_path        = feat.format_path(save_path)
    create_multi_graph(output,save_path)
