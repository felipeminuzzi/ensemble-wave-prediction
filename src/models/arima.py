import pandas as pd
import numpy as np
import warnings

from pmdarima.arima import auto_arima
from src.data import format_data as data_format 
from src.features import features as feat
warnings.filterwarnings('ignore')

class ArimaModel():
    def __init__(self, mod, features, target, dates, forecast, npredict, lead, num_features, val_size, flag, is_seazonal):
        self.model           = mod
        self.df_features     = features
        self.df_target       = target
        self.forecast        = forecast
        self.npredict        = npredict
        self.lead            = lead
        self.num_features    = num_features
        self.dates           = dates
        self.val_size        = val_size
        self.flag            = flag
        self.is_seazonal     = is_seazonal

    def search_model(self, ts):
        model                = auto_arima(ts, X=None,  max_p=10, max_d=2, max_q=10, start_P=1, D=None, 
                                            start_Q=1, max_P=5, max_D=2, max_Q=5, max_order=5, m=1,
                                            seasonal=self.is_seazonal, information_criterion='aic', alpha=0.05,
                                            test='kpss', seasonal_test='ocsb', stepwise=True, n_jobs=1 ,
                                            maxiter=100, random_state=None, n_fits=50, trace=False)
        
        return model

    def create_future(self):
        train_input, train_target, test_input, test_target    = data_format.create_train_test(self.df_target, self.df_target.values, self.npredict, self.lead)
        
        n                    = int(train_target.shape[0]*self.val_size)
        ts_train             = train_target[0:- n]
        ts_val               = train_target[- n:]
        
        model                = self.search_model(ts_train)
        val_prevs            = model.predict(n_periods=n)
        self.best_metric     = feat.mape(ts_val, val_prevs)
        x_test, y_test       = data_format.split_sequence(test_target.reshape(-1,1), test_target, self.forecast, self.lead, self.flag)

        predictions          = []
        for i in range(x_test.shape[0]):
            try:
                self.best_model  = model.fit(x_test[i])
                aux_res          = self.best_model.predict(n_periods=1)
                predictions.append(aux_res[0])
            except:
                aux_res          = predictions[-1]
                print(i)
                predictions.append(aux_res)
        result                   = self.create_output(predictions, y_test, self.dates)
        metric                   = None

        return result, metric

    def create_output(self, pred, lab, df):
        result                     = pd.DataFrame()
        result['Data']             = df[-len(lab):]
        result['label']            = lab
        result['predict']          = pred
                
        return result
