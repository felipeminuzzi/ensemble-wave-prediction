import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.base import clone
from skopt import gp_minimize
from functools import partial

from src.features import features as feat
from src.data import format_data as data_format 

class SklearnClass():
    def __init__(self, regressor, features, ts, dates, forecast, npredict, lead, num_features, val_size, n_calls, space,  
                n_initial_points, parameters, model_name, flag):
        self.reg             = regressor
        self.df_features     = features
        self.df_target       = ts
        self.dates           = dates
        self.lead            = lead
        self.forecast        = forecast
        self.num_features    = num_features
        self.npredict        = npredict
        self.n_calls         = n_calls
        self.space           = space
        self.n_initial_points= n_initial_points
        self.parameters      = parameters
        self.val_size        = val_size
        self.model_name      = model_name
        self.flag            = flag
        
        self.format_ts_train()
        self.optmizer()

    def format_ts_train(self):
        x_train              = pd.DataFrame()
        for cols in self.df_features:
            self.input_cols  = [f'lag{i}_{cols}' for i in reversed(range(1, self.lead+1))]
            ts_lagged        = feat.create_windowing(self.df_features[cols], cols, self.lead-1).fillna(0)
            ts_lagged.columns= self.input_cols
            x_train          = pd.concat([x_train,ts_lagged], axis=1)
        train_input, train_target, test_input, test_target    = data_format.create_train_test(x_train, self.df_target.values, self.npredict, self.lead)
        
        train_input          = pd.DataFrame(train_input, columns = x_train.columns)
        train_target         = pd.DataFrame(train_target) 
        
        self.y_train         = train_target.values.reshape(-1,1)
        self.x_train         = train_input.to_numpy()

    def return_model_assessment(self, args, X_train, y_train, X_val, y_val, hyper_params):
        """Implements the evaluation of the model on the test set.

           Arguments:
                     args - values of the model's parameters
                     X_train - train set features
                     y_train - train set target
                     X_val - validation set features
                     y_val - validation set target
                     hyper_params - model's hyperparameters
           Output:
                     test_score - score of the model on the validation set
        """
        score                = feat.mape
        params               = {hyper_params[i]: args[i] for i, j in enumerate(hyper_params)}

        model                = clone(self.reg)
        model.set_params(**params)

        model.fit(X_train, y_train)
        test_predictions     = model.predict(X_val)
        test_score = score(y_val.flatten(), test_predictions)
        return test_score

    def optmizer(self):
        """ Applies Bayesian optimization in order to retrieve the optimal parameters
            of the model.
            Output: uses the best parameters to configure the model an runs it on training data
        """
        n                    = int(len(self.x_train)*self.val_size)
        x_train              = self.x_train[0:- n]
        y_train              = self.y_train[0:- n]

        x_val                = self.x_train[-n:]
        y_val                = self.y_train[-n:]

        objective_function   = partial(self.return_model_assessment, X_train=x_train,
                                     y_train=y_train, X_val=x_val,
                                     y_val=y_val, hyper_params=self.parameters)
        if self.model_name == 'tree':
            acq_opt='sampling'
        else:
            acq_opt='lbfgs'

        results              = gp_minimize(objective_function, self.space, acq_optimizer=acq_opt,
                                           base_estimator=None, n_calls=self.n_calls ,
                                           n_initial_points=self.n_initial_points,
                                           random_state=0,  n_jobs=1)
        best_params          =  dict()

        for i,par in enumerate(self.parameters):
            best_params[par] = results['x'][i]

        mape_val             = results['fun']

        #runs with the best model
        self.best_model      = clone(self.reg)
        self.best_model.set_params(**best_params)
        self.best_metric     = mape_val
        self.best_params     = best_params
        self.best_model.fit(self.x_train, self.y_train)
        
    def create_future(self):
        x_train              = pd.DataFrame()
        for cols in self.df_features:
            self.input_cols  = [f'lag{i}_{cols}' for i in reversed(range(1, self.lead+1))]
            ts_lagged        = feat.create_windowing(self.df_features[cols], cols, self.lead-1).fillna(0)
            ts_lagged.columns= self.input_cols
            x_train          = pd.concat([x_train,ts_lagged], axis=1)
        train_input, train_target, test_input, test_target    = data_format.create_train_test(x_train, self.df_target.values, self.npredict, self.lead)
        x_test, y_test                                        = data_format.split_sequence(test_input, test_target, self.forecast, self.lead, self.flag)

        predictions          = [] 
        for i in range(x_test.shape[0]):
            res              = self.best_model.predict(x_test[i])
            predictions.append(res[0])
        if self.model_name == 'knn':
            predictions = np.hstack(predictions) 
        result, metric       = self.create_output(predictions, y_test, self.dates)
        return result, metric

    def create_output(self, pred, lab, df):
        result                          = pd.DataFrame()
        result['Data']                  = df[-len(pred):]
        result['Hs_real']               = lab[-len(pred):]
        result[f'Hs_{self.model_name}'] = pred
        
        mape_model                      = feat.mape(result['Hs_real'], result[f'Hs_{self.model_name}'])

        return result, mape_model