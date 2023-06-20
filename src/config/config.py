from skopt.space import Real, Integer, Categorical
import numpy as np
import xgboost
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from numpy.random import seed
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

seed(1)

class Config():
    def __init__(self, n_calls=4, n_initial_points=3, val_size = 0.2, epochs = 5000, lag_size=7):
        self.n_calls          = n_calls
        self.n_initial_points = n_initial_points
        self.val_size         = val_size
        self.epochs           = epochs
        self.lag_size         = lag_size
        self.target           = ['Wvht']
        self.features         = ['Wspd','Wdir','Dpd']
        self.forecast         = 12
        self.predict          = 120
        self.leads            = [6,12,18,24]
        self.machine          = ['lstm','rnn','cnn-lstm','cnn','dense']
        self.n_jobs           = -1
        ##################################
        self.activation       = 'tanh'#'tanh' 
        self.var_to_error     = 'mean'
        self.use_error        = True
        self.use_era          = False
        self.use_spaced       = False     
        self.future           = False
        self.multi_target     = True
        
        self.models = {
                    'svr': {
                        'reg': SVR(kernel='rbf', gamma='scale', max_iter = 1000000), 
                        'space': [
                            Real(0.001, 0.1, name="tol"),
                            Integer(1, 1000, name="C"),
                            Real(0.001, 0.1, name="epsilon"),
                        
                        ],
                        'hyper_params' : ['tol', 'C', 'epsilon']
                    },
                    'xgb':{
                        'reg': xgboost.XGBRegressor(n_jobs=1, missing=0, random_state=0), 
                        'space': [
                            Real(0.6, 1, name="colsample_bytree"),
                            Real(0.001, 1, name="gamma"),
                            Integer(2, 6, name="max_depth"),
                            Real(1, 7, name="min_child_weight"),
                            Integer(100, 200, name="n_estimators"),
                            Real(0.6, 1, name="subsample"),
                        ],
                        'hyper_params' : ['colsample_bytree', 'gamma', 'max_depth', 'min_child_weight', 
                                    'n_estimators', 'subsample']
                    },
                    'knn' : {
                        'reg' : neighbors.KNeighborsRegressor(n_jobs = 1, algorithm = 'auto'),
                        'space' : [
                            Integer(5, 100, name = 'n_neighbors'),
                            Categorical(['uniform', 'distance'], name = 'weights'),
                            Integer(1,5, name = 'p')
                        ],
                        'hyper_params' : ['n_neighbors', 'weights', 'p']
                    },
                    'tree' : {
                        'reg' : DecisionTreeRegressor(max_features = None, min_samples_split = 100, max_depth = None),
                        'space' : [
                            Categorical(['squared_error', 'friedman_mse', 'absolute_error'], name = 'criterion'),
                            Categorical(['best', 'random'], name = 'splitter')
                        ],
                        'hyper_params' : ['criterion','splitter']
                    },
                    'mlp' : {
                        'reg' : MLPRegressor(early_stopping=True, solver = 'adam', hidden_layer_sizes = (64,48,32)),
                        'space' : [
                            Categorical(['identity', 'logistic', 'tanh', 'relu'], name = 'activation'),
                            Real(0.1, 1.0, name= 'alpha')
                        ],
                        'hyper_params' : ['activation','alpha']
                    }
                }
