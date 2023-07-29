import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
import warnings
from src.data import format_data as data_format 
from src.config.config import Config

warnings.filterwarnings('ignore')

class TFlow():
    def __init__(self, mod, features, target, dates, forecast, npredict, lead, num_features, epochs, val_size, flag, future):
        self.model           = mod
        self.df_features     = features
        self.df_target       = target
        self.forecast        = forecast
        self.npredict        = npredict
        self.lead            = lead
        self.num_features    = num_features
        self.num_epochs      = epochs
        self.dates           = dates
        self.val_size        = val_size
        self.flag            = flag
        self.pred_fut        = future

    def compile_and_fit(self, model, X, y, patience=100):
        early_stopping           = tf.keras.callbacks.EarlyStopping(
            monitor              = 'loss',
            patience             = patience,
            mode                 = 'min'
        )
        model.compile(
            loss                 = 'mean_absolute_error',
            optimizer            = 'adam',
            metrics              = ['mean_squared_error','mean_absolute_percentage_error']
        )
        history                  = model.fit(
            X,
            y,
            epochs               = self.num_epochs,
            validation_split     = self.val_size,
            callbacks            = [early_stopping],
            verbose              = 0
        )
        return history

    def prediction(self, model, data_in):
        if self.model == 'cnn-lstm':
            x_predict                    = data_in.reshape(1, 1, len(data_in), self.num_features)
        else:
            x_predict                    = data_in.reshape(1, len(data_in), self.num_features)
        
        predict                      = model.predict(x_predict)
        if self.model == 'dense':
            predict                   = predict[0,1].reshape(-1,1) 
 #       predict                      = data_format.inverse_transform(predict[0,-1],True)        

        return predict[0,-1]

    def create_future(self):
        train_input, train_target, test_input, test_target    = data_format.create_train_test(self.df_features, self.df_target.values, self.npredict, self.lead)
        x_train, y_train                                      = data_format.split_sequence(train_input, train_target, self.forecast, self.lead, self.flag)
        if self.flag:
            y_train = y_train.reshape(-1,1)
        X, y                                                  = data_format.prepare_data_lstm(x_train, y_train, self.flag, self.num_features)
        
        if self.model == 'cnn-lstm':
            X                                                 = data_format.convert_cnn_lstm(X)
        
        if self.pred_fut:
            model                                             = self.get_model(self.npredict, self.forecast)
        else:
            if self.flag:
                model                                         = self.get_model(self.lead+1, self.forecast+self.lead)
            else:
                model                                         = self.get_model(1, self.forecast)
        
        history                                               = self.compile_and_fit(model, X, y)
        val_metric                                            = history.history

        x_test, y_test                                        = data_format.split_sequence(test_input, test_target, self.forecast, self.lead, self.flag)
        if self.flag:
            y_test = y_test.reshape(-1,1)
        x_in, labels                                          = data_format.prepare_data_lstm(x_test, y_test, self.flag, self.num_features)
        
        if self.pred_fut:
            if self.model == 'cnn-lstm':
                x_in                                          = x_in.reshape(x_in.shape[0],1,self.forecast,self.num_features)
            predictions                                       = model.predict(x_in)
            if self.model == 'dense':
                predictions                                   = predictions[:,0][:,0].tolist()
            else:
                predictions                                   = predictions[:,0].tolist()
        else:
            predictions                                       = [self.prediction(model, dado) for dado in x_in]
        #labels                                                = data_format.inverse_transform(labels[:,-1], False)
        labels                                                = labels[:,-1]
        result                                                = self.create_output(predictions, labels, self.dates)
        
        return result, val_metric
    
    def create_multi_output(self):
        train_input, train_target, test_input, test_target    = data_format.create_train_test_multi(self.df_features, self.df_target, 4)
        x_train, y_train                                      = data_format.split_sequence(train_input, train_target, 1, self.lead, self.flag)

        if self.model == 'cnn-lstm':
           x_train                                            = data_format.convert_cnn_lstm(x_train)
        
        model                                                 = self.get_model(self.npredict, 1)
        history                                               = self.compile_and_fit(model, x_train, y_train)
        val_metric                                            = history.history
        
        x_test, y_test                                        = data_format.split_sequence(test_input, test_target, 1, self.lead, self.flag)
        x_in                                                  = x_test[0].reshape(1, len(x_test[0]), self.num_features)

        if self.model == 'cnn-lstm':
            x_in                                              = x_in.reshape(x_in.shape[0],1,1,self.num_features)
        predictions                                           = model.predict(x_in)
        
        if self.model == 'dense':
            predictions                                       = predictions[0][0].tolist()
        else:
            predictions                                       = predictions[0].tolist()

        result                                                = pd.DataFrame()
        result['Data']                                        = self.dates
        result['label']                                       = y_test[0].tolist()
        result['predict']                                     = predictions
        
        return result, val_metric

    def create_output(self, pred, lab, df):
        result                   = pd.DataFrame()
        if self.flag:
            result['Data']       = self.df_features.index[-self.npredict+self.forecast+5+self.lead:]
        elif self.lead == 0:
            result['Data']       = self.df_features.index[-len(pred):]
        else:
            result['Data']       = self.df_features.index[-self.npredict+self.forecast+5:]
        
        result['label']          = lab
        result['predict']        = pred
    
        return result

    def get_model(self, num_prev, shape1):
        config                   = Config()
        act_function             = config.activation
        ka                       = 1
        if self.model == 'lstm':
            modelo             = tf.keras.Sequential([
                tf.keras.layers.LSTM(512,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.LSTM(256,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.LSTM(128,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.LSTM(64,   activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),                                
                tf.keras.layers.LSTM(32,   activation=act_function, return_sequences=False),
                tf.keras.layers.Dense(num_prev, kernel_initializer=tf.initializers.zeros)])
        if self.model == 'rnn':
            modelo             = tf.keras.Sequential([
                tf.keras.layers.SimpleRNN(512,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),                
                tf.keras.layers.SimpleRNN(256,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.SimpleRNN(128,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.SimpleRNN(64,   activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.SimpleRNN(48,   activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.SimpleRNN(32,   activation=act_function, return_sequences=False),
                tf.keras.layers.Dense(num_prev, kernel_initializer=tf.initializers.zeros)])
        if self.model == 'cnn-lstm':
            modelo             = tf.keras.Sequential([
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(512, kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(256, kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(128, kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64,  kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(48,  kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32,  kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=1)),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.LSTM(512,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.LSTM(256,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.LSTM(128,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.LSTM(64,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.LSTM(48,  activation=act_function, return_sequences=True, input_shape=(shape1, self.num_features)),
                tf.keras.layers.LSTM(32,  activation=act_function, return_sequences=False),
                tf.keras.layers.Dense(num_prev, kernel_initializer=tf.initializers.zeros)])
        if self.model == 'cnn':
            modelo             = tf.keras.Sequential([
                tf.keras.layers.Conv1D(512, kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Conv1D(256, kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Conv1D(128, kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Conv1D(64,  kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Conv1D(48,  kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Conv1D(32,  kernel_size = ka, activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.MaxPooling1D(pool_size=1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(num_prev, kernel_initializer=tf.initializers.zeros)])
        if self.model == 'dense':
            modelo             = tf.keras.Sequential([
                tf.keras.layers.Dense(512,  activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Dense(256,  activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Dense(128,  activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Dense(64,   activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Dense(48,   activation=act_function, input_shape=(shape1, self.num_features)),
                tf.keras.layers.Dense(32,   activation=act_function),
                tf.keras.layers.Dense(num_prev, kernel_initializer=tf.initializers.zeros)])            

        return modelo
