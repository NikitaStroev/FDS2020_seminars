import time
import os
import dask
import dask.dataframe as dd
import pandas as pd
import time
import dask_ml
import joblib
import numpy as np
import dask_ml.xgboost

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from dask_ml.model_selection import GridSearchCV as GridSearchCV_dask
from scipy_utils import make_cluster
from dask.distributed import Client
from xgboost import XGBRegressor


class Dataframe_NYC():
    def __init__(self, url, parameters):
        self.parameters = parameters
        
    def Data_prep(): #Data preprocessing
        df = dd.read_csv(os.path.join('data', 'nycflights', '*.csv'),
                    parse_dates={'Date': [0, 1, 2]},
                    dtype={'TailNum': str,
                            'CRSElapsedTime': float,
                            'Cancelled': bool})

        df = df.query("Cancelled == False")
        df = df.drop(['TailNum','Cancelled','TaxiIn','TaxiOut'], axis=1)
        df = df.set_index('Date')
        df = df.compute()
        df = df.select_dtypes(exclude = 'object')
        df.fillna(value=0, inplace=True)
        return df

    def split_train_test(pd_data, test_size, random_state):
        X = pd_data.drop(['DepDelay'],axis=1)
        y = pd_data.DepDelay
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test 

class Dask_RandomForest():
    def __init__(self):
        pass 
    def training(grid_param, X_train, X_test, y_train, y_test):
        #Create cluster/client
        cluster = make_cluster()
        cluster
        client = Client(cluster)
        client
        #Construct Dask DataFrame
        X_train = dd.from_pandas(X_train, npartitions=4)       
        y_train = dd.from_pandas(y_train, npartitions=4)
        X_test = dd.from_pandas(X_test, npartitions=4)        
        y_test = dd.from_pandas(y_test, npartitions=4)

        estimator = RandomForestRegressor()
        #Train model
        train_time = time.time()
        grid_search = GridSearchCV_dask(estimator, grid_param, cv=2, n_jobs=-1)

        with joblib.parallel_backend("dask", scatter=[X_train, y_train]):
            grid_search.fit(X_train, y_train)
        grid_search.score(X_test, y_test)
        train_time = time.time() - train_time
        #Predictions
        acc_r2 = grid_search.best_estimator_.score(X_test, y_test)
        acc_mse = mean_squared_error(grid_search.best_estimator_.predict(X_test), y_test)
        return acc_r2, acc_mse, train_time


class Dask_XGB():
    def __init__(self):
        pass 
    def boosting(X_train, X_test, y_train, y_test):
        #Create cluster/client
        cluster = make_cluster()
        cluster
        client = Client(cluster)
        client
        #Construct Dask DataFrame
        X_train = dd.from_pandas(X_train, npartitions=4)        
        y_train = dd.from_pandas(y_train, npartitions=4)
        X_test = dd.from_pandas(X_test, npartitions=4)
        y_test = dd.from_pandas(y_test, npartitions=4)

        param = {'objective': 'reg:squarederror',
                'colsample_bytree':0.5,
                'learning_rate':0.03,
                'min_child_weight':1.5,
                'max_depth':6,
                'n_jobs':-1,
                'max_depth': 12}
        #Train model
        train_time = time.time()
        xgb_model = dask_ml.xgboost.train(client, param, X_train, y_train)
        train_time = time.time() - train_time
        #Predictions
        pred = dask_ml.xgboost.predict(client, xgb_model, X_test)
        acc_mse = mean_squared_error(pred.compute(), y_test)

        return acc_mse, train_time