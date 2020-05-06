#!/usr/bin/env python
import time
import json
import argparse

import pandas as pd
import numpy as np

from load import Flights
from methods import Dask_XGB, Dask_RandomForest, Dataframe_NYC


def main():
    pass
    #Parser block
    parser = argparse.ArgumentParser(description='Args parser')
    parser.add_argument('--parameters', default='parameters.json', dest='config', type=str) #parameters_trial.json #testing set of parameters
    parser.add_argument('--test_size', default=0.3, dest='test_size', action='store', type=float)
    parser.add_argument('--results_path', default='results.csv', dest='results_path', type=str)

    args = parser.parse_args()
    test_size = args.test_size
    result_csv = str(args.results_path)
    
    with open(args.config) as json_file:
        parameters = json.load(json_file)
    data_dir = parameters['Data_folder']
    url = parameters['URL']
    number_of_rows = parameters['Number_of_rows']

    #Load and preprocessing
    Flights(data_dir, url, number_of_rows)
    pd_df = Dataframe_NYC.Data_prep()
    X_train, X_test, y_train, y_test = Dataframe_NYC.split_train_test(pd_df, test_size, random_state=42)
    print('** Preprocessing finished! **')

    # Setting output file
    result_df = pd.DataFrame(columns=['Time spent','MSE', 'R_2'], index=['DASK_RandomForest', 'DASK_ML_XGBoost'])
    
    acc_r2, acc_mse, train_time = Dask_RandomForest.training((parameters['models']['RandomForest'][0]), X_train, X_test, y_train, y_test)
    result_df.iloc[0] = train_time, acc_mse, acc_r2
    print('** DASK_ML_RandomForest finished! **')

    acc_mse, train_time = Dask_XGB.boosting(X_train, X_test, y_train, y_test)
    result_df.iloc[1] = train_time, acc_mse, 0.
    print('** DASK_ML_XGBoost finished! **')

    result_df = result_df.round(2)
    pd.DataFrame.to_csv(result_df, result_csv, sep=',')
    print('** Saving results finished! **')

if __name__ == '__main__':

    main()   