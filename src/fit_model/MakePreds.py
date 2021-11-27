"""
Makes prediction using previously trained model
"""
import pandas as pd
from pathlib import Path
import pickle
import importlib
import sys
path = str(Path().resolve())

path = str(Path().resolve())
sys.path.insert(0, path + '\\src\\fit_model\\')
import DataPreparation as data_prep

importlib.reload(data_prep)

def main():

    # Load data to predict
    path_data = str(Path().resolve().parent) + '\\src\\data\\'
    df = pd.read_csv(path_data + '/rejected_application_data.csv')

    # format data to predict
    X = data_prep.format_data(df)

    # load model
    model = pickle.load(open(path + '\\src\\fit_model\\rf.pkl', 'rb'))
    y_pred = model.predict_proba(X)
    # add predicted probabilites to df
    df['preds'] = y_pred

    return df