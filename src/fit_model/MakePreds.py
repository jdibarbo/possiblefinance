"""
Makes prediction using previously trained model
"""
import pandas as pd
from pathlib import Path
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

