"""
Data preparation required to train the model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def format_data(df):
    """
    :param: df : dataset to format
    :return: formatted df
    """
    # put all states that have too few observations into the other category
    df['state'] = np.where(df['state'].isin(['"ID"', '"KS"', '"GA"', '"FL"']),
                                              'other',
                                              df['state'])

    # cast categorical features into dummies
    df = pd.get_dummies(df)

    return df

def x_y_split(df):
    """
    :param: df : formatted dataset to split
    :return: X and y sets
    """
    df = format_data(df)

    # split into X and y
    X = df.loc[:, df.columns != 'status']
    y = df['status']

    return X, y

def train_test_split(X, y):
    """
    :param: df : formatted dataset to split
    :return: train and test datasets
    """
    # split into train and test


    return X_train, X_test, y_train, y_test
