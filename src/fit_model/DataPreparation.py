"""
Data preparation required to train the model

TODO: remove hard coded features to drop
"""

import numpy as np
import pandas as pd


def format_data(df, train = True):
    """
    :param: df : dataset to format
    :return: formatted df
    """

    # put all states that have too few observations into the other category
    # we put all states that are not within this list into other in case there are more states
    # in the test set
    df['state'] = np.where(~df['state'].isin(['"CA"', '"WA"', '"UT"']),
                                              'other',
                                              df['state'])

    # cast categorical features into dummies
    df = pd.get_dummies(df)

    if train == False:
        # columns selected to be dropped
        cols_drop_sel = ['gig_economy_weekly_income',
                         'government_aid_weekly_income',
                         'large_deposits_weekly_income',
                         'reverse_transactions',
                         'mortgage',
                         'auto_loan',
                         'student_loan',
                         'traditional_single_payment',
                         'installment_loan',
                         'predatory_installment_loan',
                         'title_loan',
                         'traditional_single_payment_mean',
                         'traditional_single_payment_trend']
        # columns not to be used
        cols_to_drop = ['Unnamed: 0','loan_amount', 'status']
        df = df.loc[:,~df.columns.isin(cols_drop_sel + cols_to_drop)]

    return df

def x_y_split(df, train=True):
    """
    :param: df : formatted dataset to split
    :return: X and y sets
    """
    df = format_data(df, train)

    # split into X and y
    X = df.loc[:,~df.columns.isin(['Unnamed: 0','loan_amount', 'status'])]
    y = df['status']

    return X, y

