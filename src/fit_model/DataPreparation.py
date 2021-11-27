"""
Data preparation required to train the model
"""
class DataCleaningObject:
    def __init__(self, cols_to_use):

    def format_data(self, df):
        """
        :param: df : dataset to format
        :return: fitted Isolation Forest model
        """
        df['state'] = np.where(df['state'].isin(['"ID"', '"KS"', '"GA"', '"FL"']),
                                                  'other',
                                                  df['state'])

        return df

    def train_test_split(df):

        return df