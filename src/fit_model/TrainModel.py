"""
Trainig of final model
"""
import sys
import pickle
import importlib
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

path = str(Path().resolve())
sys.path.insert(0, path + '\\src\\fit_model\\')
import DataPreparation as data_prep

importlib.reload(data_prep)
seed = 1984

# Load data
path_data = str(Path().resolve()) + '\\src\\data\\'
df = pd.read_csv(path_data + 'approved_application_data.csv')

########### Feature selection
# split data into X and y
X, y = data_prep.x_y_split(df)

# train model to use as selector
clf = RandomForestClassifier(n_estimators=50, random_state = seed)
clf = clf.fit(X, y)

# selecting top features
selector = SelectFromModel(clf, prefit=True, threshold = 'median')
X_new = selector.transform(X)

cols_del_rf = pd.DataFrame({'col': X.columns, 'keep' : selector.get_support()})
cols_to_drop = cols_del_rf[cols_del_rf.keep == False]['col'].tolist()[:-4]
X_feat_sel = X.loc[:,~X.columns.isin(cols_to_drop)]

########## Model training
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=seed)

# Train model
rfc_gs = RandomForestClassifier(random_state=seed,
                                n_estimators = 1200,
                                min_samples_split = 10,
                                min_samples_leaf = 1,
                                max_features = 'auto',
                                max_depth = 10,
                                bootstrap = False)
rf_gs_mdl = rfc_gs.fit(X_train, y_train)

# Export model pkl