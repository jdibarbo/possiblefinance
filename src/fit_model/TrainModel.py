"""
Trainig of final model
"""
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

import DataPreparation as data_cleaning

seed = 1984

# Train test split


# Train model
rfc_gs = RandomForestClassifier(random_state=seed,
                                n_estimators = 1200,
                                min_samples_split = 10,
                                min_samples_leaf = 1,
                                max_features = 'auto',
                                max_depth = 10,
                                bootstrap = False)
rf_gs_mdl = rfc_gs.fit(X_train, y_train)