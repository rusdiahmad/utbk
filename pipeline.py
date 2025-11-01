import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

def train_and_save(df, feature_cols, target_col, out_path="model/utbk_model.pkl"):
    X = df[feature_cols]
    y = df[target_col]
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, feature_cols)], remainder='drop')
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
    clf.fit(X, y)
    with open(out_path, 'wb') as f:
        pickle.dump(clf, f)
    print("Saved model to", out_path)