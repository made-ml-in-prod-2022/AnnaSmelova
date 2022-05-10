""" Make feature for train model """


import sys
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def build_feature_transformer(params):
    num_transformer = Pipeline(steps=[
        ('minmax', MinMaxScaler())])
    cat_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, params.num_features),
            ('cat', cat_transformer, params.cat_features)])

    preprocessing = Pipeline(steps=[('preprocessor', preprocessor)])
    return preprocessing


def fit_feature_transformer(preprocessing: ColumnTransformer, df: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessor with input pd.DataFrame"""
    return preprocessing.fit(df)


def make_features(preprocessing: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """Make transform with input pd.DataFrame"""
    return preprocessing.transform(df)
