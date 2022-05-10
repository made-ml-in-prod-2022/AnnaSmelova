""" Make custom transformer """

from typing import NoReturn
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CustomTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer class"""
    def __init__(self, params) -> NoReturn:
        self.num_features = params.num_features
        self.cat_features = params.cat_features

        self.num_transformer = Pipeline(steps=[
            ('minmax', MinMaxScaler())])
        self.cat_transformer = Pipeline(steps=[
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.num_transformer, self.num_features),
                ('cat', self.cat_transformer, self.cat_features)])

        self.preprocessing = Pipeline(steps=[('preprocessor', self.preprocessor)])

    def fit(self, df: pd.DataFrame):
        """ Fitting transformer"""
        self.preprocessing.fit(df)
        return self

    def transform(self, df: pd.DataFrame):
        """Transform features"""
        return self.preprocessing.transform(df)
