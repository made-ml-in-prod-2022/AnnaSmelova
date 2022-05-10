import sys
import logging
from typing import Union

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


SklearnClassificationModel = Union[LogisticRegression, MLPClassifier]

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_model(df: pd.DataFrame, target: pd.Series, train_params) -> SklearnClassificationModel:
    """ Make training model """
    logger.info('Start loading %s model...', train_params.model_type)

    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif train_params.model_type == 'MLPClassifier':
        model = MLPClassifier()
    else:
        logger.exception('Model is incorrect')
        raise NotImplementedError()

    logger.info('Finished loading model')
    logger.info('Start model fitting...')
    model.fit(df, target)
    logger.info('Finished model fitting')
    return model
