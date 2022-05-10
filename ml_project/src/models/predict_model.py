import sys
import logging
from typing import Union

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


SklearnClassificationModel = Union[LogisticRegression, MLPClassifier]

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_model(model: SklearnClassificationModel, df: pd.DataFrame) -> np.ndarray:
    """Make predict model"""
    logger.info('Start model predict...')
    predict = model.predict(df)
    logger.info('Finished model predict')
    return predict
