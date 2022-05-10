import pickle
import sys
import logging
from typing import Union, NoReturn

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

SklearnClassificationModel = Union[LogisticRegression, MLPClassifier]

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def save_model(model: SklearnClassificationModel, path: str) -> NoReturn:
    """Save model to file"""
    with open(path, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Model saved.')


def load_model(path: str):
    """Load model from file"""
    logger.info('Start loading model...')
    with open(path, 'rb') as model:
        return pickle.load(model)


def save_transformer(transformer, path: str):
    """Save transformer to file"""
    with open(path, 'wb') as file:
        pickle.dump(transformer, file, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Transformer saved.')


def load_transformer(path: str):
    """Load transformer from file"""
    logger.info('Start loading transformer...')
    with open(path, 'rb') as transformer:
        return pickle.load(transformer)
