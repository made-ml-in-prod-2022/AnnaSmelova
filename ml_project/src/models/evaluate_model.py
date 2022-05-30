import sys
import logging
from typing import Dict

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def evaluate_model(predict: np.ndarray, target: pd.Series) -> Dict[str, float]:
    """ Make evaluate model """
    logger.info('Start calculate metrics for model...')
    acc = accuracy_score(target, predict)
    f1 = f1_score(target, predict, average='macro')
    roc = roc_auc_score(target, predict)
    logger.info('Finished calculate metrics.')
    return {
        'acc': acc,
        'f1': f1,
        'roc_auc': roc,
    }
