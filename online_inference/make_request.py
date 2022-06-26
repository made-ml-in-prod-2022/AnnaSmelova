import logging
import os
import sys
from typing import Tuple

import pandas as pd
import requests
from sklearn.compose import ColumnTransformer

from src.prepare_data_utils import prepare_data, make_features, PredictPipelineParams
from src.get_data_utils import get_data, get_s3_instance, get_transformer, \
    get_config_for_prediction, read_config_params


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

CONFIG_PATH = 's3_config.yaml'

s3_bucket = None
transformer = None
df = None
path_to_target = None
predict_config = None


def prepare_df(df: pd.DataFrame, transformer: ColumnTransformer, config) -> Tuple[pd.DataFrame, pd.Index]:
    df = prepare_data(df, config.prepare_params)
    df_index = df.index
    features = make_features(transformer, df)
    return features, df_index


def upload_predictions(s3_bucket: str, local_path_to_target: str, path_to_target: str) -> str:
    """Function to upload predictions to s3 yandex storage"""
    s3 = get_s3_instance()
    s3.upload_file(local_path_to_target, s3_bucket, path_to_target)
    logger.info('Predictions was uploaded to %s:%s', s3_bucket, path_to_target)

    if os.path.exists(local_path_to_target) and os.path.isfile(local_path_to_target):
        os.remove(local_path_to_target)
        logger.info('Local file with predictions deleted')

    return f'Data was loaded from {s3_bucket}/{os.getenv("PATH_TO_DATA")}. ' \
           f'Predictions was uploaded to {s3_bucket}/{path_to_target}'


def predict():
    config_params = read_config_params(CONFIG_PATH)
    s3_bucket = config_params.s3_bucket
    path_to_transformer = config_params.path_to_transformer
    path_to_data = config_params.path_to_data
    path_to_target = config_params.path_to_target
    path_to_predict_config = config_params.path_to_predict_config

    transformer = get_transformer(s3_bucket, path_to_transformer)
    df = get_data(s3_bucket, path_to_data)
    predict_config = get_config_for_prediction(s3_bucket, path_to_predict_config, PredictPipelineParams)

    logger.info('Preparing data...')
    df_prepared, df_index = prepare_df(df, transformer, predict_config)
    logger.info('df_prepared: %s', df_prepared.shape)
    df_json = df_prepared.tolist()

    response = requests.post("http://0.0.0.0:8000/predict/", json={"data": df_json})
    print(response.status_code)

    path_to_save_predictions = './predict.csv'
    pd.Series(response.json(), index=df_index, name='Predict').to_csv(path_to_save_predictions)
    logger.info('Predictions was saved to local path: %s', path_to_save_predictions)
    upload_predictions(s3_bucket, path_to_save_predictions, path_to_target)
    logger.info('Predictions was uploaded to s3')


if __name__ == "__main__":
    predict()
