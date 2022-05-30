import logging
import os
import sys
from typing import List, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline
from src.prepare_data_utils import prepare_data, make_features, PredictPipelineParams
from src.get_data_utils import get_model, get_data, get_s3_instance, get_transformer, \
    get_config_for_prediction, read_config_params


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

CONFIG_PATH = 's3_config.yaml'

s3_bucket = None
model = None
transformer = None
df = None
path_to_target = None
predict_config = None


class DataModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=12, max_items=12)]


class PredictResponse(BaseModel):
    id: float
    heart_disease: float


def prepare_df(df: pd.DataFrame, transformer: Pipeline, config):
    df = prepare_data(df, config.prepare_params)
    df_index = df.index
    features = make_features(transformer, df)
    return features, df_index


def get_predictions(df: pd.DataFrame, model: Pipeline) -> str:
    """Function to make predict and save it to local folder"""
    logger.info('Preparing data...')
    df_prepared, df_index = prepare_df(df, transformer, predict_config)
    predictions = model.predict(df_prepared)
    path_to_save_predictions = './predict.scv'
    pd.Series(predictions, index=df_index, name='Predict').to_csv(path_to_save_predictions)
    logger.info('Predictions was saved to local path: %s', path_to_save_predictions)
    return path_to_save_predictions, predictions


def upload_predictions(s3_bucket: str, local_path_to_target: str, path_to_target: str):
    """Function to upload predictions to s3 yandex storage"""
    s3 = get_s3_instance()
    s3.upload_file(local_path_to_target, s3_bucket, path_to_target)
    logger.info('Predictions was uploaded to %s:%s', s3_bucket, path_to_target)

    if os.path.exists(local_path_to_target) and os.path.isfile(local_path_to_target):
        os.remove(local_path_to_target)
        logger.info('Local file with predictions deleted')

    return f'Data was loaded from {s3_bucket}/{os.getenv("PATH_TO_DATA")}. ' \
           f'Predictions was uploaded to {s3_bucket}/{path_to_target}'


app = FastAPI()


@app.get("/")
def main():
    return "Hello, WORLD! It is entry point of our predictor."


@app.on_event("startup")
def get_config():
    config_params = read_config_params(CONFIG_PATH)
    os.environ['S3_BUCKET'] = config_params.s3_bucket
    os.environ['PATH_TO_MODEL'] = config_params.path_to_model
    os.environ['PATH_TO_TRANSFORMER'] = config_params.path_to_transformer
    os.environ['PATH_TO_DATA'] = config_params.path_to_data
    os.environ['PATH_TO_TARGET'] = config_params.path_to_target
    os.environ['PATH_TO_PREDICT_CONFIG'] = config_params.path_to_predict_config


@app.on_event("startup")
def get_s3():
    global s3_bucket
    s3_bucket = os.getenv("S3_BUCKET")
    if s3_bucket is None:
        err = f"S3_BUCKET {s3_bucket} is None"
        logger.error(err)
        raise RuntimeError(err)


@app.on_event("startup")
def load_model():
    global model
    path_to_model = os.getenv("PATH_TO_MODEL")

    if path_to_model is None:
        err = f"PATH_TO_MODEL {path_to_model} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = get_model(s3_bucket, path_to_model)


@app.on_event("startup")
def load_transformer():
    global transformer
    path_to_transformer = os.getenv("PATH_TO_TRANSFORMER")

    if path_to_transformer is None:
        err = f"PATH_TO_MODEL {path_to_transformer} is None"
        logger.error(err)
        raise RuntimeError(err)

    transformer = get_transformer(s3_bucket, path_to_transformer)


@app.on_event("startup")
def load_data():
    global df
    path_to_data = os.getenv("PATH_TO_DATA")

    if path_to_data is None:
        err = f"PATH_TO_DATA {path_to_data} is None"
        logger.error(err)
        raise RuntimeError(err)

    df = get_data(s3_bucket, path_to_data)


@app.on_event("startup")
def get_path_to_target():
    global path_to_target
    path_to_target = os.getenv("PATH_TO_TARGET")
    if path_to_target is None:
        err = f"PATH_TO_TARGET {path_to_target} is None"
        logger.error(err)
        raise RuntimeError(err)


@app.on_event("startup")
def get_predict_config():
    global predict_config
    path_to_config = os.getenv("PATH_TO_PREDICT_CONFIG")
    if path_to_config is None:
        err = f"PATH_TO_PREDICT_CONFIG {path_to_config} is None"
        logger.error(err)
        raise RuntimeError(err)

    predict_config = get_config_for_prediction(s3_bucket, path_to_config, PredictPipelineParams)


@app.get("/health")
def health() -> bool:
    return not (model is None) and \
           not (df is None) and \
           not (s3_bucket is None) and \
           not (path_to_target is None) and \
           not (transformer is None) and \
           not (predict_config is None)


@app.get("/predict")
def predict():
    local_predictions_path, _ = get_predictions(df, model)
    return upload_predictions(s3_bucket, local_predictions_path, path_to_target)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
