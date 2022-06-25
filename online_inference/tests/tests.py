import os
import pandas as pd
from fastapi.testclient import TestClient

from app import app
from src.prepare_data_utils import prepare_data, make_features, PredictPipelineParams
from src.get_data_utils import get_model, get_transformer, \
    get_config_for_prediction, read_config_params


DATA_FOR_TEST = pd.DataFrame(data={'Age': [40, 37], 'Sex': ['M', 'M'],
                                   'ChestPainType': ['ATA', 'ATA'], 'RestingBP': [140, 130],
                                   'Cholesterol': [289, 283], 'FastingBS': [0, 0],
                                   'RestingECG': ['Normal', 'ST'], 'MaxHR': [172, 98],
                                   'ExerciseAngina': ['N', 'N'], 'Oldpeak': [0, 0],
                                   'ST_Slope': ['Up', 'Up'], 'HeartDisease': [0, 0]})

EXPECTED_TARGET = pd.Series([0, 0], copy=False)

CONFIG_PATH = 's3_config.yaml'


def test_entrypoint():
    with TestClient(app) as client:
        response = client.get('/')
        assert 200 == response.status_code, f'Entrypoint test failed: {response.status_code}'


def test_healthpoint():
    with TestClient(app) as client:
        response = client.get('/health')
        assert 200 == response.status_code, f'Healthtest test failed: {response.status_code}'
        assert response.json() is True, f'Bad response: {response.json()}'


def test_bad_endpoint():
    with TestClient(app) as client:
        response = client.get('/chtoto')
        assert 404 == response.status_code, f'Bad endpoint test failed: {response.status_code}'


def test_get_model():
    path_to_model = os.getenv("PATH_TO_MODEL")
    s3_bucket = os.getenv("S3_BUCKET")
    model = get_model(s3_bucket, path_to_model)
    assert not (model is None), f'Model is None'


def test_predict_endpoint():
    with TestClient(app) as client:
        config_params = read_config_params(CONFIG_PATH)
        s3_bucket = config_params.s3_bucket
        path_to_transformer = config_params.path_to_transformer
        path_to_predict_config = config_params.path_to_predict_config

        transformer = get_transformer(s3_bucket, path_to_transformer)
        df = DATA_FOR_TEST
        predict_config = get_config_for_prediction(s3_bucket, path_to_predict_config, PredictPipelineParams)

        df = prepare_data(df, predict_config.prepare_params)
        df_prepared = make_features(transformer, df)
        df_json = df_prepared.tolist()
        response = client.post("/predict/", json={"data": df_json})
        assert 200 == response.status_code, f'Predict endpoint test failed: {response.status_code}'
        assert len(df_prepared) == len(response.json())
