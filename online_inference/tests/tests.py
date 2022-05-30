import os
import pandas as pd
from fastapi.testclient import TestClient

from app import app, get_predictions
from src.get_data_utils import get_model


DATA_FOR_TEST = pd.DataFrame(data={'Age': [40, 37], 'Sex': ['M', 'M'],
                                   'ChestPainType': ['ATA', 'ATA'], 'RestingBP': [140, 130],
                                   'Cholesterol': [289, 283], 'FastingBS': [0, 0],
                                   'RestingECG': ['Normal', 'ST'], 'MaxHR': [172, 98],
                                   'ExerciseAngina': ['N', 'N'], 'Oldpeak': [0, 0],
                                   'ST_Slope': ['Up', 'Up'], 'HeartDisease': [0, 0]})

EXPECTED_TARGET = pd.Series([0, 0], copy=False)


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


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.get('/predict')
        assert 200 == response.status_code, f'Predict endpoint test failed: {response.status_code}'
        assert 'Data was loaded' in response.json(), f'Bad response: {response.json()}'


def test_get_model():
    path_to_model = os.getenv("PATH_TO_MODEL")
    s3_bucket = os.getenv("S3_BUCKET")
    model = get_model(s3_bucket, path_to_model)
    assert not (model is None), f'Model is None'


def test_get_predictions():
    path_to_model = os.getenv("PATH_TO_MODEL")
    s3_bucket = os.getenv("S3_BUCKET")
    model = get_model(s3_bucket, path_to_model)
    local_path_to_target, predictions = get_predictions(DATA_FOR_TEST, model)
    if os.path.exists(local_path_to_target) and os.path.isfile(local_path_to_target):
        os.remove(local_path_to_target)
    assert EXPECTED_TARGET[0] == predictions[0], (
        f'Bad predictions',
        f'Expected: {EXPECTED_TARGET}',
        f'Got: {predictions}'
    )
    assert EXPECTED_TARGET[1] == predictions[1], (
        f'Bad predictions',
        f'Expected: {EXPECTED_TARGET}',
        f'Got: {predictions}'
    )
    assert predictions.shape[0] == DATA_FOR_TEST.shape[0], f'Bad predictions shape'
