import logging
import pickle
import sys

import boto3
import yaml
import pandas as pd
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from marshmallow_dataclass import class_schema


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@dataclass
class S3ConfigParams:
    """ Structure for config parameters """
    s3_bucket: str
    path_to_model: str
    path_to_transformer: str
    path_to_data: str
    path_to_target: str
    path_to_predict_config: str


ConfigParamsSchema = class_schema(S3ConfigParams)


def read_config_params(path: str):
    """Read config for model predicting"""
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = ConfigParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


def get_s3_instance():
    session = boto3.session.Session()
    return session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net'
    )


def get_model(s3_bucket: str, path_to_model: str) -> Pipeline:
    """Function to get model.pkl from s3 yandex storage"""
    s3 = get_s3_instance()
    logger.info('Loading model from %s:%s...', s3_bucket, path_to_model)
    get_object_response = s3.get_object(Bucket=s3_bucket, Key=path_to_model)
    model = pickle.load(get_object_response['Body'])
    return model


def get_transformer(s3_bucket: str, path_to_transformer: str) -> Pipeline:
    """Function to get transformer.pkl from s3 yandex storage"""
    s3 = get_s3_instance()
    logger.info('Loading transformer from %s:%s...', s3_bucket, path_to_transformer)
    get_object_response = s3.get_object(Bucket=s3_bucket, Key=path_to_transformer)
    transformer = pickle.load(get_object_response['Body'])
    return transformer


def get_data(s3_bucket: str, path_to_data: str):
    """Function to get dataframe for prediction from s3 yandex storage"""
    s3 = get_s3_instance()
    logger.info('Loading data for predict from %s:%s...', s3_bucket, path_to_data)
    get_object_response = s3.get_object(Bucket=s3_bucket, Key=path_to_data)
    df = pd.read_csv(get_object_response['Body'])
    return df


def get_config_for_prediction(s3_bucket: str, path_to_config: str, pipeline_params):
    """Function to get config for data preparing from s3 yandex storage"""
    s3 = get_s3_instance()
    logger.info('Loading config for data preparing from %s:%s...', s3_bucket, path_to_config)
    get_object_response = s3.get_object(Bucket=s3_bucket, Key=path_to_config)
    TrainingPipelineParamsSchema = class_schema(pipeline_params)
    schema = TrainingPipelineParamsSchema()
    return schema.load(yaml.safe_load(get_object_response['Body']))
