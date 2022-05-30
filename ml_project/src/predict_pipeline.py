""" Predicting pipeline for ml model """
import logging
import sys

import click
import pandas as pd
from pandas import DataFrame

from entity.predict_pipeline_params import PredictPipelineParams, read_predict_pipeline_params
from data.make_dataset import read_raw_data, prepare_data
from features.build_features import make_features
from models.predict_model import predict_model
from models.save_load_models import load_model, load_transformer


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    """ Predict pipeline """
    predict_pipeline_params = read_predict_pipeline_params(config_path)
    return predict_pipeline_run(predict_pipeline_params)


def predict_pipeline_run(predict_pipeline_params: PredictPipelineParams):
    logger.info('Start predict pipeline...')

    df: pd.DataFrame = read_raw_data(predict_pipeline_params.input_data_path)
    df, target = prepare_data(df, predict_pipeline_params.prepare_params)

    transformer = load_transformer(predict_pipeline_params.transformer_path)

    features: DataFrame = make_features(transformer, df)

    model = load_model(predict_pipeline_params.model_path)

    logger.info('Making predict...')
    predict = predict_model(model, features)

    logger.info('Writing to file...')
    pd.Series(predict, index=df.index, name='Predict').to_csv(predict_pipeline_params.predict_path)


@click.command(name='predict_pipeline')
@click.argument('config_path')
def predict_pipeline_command(config_path: str):
    """ Make start for terminal """
    predict_pipeline(config_path)


if __name__ == '__main__':
    predict_pipeline_command()
