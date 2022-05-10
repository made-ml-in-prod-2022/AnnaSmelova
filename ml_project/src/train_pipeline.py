"""Train pipeline"""
import json
import logging
import sys

import click
import pandas as pd

from entity.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from data.make_dataset import read_raw_data, prepare_data, split_data
from features.build_features import build_feature_transformer, fit_feature_transformer, make_features
from features.custom_transformer import CustomTransformer
from models.train_model import train_model
from models.predict_model import predict_model
from models.evaluate_model import evaluate_model
from models.save_load_models import save_model, save_transformer


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    """ train pipeline """
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(config_path)

    logger.info('Start train pipeline with %s...', training_pipeline_params.train_params.model_type)

    df: pd.DataFrame = read_raw_data(training_pipeline_params.input_data_path)
    df, target = prepare_data(df, training_pipeline_params.prepare_params)

    logger.info('Splitting data to train and test...')

    df_train, df_test, train_target, test_target = split_data(df,
                                                              target,
                                                              training_pipeline_params.splitting_params
                                                              )

    logger.info('Train part shape is %s', df_train.shape)
    logger.info('Test part shape is %s', df_test.shape)

    if training_pipeline_params.custom_transformer_params.use_custom_transformer:
        transformer = CustomTransformer(training_pipeline_params.feature_params)
        transformer.fit(df_train)
    else:
        transformer = build_feature_transformer(training_pipeline_params.feature_params)
        transformer = fit_feature_transformer(transformer, df_train)

    save_transformer(transformer, training_pipeline_params.save_transformer)

    train_features = make_features(transformer, df_train)

    model = train_model(train_features, train_target, training_pipeline_params.train_params)

    test_features = make_features(transformer, df_test)

    predict = predict_model(model, test_features)

    metrics = evaluate_model(predict, test_target)

    logger.info('Metrics: %s', metrics)

    with open(training_pipeline_params.metric_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f)

    save_model(model, training_pipeline_params.save_model)
    return metrics


@click.command(name='train_pipeline')
@click.argument('config_path', default='configs/train_config_logreg.yaml')
def train_pipeline_command(config_path: str):
    """ Make start for terminal """
    train_pipeline(config_path)


if __name__ == '__main__':
    train_pipeline_command()
