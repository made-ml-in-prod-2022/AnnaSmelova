import os
import numpy as np

from src.models import train_model, predict_model, load_model, save_model
from src.features import build_feature_transformer, fit_feature_transformer, make_features
from sklearn.linear_model import LogisticRegression
from src.entity import TrainingPipelineParams
from src.data import read_raw_data, prepare_data, split_data


def test_train_model(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    df, target = prepare_data(df, params.prepare_params)
    df_train, df_test, train_target, test_target = split_data(df, target, params.splitting_params)
    preprocessor = build_feature_transformer(params.feature_params)
    transformed_data = make_features(fit_feature_transformer(preprocessor, df_train), df_train)
    model = train_model(transformed_data, train_target, params.train_params)
    assert isinstance(model, LogisticRegression)


def test_save_model(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    df, target = prepare_data(df, params.prepare_params)
    df_train, df_test, train_target, test_target = split_data(df, target, params.splitting_params)
    preprocessor = build_feature_transformer(params.feature_params)
    transformed_data = make_features(fit_feature_transformer(preprocessor, df_train), df_train)
    model = train_model(transformed_data, train_target, params.train_params)
    save_model(model, params.save_model)
    assert os.path.exists(params.save_model)


def test_load_model(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    df, target = prepare_data(df, params.prepare_params)
    df_train, df_test, train_target, test_target = split_data(df, target, params.splitting_params)
    preprocessor = build_feature_transformer(params.feature_params)
    transformed_data = make_features(fit_feature_transformer(preprocessor, df_train), df_train)
    model = train_model(transformed_data, train_target, params.train_params)
    save_model(model, params.save_model)

    model = load_model(params.save_model)
    assert isinstance(model, LogisticRegression)


def test_predict_model(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    df, target = prepare_data(df, params.prepare_params)
    df_train, df_test, train_target, test_target = split_data(df, target, params.splitting_params)
    preprocessor = build_feature_transformer(params.feature_params)
    transformed_data = make_features(fit_feature_transformer(preprocessor, df_train), df_train)
    transformed_test = make_features(fit_feature_transformer(preprocessor, df_train), df_test)
    model = train_model(transformed_data, train_target, params.train_params)
    predict = predict_model(model, transformed_test)
    assert type(predict) == np.ndarray
    assert len(predict) == len(df_test)
