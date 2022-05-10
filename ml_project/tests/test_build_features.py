import numpy as np
from src.data import read_raw_data, prepare_data, split_data
from src.features import build_feature_transformer, fit_feature_transformer, make_features
from src.entity import TrainingPipelineParams


def test_make_features(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    df, target = prepare_data(df, params.prepare_params)
    df_train, df_test, train_target, test_target = split_data(df, target, params.splitting_params)
    preprocessor = build_feature_transformer(params.feature_params)
    transformed_data = make_features(fit_feature_transformer(preprocessor, df_train), df_train)
    assert transformed_data.shape[1] == 26  # 4values*5cat_feats + 6num_feats
    assert np.ndarray == type(transformed_data), (
        f'Wrong data type'
        f'Result is {type(transformed_data)}'
        f'Expected type is {np.ndarray}'
    )
