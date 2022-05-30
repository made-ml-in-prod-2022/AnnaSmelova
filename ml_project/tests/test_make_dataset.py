from src.data import read_raw_data, prepare_data, split_data
from src.entity import TrainingPipelineParams


def test_read_data(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    assert df.shape[1] == 12


def test_prepare_data(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    df, target = prepare_data(df, params.prepare_params)
    assert df.shape[1] == 11
    assert 'HeartDisease' not in df.columns


def test_split_data(params: TrainingPipelineParams):
    df = read_raw_data(params.input_data_path)
    df, target = prepare_data(df, params.prepare_params)
    df_train, df_test, train_target, test_target = split_data(df, target, params.splitting_params)
    test_size = params.splitting_params.test_size
    expected_test_size = int(test_size * df.shape[0])
    expected_train_size = int((1 - test_size) * df.shape[0])
    cat_features = params.feature_params.cat_features
    num_features = params.feature_params.num_features
    features_count = len(cat_features) + len(num_features)

    assert df_train.shape[0] == expected_train_size
    assert df_train.shape[1] == features_count
    assert df_test.shape[0] == expected_test_size
    assert df_test.shape[1] == features_count
    assert len(train_target) == expected_train_size
    assert len(test_target) == expected_test_size
