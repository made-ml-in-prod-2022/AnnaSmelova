import pytest
from typing import List
import numpy as np
import pandas as pd

from src.entity import (TrainingPipelineParams,
                        PredictPipelineParams,
                        FeatureParams,
                        SplittingParams,
                        PreparingParams,
                        TrainingParams,
                        TransformerParams
                        )


@pytest.fixture(scope="session")
def target():
    return "HeartDisease"


@pytest.fixture(scope="session")
def cat_features() -> List[str]:
    return ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


@pytest.fixture(scope="session")
def num_features() -> List[str]:
    return ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]


@pytest.fixture()
def params(cat_features, num_features, target, tmpdir) -> TrainingPipelineParams:
    np.random.seed(0)
    rows_number = 100
    data = pd.DataFrame()

    for col in cat_features:
        values = [0, 1, 2, 3]
        column = np.random.choice(values, rows_number)
        data[col] = column

    for col in num_features:
        column = np.random.randint(300.0, size=rows_number)
        data[col] = column

    test_filename = tmpdir.mkdir("tmpdir").join("test_data.csv")
    train_filename = tmpdir.join("tmpdir/train_data.csv")
    transformer_path = tmpdir.join("tmpdir/transformer.pkl")
    model_path = tmpdir.join("tmpdir/model.pkl")
    metric_path = tmpdir.join("tmpdir/metric.json")
    data.to_csv(test_filename, index_label=False)
    data[target] = np.random.choice([0, 1], rows_number)
    data.to_csv(train_filename, index_label=False)

    features = FeatureParams(
        cat_features=cat_features,
        num_features=num_features,
        target=target,
    )
    params = TrainingPipelineParams(
        input_data_path=train_filename,
        metric_path=metric_path,
        save_model=model_path,
        save_transformer=transformer_path,
        splitting_params=SplittingParams(test_size=0.3, random_state=0),
        feature_params=features,
        prepare_params=PreparingParams(outliers_field='RestingBP',
                                       nulls_field='Cholesterol',
                                       target_field='HeartDisease'),
        train_params=TrainingParams(model_type='LogisticRegression', random_state=0),
        custom_transformer_params=TransformerParams(use_custom_transformer=False)
    )
    return params
