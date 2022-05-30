""" Predict model pipeline """

from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from .prepare_params import PreparingParams


@dataclass()
class PredictPipelineParams:
    """ Structure for pipeline parameters """
    input_data_path: str
    model_path: str
    predict_path: str
    transformer_path: str
    target_col: str
    prepare_params: PreparingParams


TrainingPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str):
    """Read config for model predicting"""
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
