"""Training model pipeline"""

from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from .split_params import SplittingParams
from .prepare_params import PreparingParams
from .feature_params import FeatureParams
from .custom_transformer_params import TransformerParams
from .train_params import TrainingParams


@dataclass
class TrainingPipelineParams:
    """Structure for pipeline parameters"""
    input_data_path: str
    metric_path: str
    save_model: str
    save_transformer: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    prepare_params: PreparingParams
    train_params: TrainingParams
    custom_transformer_params: TransformerParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str):
    """Read config for model training"""
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
