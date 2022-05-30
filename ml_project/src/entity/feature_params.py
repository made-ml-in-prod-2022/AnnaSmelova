""" Feature params """

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeatureParams:
    """ Structure contain categorical and numerical params in dataset"""
    cat_features: List[str]
    num_features: List[str]
    target: Optional[str]
