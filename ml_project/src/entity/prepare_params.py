""" Preparing data  params"""

from dataclasses import dataclass, field


@dataclass
class PreparingParams:
    """ Structure contain parameters for preparing data """
    outliers_field: str = field(default='RestingBP')
    nulls_field: str = field(default='Cholesterol')
    target_field: str = field(default='HeartDisease')
