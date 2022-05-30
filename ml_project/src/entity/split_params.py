"""Splitting data  params"""

from dataclasses import dataclass, field


@dataclass
class SplittingParams:
    """ Structure contain parameters for splitting data """
    test_size: float = field(default=0.3)
    random_state: int = field(default=0)
