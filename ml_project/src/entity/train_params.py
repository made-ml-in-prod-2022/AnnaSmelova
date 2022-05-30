"""Train params"""

from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    """ Structure for train model parameters """
    model_type: str = field(default='LogisticRegression')
    random_state: int = field(default=0)
