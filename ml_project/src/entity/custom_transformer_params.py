""" Custom transformer """

from dataclasses import dataclass


@dataclass
class TransformerParams:
    """ Structure contain switch for custom transformer """
    use_custom_transformer: bool
