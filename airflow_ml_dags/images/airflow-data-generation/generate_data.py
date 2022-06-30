import os

import click
import pandas as pd
from sklearn.datasets import make_classification


DATA_PATH = "data.csv"
TARGETS_PATH = "target.csv"


@click.command("generate")
@click.option("--output_dir")
def generate_data(output_dir: str) -> None:
    data, targets = make_classification(n_features=12, random_state=0)
    os.makedirs(output_dir, exist_ok=True)
    data = pd.DataFrame(data)
    targets = pd.DataFrame(targets, columns=["target"])
    data.to_csv(os.path.join(output_dir, DATA_PATH), index=False)
    targets.to_csv(os.path.join(output_dir, TARGETS_PATH), index=False)


if __name__ == '__main__':
    generate_data()
