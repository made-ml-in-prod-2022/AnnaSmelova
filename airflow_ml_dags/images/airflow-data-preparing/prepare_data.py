import os

import click
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = "data.csv"
TARGET_PATH = "target.csv"
SCALED_DATA_PATH = "data_scaled.csv"


@click.command("prepare")
@click.option("--input_dir")
@click.option("--output_dir")
def prepare_data(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, DATA_PATH))
    target = pd.read_csv(os.path.join(input_dir, TARGET_PATH))
    scaler = MinMaxScaler()
    cols = data.columns
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data=data_scaled, columns=cols)
    os.makedirs(output_dir, exist_ok=True)
    data_scaled.to_csv(os.path.join(output_dir, SCALED_DATA_PATH), index=False)
    target.to_csv(os.path.join(output_dir, TARGET_PATH), index=False)


if __name__ == '__main__':
    prepare_data()
