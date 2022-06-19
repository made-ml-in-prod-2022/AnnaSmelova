import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "data_scaled.csv"
TARGET_PATH = "target.csv"

TRAIN_DATA_PATH = "train_data.csv"
VAL_DATA_PATH = "val_data.csv"

TRAIN_TARGET_PATH = "train_target.csv"
VAL_TARGET_PATH = "val_target.csv"

TEST_SIZE = 0.3


@click.command("split")
@click.option("--input_dir")
def split_data(input_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, DATA_PATH))
    target = pd.read_csv(os.path.join(input_dir, TARGET_PATH))
    X_train, X_val, y_train, y_val = train_test_split(
        data,
        target,
        test_size=TEST_SIZE,
        random_state=0
    )
    X_train.to_csv(os.path.join(input_dir, TRAIN_DATA_PATH), index=False)
    y_train.to_csv(os.path.join(input_dir, TRAIN_TARGET_PATH), index=False)
    X_val.to_csv(os.path.join(input_dir, VAL_DATA_PATH), index=False)
    y_val.to_csv(os.path.join(input_dir, VAL_TARGET_PATH), index=False)


if __name__ == '__main__':
    split_data()
