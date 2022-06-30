import os
import pickle

import click
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

VAL_DATA_PATH = "val_data.csv"
VAL_TARGET_PATH = "val_target.csv"
MODEL_PATH = 'model.pkl'
METRICS_FILEPATH = 'metrics.txt'


@click.command()
@click.option("--model_dir")
@click.option("--data_dir")
def validate(model_dir: str, data_dir: str) -> None:
    data = pd.read_csv(os.path.join(data_dir, VAL_DATA_PATH))
    target = pd.read_csv(os.path.join(data_dir, VAL_TARGET_PATH))

    with open(os.path.join(model_dir, MODEL_PATH), 'rb') as fin:
        model = pickle.load(fin)

    target = target["target"]
    predictions = model.predict(data)

    roc_auc = roc_auc_score(target, predictions)
    accuracy = accuracy_score(target, predictions)
    f1_score_value = f1_score(target, predictions)

    with open(os.path.join(model_dir, METRICS_FILEPATH), "w") as fout:
        fout.write("roc_auc: {}, accuracy: {}, f1_score {}".format(roc_auc, accuracy, f1_score_value))


if __name__ == '__main__':
    validate()
