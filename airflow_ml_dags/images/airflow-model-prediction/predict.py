import os
import pickle

import click
import pandas as pd


INFERENCE_DATA_PATH = "data_scaled.csv"
MODEL_NAME = "model.pkl"
PREDS_PATH = "predictions.csv"


@click.command()
@click.option("--input_dir")
@click.option("--model_dir")
@click.option("--preds_dir")
def predict(input_dir: str, model_dir: str, preds_dir: str) -> None:
    path = os.path.join(input_dir, INFERENCE_DATA_PATH)
    data = pd.read_csv(path)

    model_path = os.path.join(model_dir, MODEL_NAME)
    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)
    predictions = model.predict(data)

    os.makedirs(preds_dir, exist_ok=True)
    pred = pd.DataFrame(predictions)
    pred_path = os.path.join(preds_dir, PREDS_PATH)
    pred.to_csv(pred_path, index=False)


if __name__ == '__main__':
    predict()
