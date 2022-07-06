import logging
import os
import sys
import time
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.get_data_utils import get_model, read_config_params


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

CONFIG_PATH = 's3_config.yaml'

s3_bucket = None
model = None


class DataModel(BaseModel):
    data: List[List[float]]


class PredictResponse(BaseModel):
    heart_disease: int


app = FastAPI()

start_time = time.time()


@app.get("/")
def main():
    return "Hello, WORLD! It is entry point of our predictor."


@app.on_event("startup")
def get_config():
    time.sleep(25)
    config_params = read_config_params(CONFIG_PATH)
    os.environ['S3_BUCKET'] = config_params.s3_bucket
    os.environ['PATH_TO_MODEL'] = config_params.path_to_model


@app.on_event("startup")
def get_s3():
    global s3_bucket
    s3_bucket = os.getenv("S3_BUCKET")
    if s3_bucket is None:
        err = f"S3_BUCKET {s3_bucket} is None"
        logger.error(err)
        raise RuntimeError(err)


@app.on_event("startup")
def load_model():
    global model
    path_to_model = os.getenv("PATH_TO_MODEL")

    if path_to_model is None:
        err = f"PATH_TO_MODEL {path_to_model} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = get_model(s3_bucket, path_to_model)


@app.get("/health")
def health() -> bool:
    global start_time
    if time.time() - start_time > 120:
        raise RuntimeError('Application runtime limit exceeded')
    return not (model is None) and \
           not (s3_bucket is None)


@app.post("/predict/", response_model=List[PredictResponse])
def predict(request: DataModel) -> List[PredictResponse]:
    predictions = model.predict(np.array(request.data))
    return [PredictResponse(heart_disease=p) for p in predictions]


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
