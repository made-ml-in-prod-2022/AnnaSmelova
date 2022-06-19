from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable
from docker.types import Mount


PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
MODEL_PATH = Variable.get("MODEL_PATH")
PREDICTIONS_PATH = "/data/predictions/{{ ds }}"
MOUNT_SOURCE = Mount(
    source="/Users/annasmelova/Documents/MADE/ProdML/prodml_project/airflow_ml_dags/data/",
    target="/data",
    type='bind'
    )


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=pendulum.today('UTC').add(days=0),
) as dag:
    predict = DockerOperator(
        image="airflow-model-prediction",
        command=f"--input_dir {PROCESSED_DATA_PATH} --model_dir {MODEL_PATH} --preds_dir {PREDICTIONS_PATH}",
        task_id="docker-airflow-prediction",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    predict
