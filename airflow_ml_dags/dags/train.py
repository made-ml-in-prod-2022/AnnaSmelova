from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


GENERATED_DATA_PATH = "/data/raw/{{ ds }}"
PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
MODEL_PATH = "/data/models/lr/{{ ds }}"
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
        "train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=pendulum.today('UTC').add(days=0),
) as dag:

    prepare_data = DockerOperator(
        image="airflow-data-preparing",
        command=f"--input_dir {GENERATED_DATA_PATH} --output_dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-preparing",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    split_data = DockerOperator(
        image="airflow-data-splitting",
        command=f"--input_dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-splitting",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    train_model = DockerOperator(
        image="airflow-model-training",
        command=f"--input_dir {PROCESSED_DATA_PATH} --output_dir {MODEL_PATH}",
        task_id="docker-airflow-training",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    val_model = DockerOperator(
        image="airflow-model-validation",
        command=f"--model_dir {MODEL_PATH} --data_dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-validation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    prepare_data >> split_data >> train_model >> val_model
