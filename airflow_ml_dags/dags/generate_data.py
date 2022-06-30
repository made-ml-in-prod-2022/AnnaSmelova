import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


GENERATED_DATA_PATH = "/data/raw/{{ ds }}"
default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "data_generator",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=pendulum.today('UTC').add(days=0),
) as dag:
    generate = DockerOperator(
        image="airflow-data-generation",
        command=f"--output_dir {GENERATED_DATA_PATH}",
        task_id="docker-airflow-generation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/annasmelova/Documents/MADE/ProdML/prodml_project/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    generate
