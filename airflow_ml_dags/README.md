Airflow
==============================
### Сборка ml-base локально:

    DOCKER_BUILDKIT=0 docker build -t airflow-ml-base:latest .
  
### Изменение модели для инференса:

    AIRFLOW_VAR_MODEL_PATH=/data/models/lr/2022-06-18
  
### Для корректной работы с переменными, созданными из UI:

    export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")

### Запуск приложения:

    docker compose up --build

