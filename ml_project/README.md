Production ready проект для решения задачи классификации
==============================
**Материалы к проекту (файлы):**
heart.csv
<a href='https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction'>Источник данных Kaggle</a>

Установка:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Обучение модели(2 конфигурации):

    python src/train_pipeline.py configs/train_config_logreg.yaml
    python src/train_pipeline.py configs/train_config_mlpclassifier.yaml

Создание предикта(2 модели):

    python src/predict_pipeline.py configs/predict_config_logreg.yaml
    python src/predict_pipeline.py configs/predict_config_mlpclassifier.yaml

Запуск тестов:

    python -m pytest tests


Структура проекта:
------------

    ├── configs            <- Сonfiguration files
    ├── data
    │   ├── predicted      <- predicted .scv files
    │   └── raw            <- original .scv files
    │
    ├── models             <- Trained model, transformers and metrics
    │
    ├── notebooks          <- Jupyter notebooks - EDA
    │
    ├── requirements.txt   <- Requirements
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── entity         <- Scripts to read configuration file
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │   └── custom_transformer.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                    predictions
    │   ├── predict_pipeline.py <- Pipeline for prediction
    │   ├── train_pipeline.py <- Pipeline for training model
    ├── tests                <- Tests folder

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
