Online inference модели проекта для решения задачи классификации
==============================
**Материалы к проекту (файлы):**
heart.csv
<a href='https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction'>Источник данных Kaggle</a>

Данные и модель грузятся из Yandex Object Storage. Предсказания также загружаются в хранилище. 
### Сборка docker:
    
    DOCKER_BUILDKIT=0 docker build -t annasmelova/online_inference:v1 .
    
    docker run -p 8000:8000 annasmelova/online_inference:v1

### Публикация образа в https://hub.docker.com/

    docker push annasmelova/online_inference:v1

### Загрузка из https://hub.docker.com/ и запуск сервиса в docker:

    docker pull annasmelova/online_inference:v1
    
    docker run -p 8000:8000 annasmelova/online_inference:v1

### Оптимизация размера docker image:
* Заменила базовый образ python:3.9 на более легкий python:3.9.13-slim-buster: pазмер уменьшился с 1.4GB до 630MB
* Уменьшила количество слоев в Dockerfile: особо на размер не повлияло
* Добавила --no-cache-dir: размер уменьшился до 611MB
* Также вместо того, чтобы использовать препроцессинг из проекта ml_project, сделала отдельный скрипт в online_inference с необходимым функционалом. Получилась немного копипаста, зато не надо было копировать весь ml_project в docker, чтобы импортировать оттуда функционал.
