FROM python:3.9.13-slim-buster

COPY requirements.txt app.py s3_config.yaml ./
COPY src ./src
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

ENV AWS_ACCESS_KEY_ID=YCAJEBaPYLyNows9Gp1ijDPFY \
AWS_SECRET_ACCESS_KEY=YCMCKvfDCLX4NgKzMnvaV03RmrUXdF-yAjujPuyw \
AWS_DEFAULT_REGION=ru-central1

WORKDIR .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
