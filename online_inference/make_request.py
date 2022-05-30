import requests

if __name__ == "__main__":
    response = requests.get("http://0.0.0.0:8000/predict")
    print(response.status_code)
    print(response.json())
