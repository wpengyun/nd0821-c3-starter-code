from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_inference_1():
    data = {
        "workclass": "Private",
        "education": 11,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": 1.11,
        "native-country": "United-States",
    }
    response = client.post("/inference", json=data)
    response.status_code == 200
    assert response.json() == 0


def test_inference_2():
    data = {
        "workclass": "Private",
        "education": 11,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "native-country": "United-States",
    }
    response = client.post("/inference", json=data)
    response.status_code == 200
    assert response.json() == 0
