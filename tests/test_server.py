from time import sleep

from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_predict():
    request_response = client.post("/model", json={"inputs": "hey",})
    assert request_response.status_code == 200
