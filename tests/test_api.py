from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]


def test_predict_returns_a_product():
    response = client.post(
        "/predict",
        json={
            "text": (
                "My checking account was overdrafted twice and the bank will not reverse the fees."
            ),
            "state": "TX",
            "tags": "Servicemember",
            "date_received": "2024-03-15",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["product"]
    assert body["model"]
    if body["scores"] is not None:
        assert body["product"] in body["scores"]
        assert abs(sum(body["scores"].values()) - 1.0) < 1e-3


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
