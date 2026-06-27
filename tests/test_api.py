import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Identifiants valides pour les tests
VALID_AUTH = ("admin_agence", "SuperMotDePasseSecurise123!")
INVALID_AUTH = ("hacker", "wrongpassword")

def test_read_root():
    """Vérifie que la racine de l'API répond correctement sans auth."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_etl_endpoint_unauthorized():
    """Vérifie que la route /etl rejette les requêtes sans authentification."""
    response = client.post("/etl")
    assert response.status_code == 401

def test_etl_endpoint_bad_credentials():
    """Vérifie que la route /etl rejette les mauvaises informations d'authentification."""
    response = client.post("/etl", auth=INVALID_AUTH)
    assert response.status_code == 401

def test_train_endpoint_unauthorized():
    """Vérifie que la route /train rejette les requêtes sans authentification."""
    response = client.post("/train", json={"n_estimators": 10, "random_state": 42})
    assert response.status_code == 401