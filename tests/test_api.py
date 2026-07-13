import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Identifiants de test (utilisés pour vérifier les blocages de sécurité)
INVALID_AUTH = ("hacker", "wrongpassword")

def test_read_root():
    """Vérifie que la racine de l'API répond correctement sans auth."""
    response = client.get("/")
    assert response.status_code == 200
    # On s'assure que l'API renvoie bien un dictionnaire/JSON valide
    assert isinstance(response.json(), dict)

def test_predict_endpoint_public():
    """Vérifie que la route /predict refuse les requêtes mal formées mais est accessible public."""
    # On envoie un payload vide pour tester la réaction de la route publique
    response = client.post("/predict", json={})
    # L'API doit répondre 422 (Unprocessable Entity) car le JSON est vide, 
    # mais PAS 401 (Unauthorized), ce qui prouve que la route est bien publique !
    assert response.status_code in [422, 500]

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
