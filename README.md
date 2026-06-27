# Agence_Dvpt

Projet de prédiction des engagements de l'Aide Publique au Développement (APD).

## Fonctionnalités

- Prétraitement des données APD
- Entraînement d'un modèle Random Forest
- API FastAPI
- Endpoint d'entraînement
- Endpoint de prédiction
- Documentation Swagger

## Installation

```bash
git clone https://github.com/AFolsig/Agence_Dvpt.git
cd Agence_Dvpt
pip install -r requirements.txt

## Lancement

python -m uvicorn api.main:app --reload

## Documentation
Accéder à: 

http://127.0.0.1:8000/docs

## Endpoints
POST/train

Entraîne le modèle.

POST/predict

Retourne une prédiction des engagements APD.

## Auteurs

Augustin FAYE
Mohamed AFIRI
