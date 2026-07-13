import requests, time

print('🚀 Lancement de la simulation d''anomalies...')

# 1. Générer des erreurs 404 (URL introuvable)
for _ in range(5):
    requests.get('http://localhost:8000/une-route-qui-n-existe-pas')

# 2. Générer des erreurs 401 (Mauvaise authentification sur l'ETL)
for _ in range(5):
    requests.post('http://localhost:8000/etl', auth=('pirate', 'wrong_password'))

# 3. Générer des erreurs 500 (Crash de l'inférence)
# En envoyant un dictionnaire vide ou cassé, le script 'make_prediction' va lever
# une exception, ce qui va déclencher le bloc d'erreur 500 de ton API.
payload_invalide = {'colonne_inexistante': 'boom'}
for _ in range(10):
    requests.post('http://localhost:8000/predict', json=payload_invalide)

print('✅ Simulation terminée !')