import sys
import time
import requests
import random  # Ajout de l'import random
from concurrent.futures import ThreadPoolExecutor

URL_API = "http://localhost:8000"

def get_valid_afd_payload():
    """Renvoie un dictionnaire complet avec les 49 variables pour éviter les crashs de l'API."""
    return {
        "Agence": "Agence française de développement",
        "Nature de l'activite": "Activité nouvellement notifiée",
        "Pays beneficiaire": "Sénégal",
        "Catégorie CAD": "PMA",
        "Catégorie Banque mondiale": "WorldBank Low income countries",
        "Région": "Afrique sub-saharienne",
        "Sous-région": "Afrique de l'Ouest",
        "Canal de transfert": "Action contre la faim",
        "Canal agrege": "ONG basée dans un pays donneur",
        "Bi/Multi.1": "Bilatéral",
        "Type de flux": "Aide Publique au Développement",
        "Type de financement": "Prêt",
        "Modalites de cooperation": "Interventions de type projet",
        "Objet": "Approvisionnement en eau potable – dispositifs de base",
        "Secteur": "Eau et assainissement",
        "Genre": "1.0",
        "Aide a l'environnement": "2.0",
        "Gouvernance": "0.0",
        "Développement du commerce": "0.0",
        "Santé genesique, maternelle, neonatale et infantile (SGMNI)": "Non",
        "Reduction du risque de catastrophe": "Non",
        "Nutrition": "Non",
        "Inclusion des personnes en situation de handicap": "Oui",
        "FTC": "Non",
        "Biodiversite": "1.0",
        "Attenuation du changement climatique": "2.0",
        "Adaptation au changement climatique": "2.0",
        "Desertification": "0.0",
        "Priorité CICID": "Crises et fragilités",
        "Modalités agrégées": "Interventions de type projet",
        "Marqueurs": "adaptation,atténuation",
        "log_engagements": 7.31,
        "ODD_Pas_de_pauvrete": 1,
        "ODD_Faim_zero": 0,
        "ODD_Sante": 0,
        "ODD_Education": 0,
        "ODD_Egalite_sexes": 1,
        "ODD_Eau": 1,
        "ODD_Energie": 0,
        "ODD_Travail": 0,
        "ODD_Industrie": 0,
        "ODD_Inegalites": 0,
        "ODD_Villes": 1,
        "ODD_Consommation": 0,
        "ODD_Climat": 1,
        "ODD_Vie_aquatique": 0,
        "ODD_Vie_terrestre": 0,
        "ODD_Paix": 0,
        "ODD_Partenariats": 1,
        "Engagements (K EUR)": 1500.0
    }

def trigger_traffic_spike():
    """Simule un pic de trafic critique (Flood de requêtes valides)."""
    print("🚀 [1/3] Lancement de la simulation : Pic de Trafic (Traffic Spike)...")
    payload = get_valid_afd_payload()
    
    def send_req(_):
        try:
            requests.post(f"{URL_API}/predict", json=payload, timeout=2)
        except:
            pass

    with ThreadPoolExecutor(max_workers=15) as executor:
        executor.map(send_req, range(200))
    print("✅ Pic de trafic envoyé avec succès. Surveillez Grafana !")

def trigger_http_errors():
    """Simule un mélange d'erreurs 500 (Crash API) et 422/400 (Mauvaise requête client)."""
    print("🚨 [2/3] Lancement de la simulation : Mix d'erreurs HTTP 400 et 500...")
    
    def send_error_req(_):
        # Tirage au sort : 50% de chance d'envoyer un payload qui génère une 500, 50% pour une 422/400
        is_500_error = random.choice([True, False])
        
        try:
            if is_500_error:
                # Payload incomplet qui passe Pydantic mais fait crasher le modèle de Machine Learning
                broken_payload = {"Agence": "Agence française de développement", "Pays beneficiaire": "Sénégal"}
                requests.post(f"{URL_API}/predict", json=broken_payload, timeout=2)
            else:
                # Payload totalement invalide (mauvais type) que FastAPI va bloquer avant même le modèle
                bad_type_payload = {"Engagements (K EUR)": "Ceci n'est pas un nombre"}
                requests.post(f"{URL_API}/predict", json=bad_type_payload, timeout=2)
        except:
            pass

    # On envoie une rafale de 40 requêtes en erreur
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(send_error_req, range(40))
        
    print("✅ Rafale d'erreurs 400 et 500 envoyée. La jauge Grafana va s'affoler !")

def trigger_latency_spike():
    """Simule une surcharge provoquant une hausse des temps de réponse."""
    print("⏳ [3/3] Lancement de la simulation : Surcharge de la Latence...")
    payload = get_valid_afd_payload()
    
    def send_slow_req(_):
        try:
            requests.post(f"{URL_API}/predict", json=payload, timeout=5)
        except:
            pass

    with ThreadPoolExecutor(max_workers=30) as executor:
        executor.map(send_slow_req, range(150))
    print("✅ Vague de latence envoyée.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Spécifiez l'alerte à simuler : 'trafic', 'erreurs' ou 'latence'")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    if action == "trafic":
        trigger_traffic_spike()
    elif action == "erreurs":
        trigger_http_errors() # Appel de la nouvelle fonction
    elif action == "latence":
        trigger_latency_spike()