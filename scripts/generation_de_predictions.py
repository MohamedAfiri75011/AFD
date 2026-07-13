import requests
import pandas as pd
from sqlalchemy import create_engine
import time

# 1. Configuration de la base de données (Remplace par ton URI Supabase)
# Tu peux trouver cette URI dans ton fichier .env ou dans main.py
DATABASE_URL = "postgresql://postgres.vsusfuhifwtuxohnbmwi:Uv7K6MelZ4xMVcDS@aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require"

# URL de ton API locale
API_URL = "http://localhost:8000/predict"

try:
    print("🔌 Connexion à la base de données...")
    engine = create_engine(DATABASE_URL)
    
    # 2. Récupérer 30 lignes aléatoires de la table d'entraînement (afd)
    # ORDER BY RANDOM() garantit qu'on a un échantillon très varié à chaque fois
    query = "SELECT * FROM afd ORDER BY RANDOM() LIMIT 300;"
    df_sain = pd.read_sql(query, engine)
    
    print(f"📊 {len(df_sain)} lignes récupérées avec succès !")

    # 3. Supprimer la colonne cible (celle que le modèle est censé deviner)
    # Remplacer 'target_column' par le vrai nom de la variable à prédire (ex: montant, statut...)
    # Sinon ton API risque de renvoyer une erreur 422 car elle n'attend pas cette information
    colonne_a_predire = 'prediction_engagement_k_eur' 
    if colonne_a_predire in df_sain.columns:
        df_sain = df_sain.drop(columns=[colonne_a_predire])

    # 4. Envoi des requêtes à l'API
    success_count = 0
    print("🚀 Début de l'envoi des prédictions à l'API...")

    for index, row in df_sain.iterrows():
        # Convertir la ligne en dictionnaire pour le format JSON
        # dropna() permet d'enlever les valeurs nulles qui pourraient faire planter l'API
        payload = row.dropna().to_dict() 
        
        # Envoi de la requête POST
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            success_count += 1
            print(f"✅ Prédiction {success_count}/30 réussie")
        else:
            print(f"❌ Erreur sur la ligne {index}: {response.text}")
            
        # Petite pause pour simuler un vrai trafic utilisateur et ne pas noyer l'API
        time.sleep(0.5)

    print(f"\n🎉 Terminé ! {success_count} requêtes saines ont été enregistrées dans predict_logs.")

except Exception as e:
    print(f"❌ Une erreur est survenue : {e}")