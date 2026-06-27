import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Configuration des chemins et connexions
CSV_PATH = "data/raw/fichier_brut.csv"
RENAMED_CSV_PATH = "data/raw/import_reussi_fichier_brut.csv"
SUPABASE_CONN_STRING = "postgresql://postgres.vsusfuhifwtuxohnbmwi:Uv7K6MelZ4xMVcDS@aws-0-eu-west-1.pooler.supabase.com:5432/postgres?sslmode=require"
TABLE_NAME = "afd"

def clean_column_name(col: str) -> str:
    """Nettoie uniformement le nom d'une colonne en snake_case."""
    return (
        str(col).lower()
        .strip()
        .replace("é", "e")
        .replace("è", "e")
        .replace("à", "a")
        .replace("ô", "o")
        .replace("'", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(".", "_")
        .replace(",", "_")
    )

def run_etl():
    print(f"Vérification du fichier brut à l'emplacement : {os.path.abspath(CSV_PATH)}", flush=True)
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Le fichier brut est introuvable à l'emplacement '{CSV_PATH}'."
        )

    print("1. Chargement du fichier brut...", flush=True)
    try:
        df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8-sig', low_memory=False)
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier CSV : {str(e)}")
        
    print(f"Dimensions initiales : {df.shape}", flush=True)

    print("2. Nettoyage standardise des noms de colonnes...", flush=True)
    df.columns = [clean_column_name(col) for col in df.columns]

    df = df.rename(columns={
        "sante_genesique__maternelle__neonatale_et_infantile_sgmni": "sante_genesique_maternelle_neonatale_et_infantile_sgmni"
    })

    print("3. Filtrage et nettoyage de la variable cible (engagements_k_eur)...", flush=True)
    if "engagements_k_eur" in df.columns:
        if df["engagements_k_eur"].dtype == object:
            df["engagements_k_eur"] = df["engagements_k_eur"].astype(str).str.replace(',', '.', regex=False)
        df["engagements_k_eur"] = pd.to_numeric(df["engagements_k_eur"], errors="coerce")
    else:
        raise KeyError("La colonne 'engagements_k_eur' est introuvable dans le fichier d'origine.")
    
    df = df.dropna(subset=["engagements_k_eur"])
    df = df[df["engagements_k_eur"] > 0]

    print("4. Application de la transformation logarithmique (Target Engineering)...", flush=True)
    df["log_engagements"] = np.log1p(df["engagements_k_eur"])

    print("5. Calcul dynamique de la variable 'nb_odd'...", flush=True)
    odd_cols = [col for col in df.columns if col.startswith("odd_") and col != "odd_agrege"]
    if odd_cols:
        df["nb_odd"] = df[odd_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)
    else:
        df["nb_odd"] = 0

    print("6. Selection stricte de la liste des colonnes demandees...", flush=True)
    target_columns = [
        "agence", "nature_de_l_activite", "pays_beneficiaire", "categorie_cad",
        "categorie_banque_mondiale", "region", "sous_region", "canal_de_transfert",
        "canal_agrege", "bi_multi_1", "type_de_flux", "type_de_financement",
        "modalites_de_cooperation", "objet", "secteur", "genre",
        "aide_a_l_environnement", "gouvernance", "developpement_du_commerce",
        "sante_genesique_maternelle_neonatale_et_infantile_sgmni",
        "reduction_du_risque_de_catastrophe", "nutrition",
        "inclusion_des_personnes_en_situation_de_handicap", "ftc",
        "biodiversite", "attenuation_du_changement_climatique",
        "adaptation_au_changement_climatique", "desertification",
        "engagements_k_eur", "priorite_cicid", "modalites_agregees", "marqueurs",
        "log_engagements",
        "odd_pas_de_pauvrete", "odd_faim_zero", "odd_sante", "odd_education",
        "odd_egalite_sexes", "odd_eau", "odd_energie", "odd_travail",
        "odd_industrie", "odd_inegalites", "odd_villes", "odd_consommation",
        "odd_climat", "odd_vie_aquatique", "odd_vie_terrestre", "odd_paix",
        "odd_partenariats"
    ]

    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan

    df_clean = df[target_columns].copy()

    # 🚀 ÉTAPE 6.5 : Remplacement des NULL / NaN par des 0 pour les ODD et les Marqueurs numériques
    print("6.5. Traitement des valeurs manquantes pour les ODD et Marqueurs (Conversion en 0/1)...", flush=True)
    colonnes_marqueurs = [
        'genre', 'gouvernance', 'aide_a_l_environnement', 'developpement_du_commerce',
        'sante_genesique_maternelle_neonatale_et_infantile_sgmni', 'reduction_du_risque_de_catastrophe',
        'nutrition', 'inclusion_des_personnes_en_situation_de_handicap', 'ftc', 'biodiversite',
        'attenuation_du_changement_climatique', 'adaptation_au_changement_climatique', 'desertification'
    ]
    colonnes_odd = [col for col in df_clean.columns if col.startswith('odd_')]
    colonnes_a_traiter = colonnes_marqueurs + colonnes_odd

    for col in colonnes_a_traiter:
        if col in df_clean.columns:
            # force la conversion numérique (les textes ou vides deviennent NaN) puis remplace par 0
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna("Non renseigne").astype(str).str.strip()

    print(f"Dimensions de la table finale : {df_clean.shape}", flush=True)

    print(f"7. Envoi des donnees vers la table Supabase '{TABLE_NAME}' (Mode Append Sécurisé)...", flush=True)
    try:
        engine = create_engine(SUPABASE_CONN_STRING)
        
        # 1. Inspecter la table existante sur Supabase pour récupérer ses vrais noms de colonnes
        from sqlalchemy import inspect
        inspector = inspect(engine)
        vrais_noms_db = [col['name'] for col in inspector.get_columns(TABLE_NAME)]
        
        # 2. Créer dynamiquement le dictionnaire de correspondance (Nom nettoyé -> Vrai nom en base)
        mapping_colonnes = {}
        for col_db in vrais_noms_db:
            nom_nettoye = clean_column_name(col_db)
            mapping_colonnes[nom_nettoye] = col_db
            
        # Correctif manuel de sécurité pour la colonne complexe SGMNI
        mapping_colonnes["sante_genesique_maternelle_neonatale_et_infantile_sgmni"] = "Santé genesique, maternelle, neonatale et infantile (SGMNI)"
        
        # 3. Renommer temporairement les colonnes du dataframe pour coller EXACTEMENT à la DB
        df_pret_pour_db = df_clean.rename(columns=mapping_colonnes)
        
        # 4. Par sécurité, ne garder QUE les colonnes qui existent réellement dans ta table Supabase
        colonnes_communes = [col for col in df_pret_pour_db.columns if col in vrais_noms_db]
        df_pret_pour_db = df_pret_pour_db[colonnes_communes]
        
        # 5. Insertion en base avec 'append' -> L'historique est 100% préservé !
        df_pret_pour_db.to_sql(TABLE_NAME, con=engine, if_exists='append', index=False)
        print("Données ajoutées avec succès à la table afd (historique préservé) !", flush=True)
        
        # --- ACTION : Renommer le fichier uniquement en cas de succes de l'envoi ---
        print(f"8. Pipeline reussi. Renommage du fichier brut vers : {RENAMED_CSV_PATH}", flush=True)
        if os.path.exists(RENAMED_CSV_PATH):
            os.remove(RENAMED_CSV_PATH)
        os.rename(CSV_PATH, RENAMED_CSV_PATH)
        print("Fichier renomme avec succes.", flush=True)
        
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la sauvegarde sur Supabase : {str(e)}")
    
if __name__ == "__main__":
    run_etl()