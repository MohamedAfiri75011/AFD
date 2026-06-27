# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import datetime
from sqlalchemy import create_engine, text

# Configuration globale de la page
st.set_page_config(
    page_title="Dashboard MLOps — Industrialisation AFD",
    page_icon="🚀",
    layout="wide"
)

# ==============================================================================
# 1. CONFIGURATION DES ACCÈS
# ==============================================================================
SUPABASE_DB_URI = "postgresql://postgres.vsusfuhifwtuxohnbmwi:Uv7K6MelZ4xMVcDS@aws-0-eu-west-1.pooler.supabase.com:5432/postgres?sslmode=require"
engine = create_engine(SUPABASE_DB_URI)

API_URL = "http://agence_api:8000"

# Fonction de récupération dynamique avec cache
@st.cache_data(ttl=600)
def get_unique_values(column_name, table_name="afd"):
    """Récupère la liste triée des valeurs uniques d'une colonne dans Supabase."""
    try:
        query = f'SELECT DISTINCT "{column_name}" FROM public."{table_name}" WHERE "{column_name}" IS NOT NULL ORDER BY "{column_name}";'
        df = pd.read_sql(query, con=engine)
        return df[column_name].tolist()
    except Exception as e:
        st.error(f"Erreur lors du chargement de la colonne '{column_name}': {e}")
        return ["Erreur de chargement"]

# Fonction pour charger l'historique des logs (sans cache persistant pour voir le direct)
def get_prediction_logs(limit=10):
    """Récupère les dernières lignes enregistrées dans la table predict_logs."""
    try:
        query = f'SELECT created_at, agence, secteur, bi_multi_1, type_de_financement, pays_beneficiaire, prediction_engagement_k_eur FROM public.predict_logs ORDER BY created_at DESC LIMIT {limit};'
        df = pd.read_sql(query, con=engine)
        # Formatage de la date pour une lecture plus agréable
        if not df.empty and 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%d/%m/%Y %H:%M:%S')
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'historique : {e}")
        return pd.DataFrame()

# ==============================================================================
# 2. STRUCTURE EN ONGLETS POUR LE JURY
# ==============================================================================
st.title("🚀 Plateforme MLOps Industrielle — Prédiction des Engagements")
st.markdown("---")

tab_jury, tab_etl, tab_train, tab_predict = st.tabs([
    "👨‍⚖️ Soutenance / Présentation", 
    "🔄 Pipeline ETL", 
    "🧠 Entraînement & Tracking", 
    "🔮 Prévisions & Historique"
])

# ==============================================================================
# ONGLET 1 : PRÉSENTATION JURY
# ==============================================================================
with tab_jury:
    st.header("🎯 Présentation du Projet Devant le Jury")
    
    try:
        response = requests.get(f"{API_URL}/model-metadata", timeout=5)
        if response.status_code == 200 and response.json().get("status") == "success":
            metadata = response.json()
        else:
            metadata = None
    except Exception:
        metadata = None

    if metadata:
        st.markdown(f"### 🏆 Modèle en Production : `{metadata['model_name']}` (v{metadata['version']})")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Version Active", f"v{metadata['version']}")
        kpi2.metric("Score $R^2$ (Validation)", f"{metadata['metrics']['val_r2']:.3f}")
        kpi3.metric("Erreur Moyenne (MAE)", f"{metadata['metrics']['mae']:.0f} k€")
        kpi4.metric("Dernier Commit Git", f"🛠️ {metadata['git_commit']}")
        st.markdown("---")
    else:
        st.warning("⚠️ Aucun modèle marqué comme 'champion' n'a été détecté dans le registre MLflow. Lancez un entraînement pour l'initialiser.")
    
    col_intro, col_stats = st.columns([2, 1])
    
    with col_intro:
        st.subheader("Contexte Métier")
        st.write(
            "Ce projet répond à un besoin crucial d'automatisation et de pilotage des engagements financiers "
            "de l'**Agence Française de Développement (AFD)**. En nous basant sur l'historique des données, "
            "notre solution permet d'estimer instantanément l'enveloppe budgétaire requise pour un nouveau projet."
        )
        
        st.subheader("Architecture Technique (MLOps)")
        st.markdown(
            """
            * **Collecte & Stockage** : Base de données de production **Supabase (PostgreSQL)** sécurisée via Pooler IPv4.
            * **Pipeline Data** : Processus d'extraction et nettoyage (ETL) automatisé via un planificateur (**Scheduler**).
            * **Serveur API** : Conteneurisation de modèles via **FastAPI** avec validation stricte des données (**Pydantic**).
            * **Tracking & Registre** : **MLflow** pour le suivi des métriques ($R^2$, $MAE$) et la gouvernance du modèle 'Champion'.
            * **Conteneurisation** : Isolation complète de l'écosystème sous **Docker Compose**.
            """
        )

    with col_stats:
        st.subheader("Indicateurs Clés du Pipeline")
        try:
            with engine.connect() as conn:
                count_rows = conn.execute(text("SELECT COUNT(*) FROM public.afd;")).scalar()
                count_logs = conn.execute(text("SELECT COUNT(*) FROM public.predict_logs;")).scalar()
            
            st.metric("Lignes dans la base (AFD)", f"{count_rows:,}".replace(",", " "))
            st.metric("Requêtes API Loggées", f"{count_logs:,}".replace(",", " "))
        except Exception:
            st.metric("Lignes dans la base (AFD)", "Connecté")
            st.metric("Requêtes API Loggées", "Actif")

# ==============================================================================
# ONGLET 2 : SECTION ETL
# ==============================================================================
with tab_etl:
    st.header("🔄 Gestion du Pipeline Extract-Transform-Load")
    st.write("Cette section permet de piloter manuellement le pipeline de synchronisation.")
    
    if st.button("🚀 Déclencher le Pipeline ETL", key="btn_etl", type="primary"):
        try:
            with st.spinner("Exécution de l'ETL en cours..."):
                res = requests.post(f"{API_URL}/etl", timeout=30)
            if res.status_code == 200:
                st.success(f"✅ ETL exécuté avec succès à {datetime.datetime.now().strftime('%H:%M:%S')} !")
                st.json(res.json())
            else:
                st.error(f"Échec de l'ETL (Code {res.status_code})")
        except Exception as e:
            st.error(f"Erreur de communication avec l'API : {e}")

# ==============================================================================
# ONGLET 3 : SECTION TRAIN
# ==============================================================================
with tab_train:
    st.header("🧠 Entraînement Continu & Statut du Modèle")
    
    col_param, col_status = st.columns(2)
    
    with col_param:
        st.subheader("Paramètres d'entraînement")
        n_estimators = st.slider("Nombre d'estimateurs (Arbres)", 10, 50, 20, step=10)
        random_state = st.number_input("Random State (Reproductibilité)", value=42)
        
        if st.button("🏋️ Lancer le Ré-entraînement", type="primary"):
            try:
                res = requests.post(f"{API_URL}/train", json={"n_estimators": n_estimators, "random_state": random_state}, timeout=5)
                if res.status_code == 200:
                    st.success("🚀 Entraînement démarré en arrière-plan sur le serveur !")
                    st.json(res.json())
                else:
                    st.error("Erreur lors de l'initialisation de l'entraînement.")
            except Exception as e:
                st.error(f"Impossible de joindre l'API : {e}")
                
    with col_status:
        st.subheader("🔍 Vérification du Statut")
        st.write("Consultez l'état d'avancement de la tâche de fond exécutée par l'API.")
        
        if st.button("🔄 Actualiser le statut (Docs /train-status)", type="secondary"):
            try:
                res = requests.get(f"{API_URL}/train-status")
                if res.status_code == 200:
                    status_data = res.json()
                    current_status = status_data.get('status', 'unknown').upper()
                    
                    if current_status == "RUNNING":
                        st.warning(f"⏳ **Statut** : {current_status}")
                    elif current_status == "SUCCESS":
                        st.success(f"✅ **Statut** : {current_status}")
                    elif current_status == "FAILED":
                        st.error(f"❌ **Statut** : {current_status}")
                    else:
                        st.info(f"ℹ️ **Statut** : {current_status}")
                        
                    st.write(f"**Dernier lancement** : {status_data.get('last_run')}")
                    st.write(f"**Détails du serveur** : {status_data.get('details')}")
                else:
                    st.error("Impossible de récupérer le statut.")
            except Exception as e:
                st.error(f"Erreur : {e}")

# ==============================================================================
# ONGLET 4 : PREVISIONS EN DIRECT & HISTORIQUE DES LOGS
# ==============================================================================
with tab_predict:
    st.header("🔮 Simulateur d'Engagement en Temps Réel")
    
    with st.spinner("Chargement des options depuis Supabase..."):
        options_agence = get_unique_values("Agence")
        options_secteur = get_unique_values("Secteur")
        options_bi_multi = get_unique_values("Bi/Multi.1")
        options_financement = get_unique_values("Type de financement")
        options_pays = get_unique_values("Pays beneficiaire")

    col1, col2 = st.columns(2)
    with col1:
        agence = st.selectbox("Agence en charge", options=options_agence)
        secteur = st.selectbox("Secteur d'intervention", options=options_secteur)
        bi_multi_1 = st.selectbox("Type de canal (Bi/Multi.1)", options=options_bi_multi)
    with col2:
        type_de_financement = st.selectbox("Type de financement", options=options_financement)
        pays_beneficiaire = st.selectbox("Pays Bénéficiaire", options=options_pays)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Calculer la prédiction d'engagement", key="btn_predict", type="primary"):
        payload = {
            "agence": agence,
            "secteur": secteur,
            "Bi/Multi.1": bi_multi_1,
            "Type de financement": type_de_financement,
            "Pays beneficiaire": pays_beneficiaire
        }
        try:
            with st.spinner("Calcul de la prédiction..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction_engagement_k_eur")
                st.success("✅ Prédiction calculée et enregistrée dans Supabase !")
                st.metric(label="Estimation de l'Engagement Financier", value=f"{prediction:,.2f} k€".replace(",", " "))
            else:
                st.error(f"❌ Erreur Backend (Code {response.status_code})")
                st.json(response.json())
        except Exception as e:
            st.error(f"🚨 Erreur de communication : {str(e)}")

    # 🎯 NOUVELLE SECTION : Historique de la table predict_logs
    st.markdown("---")
    col_hist_title, col_hist_btn = st.columns([4, 1])
    with col_hist_title:
        st.subheader("📋 Historique des dernières requêtes (Table `predict_logs`)")
    with col_hist_btn:
        # Bouton pour forcer le rafraîchissement des données PostgreSQL
        refresh_logs = st.button("🔄 Rafraîchir les logs", type="secondary", use_container_width=True)

    # Récupération des logs
    limit_logs = st.slider("Nombre de prévisions à afficher", min_value=5, max_value=50, value=10, step=5)
    logs_df = get_prediction_logs(limit=limit_logs)

    if not logs_df.empty:
        # Renommer les colonnes pour que ce soit plus propre sur l'interface du jury
        logs_df_clean = logs_df.rename(columns={
            "created_at": "Date & Heure",
            "agence": "Agence",
            "secteur": "Secteur",
            "bi_multi_1": "Canal (Bi/Multi)",
            "type_de_financement": "Financement",
            "pays_beneficiaire": "Pays",
            "prediction_engagement_k_eur": "Prédiction (k€)"
        })
        
        # Affichage sous forme de tableau interactif Streamlit bien formaté
        st.dataframe(
            logs_df_clean, 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("ℹ️ Aucune prédiction enregistrée dans la table `predict_logs` pour le moment.")