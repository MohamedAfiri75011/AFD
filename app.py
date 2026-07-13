import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sqlalchemy import create_engine
import os

# CONFIGURATIONS & INITIALISATIONS (VARIABLES D'ENVIRONNEMENT)
st.set_page_config(
    page_title="Plateforme MLOps - Agence (Suivi AFD)",
    layout="wide"
)

# Chargement dynamique depuis le fichier .env via Docker
API_URL = os.getenv("API_URL", "http://fastapi_server:8000")
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI", "postgresql://postgres.vsusfuhifwtuxohnbmwi:Uv7K6MelZ4xMVcDS@aws-0-eu-west-1.pooler.supabase.com:5432/postgres?sslmode=require")

ADMIN_USER = os.getenv("ADMIN_USER", "admin_agence")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "AFD2026!")
AUTH_CREDENTIALS = (ADMIN_USER, ADMIN_PASSWORD)

# Moteur SQL de secours pour calculer les medianes/modes
@st.cache_resource
def get_db_engine():
    return create_engine(SUPABASE_DB_URI)

@st.cache_data
def load_json(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Impossible de charger {filename} : {e}")
        return {}

@st.cache_data
def load_data():
    """Charge un echantillon pour les fallbacks de donnees numeriques"""
    try:
        engine = get_db_engine()
        return pd.read_sql_query('SELECT * FROM public.afd LIMIT 200;', con=engine)
    except Exception:
        return pd.DataFrame()

# BARRE LATERALE DE NAVIGATION
st.sidebar.title("Navigation MLOps")
page = st.sidebar.radio(
    "Selectionnez une page :", 
    [
        "Prédictions", 
        "Drift", 
        "Re-entrainement Modele",
        "Tableau de bord MLOps",
        "Architecture MLOps",
        "Données & Supabase",
        "Améliorations futures"
    ]
)

# Affichage rapide de l'etat de l'API dans la sidebar
try:
    api_health = requests.get(f"{API_URL}/").json()
    status_api = api_health.get('status', 'online')
    st.sidebar.success(f"API : {status_api.upper()}")
    if api_health.get('training_status') == "busy":
        st.sidebar.warning("Entrainement actif")
except Exception:
    st.sidebar.error("API : Deconnectee")

# AFFICHAGE DU MODELE CHAMPION DANS LA SIDEBAR
st.sidebar.markdown("---")
st.sidebar.subheader("Modele Champion Actif")
try:
    meta_res = requests.get(f"{API_URL}/model-metadata", timeout=3).json()
    version = meta_res.get('model_version', meta_res.get('version', 'Non definie'))
    r2_score = meta_res.get('r2', 'N/A')
    rmse_score = meta_res.get('rmse', 'N/A')
    
    if isinstance(r2_score, float): r2_score = f"{r2_score:.3f}"
    if isinstance(rmse_score, float): rmse_score = f"{rmse_score:.3f}"

    st.sidebar.info(f"Version : {version}")
    st.sidebar.write(f"**Score R2 :** {r2_score}")
    st.sidebar.write(f"**RMSE :** {rmse_score}")
except Exception:
    st.sidebar.warning("Metadonnees du modele indisponibles")

# =====================================================================
# 1. PAGE : Prédictions
# =====================================================================
if page == "Prédictions":
    st.title("Estimation des Engagements Financiers")
    st.markdown("Renseignez les caracteristiques d'un projet pour obtenir une estimation du montant engage (en K Euros).")
    st.markdown("---")

    meta = load_json("meta.json")
    dropdowns = load_json("dropdowns.json")
    df = load_data()

    with st.form("prediction_form"):
        st.subheader("Caracteristiques du projet")

        col1, col2, col3 = st.columns(3)
        inputs = {}

        key_cats = ['Agence', 'Type de financement', 'Pays beneficiaire', 'Region',
                    'Secteur', 'Categorie CAD', 'Bi/Multi.1']
        
        with col1:
            for c in key_cats[:3]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)
        with col2:
            for c in key_cats[3:6]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)
        with col3:
            for c in key_cats[6:]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)

        with st.expander("+ Parametres avances (optionnel)", expanded=False):
            c1, c2, c3 = st.columns(3)
            cat_low = meta.get('cat_low', [])
            cat_high = meta.get('cat_high', [])
            advanced_cats = [c for c in cat_low + cat_high if c not in key_cats and c in dropdowns]
            
            for i, c in enumerate(advanced_cats[:12]):
                with [c1, c2, c3][i % 3]:
                    inputs[c] = st.selectbox(c, ["(par defaut)"] + dropdowns.get(c, []), index=0)

        submitted = st.form_submit_button("Estimer le montant d'engagement", type="primary", use_container_width=True)

    if submitted:
        row = {}
        feature_cols = meta.get('feature_cols', [])
        num_cols = meta.get('num_cols', [])
        
        for c in feature_cols:
            if c in inputs and inputs[c] != "(par defaut)":
                row[c] = inputs[c]
            elif c in num_cols:
                if not df.empty and c in df.columns:
                    median_val = pd.to_numeric(df[c], errors='coerce').median()
                    row[c] = float(median_val) if pd.notna(median_val) else 0.0
                else:
                    row[c] = 0.0
            else:
                row[c] = df[c].mode().iloc[0] if not df.empty and c in df.columns else "Unknown"

        try:
            with st.spinner("Calcul en cours par l'API Championne..."):
                response = requests.post(f"{API_URL}/predict", json=row)
            
            if response.status_code == 200:
                result = response.json()
                montant_keur = result.get('prediction', 0.0)
                montant_eur = montant_keur * 1000
                log_pred = np.log1p(montant_keur)

                st.markdown("---")
                st.subheader("Resultat de l'estimation de l'engagement")
                c1, c2, c3 = st.columns(3)
                c1.metric("log(1 + engagement)", f"{log_pred:.2f}")
                
                if montant_keur < 1000:
                    c2.metric("Montant estime", f"{montant_keur:,.1f} K EUR")
                else:
                    c2.metric("Montant estime", f"{montant_keur/1000:,.2f} M EUR")
                c3.metric("En euros", f"{montant_eur:,.0f} EUR")
            else:
                st.error(f"Erreur renvoyee par l'API ({response.status_code}) : {response.text}")
        except Exception as e:
            st.error(f"Echec de la communication avec l'API FastAPI : {str(e)}")

    # ---------------------------------------------------------------------
    # SECTION : Affichage de la table predict_logs (Toutes les colonnes)
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Historique complet de la table predict_logs")
    
    try:
        history_response = requests.get(f"{API_URL}/prediction-history?limit=15", timeout=10)
        
        if history_response.status_code == 200:
            history_data = history_response.json()
            
            if len(history_data) == 0:
                st.info("Aucune prediction enregistree pour le moment dans Supabase.")
            else:
                # Pandas transforme automatiquement le JSON complet en un tableau contenant toutes les colonnes
                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, use_container_width=True)
        else:
            st.error("Impossible de recuperer la table predict_logs depuis l'API.")
    except Exception as e:
        st.error(f"Erreur lors de la recuperation des logs de prediction : {str(e)}")

# =====================================================================
# 2. PAGE : Drift
# =====================================================================
elif page == "Drift":
    st.title("Surveillance de la qualité des données (Evidently AI)")
    st.write("Comparez les données reçues aujourd'hui avec vos données historiques pour vérifier si le comportement des utilisateurs a changé.")
    st.markdown("---")
    
    if st.button("Lancer le diagnostic de santé du modèle", use_container_width=True, type="primary"):
        try:
            with st.spinner("Analyse des distributions et calcul des distances statistiques..."):
                res = requests.post(f"{API_URL}/drift", auth=AUTH_CREDENTIALS)
            
            if res.status_code == 200:
                drift_res = res.json()
                if drift_res.get("status") == "success":
                    st.success("Analyse completée avec succès !")
                    global_drift = drift_res.get("global_drift", 0.0)
                    
                    # Affichage direct de la métrique sans division en colonnes
                    st.metric("Score de Drift Global", f"{global_drift * 100:.2f} %")
                    
                    if global_drift > 0.05: # Le seuil de 5% pour ta démo
                        st.error("ALERT : Le seuil critique a ete franchi ! Réentrainement lancé.")
                    else:
                        st.success("Stabilite confirmée.")
        except Exception as e:
            st.error(f"Erreur de communication : {e}")

# =====================================================================
# 3. PAGE : RE-ENTRAINEMENT
# =====================================================================
elif page == "Re-entrainement Modele":
    st.title("Pipeline de Re-entrainement Asynchrone")
    st.markdown("---")
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        n_estimators = st.slider("Nombre d'arbres (n_estimators)", min_value=10, max_value=300, value=100, step=10)
    with col_h2:
        random_state = st.number_input("Graine Aleatoire (random_state)", value=42, step=1)

    if st.button("Declencher la mise a jour globale du modele", use_container_width=True, type="primary"):
        try:
            res = requests.post(f"{API_URL}/train", json={"n_estimators": n_estimators, "random_state": random_state}, auth=AUTH_CREDENTIALS)
            if res.status_code == 200:
                st.warning("Re-entrainement en cours en tache de fond.")
        except Exception as e:
            st.error(f"Erreur reseau : {str(e)}")

# =====================================================================
# 4. PAGE : TABLEAU DE BORD MLOPS
# =====================================================================
elif page == "Tableau de bord MLOps":

   st.title("🏠 Tableau de bord MLOps")
   st.header("Projet APD - Aide Publique au Développement")

   st.markdown("""
   Bienvenue sur le tableau de bord du projet MLOps.

   Cette interface permet de suivre l'ensemble de la chaîne de traitement,
   depuis la collecte des données APD jusqu'à la prédiction finale via l'API FastAPI
   et l'interface Streamlit.
   """)

   st.divider()

   st.subheader("📊 Indicateurs principaux")

   col1, col2, col3, col4 = st.columns(4)

   with col1:
       st.metric(
           label="📁 Données brutes",
           value="106 519",
           delta="103 colonnes"
       )

   with col2:
       st.metric(
           label="🧹 Données nettoyées",
           value="72 835",
           delta="35 variables"
       )

   with col3:
       st.metric(
           label="🤖 Modèle",
           value="Random Forest",
           delta="Champion MLflow"
       )

   with col4:
       st.metric(
           label="⚙️ Pipeline",
           value="100 %",
           delta="Opérationnel"
       )

   st.success("""
   ✅ **État du pipeline**

   Toutes les composantes principales du projet MLOps sont opérationnelles :
   collecte, stockage, prétraitement, entraînement, tracking MLflow, API FastAPI
   et interface Streamlit.
   """)

   st.divider()

   st.subheader("🔁 Chaîne MLOps du projet")

   col1, col2 = st.columns(2)

   with col1:
       st.info("""
       ### 📥 Données

       - Collecte des données APD
       - Nettoyage
       - Prétraitement
       - Stockage dans Supabase
       """)

   with col2:
       st.info("""
       ### 🤖 Machine Learning

       - Modèle Random Forest
       - Prédiction des engagements en K EUR
       - Suivi des expériences avec MLflow
       - Modèle Champion enregistré
       """)

   col3, col4 = st.columns(2)

   with col3:
       st.info("""
       ### 🚀 Déploiement

       - API FastAPI sécurisée
       - Endpoint `/predict`
       - Appel du modèle Champion
       - Temps de réponse affiché
       """)

   with col4:
       st.info("""
       ### 🖥 Interface utilisateur

       - Application Streamlit
       - Formulaire de prédiction
       - Dashboard MLOps
       - Visualisation du pipeline
       """)

   st.divider()

   st.subheader("🧭 Vue synthétique du pipeline")

   st.code("""
📁 Données APD
       │
       ▼
🗄 Supabase
       │
       ▼
⚙ Prétraitement
       │
       ▼
🤖 Random Forest
       │
       ▼
📊 MLflow
       │
       ▼
🏆 Champion
       │
       ▼
🚀 FastAPI
       │
       ▼
🖥 Streamlit
   """)

   st.success("""
   ✅ Le modèle actuellement utilisé par l'application est l'alias **Champion**
   enregistré dans **MLflow Model Registry**.
   """)

   st.divider()

   st.subheader("📌 Informations du projet")

   col1, col2, col3 = st.columns(3)

   with col1:
       st.markdown("""
       **👥 Réalisé par**

       Augustin FAYE  
       Mohamed AFIRI
       """)

   with col2:
       st.markdown("""
       **🎓 Formation**

       Machine Learning Engineer / MLOps  
       École Liora *(ex DataScientest)*
       """)

   with col3:
       st.markdown("""
       **📅 Version**

       Juillet 2026  
       Projet fil rouge MLOps
       """)

# =====================================================================
# 5. PAGE : ARCHITECTURE MLOPS
# =====================================================================
elif page == "Architecture MLOps":

   st.title("🏗️ Architecture MLOps complète")

   st.markdown("""
Cette application suit une architecture **MLOps moderne** permettant
d'automatiser toute la chaîne de traitement des données APD.

Chaque composant possède un rôle précis depuis la collecte des données
jusqu'à la prédiction finale.
""")

   st.subheader("📊 Chiffres clés du projet")

   m1, m2, m3 = st.columns(3)
   m1.metric("Lignes", "72 835")
   m2.metric("Colonnes", "35")
   m3.metric("Base", "PostgreSQL")

   m4, m5, m6 = st.columns(3)
   m4.metric("Modèle", "Random Forest")
   m5.metric("Registry", "Champion")
   m6.metric("API", "FastAPI")

   st.caption("Ces indicateurs donnent une vue rapide du projet MLOps.")
   st.divider()

   st.subheader("🧩 Schéma d'architecture MLOps")

   st.markdown("""
   Ce schéma présente les interactions principales entre les données, le modèle,
   MLflow, l’API FastAPI et l’interface Streamlit.
   """)

   mermaid_code = """
   flowchart TB

       A["📂 Données APD<br/>106 519 lignes<br/>103 colonnes"]
       B["📦 Collecte automatique<br/>Batch APD"]
       C["🗄️ Supabase<br/>PostgreSQL"]
       D["⚙️ Prétraitement<br/>Nettoyage + encodage<br/>72 835 lignes / 35 variables"]
       E["🤖 Entraînement<br/>Random Forest Regressor"]
       F["📈 MLflow<br/>Tracking des runs<br/>métriques + artefacts"]
       G["🏆 Model Registry<br/>Alias Champion"]
       H["🌐 FastAPI<br/>API sécurisée<br/>/predict"]
       I["📊 Streamlit<br/>Interface utilisateur"]
       J["👤 Utilisateur<br/>Prédiction finale"]

       A --> B --> C --> D --> E --> F --> G --> H --> I --> J

       classDef data fill:#EAF3FF,stroke:#1F4E79,stroke-width:2px,color:#0B2545;
       classDef ml fill:#EAF7EA,stroke:#2E7D32,stroke-width:2px,color:#103B16;
       classDef tracking fill:#FFF3D6,stroke:#B7791F,stroke-width:2px,color:#4A2C00;
       classDef deploy fill:#F1EAFE,stroke:#6B46C1,stroke-width:2px,color:#2D1B69;
       classDef user fill:#FFFFFF,stroke:#555555,stroke-width:2px,color:#111111;

       class A,B,C data;
       class D,E ml;
       class F,G tracking;
       class H,I deploy;
       class J user;
   """

   components.html(
       f"""
       <div class="mermaid">
       {mermaid_code}
       </div>

       <script type="module">
           import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
           mermaid.initialize({{
               startOnLoad: true,
               theme: "base",
               flowchart: {{
                   curve: "basis",
                   nodeSpacing: 45,
                   rankSpacing: 65
               }},
               themeVariables: {{
                   fontFamily: "Arial",
                   fontSize: "18px",
                   primaryBorderColor: "#1F4E79",
                   lineColor: "#333333"
               }}
           }});
       </script>
       """,
       height=850,
       scrolling=True
   )

   st.success(
       "✅ Le modèle utilisé par l'API est toujours l'alias **Champion** enregistré dans **MLflow Model Registry**."
   )
     
         
   st.divider()

   st.subheader("🌐 Endpoints FastAPI")

   st.markdown("""
   Les principaux endpoints exposés par l'API sont présentés ci-dessous.
   """)

   st.table({
       "Endpoint": [
           "/",
           "/health",
           "/collect",
           "/train",
           "/predict",
           "/pipeline",
           "/metrics",
           "/data-stats",
           "/prediction-history",
           "/docs",
       ],
       "Description": [
           "Page d'accueil de l'API",
           "Vérifie que l'API est disponible",
           "Collecte automatique des données APD",
           "Entraîne le modèle Random Forest",
           "Prédit les engagements (K EUR)",
           "Exécute un scénario complet de prédiction",
           "Récupère les métriques du modèle Champion",
           "Récupère les statistiques des données Supabase",
           "Affiche l'historique des prédictions enregistrées",
           "Documentation interactive Swagger UI",
       ],
   })

   st.divider()

   st.markdown("---")

   st.subheader("🛠️ Technologies utilisées")

   col1, col2, col3, col4 = st.columns(4)

   with col1:
       st.markdown("### 🐍 Python")
       st.caption("Langage principal")

   with col2:
       st.markdown("### ⚡ FastAPI")
       st.caption("API REST")

   with col3:
       st.markdown("### 📊 Streamlit")
       st.caption("Interface")

   with col4:
       st.markdown("### 📈 MLflow")
       st.caption("Tracking")

   col1, col2, col3, col4 = st.columns(4)

   with col1:
       st.markdown("### 🗄️ Supabase")
       st.caption("PostgreSQL")

   with col2:
       st.markdown("### 🌲 Scikit-Learn")
       st.caption("Machine Learning")

   with col3:
       st.markdown("### 🐳 Docker")
       st.caption("Déploiement")

   with col4:
       st.markdown("### 🐙 GitHub")
       st.caption("Versioning")

   st.divider()

   st.header("📦 Description des composants")

   col1, col2 = st.columns(2)

   with col1:

       st.info("""
   ### 📦 Collect
   - récupération automatique des données APD
   - exécution par batch
   - alimentation automatique de Supabase
   """)

       st.info("""
   ### 🗄️ Supabase
   - Base PostgreSQL
   - stockage centralisé
   - historique des données
   - source unique pour l'entraînement
   """)

       st.info("""
   ### ⚙️ Prétraitement
   - nettoyage
   - suppression des valeurs inutiles
   - encodage
   - préparation du DataFrame
   """)

   with col2:

       st.info("""
   ### 🤖 Machine Learning
   - Random Forest Regressor
   - prédiction des engagements (K EUR)
   - pipeline Scikit-Learn
   """)

       st.info("""
   ### 📈 MLflow
   - suivi des expériences
   - métriques
   - artefacts
   - Model Registry
   - alias Champion
   """)

       st.info("""
   ### 🌐 FastAPI
   - API sécurisée (x-api-key)
   - Collecte des données
   - Entraînement du modèle
   - Prédiction
   - Métriques et statistiques
   - Historique des prédictions
   - Documentation Swagger (`/docs`)
   """)

   st.markdown("---")

   st.success("""
   ## ✅ Pipeline entièrement automatisé

   **Collect → Supabase → Prétraitement → Random Forest → MLflow → Champion → FastAPI → Streamlit**

   L'ensemble de la chaîne MLOps est automatisé, depuis la collecte des données APD jusqu'à la prédiction finale via l'interface utilisateur. 
   """)
	
   st.info(
       "🚀 Projet MLOps APD : pipeline de Machine Learning automatisé avec suivi MLflow et déploiement FastAPI/Streamlit."
   )
# =====================================================================
# 6. PAGE : DONNEES & SUPABASE
# =====================================================================
elif page == "Données & Supabase":

   st.title("🗄Gestion des données & Supabase")

   st.markdown("""
   Cette page présente la gestion des données du projet APD : collecte, nettoyage,
   transformation, stockage dans Supabase puis utilisation pour l'entraînement du modèle.
   """)

   st.markdown("## 🟢 État de la base de données")

   col1, col2, col3 = st.columns(3)

   with col1:
       st.metric(
           label="🗄️ Base",
           value="PostgreSQL",
           delta="Supabase"
       )

   with col2:
       st.metric(
           label="📋 Table principale",
           value="donnees",
           delta="Disponible"
       )

   with col3:
       st.metric(
           label="✅ Statut",
           value="Opérationnelle",
           delta="Prête ML"
       )

   st.success("""
   ✅ **Base PostgreSQL opérationnelle**  
   ✅ **Données APD disponibles**  
   ✅ **Table principale :** `donnees`  
   ✅ **Données nettoyées et transformées**  
   ✅ **Source unique utilisée pour l'entraînement du modèle**
   """)

   st.divider()

   st.subheader("🔁 Pipeline de traitement des données")

   st.caption(
       "Ce pipeline synthétise le parcours complet des données, de la collecte jusqu'à la prédiction utilisateur."
   )

   pipeline_dot = """
   digraph {
       rankdir=TB;
       graph [bgcolor="transparent"];
       node [
           shape=box,
           style="rounded,filled",
           fontname="Helvetica",
           fontsize=16,
           margin="0.35,0.25",
           width=3.6,
           height=1.0
       ];
       edge [
           color="#111827",
           arrowsize=0.9
       ];

       A [label="📁 Données APD\\n106 519 lignes\\n103 colonnes", fillcolor="#EAF2FF"];
       B [label="⚙️ Préparation des données\\nCollecte + nettoyage + transformation", fillcolor="#EAF7EA"];
       C [label="🗄️ Supabase\\nBase PostgreSQL centrale", fillcolor="#FFF7D6"];
       D [label="🤖 Modélisation\\nRandom Forest Regressor", fillcolor="#F3E8FF"];
       E [label="📈 MLflow\\nTracking + Model Registry\\nAlias Champion", fillcolor="#FDEBD0"];
       F [label="🚀 FastAPI + Streamlit\\nAPI /predict + interface utilisateur", fillcolor="#E0F2FE"];
       G [label="👤 Utilisateur\\nPrédiction finale", fillcolor="#FFFFFF"];

       A -> B -> C -> D -> E -> F -> G;
   }
   """

   st.graphviz_chart(pipeline_dot, use_container_width=True)

   st.success(
       "✅ Les données suivent un pipeline complet : préparation, stockage PostgreSQL, entraînement, suivi MLflow, déploiement API et prédiction via Streamlit."
   )

  
          
   st.divider()

   st.subheader("🗄️ Rôle de Supabase")

   col1, col2 = st.columns(2)

   with col1:
       st.markdown("""
       ### Base PostgreSQL

       - Stockage centralisé des données APD
       - Historique des données nettoyées
       - Table principale : `donnees`
       - Source utilisée pour l'entraînement
       """)

   with col2:
       st.markdown("""
       ### Utilité MLOps

       - Remplace le fichier CSV local
       - Facilite l'automatisation
       - Rend le pipeline plus reproductible
       - Prépare le projet à un usage plus professionnel
       """)

   st.divider()

   st.subheader("⚙️ Rôle des batchs")

   st.markdown("""
   Les batchs permettent d'automatiser le traitement des données.

   Ils servent à :

   - récupérer les données APD ;
   - nettoyer et transformer les données ;
   - insérer les données dans Supabase ;
   - fournir une base propre pour l'entraînement du modèle.
   """)
   st.markdown("#### 🔄 Pipeline automatisé des batches")
   st.caption("Les traitements sont exécutés automatiquement avant l'entraînement du modèle.")
   col1, col2, col3 = st.columns([1,2,1])
   with col2:
       st.graphviz_chart("""
       digraph {
           rankdir=TB;
           node [shape=box, style="rounded,filled", fontname="Arial", fontsize=13];

           A [label="📥 Collecte\\nDonnées APD", fillcolor="#E8EEF7"];
           B [label="🧹 Nettoyage\\nValeurs inutiles", fillcolor="#EAF7EA"];
           C [label="⚙️ Transformation\\nEncodage + sélection", fillcolor="#EAF7EA"];
           D [label="🗄️ Supabase\\nPostgreSQL", fillcolor="#FFF4D6"];
           E [label="🤖 Random Forest\\nEntraînement", fillcolor="#EDE7F6"];

           A -> B -> C -> D -> E;
       }
       """)

   st.divider()

   st.subheader("✅ Qualité des données")

   col1, col2 = st.columns(2)

   with col1:
       st.markdown("""
       ### Nettoyage effectué

       - Nettoyage des données
       - Traitement des valeurs manquantes
       - Sélection des variables utiles
       - Encodage des variables catégorielles
       """)

   with col2:
       st.markdown("""
       ### Données prêtes pour le ML

       - Dataset de 72 835 lignes
       - 35 variables conservées
       - Cible définie : `Engagements (K EUR)`
       - Compatible avec le pipeline Scikit-Learn
       """)

   st.success("""
   ✅ Les données APD ont été nettoyées, transformées et préparées
   avant leur utilisation par le pipeline d'entraînement Random Forest.
   """)

# =====================================================================
# 7. PAGE : AMELIORATIONS FUTURES
# =====================================================================
elif page == "Améliorations futures":

   st.title("🚀 Améliorations futures du projet")

   st.markdown("""
   Cette page présente les évolutions qui auraient pu être ajoutées avec plus de temps
   afin de rapprocher le projet d'une architecture MLOps professionnelle complète.
   """)

   st.divider()

   st.header("⏱️ Orchestration avec Airflow")

   st.markdown("""
   **Airflow** permettrait d'automatiser toute la chaîne de traitement :

   - collecte automatique des données APD ;
   - chargement dans Supabase ;
   - préparation des données ;
   - entraînement du modèle ;
   - enregistrement des métriques dans MLflow ;
   - promotion automatique du meilleur modèle.
   """)

   st.success("Objectif : remplacer les lancements manuels par un pipeline planifié et reproductible.")

   st.divider()

   st.markdown("## 📈 Évolutions du monitoring")

   st.write("""
   Le système de monitoring basé sur Evidently détecte automatiquement les dérives des données. 
   Lorsqu'un seuil de dérive est dépassé, un réentraînement du modèle peut être déclenché 
   afin de restaurer ses performances.

   Les prochaines évolutions pourraient inclure :
   """)

   st.markdown("""
   - Alertes automatiques (email, Slack ou Teams) en cas de dérive importante.
   - Tableau de bord temps réel pour suivre l'évolution des métriques du modèle.
   - Déclenchement automatique du réentraînement via Airflow après détection d'un drift.
   - Suivi continu des performances du modèle en production.
   - Historisation avancée des rapports de monitoring et des actions de remédiation.
   """)

   st.success(
       "Objectif : rendre le monitoring entièrement automatisé et proactif tout au long du cycle de vie du modèle."
   )

   st.divider()

   st.header("☸️ Déploiement avec Kubernetes")

   st.markdown("""
   **Kubernetes** permettrait de rendre l'application plus robuste et scalable :

   - déploiement de FastAPI dans un conteneur ;
   - déploiement de Streamlit dans un autre conteneur ;
   - gestion automatique des redémarrages ;
   - montée en charge si le trafic augmente ;
   - meilleure séparation entre API, interface et base de données.
   """)

   st.success("Objectif : rendre le projet plus proche d'un environnement de production réel.")

   st.divider()

   st.header("🚀 Déploiement et industrialisation")

   st.markdown("""
   Améliorations envisageables en environnement de production :

   - gestion des secrets avec un coffre sécurisé (Vault, Secrets Manager) ;
   - authentification OAuth2/JWT à la place d'une simple clé API ;
   - déploiement continu (CD) après validation de la CI GitHub Actions ;
   - publication automatique des images Docker ;
   - validation automatique des données entrantes ;
   - alertes automatiques en cas d'échec du pipeline.
   """)

   st.divider()

   st.markdown("## 🎯 Conclusion")

   st.info("""
   Le projet met désormais en œuvre les principales briques d'un pipeline MLOps moderne :

   - collecte automatisée des données ;
   - stockage dans Supabase PostgreSQL ;
   - prétraitement et entraînement du modèle ;
   - suivi des expériences avec MLflow ;
   - Model Registry avec alias Champion ;
   - API FastAPI sécurisée ;
   - interface Streamlit ;
   - intégration continue (CI) avec GitHub Actions ;
   - monitoring des données avec Evidently ;
   - détection automatique du drift ;
   - génération de rapports HTML et JSON ;
   - mécanisme de remédiation avec possibilité de déclencher le réentraînement du modèle ;
   - journalisation des actions de monitoring et de remédiation.

   Les évolutions proposées (Airflow, alertes automatiques, Kubernetes, déploiement cloud et automatisation complète du réentraînement) permettront de faire évoluer ce prototype vers une plateforme MLOps encore plus robuste et industrialisée.
   """)
