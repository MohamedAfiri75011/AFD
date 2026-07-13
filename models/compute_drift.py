import os
import pandas as pd
import json
import mlflow
import mlflow.data 
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

try:
    from api.database import engine
except ImportError:
    SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI", "postgresql://postgres:postgres@localhost:5432/postgres")
    engine = create_engine(SUPABASE_DB_URI, poolclass=NullPool)

def calculate_data_drift():
    print("SCRIPT V6.1 CHARGE : CORRECTION DU SCOPE MLFLOW")

    if mlflow.active_run():
        mlflow.end_run()

    try:
        df_ref = pd.read_sql_query('SELECT * FROM public.afd ORDER BY random() LIMIT 5000;', con=engine)
        df_prod_raw = pd.read_sql_query("SELECT inputs FROM public.predict_logs WHERE status = 'success';", con=engine)
        
        if df_prod_raw.empty or len(df_prod_raw) < 10:
            return {"status": "skipped", "message": "Pas de donnees."}

        # Conversion du JSON
        inputs_list = []
        for val in df_prod_raw['inputs']:
            if isinstance(val, str):
                inputs_list.append(json.loads(val))
            else:
                inputs_list.append(val)
                
        df_prod = pd.json_normalize(inputs_list)

        for df in [df_ref, df_prod]:
            df.columns = (
                df.columns.str.lower().str.strip().str.normalize("NFKD")
                .str.encode("ascii", errors="ignore").str.decode("utf-8")
                .str.replace(r"[^\w]+", "_", regex=True)
            )

        cols_to_exclude = ["log_engagements", "engagements_k_eur_", "engagements_k_eur", "id", "created_at", "index"]
        
        features_du_modele = [c for c in df_ref.columns if c not in cols_to_exclude and not c.startswith("odd_")]
        features_valides = [col for col in features_du_modele if col in df_prod.columns]
        
        if not features_valides:
            return {"status": "failed", "error": "Aucune feature valide."}
        
        df_ref = df_ref[features_valides]
        df_prod = df_prod[features_valides]
        
        # Alignement des types de donnees
        print("Alignement des types de donnees (SQL vs JSON)...")
        for col in features_valides:
            try:
                df_prod[col] = df_prod[col].fillna(pd.NA)
                df_prod[col] = df_prod[col].astype(df_ref[col].dtype)
            except Exception:
                df_ref[col] = df_ref[col].astype(str)
                df_prod[col] = df_prod[col].astype(str)

        print(f"Analyse sur {len(features_valides)} features metiers.")
        
        data_drift_report = Report(metrics=[DataDriftPreset(drift_share=0.3)])
        data_drift_report.run(reference_data=df_ref, current_data=df_prod)
        
        report_dict = data_drift_report.as_dict()
        result = report_dict["metrics"][0]["result"]
        
        colonnes_en_derive = result["number_of_drifted_columns"]
        colonnes_totales = result["number_of_columns"]
        vrai_score_drift = colonnes_en_derive / colonnes_totales
        
        print(f"VRAI Score trouve : {vrai_score_drift:.3f}")
        
        html_file_path = "evidently_data_drift_report.html"
        data_drift_report.save_html(html_file_path)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_SERVER", "http://localhost:5000"))
        mlflow.set_experiment("Data_Drift_Monitoring")
        
        with mlflow.start_run(run_name=f"drift_check_{datetime.now().strftime('%H%M%S')}"):
            # 1. Metriques et Parametres principaux
            mlflow.log_metric("drift_share", vrai_score_drift)
            mlflow.log_param("features_analyzed", colonnes_totales)
            mlflow.log_param("drifted_features", colonnes_en_derive)
            
            # 2. ALIMENTATION DE L'ONGLET DATASET
            try:
                ref_dataset = mlflow.data.from_pandas(df_ref.head(5), name=f"afd_reference_sample ({len(df_ref)} rows)", source="public.afd")
                prod_dataset = mlflow.data.from_pandas(df_prod.head(5), name=f"predict_logs_production ({len(df_prod)} rows)", source="public.predict_logs")
                
                mlflow.log_input(ref_dataset, context="reference")
                mlflow.log_input(prod_dataset, context="production")
            except Exception as dataset_err:
                print(f"Dataset non supporte : {dataset_err}")

            # 3. REMPLISSAGE CONTEXTUEL DES COLONNES
            mlflow.set_tag("monitored_model", "RandomForestRegressor")
            mlflow.set_tag("target_version", "production_champion")

            # 4. Artifact HTML
            mlflow.log_artifact(html_file_path)

        if os.path.exists(html_file_path):
            os.remove(html_file_path)

        return {"status": "success", "global_drift": vrai_score_drift}

    except Exception as e:
        print(f"ERREUR : {e}")
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    calculate_data_drift()