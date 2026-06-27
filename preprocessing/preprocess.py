import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv("ingestion/.env")

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM donnees", engine)

print("Dimensions avant nettoyage :", df.shape)

# Suppression doublons
df = df.drop_duplicates()

# Conversion date
if "Date d'engagement" in df.columns:
   df["Date d'engagement"] = pd.to_datetime(df["Date d'engagement"], errors="coerce")
   df["annee_engagement"] = df["Date d'engagement"].dt.year
   df["mois_engagement"] = df["Date d'engagement"].dt.month
   df["trimestre_engagement"] = df["Date d'engagement"].dt.quarter

# Cible régression
target_reg = "Engagements (K EUR)"

if target_reg in df.columns:
   df = df[df[target_reg].notna()]
   df = df[df[target_reg] > 0]

# Remplissage valeurs manquantes
cat_cols = df.select_dtypes(include=["object"]).columns
num_cols = df.select_dtypes(include=["number"]).columns

df[cat_cols] = df[cat_cols].fillna("Non renseigné")
df[num_cols] = df[num_cols].fillna(0)

# Sauvegarde
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
   "data/processed/apd_ml_ready.csv",
   index=False
)

print("Dimensions après preprocessing :", df.shape)
print("Fichier ML sauvegardé : data/processed/apd_ml_ready.csv")
