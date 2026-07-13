import pandas as pd
import json
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres.vsusfuhifwtuxohnbmwi:Uv7K6MelZ4xMVcDS@aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require"
engine = create_engine(DATABASE_URL)

print("🧹 Nettoyage de la table predict_logs...")
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE public.predict_logs;"))

print("📥 Extraction de 2500 lignes aléatoires...")
df = pd.read_sql("SELECT * FROM afd ORDER BY RANDOM() LIMIT 2500;", engine)

# Convertir les NaN en None (Null) pour un format JSON parfait
df = df.where(pd.notnull(df), None)

print("🚀 Injection directe...")
with engine.begin() as conn:
    for _, row in df.iterrows():
        payload = row.to_dict()
        conn.execute(
            text("INSERT INTO public.predict_logs (inputs, status) VALUES (:inputs, 'success');"),
            {"inputs": json.dumps(payload)}
        )
print("🎉 Succès ! 2500 lignes complètes insérées.")