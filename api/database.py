# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, text

# Solution standard pour le Transaction Pooler (6543) : l'ID du projet réintègre l'utilisateur
# Le point est encodé en %2E pour éviter que SQLAlchemy ne coupe la chaîne.
SUPABASE_CONN_STRING = "postgresql://postgres%2Evsusfuhifwtuxohnbmwi:Uv7K6MelZ4xMVcDS@aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require"

engine = create_engine(
    SUPABASE_CONN_STRING, 
    pool_pre_ping=True
)

def test_connection():
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT version();")).scalar()
    except Exception:
        return None