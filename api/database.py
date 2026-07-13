import os
from sqlalchemy import create_engine, text

# Configuration et Connexion Supabase
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")

if not SUPABASE_DB_URI:
    raise ValueError(
        "Erreur : La variable d'environnement SUPABASE_DB_URI n'est pas définie ! "
        "Vérifiez le fichier .env et la configuration Docker Compose."
    )

engine = create_engine(SUPABASE_DB_URI)

def test_connection():
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT version();")).scalar()
    except Exception:
        return None