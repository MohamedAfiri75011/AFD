import pandas as pd

# On ajoute sep=";" pour forcer la lecture avec le bon séparateur
try:
    # On tente de le lire en UTF-8 si c'est son encodage initial masqué
    df = pd.read_csv("adp.csv", sep=";", encoding="utf-8")
except (UnicodeDecodeError, pd.errors.ParserError):
    # Si ça échoue, on le lit en Latin-1 (encodage standard d'Excel en Europe)
    df = pd.read_csv("adp.csv", sep=";", encoding="latin1")

# On nettoie les espaces vides éventuels dans les noms de colonnes
df.columns = [col.strip() for col in df.columns]

# On le sauvegarde en vrai UTF-8 universel avec des VIRGULES (standard pour Supabase et PostgreSQL)
df.to_csv("adp_train_propre.csv", index=False, sep=",", encoding="utf-8-sig")

print("✨ Fichier 'adp_train_propre.csv' généré avec succès !")
print(f"Dimensions : {df.shape[0]} lignes et {df.shape[1]} colonnes.")