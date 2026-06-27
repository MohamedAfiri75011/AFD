import pandas as pd
from database import engine

CSV_PATH = "data/raw/aide-publique-au-developpement_clean.csv"

df = pd.read_csv(
   CSV_PATH,
   sep=";",
   encoding="utf-8",
   low_memory=False
)

print(df.shape)

df.to_sql(
   "donnees",
   con=engine,
   if_exists="replace",
   index=False,
   chunksize=1000
)

print("Données chargées dans Supabase")
