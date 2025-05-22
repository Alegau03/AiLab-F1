import os
import pandas as pd
from gare import estrai_gara

# Carica il file delle gare
gare_df = pd.read_csv("../data/gare_per_anno.csv")

# Ordina per anno e round
gare_df = gare_df.sort_values(by=["anno", "round"])

# Elimina i file esistenti
for anno in [2023, 2024]:
    dir_path = f"../data/Gare{anno}"
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            if file.endswith(".csv"):
                os.remove(os.path.join(dir_path, file))
        print(f"✅ Eliminati i file in {dir_path}")

# Rigenera i file con i pit stop
for _, row in gare_df.iterrows():
    anno = int(row['anno'])
    nome_gp = row['nome_gran_premio']
    print(f"\nProcesso {nome_gp} {anno}...")
    estrai_gara(anno, nome_gp) 