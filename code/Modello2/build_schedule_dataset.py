import fastf1
import pandas as pd

# Attiva la cache a piacimento
fastf1.Cache.enable_cache('./cache')  # opzionale, utile per velocizzare

anni = [2025]
tutte_le_gare = []

for anno in anni:
    schedule = fastf1.get_event_schedule(anno)

    for _, row in schedule.iterrows():
        tutte_le_gare.append({
            'anno': anno,
            'round': row['RoundNumber'],
            'nome_gran_premio': row['EventName'],
            'location': row['Location'],
            'data_gara': row['Session5Date'].date() if pd.notna(row['Session5Date']) else None
        })

df_gare = pd.DataFrame(tutte_le_gare)
df_gare = df_gare.sort_values(by=['anno', 'round']).reset_index(drop=True)

df_gare.to_csv("gare_2025.csv", index=False)
print("✅ Dataset salvato in 'gare_per_anno.csv'")
