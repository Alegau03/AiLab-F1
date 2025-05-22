import os
import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm

# Cache FastF1
os.makedirs('./cache', exist_ok=True)
fastf1.Cache.enable_cache('./cache')

# Cartelle dati
#os.makedirs('../data/Gare2023', exist_ok=True)
os.makedirs('../data/Gare2025', exist_ok=True)

# Anni da estrarre
ANNI = [2025]

df_gare = pd.read_csv("data/gare_2025.csv").sort_values(by=["anno", "round"])

def process_telemetry(telemetry):
    total = len(telemetry)
    if total == 0:
        return {k: np.nan for k in ['speed_avg', 'throttle_pct', 'brake_pct', 'drs_pct', 'gear_avg', 'hard_brakes']}
    return {
        'speed_avg': telemetry['Speed'].mean(),
        'throttle_pct': (telemetry['Throttle'] > 80).sum() / total,
        'brake_pct': (telemetry['Brake'] > 0).sum() / total,
        'drs_pct': (telemetry['DRS'] > 0).sum() / total,
        'gear_avg': telemetry['nGear'].mean(),
        'hard_brakes': ((telemetry['Brake'] > 80) & (telemetry['Speed'] > 150)).sum()
    }

def estrai_gara(anno, nome_gp):
    nome_file = f"../data/Gare{anno}/data_{nome_gp.replace(' ', '_')}_{anno}.csv"
    if os.path.exists(nome_file): return

    try:
        session = fastf1.get_session(anno, nome_gp, 'R')
        session.load()
        laps = session.laps
        drivers = laps['Driver'].unique()
        data = []

        # Calcola i pit stop per ogni pilota
        pit_data = laps[['Driver', 'PitInTime']].dropna()
        pit_counts = pit_data.groupby('Driver').count().PitInTime.to_dict()

        for drv in tqdm(drivers, desc=f"{nome_gp} {anno}"):
            drv_laps = laps.pick_driver(drv).sort_values(by='LapNumber')
            try:
                final_pos = session.results.loc[session.results['Abbreviation'] == drv, 'Position'].values[0]
            except IndexError:
                final_pos = np.nan

            # Ottieni il numero di pit stop per questo pilota
            pit_stop_count = pit_counts.get(drv, 0)

            for _, lap in drv_laps.iterrows():
                try:
                    telem = lap.get_car_data()
                    stats = process_telemetry(telem)
                    data.append({
                        'driver': drv,
                        'lap_number': lap['LapNumber'],
                        'compound': lap['Compound'],
                        'track_status': lap['TrackStatus'],
                        'lap_time': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
                        'position': lap['Position'],
                        'gap_to_leader': lap['Time'].total_seconds() if pd.notna(lap['Time']) else None,
                        'final_position': final_pos,
                        'pit_stop_count': pit_stop_count,
                        **stats
                    })
                except:
                    continue

        pd.DataFrame(data).to_csv(nome_file, index=False)
        print(f"\u2705 Salvato: {nome_file}")
    except Exception as e:
        print(f"Errore {nome_gp} {anno}: {e}")

for anno in ANNI:
    for _, row in df_gare[df_gare["anno"] == anno].iterrows():
        estrai_gara(anno, row['nome_gran_premio'])