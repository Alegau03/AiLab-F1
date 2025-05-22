import os
import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm

# Crea la cache se non esiste
os.makedirs('./cache', exist_ok=True)
fastf1.Cache.enable_cache('./cache')


def process_telemetry(telemetry):
    total_points = len(telemetry)
    if total_points == 0:
        return {
            'speed_avg': np.nan,
            'throttle_pct': np.nan,
            'brake_pct': np.nan,
            'drs_pct': np.nan,
            'gear_avg': np.nan,
            'hard_brakes': np.nan
        }

    speed_avg = telemetry['Speed'].mean()
    throttle_pct = (telemetry['Throttle'] > 80).sum() / total_points
    brake_pct = (telemetry['Brake'] > 0).sum() / total_points
    drs_pct = (telemetry['DRS'] > 0).sum() / total_points
    gear_avg = telemetry['nGear'].mean()
    hard_brakes = ((telemetry['Brake'] > 80) & (telemetry['Speed'] > 150)).sum()

    return {
        'speed_avg': speed_avg,
        'throttle_pct': throttle_pct,
        'brake_pct': brake_pct,
        'drs_pct': drs_pct,
        'gear_avg': gear_avg,
        'hard_brakes': hard_brakes
    }


def extract_gara(year=2024, gp_name='Saudi Arabia', session_type='R', max_laps=15):
    session = fastf1.get_session(year, gp_name, session_type)
    session.load()

    laps = session.laps
    drivers = laps['Driver'].unique()
    data = []

    print(f"Estrazione dati per {gp_name} {year} - Sessione: {session_type}")

    for drv in tqdm(drivers):
        drv_laps = laps.pick_drivers([drv]).pick_quicklaps()
        drv_laps = drv_laps.sort_values(by='LapNumber').head(max_laps)

        try:
            final_position = session.results.loc[session.results['Abbreviation'] == drv, 'Position'].values[0]
        except IndexError:
            final_position = np.nan

        for _, lap in drv_laps.iterrows():
            try:
                car_data = lap.get_car_data()

                telemetry_stats = process_telemetry(car_data)

                lap_data = {
                    'driver': drv,
                    'lap_number': lap['LapNumber'],
                    'compound': lap['Compound'],
                    'track_status': lap['TrackStatus'],
                    'lap_time': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
                    'position': lap['Position'],
                    'gap_to_leader': lap['Time'].total_seconds() if pd.notna(lap['Time']) else None,
                    'final_position': final_position,
                    **telemetry_stats
                }

                data.append(lap_data)

            except Exception as e:
                print(f"Errore su {drv} lap {lap['LapNumber']}: {e}")

    df = pd.DataFrame(data)
    filename = f'data_{gp_name.replace(" ", "_")}_{year}.csv'
    df.to_csv(filename, index=False)
    print(f"\n✅ Dataset salvato in: {filename}")


if __name__ == "__main__":
    # Cambia il GP e l’anno se vuoi un'altra gara
    extract_gara(year=2024, gp_name='Saudi Arabia', session_type='R', max_laps=15)
