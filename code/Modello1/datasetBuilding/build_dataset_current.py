# build_dataset_current_season.py

import os
from glob import glob
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import fastf1

# Assumendo che utils.elo sia accessibile (nella stessa cartella o nel PYTHONPATH)
from utils.elo import compute_dynamic_elo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for Current Season ---
fastf1.Cache.enable_cache('./cache')

YEAR_TO_PROCESS = 2025 # <--- ANNO SPECIFICO DA PROCESSARE PER QUESTO SCRIPT
ALL_YEARS_FOR_ELO = [2023, 2024, YEAR_TO_PROCESS] # <--- ANNI PER CALCOLO ELO COMPLETO

DATA_DIR = "data"
# GP_DIRS ora è specifico per YEAR_TO_PROCESS per trovare i file CSV sorgente
GP_DIRS = {YEAR_TO_PROCESS: os.path.join(DATA_DIR, f"Gare{YEAR_TO_PROCESS}")}
CALENDAR_PATH = os.path.join(DATA_DIR, "gare_per_anno.csv") # Usato per tutti gli anni in ELO_CALCULATION_YEARS
OUTPUT_PATH = os.path.join(DATA_DIR, f"training_dataset_{YEAR_TO_PROCESS}.csv") # Output specifico
MAX_LAP = 30

PCT_COLS_FROM_CSV = ['lap_time', 'gap_to_leader']
COLS_TO_IMPUTE_MEDIAN = [
    'lap_time', 'sector1', 'sector2', 'sector3',
    'speed_avg', 'throttle_pct', 'brake_pct', 'drs_pct', 'gear_avg',
    'position', 'lap_time_pct'
]
# --- End Configuration ---

def safe_total_seconds(td):
    if pd.isna(td): return np.nan
    return td.total_seconds()

def process_telemetry(car_data):
    required_cols = {'Speed', 'Throttle', 'Brake', 'DRS', 'nGear'}
    if car_data is None or car_data.empty or not required_cols.issubset(car_data.columns):
        logging.debug("Telemetry data missing or incomplete for telemetry processing.")
        return dict.fromkeys(['speed_avg','throttle_pct','brake_pct','drs_pct','gear_avg'], np.nan)
    total = len(car_data)
    if total == 0: return dict.fromkeys(['speed_avg','throttle_pct','brake_pct','drs_pct','gear_avg'], np.nan)
    try:
        return {
            'speed_avg': car_data['Speed'].mean(),
            'throttle_pct': (car_data['Throttle'] > 80).sum() / total if total else 0,
            'brake_pct': (car_data['Brake'] > 0).sum() / total if total else 0,
            'drs_pct': (car_data['DRS'] > 0).sum() / total if total else 0,
            'gear_avg': car_data['nGear'].mean(),
        }
    except Exception as e:
        logging.error(f"Error processing telemetry: {e}")
        return dict.fromkeys(['speed_avg','throttle_pct','brake_pct','drs_pct','gear_avg'], np.nan)

def main():
    logging.info(f"Starting dataset build process for YEAR {YEAR_TO_PROCESS}...")
    try:
        cal_full = pd.read_csv(CALENDAR_PATH)
        logging.info("Full calendar loaded for ELO calculation.")
        # Filtra il calendario solo per gli anni necessari all'Elo
        cal_for_elo = cal_full[cal_full['anno'].isin(ALL_YEARS_FOR_ELO)].copy()
        if cal_for_elo.empty:
            logging.error(f"Calendar does not contain data for required ELO years: {ALL_YEARS_FOR_ELO}")
            return
    except FileNotFoundError:
        logging.error(f"Calendar file not found at {CALENDAR_PATH}")
        return

    logging.info(f"Computing dynamic ELO ratings across years: {ALL_YEARS_FOR_ELO}...")
    # Passa il calendario filtrato per gli anni ELO a compute_dynamic_elo
    elo_drv, elo_team = compute_dynamic_elo(cal_for_elo)
    logging.info("ELO ratings computed.")

    records = []
    # Il loop esterno ora itera solo su YEAR_TO_PROCESS per la raccolta dati dei GP
    logging.info(f"Processing GP data for year {YEAR_TO_PROCESS}...")
    
    # Assicurati che GP_DIRS[YEAR_TO_PROCESS] esista
    year_gp_dir = GP_DIRS.get(YEAR_TO_PROCESS)
    if not year_gp_dir or not os.path.exists(year_gp_dir):
        logging.warning(f"No data directory found for year {YEAR_TO_PROCESS} at {year_gp_dir}. No data will be processed.")
        gp_files = []
    else:
        gp_files = sorted(glob(os.path.join(year_gp_dir, "data_*.csv")))

    if not gp_files:
        logging.warning(f"No data CSV files found for year {YEAR_TO_PROCESS} in {year_gp_dir}")
        # Non terminare lo script, potrebbe comunque voler creare un file vuoto se non ci sono gare
    
    # Filtra il calendario per l'anno specifico che stiamo processando per i dati dei GP
    cal_year_to_process = cal_full[cal_full['anno'] == YEAR_TO_PROCESS].copy()

    for path in tqdm(gp_files, desc=f"GPs {YEAR_TO_PROCESS}"):
        try:
            gp_name_parts = Path(path).stem.split("_")[1:-1]
            if not gp_name_parts: logging.warning(f"Could not parse GP name from: {Path(path).name}"); continue
            gp = " ".join(gp_name_parts)
            
            # Usa cal_year_to_process per trovare il round per i GP dell'anno corrente
            rr_info = cal_year_to_process.query("nome_gran_premio==@gp") # anno è già filtrato
            if rr_info.empty: logging.warning(f"Round not found in calendar for {gp} {YEAR_TO_PROCESS}"); continue
            rnd = int(rr_info.iloc[0]['round'])
            gp_identifier = f"{YEAR_TO_PROCESS} Round {rnd} - {gp}"

            df_input_csv = pd.read_csv(path)
            df_input_csv = df_input_csv[df_input_csv.lap_number <= MAX_LAP].copy()
            if df_input_csv.empty: logging.info(f"No laps <= {MAX_LAP} in {Path(path).name}"); continue

            try:
                session = fastf1.get_session(YEAR_TO_PROCESS, gp, 'R')
                session.load(laps=True, telemetry=True, weather=False, messages=False)
                laps_all_f1 = session.laps
                if laps_all_f1 is None or laps_all_f1.empty: logging.warning(f"FastF1 no laps for {gp_identifier}. Skip."); continue
                
                team_map = {}
                if session.results is not None and not session.results.empty:
                    if 'Abbreviation' in session.results.columns: team_map = session.results.set_index('Abbreviation')['TeamName'].to_dict()
                    else: team_map = session.results.set_index('DriverNumber')['TeamName'].to_dict() # Fallback
                elif not laps_all_f1.empty : team_map = laps_all_f1.groupby('Driver')['Team'].first().to_dict() # Fallback 2
                else: logging.error(f"Cannot get team map for {gp_identifier}. Skip."); continue

            except Exception as e: logging.error(f"Error FastF1 session for {gp_identifier}: {e}. Skip."); continue

            csv_drivers = set(df_input_csv.driver.unique())
            missing_drivers = csv_drivers - set(team_map.keys())
            if missing_drivers:
                for missing_drv in list(missing_drivers):
                    drv_lap_data = laps_all_f1[laps_all_f1['Driver'] == missing_drv]
                    if not drv_lap_data.empty:
                        team_map[missing_drv] = drv_lap_data['Team'].iloc[0]; missing_drivers.remove(missing_drv)
            if missing_drivers: logging.error(f"Unmappable drivers for {gp_identifier}: {missing_drivers}. Skip."); continue

            for lap_no, grp_csv in df_input_csv.groupby('lap_number'):
                leader_rows_csv = grp_csv[grp_csv.position == 1]
                leader_driver = None
                if not leader_rows_csv.empty: leader_driver = leader_rows_csv.iloc[0]['driver']
                else:
                    f1_lap_data = laps_all_f1[laps_all_f1['LapNumber'] == lap_no]
                    leader_f1_rows = f1_lap_data[f1_lap_data['Position'] == 1]
                    if not leader_f1_rows.empty: leader_driver = leader_f1_rows.iloc[0]['Driver']
                if leader_driver is None: logging.warning(f"Leader not found Lap {lap_no} {gp_identifier}. Skip lap."); continue

                leader_lap_f1_series = laps_all_f1.pick_drivers(leader_driver).loc[lambda d: d.LapNumber == lap_no]
                if leader_lap_f1_series.empty: logging.warning(f"Leader {leader_driver} lap data not in FastF1 Lap {lap_no}. Skip lap."); continue
                leader_lap_f1 = leader_lap_f1_series.iloc[0]

                leader_telemetry = {}
                try: leader_telemetry = process_telemetry(leader_lap_f1.get_car_data())
                except Exception as e: logging.warning(f"Telemetry error leader {leader_driver} Lap {lap_no}: {e}")
                leader_telemetry.setdefault('speed_avg', np.nan) # Ensure all keys exist

                leader_csv_row = grp_csv[grp_csv['driver'] == leader_driver] # Find leader in current CSV group
                leader_csv_row_gap = leader_csv_row.iloc[0]['gap_to_leader'] if not leader_csv_row.empty else np.nan

                base_values = { 'lap_time': safe_total_seconds(leader_lap_f1.LapTime), 'gap_to_leader': leader_csv_row_gap, **leader_telemetry }

                pct_lookup = {col: dict(grp_csv[col].rank(pct=True, method='dense')) if col in grp_csv else {} for col in PCT_COLS_FROM_CSV}

                for idx, row_csv in grp_csv.iterrows():
                    drv = row_csv['driver']; team = team_map.get(drv, "Unknown")
                    lap_obj_f1_series = laps_all_f1.pick_drivers(drv).loc[lambda d: d.LapNumber == lap_no]
                    if lap_obj_f1_series.empty: continue
                    lap_obj_f1 = lap_obj_f1_series.iloc[0]

                    rec = {
                        'anno': YEAR_TO_PROCESS, 'gp': gp, 'round': rnd, 'lap_number': lap_no,
                        'driver': drv, 'team': team,
                        'compound': row_csv.get('compound', lap_obj_f1.get('Compound', "Unknown")),
                        'position': lap_obj_f1.get('Position', row_csv.get('position', np.nan)),
                        'final_position': row_csv['final_position'],
                        'csv_pit_stop_count': row_csv.get('pit_stop_count', np.nan),
                        'elo_driver': elo_drv.get(drv, 1500.0), # Elo is from ALL_YEARS_FOR_ELO calc
                        'elo_team': elo_team.get(team, 1500.0), # Elo is from ALL_YEARS_FOR_ELO calc
                        'stint': int(lap_obj_f1.get('StintNumber', 0)),
                        'lap_in_stint': int(lap_obj_f1.get('LapNumberInStint', 0)),
                        'pit_in': pd.notna(lap_obj_f1.get('PitInTime')),
                        'pit_out': pd.notna(lap_obj_f1.get('PitOutTime')),
                        'sector1': safe_total_seconds(lap_obj_f1.get('Sector1Time')),
                        'sector2': safe_total_seconds(lap_obj_f1.get('Sector2Time')),
                        'sector3': safe_total_seconds(lap_obj_f1.get('Sector3Time')),
                        'track_status': lap_obj_f1.get('TrackStatus', '0'),
                        'lap_time': safe_total_seconds(lap_obj_f1.LapTime),
                    }
                    driver_telemetry = {};
                    try: driver_telemetry = process_telemetry(lap_obj_f1.get_car_data())
                    except Exception as e: logging.warning(f"Telemetry error {drv} Lap {lap_no}: {e}")
                    rec.update(driver_telemetry)

                    rec['lap_time_diff'] = rec['lap_time'] - base_values['lap_time'] if pd.notna(rec['lap_time']) and pd.notna(base_values['lap_time']) else np.nan
                    rec['gap_to_leader_diff'] = row_csv['gap_to_leader'] - base_values['gap_to_leader'] if pd.notna(row_csv['gap_to_leader']) and pd.notna(base_values['gap_to_leader']) else np.nan
                    for col_tel in driver_telemetry.keys():
                        rec[f"{col_tel}_diff"] = rec[col_tel] - base_values.get(col_tel, np.nan) if pd.notna(rec[col_tel]) and pd.notna(base_values.get(col_tel)) else np.nan
                    for col_pct in PCT_COLS_FROM_CSV: rec[f"{col_pct}_pct"] = pct_lookup[col_pct].get(idx, np.nan)
                    records.append(rec)
        except pd.errors.EmptyDataError: logging.warning(f"Empty CSV: {path}")
        except KeyError as e: logging.error(f"Missing column in {path} or F1 data: {e}")
        except Exception as e: logging.error(f"Unhandled error {path}: {e}", exc_info=True)

    if not records:
        logging.warning(f"No records generated for {YEAR_TO_PROCESS}. Output file might be empty or not created if it doesn't exist.")
        # Decide if you want to create an empty file with headers or just skip
        # For now, if no records, we'll just print a message and the save step might create an empty file or skip.
        # To ensure headers for an empty file:
        # if not os.path.exists(OUTPUT_PATH): # Or always write headers if file might be overwritten
        #     pd.DataFrame(columns=[...list of expected columns...]).to_csv(OUTPUT_PATH, index=False)
        # return

    df_out = pd.DataFrame(records)
    if df_out.empty:
        logging.warning(f"DataFrame is empty after processing all files for {YEAR_TO_PROCESS}.")
    else:
        logging.info(f"Generated {len(df_out)} records for {YEAR_TO_PROCESS} initially.")

        # IMPUTATION (identica a prima)
        logging.info(f"Imputing NaNs for columns: {COLS_TO_IMPUTE_MEDIAN}")
        for col in COLS_TO_IMPUTE_MEDIAN:
            if col in df_out.columns:
                df_out[col] = df_out.groupby('driver')[col].transform(lambda x: x.fillna(x.median()))
                global_median = df_out[col].median()
                if pd.isna(global_median):
                    fill_value = 0.5 if '_pct' in col else 0
                    logging.warning(f"Global median for {col} is NaN. Filling NaNs with {fill_value}.")
                    df_out[col] = df_out[col].fillna(fill_value)
                else:
                    df_out[col] = df_out[col].fillna(global_median)
                if 'sector' in col: df_out[col] = df_out[col].apply(lambda x: max(0, x) if pd.notna(x) else 0)
                if col == 'position': df_out[col] = df_out[col].round().astype(int)
            else: logging.warning(f"Imputation column {col} not in DataFrame.")

        if 'lap_time_diff' in df_out.columns:
            nan_count = df_out['lap_time_diff'].isna().sum()
            if nan_count > 0: logging.info(f"Imputing {nan_count} NaNs in 'lap_time_diff' with 0."); df_out['lap_time_diff'] = df_out['lap_time_diff'].fillna(0)
        else: logging.warning("Colonna 'lap_time_diff' non trovata per imputazione specifica.")

        final_nan_counts = df_out.isnull().sum(); final_nan_counts = final_nan_counts[final_nan_counts > 0]
        if not final_nan_counts.empty: logging.warning("NaNs rimangono dopo imputazione:\n" + str(final_nan_counts))
        else: logging.info("✅ Nessun NaN rimanente dopo imputazione.")

    try:
        # Se df_out è vuoto ma vuoi un file con header:
        if df_out.empty and not records : # e non c'erano record nemmeno all'inizio
            # Definisci qui le colonne attese per l'header se vuoi un file vuoto con header
            # Altrimenti, se df_out è vuoto, to_csv non farà nulla o creerà un file vuoto.
            expected_cols = [ # Ricrea la lista delle colonne attese basandoti su 'rec'
                'anno', 'gp', 'round', 'lap_number', 'driver', 'team', 'compound',
                'position', 'final_position', 'csv_pit_stop_count', 'elo_driver',
                'elo_team', 'stint', 'lap_in_stint', 'pit_in', 'pit_out', 'sector1',
                'sector2', 'sector3', 'track_status', 'lap_time', 'speed_avg',
                'throttle_pct', 'brake_pct', 'drs_pct', 'gear_avg', 'lap_time_diff',
                'gap_to_leader_diff', 'speed_avg_diff', 'throttle_pct_diff',
                'brake_pct_diff', 'drs_pct_diff', 'gear_avg_diff', 'lap_time_pct',
                'gap_to_leader_pct'
            ]
            df_out = pd.DataFrame(columns=expected_cols)
            logging.info(f"Nessun record processato, verrà salvato un file CSV vuoto con header: {OUTPUT_PATH}")


        df_out.to_csv(OUTPUT_PATH, index=False)
        logging.info(f"✅ Dataset per l'anno {YEAR_TO_PROCESS} salvato in {OUTPUT_PATH} ({len(df_out)} righe)")
    except Exception as e:
        logging.error(f"Failed to save dataset to {OUTPUT_PATH}: {e}")

if __name__ == "__main__":
    # Verifica se la directory per l'anno corrente esiste, altrimenti crea un avviso
    # Questo è più per l'utente che esegue lo script
    current_year_gp_path = os.path.join(DATA_DIR, f"Gare{YEAR_TO_PROCESS}")
    if not os.path.exists(current_year_gp_path):
        logging.warning(f"La directory dei dati per l'anno {YEAR_TO_PROCESS} ('{current_year_gp_path}') non esiste.")
        logging.warning("Lo script verrà eseguito, ma probabilmente non troverà file GP da processare, risultando in un CSV vuoto.")
        # Potresti voler creare la directory qui se fa parte del workflow atteso
        # os.makedirs(current_year_gp_path, exist_ok=True)
        # print(f"Directory {current_year_gp_path} creata.")

    main()