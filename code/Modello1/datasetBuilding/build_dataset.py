# build_dataset.py (v3 - Fixes from validation)
    """
    Questo script costruisce un dataset di allenamento per un modello di previsione delle prestazioni in F1., sugli anni 2023 e 2024.
    Utilizza i dati di telemetria e le informazioni sui piloti per calcolare le caratteristiche e le etichette.
    Le caratteristiche includono tempi di giro, differenze di tempo, percentuali di telemetria e punteggi ELO.
    Il dataset finale viene salvato in un file CSV.
    Le colonne specifiche da calcolare e le strategie di imputazione sono configurabili.
    Il codice è progettato per essere eseguito in un ambiente Python con FastF1 e altre librerie necessarie installate.
    """
import os
from glob import glob
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import fastf1

from utils.elo import compute_dynamic_elo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
fastf1.Cache.enable_cache('./cache')
YEARS = [2023, 2024]
DATA_DIR = "data"
GP_DIRS = {yr: os.path.join(DATA_DIR, f"Gare{yr}") for yr in YEARS}
CALENDAR_PATH = os.path.join(DATA_DIR, "gare_per_anno.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "training_dataset_COMPLETO.csv")
MAX_LAP = 30

# Colonne da calcolare percentili (sorgente: CSV)
PCT_COLS_FROM_CSV = ['lap_time', 'gap_to_leader']

# Colonne da calcolare percentili (sorgente: F1)
COLS_TO_IMPUTE_MEDIAN = [
    'lap_time', 'sector1', 'sector2', 'sector3',
    'speed_avg', 'throttle_pct', 'brake_pct', 'drs_pct', 'gear_avg',
    'position', 'lap_time_pct' # Added position and lap_time_pct here
]
# --- End Configuration ---

# Questa funzione calcola i secondi totali da un oggetto timedelta
def safe_total_seconds(td):
    """Convert Timedelta to total seconds, returning NaN for NaT."""
    if pd.isna(td):
        return np.nan
    return td.total_seconds()

# Funzione per calcolare le caratteristiche di telemetria sintetica
def process_telemetry(car_data):

    required_cols = {'Speed', 'Throttle', 'Brake', 'DRS', 'nGear'}
    if car_data is None or car_data.empty or not required_cols.issubset(car_data.columns):
        logging.debug("Telemetry data missing or incomplete for telemetry processing.") # Changed to debug level

        return dict.fromkeys(['speed_avg','throttle_pct','brake_pct','drs_pct','gear_avg'], np.nan)

    total = len(car_data)
    if total == 0:

         return dict.fromkeys(['speed_avg','throttle_pct','brake_pct','drs_pct','gear_avg'], np.nan)

    try:
        results = {
            'speed_avg': car_data['Speed'].mean(),
            'throttle_pct': (car_data['Throttle'] > 80).sum() / total if total else 0,
            'brake_pct': (car_data['Brake'] > 0).sum() / total if total else 0,
            'drs_pct': (car_data['DRS'] > 0).sum() / total if total else 0,
            'gear_avg': car_data['nGear'].mean(),
        }
    except Exception as e:
        logging.error(f"Error processing telemetry: {e}")
        # Return dict without hard_brakes
        results = dict.fromkeys(['speed_avg','throttle_pct','brake_pct','drs_pct','gear_avg'], np.nan)

    return results

# Funzione principale per costruire il dataset
# Questa funzione esegue il processo di costruzione del dataset
# Carica i dati di telemetria, calcola le caratteristiche e salva il dataset finale
def main():
    logging.info("Starting dataset build process (v3)...")
    try:
        cal = pd.read_csv(CALENDAR_PATH)
        logging.info("Calendar loaded.")
    except FileNotFoundError:
        logging.error(f"Calendar file not found at {CALENDAR_PATH}")
        return

    logging.info("Computing dynamic ELO ratings...")
    elo_drv, elo_team = compute_dynamic_elo(cal)
    logging.info("ELO ratings computed.")

    records = []
    # --- Main Loop over Years and GPs ---
    for year in YEARS:
        gp_files = sorted(glob(os.path.join(GP_DIRS[year], "data_*.csv")))
        if not gp_files:
            logging.warning(f"No data CSV files found for year {year} in {GP_DIRS[year]}")
            continue
        logging.info(f"Processing year {year}...")

        for path in tqdm(gp_files, desc=f"GPs {year}"):
            try:
                # Extract GP name and find Round
                gp_name_parts = Path(path).stem.split("_")[1:-1]
                if not gp_name_parts:
                    logging.warning(f"Could not parse GP name from filename: {Path(path).name}")
                    continue
                gp = " ".join(gp_name_parts)
                rr_info = cal.query("anno==@year and nome_gran_premio==@gp")
                if rr_info.empty:
                    logging.warning(f"Round not found in calendar for {gp} {year}")
                    continue
                rnd = int(rr_info.iloc[0]['round'])
                gp_identifier = f"{year} Round {rnd} - {gp}"

                # --- Load Input CSV Data ---
                df_input_csv = pd.read_csv(path)
                df_input_csv = df_input_csv[df_input_csv.lap_number <= MAX_LAP].copy()
                if df_input_csv.empty:
                    logging.info(f"No laps found <= {MAX_LAP} in {Path(path).name}")
                    continue

                # --- Load FastF1 Session Data ---
                try:
                    session = fastf1.get_session(year, gp, 'R')
                    session.load(laps=True, telemetry=True, weather=False, messages=False)
                    laps_all_f1 = session.laps
                    if laps_all_f1 is None or laps_all_f1.empty:
                         logging.warning(f"FastF1 could not load laps for {gp_identifier}. Skipping GP.")
                         continue

                    # Pre-create team mapping
                    if session.results is None or session.results.empty:
                         logging.warning(f"FastF1 could not load results (for team map) for {gp_identifier}. Trying drivers...")
                         if not laps_all_f1.empty:
                              team_map = laps_all_f1.groupby('Driver')['Team'].first().to_dict()
                         else:
                              logging.error(f"Cannot get team map for {gp_identifier}. Skipping GP.")
                              continue
                    else:
                         if 'Abbreviation' in session.results.columns:
                             team_map = session.results.set_index('Abbreviation')['TeamName'].to_dict()
                         else:
                              team_map = session.results.set_index('DriverNumber')['TeamName'].to_dict()
                              logging.warning("Using DriverNumber for team map - ensure input CSV 'driver' column matches.")

                except Exception as e:
                    logging.error(f"Error loading FastF1 session for {gp_identifier}: {e}. Skipping GP.")
                    continue

                # Controlla il mapping dei team
                csv_drivers = set(df_input_csv.driver.unique())
                missing_drivers = csv_drivers - set(team_map.keys())
                if missing_drivers:
                     # Attempt to map missing drivers using F1 laps data
                    for missing_drv in list(missing_drivers):
                        driver_lap_data = laps_all_f1[laps_all_f1['Driver'] == missing_drv]
                        if not driver_lap_data.empty:
                            found_team = driver_lap_data['Team'].iloc[0]
                            team_map[missing_drv] = found_team
                            missing_drivers.remove(missing_drv)
                            logging.info(f"Mapped missing driver {missing_drv} to team {found_team} for {gp_identifier}")

                if missing_drivers:
                    logging.error(f"Unmappable drivers persist for {gp_identifier}: {missing_drivers}. Skipping GP.")
                    continue


                # --- Process Lap by Lap ---
                for lap_no, grp_csv in df_input_csv.groupby('lap_number'):


                    leader_rows_csv = grp_csv[grp_csv.position == 1]
                    leader_driver = None
                    if not leader_rows_csv.empty:
                         leader_driver = leader_rows_csv.iloc[0]['driver']
                    else: # Prova a trovare il leader in FastF1
                        f1_lap_data = laps_all_f1[laps_all_f1['LapNumber'] == lap_no]
                        leader_f1_rows = f1_lap_data[f1_lap_data['Position'] == 1]
                        if not leader_f1_rows.empty:
                             leader_driver = leader_f1_rows.iloc[0]['Driver']
                             leader_rows_csv = grp_csv[grp_csv['driver'] == leader_driver] # Find matching CSV row if possible
                             if leader_rows_csv.empty:
                                  logging.warning(f"Leader {leader_driver} found in F1 data but not CSV for Lap {lap_no} of {gp_identifier}.")
                                  # We have the driver, but maybe not the CSV gap info. Proceed carefully.

                    # If still no leader, skip lap
                    if leader_driver is None:
                        logging.warning(f"Leader not found for Lap {lap_no} in {gp_identifier}. Skipping lap.")
                        continue

                    # --- Get Leader's Data (Primary source: FastF1) ---
                    # Use pick_drivers instead of deprecated pick_driver
                    leader_lap_f1_series = laps_all_f1.pick_drivers(leader_driver).loc[lambda d: d.LapNumber == lap_no]
                    if leader_lap_f1_series.empty:
                         logging.warning(f"Leader {leader_driver}'s lap data not found in FastF1 for Lap {lap_no} of {gp_identifier}. Skipping lap.")
                         continue
                    leader_lap_f1 = leader_lap_f1_series.iloc[0]

                    # Process leader's telemetry
                    leader_telemetry = {}
                    try:
                         leader_car_data = leader_lap_f1.get_car_data()
                         leader_telemetry = process_telemetry(leader_car_data) # Returns dict without hard_brakes
                    except Exception as e:
                         logging.warning(f"Could not get/process telemetry for leader {leader_driver} Lap {lap_no}: {e}")
                         # Provide NaNs for telemetry keys (no hard_brakes)
                         leader_telemetry = dict.fromkeys(['speed_avg','throttle_pct','brake_pct','drs_pct','gear_avg'], np.nan)


                    # Baseline values for difference calculation
                    # Note: gap_to_leader taken from leader's CSV row, should ideally be 0 or reference time
                    leader_csv_row_gap = leader_rows_csv.iloc[0]['gap_to_leader'] if not leader_rows_csv.empty else np.nan

                    base_values = {
                        'lap_time': safe_total_seconds(leader_lap_f1.LapTime),
                        'gap_to_leader': leader_csv_row_gap,
                        **leader_telemetry
                    }

                    # Calculate percentiles for selected columns using the input CSV group
                    pct_lookup = {}
                    for col in PCT_COLS_FROM_CSV:
                        if col in grp_csv:
                            ranks = grp_csv[col].rank(pct=True, method='dense')
                            pct_lookup[col] = dict(ranks)
                        else:
                             pct_lookup[col] = {}


                    # --- Inner Loop: Process each Driver for the Lap ---
                    for idx, row_csv in grp_csv.iterrows():
                        drv = row_csv['driver']
                        team = team_map.get(drv, "Unknown") # Get team name

                        # Get corresponding FastF1 lap object for this driver/lap
                        # Use pick_drivers instead of deprecated pick_driver
                        lap_obj_f1_series = laps_all_f1.pick_drivers(drv).loc[lambda d: d.LapNumber == lap_no]
                        if lap_obj_f1_series.empty:
                            # logging.warning(f"Driver {drv} lap data not found in FastF1 for Lap {lap_no}. Skipping driver for this lap.") # Can be verbose
                            continue
                        lap_obj_f1 = lap_obj_f1_series.iloc[0]

                        # --- Build Record Dictionary ---
                        rec = {
                            'anno': year, 'gp': gp, 'round': rnd, 'lap_number': lap_no,
                            'driver': drv, 'team': team,
                            'compound': row_csv.get('compound', lap_obj_f1.get('Compound', "Unknown")),
                            'position': lap_obj_f1.get('Position', row_csv.get('position', np.nan)), # Get from F1, fallback CSV
                            'final_position': row_csv['final_position'],
                            'csv_pit_stop_count': row_csv.get('pit_stop_count', np.nan),
                            'elo_driver': elo_drv.get(drv, 1500.0),
                            'elo_team': elo_team.get(team, 1500.0),
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

                        # Telemetry Source: Calculated from FastF1 car data
                        driver_telemetry = {}
                        try:
                            car_data = lap_obj_f1.get_car_data()
                            driver_telemetry = process_telemetry(car_data) # Returns dict without hard_brakes
                        except Exception as e:
                             logging.warning(f"Could not get/process telemetry for {drv} Lap {lap_no}: {e}")
                             # Provide NaNs (no hard_brakes)
                             driver_telemetry = dict.fromkeys(['speed_avg','throttle_pct','brake_pct','drs_pct','gear_avg'], np.nan)
                        rec.update(driver_telemetry)

                        # --- Calculate Difference Features (no hard_brakes_diff) ---
                        rec['lap_time_diff'] = rec['lap_time'] - base_values['lap_time'] if pd.notna(rec['lap_time']) and pd.notna(base_values['lap_time']) else np.nan
                        rec['gap_to_leader_diff'] = row_csv['gap_to_leader'] - base_values['gap_to_leader'] if pd.notna(row_csv['gap_to_leader']) and pd.notna(base_values['gap_to_leader']) else np.nan

                        # Compare calculated telemetry to leader's calculated telemetry (no hard_brakes)
                        for col in driver_telemetry.keys(): # keys() now excludes hard_brakes
                            rec[f"{col}_diff"] = rec[col] - base_values[col] if pd.notna(rec[col]) and pd.notna(base_values[col]) else np.nan

                        # --- Add Percentile Features (Source: Input CSV ranks) ---
                        for col in PCT_COLS_FROM_CSV:
                             rec[f"{col}_pct"] = pct_lookup[col].get(idx, np.nan)

                        records.append(rec)

            except pd.errors.EmptyDataError:
                logging.warning(f"Input CSV file is empty: {path}")
            except KeyError as e:
                 logging.error(f"Missing expected column processing {path}: {e}")
            except Exception as e:
                logging.error(f"Unhandled error processing file {path}: {e}", exc_info=True)

    # --- Final Processing ---
    if not records:
        logging.error("No records were generated. Exiting.")
        return

    df_out = pd.DataFrame(records)
    logging.info(f"Generated {len(df_out)} records initially.")

    # Imputation Strategy:
    # 1. Median (driver -> global) for core times, telemetry, position, lap_time_pct
    # 2. Fill lap_time_diff NaN with 0 specifically
    logging.info(f"Imputing NaNs (Median strategy) for columns: {COLS_TO_IMPUTE_MEDIAN}")
    for col in COLS_TO_IMPUTE_MEDIAN:
        if col in df_out.columns:
            # Impute within driver first
            df_out[col] = df_out.groupby('driver')[col].transform(lambda x: x.fillna(x.median()))
            # Impute remaining NaNs with global median
            global_median = df_out[col].median()
            if pd.isna(global_median):
                # Fallback for edge cases where even global median is NaN (e.g., all NaNs for the column)
                fill_value = 0.5 if '_pct' in col else 0 # Use 0.5 for pct, 0 otherwise
                logging.warning(f"Global median for {col} is NaN. Filling remaining NaNs with {fill_value}.")
                df_out[col] = df_out[col].fillna(fill_value)
            else:
                df_out[col] = df_out[col].fillna(global_median)
            # Ensure sectors are non-negative after imputation
            if 'sector' in col:
                 df_out[col] = df_out[col].apply(lambda x: max(0, x) if pd.notna(x) else 0)
            # Ensure position is integer after imputation
            if col == 'position':
                 df_out[col] = df_out[col].round().astype(int)

        else:
            logging.warning(f"Column {col} specified for median imputation not found in DataFrame.")

    # Specific imputation for lap_time_diff
    if 'lap_time_diff' in df_out.columns:
         original_nan_count = df_out['lap_time_diff'].isna().sum()
         if original_nan_count > 0:
              logging.info(f"Imputing {original_nan_count} NaNs in 'lap_time_diff' with 0.")
              df_out['lap_time_diff'] = df_out['lap_time_diff'].fillna(0)
         else:
              logging.info("'lap_time_diff' had no NaNs to impute with 0.")
    else:
         logging.warning("Column 'lap_time_diff' not found for specific NaN imputation.")


    # Final checks (optional but recommended)
    final_nan_counts = df_out.isnull().sum()
    final_nan_counts = final_nan_counts[final_nan_counts > 0]
    if not final_nan_counts.empty:
         logging.warning("NaNs remain in the dataset after all imputation steps:")
         print(final_nan_counts)
    else:
         logging.info("✅ No remaining NaNs detected after all imputation steps.")

    # Save dataset
    try:
        df_out.to_csv(OUTPUT_PATH, index=False)
        logging.info(f"✅ Dataset successfully saved to {OUTPUT_PATH} ({len(df_out)} rows)")
    except Exception as e:
        logging.error(f"Failed to save dataset to {OUTPUT_PATH}: {e}")


if __name__ == "__main__":
    main()