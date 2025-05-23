
import os
from pathlib import Path
from collections import defaultdict
import logging
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import fastf1

from utils.elo import compute_elo_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Configuration ---
fastf1.Cache.enable_cache('./cache')
YEARS_FOR_PRE_RACE_MODEL = [2023, 2024, 2025]
DATA_DIR = Path("data")
CALENDAR_PATH = DATA_DIR / "gare_per_anno.csv"
OUTPUT_PRE_RACE_DATASET_PATH = DATA_DIR / "pre_race_prediction_dataset.csv"
YEARS_LOOKBACK_CIRCUIT_STATS = 2
# --- End Configuration ---

# Questa funzione calcola le statistiche per un pilota su un circuito specifico
def get_circuit_stats_for_driver(driver_abbr: str, current_gp_name: str, current_year: int,
                                 all_results_df: pd.DataFrame, lookback_years: int):
    stats = {
        f"avg_finish_pos_circuit_prev{lookback_years}y": np.nan,
        f"avg_grid_pos_circuit_prev{lookback_years}y": np.nan,
        f"avg_pos_gained_circuit_prev{lookback_years}y": np.nan,
        f"races_on_circuit_prev{lookback_years}y": 0,
        f"best_finish_pos_circuit_prev{lookback_years}y": np.nan,
        f"laps_completed_pct_circuit_prev{lookback_years}y": np.nan,
    }
    if all_results_df.empty or 'gp' not in all_results_df.columns or 'Driver' not in all_results_df.columns: # Assicurati che 'Driver' esista
        logging.debug(f"Stats df vuoto o colonne 'gp'/'Driver' mancanti per stats di {driver_abbr}")
        return stats

    prev_races = all_results_df[
        (all_results_df['gp'] == current_gp_name) &
        (all_results_df['Driver'] == driver_abbr) & # Usa 'Driver' qui
        (all_results_df['anno'] >= current_year - lookback_years) &
        (all_results_df['anno'] < current_year)
    ].copy()

    if not prev_races.empty:
        prev_races['Position_numeric'] = pd.to_numeric(prev_races['Position'], errors='coerce')
        prev_races['GridPosition_numeric'] = pd.to_numeric(prev_races['GridPosition'], errors='coerce')
        stats[f"avg_finish_pos_circuit_prev{lookback_years}y"] = prev_races['Position_numeric'].fillna(25).mean()
        stats[f"avg_grid_pos_circuit_prev{lookback_years}y"] = prev_races['GridPosition_numeric'].fillna(25).mean()
        stats[f"best_finish_pos_circuit_prev{lookback_years}y"] = prev_races['Position_numeric'].min()
        valid_finish_and_grid = prev_races[prev_races['Position_numeric'].notna() & prev_races['GridPosition_numeric'].notna()]
        if not valid_finish_and_grid.empty:
            stats[f"avg_pos_gained_circuit_prev{lookback_years}y"] = (valid_finish_and_grid['GridPosition_numeric'] - valid_finish_and_grid['Position_numeric']).mean()
        else: stats[f"avg_pos_gained_circuit_prev{lookback_years}y"] = 0.0
        stats[f"races_on_circuit_prev{lookback_years}y"] = len(prev_races)
        if 'Laps' in prev_races.columns and 'TotalLaps' in prev_races.columns:
            prev_races['Laps_numeric'] = pd.to_numeric(prev_races['Laps'], errors='coerce')
            prev_races['TotalLaps_numeric'] = pd.to_numeric(prev_races['TotalLaps'], errors='coerce')
            valid_laps_data = prev_races[prev_races['Laps_numeric'].notna() & prev_races['TotalLaps_numeric'].notna() & (prev_races['TotalLaps_numeric'] > 0)]
            if not valid_laps_data.empty:
                stats[f"laps_completed_pct_circuit_prev{lookback_years}y"] = (valid_laps_data['Laps_numeric'] / valid_laps_data['TotalLaps_numeric']).mean()
    return stats

def get_circuit_stats_for_team(team_name: str, current_gp_name: str, current_year: int,
                               all_results_df: pd.DataFrame, lookback_years: int):
    stats = {f"team_avg_finish_pos_circuit_prev{lookback_years}y": np.nan, f"team_avg_pos_gained_circuit_prev{lookback_years}y": np.nan}
    if all_results_df.empty or 'gp' not in all_results_df.columns or 'TeamName' not in all_results_df.columns:  # Assicurati che 'TeamName' esista
        logging.debug(f"Stats df vuoto o colonne 'gp'/'TeamName' mancanti per stats di {team_name}")
        return stats
        
    prev_races_team = all_results_df[
        (all_results_df['gp'] == current_gp_name) &
        (all_results_df['TeamName'] == team_name) &
        (all_results_df['anno'] >= current_year - lookback_years) &
        (all_results_df['anno'] < current_year)
    ].copy()
    if not prev_races_team.empty:
        prev_races_team['Position_numeric'] = pd.to_numeric(prev_races_team['Position'], errors='coerce').fillna(25)
        prev_races_team['GridPosition_numeric'] = pd.to_numeric(prev_races_team['GridPosition'], errors='coerce').fillna(25)
        stats[f"team_avg_finish_pos_circuit_prev{lookback_years}y"] = prev_races_team['Position_numeric'].mean()
        valid_finish_team = prev_races_team[prev_races_team['Position_numeric'].notna() & prev_races_team['GridPosition_numeric'].notna()]
        if not valid_finish_team.empty:
            stats[f"team_avg_pos_gained_circuit_prev{lookback_years}y"] = (valid_finish_team['GridPosition_numeric'] - valid_finish_team['Position_numeric']).mean()
        else: stats[f"team_avg_pos_gained_circuit_prev{lookback_years}y"] = 0.0
    return stats
# FUnzione principale per costruire il dataset pre-gara, carica i risultati delle sessioni di qualifica e gara, calcola le statistiche e salva il dataset finale
# Questa funzione costruisce il dataset pre-gara per il modello di previsione
def main():
    logging.info("Starting Pre-Race Prediction Dataset build process...")
    try: cal_full = pd.read_csv(CALENDAR_PATH); logging.info("Full calendar loaded.")
    except FileNotFoundError: logging.error(f"Calendar file not found: {CALENDAR_PATH}"); return

    schedule_for_data = cal_full[cal_full['anno'].isin(YEARS_FOR_PRE_RACE_MODEL)].sort_values(["anno", "round"]).copy()
    if schedule_for_data.empty: logging.error(f"No races in calendar for years: {YEARS_FOR_PRE_RACE_MODEL}"); return

    logging.info(f"Computing ELO history for years: {schedule_for_data['anno'].unique().tolist()}...")
    _, _, df_elo_history = compute_elo_history(schedule_for_data)
    if df_elo_history.empty: logging.error("ELO history is empty."); return
    logging.info(f"ELO history computed. Shape: {df_elo_history.shape}")

    all_session_results_list = []
    logging.info("Loading Qualifying and Race results for all relevant sessions...")
    for _, event_row in tqdm(schedule_for_data.iterrows(), total=len(schedule_for_data), desc="Loading Session Results"):
        year, gp_name = int(event_row["anno"]), str(event_row["nome_gran_premio"])
        for session_type in ['Q', 'R']:
            try:
                time.sleep(0.1)
                session = fastf1.get_session(year, gp_name, session_type)
                session.load(laps=True, telemetry=False, weather=False, messages=False) # Corretto
                if session.results is not None and not session.results.empty:
                    res = session.results.copy()
                    res['anno'] = year
                    res['gp'] = gp_name
                    res['session_type'] = session_type
                    if session_type == 'R' and hasattr(session, 'total_laps') and session.total_laps is not None:
                         res['TotalLaps'] = session.total_laps
                    else: # Assicura che la colonna esista anche se non ci sono total_laps
                         res['TotalLaps'] = np.nan
                    all_session_results_list.append(res)
                else: logging.warning(f"No results for {year} {gp_name} {session_type}")
            except Exception as e: logging.warning(f"Failed loading {session_type} for {year} {gp_name}: {e}")
    
    if not all_session_results_list:
        logging.error("No historical race/quali results loaded. Cannot compute circuit stats."); df_all_results_for_stats = pd.DataFrame()
    else:
        df_all_sessions = pd.concat(all_session_results_list, ignore_index=True)
        
        logging.info("Processing and merging Q and R session data...")
        # Estrai dati Qualifica, usa 'Abbreviation'
        q_data_cols = ['Abbreviation', 'anno', 'gp', 'Position']
        if 'TeamName' in df_all_sessions.columns: q_data_cols.append('TeamName') # Aggiungi TeamName se esiste
        
        df_q = df_all_sessions[df_all_sessions['session_type'] == 'Q'][[col for col in q_data_cols if col in df_all_sessions.columns]].copy()
        df_q.rename(columns={'Position': 'GridPosition'}, inplace=True)
        df_q = df_q.drop_duplicates(subset=['Abbreviation', 'anno', 'gp'])
        logging.info(f"  Processed Quali data: {df_q.shape[0]} rows")

        # Estrai dati Gara, usa 'Abbreviation'
        r_data_cols = ['Abbreviation', 'anno', 'gp', 'Position', 'Laps', 'Status', 'TotalLaps', 'TeamName']
        df_r = df_all_sessions[df_all_sessions['session_type'] == 'R'][[col for col in r_data_cols if col in df_all_sessions.columns]].copy()
        df_r = df_r.drop_duplicates(subset=['Abbreviation', 'anno', 'gp'])
        logging.info(f"  Processed Race data: {df_r.shape[0]} rows")

        if not df_r.empty:
            if not df_q.empty:
                # Merge: usa i dati di Gara (df_r) come base e aggiungi GridPosition e TeamName_Q (se esiste) da df_q
                # Rinomina TeamName in df_q per evitare conflitti se presente in entrambi, anche se spesso è lo stesso
                df_q_for_merge = df_q.copy()
                if 'TeamName' in df_q_for_merge.columns:
                    df_q_for_merge.rename(columns={'TeamName': 'TeamName_Q'}, inplace=True)
                
                df_all_results_for_stats = pd.merge(
                    df_r,
                    df_q_for_merge, # Contiene Abbreviation, anno, gp, GridPosition, (TeamName_Q opzionale)
                    on=['Abbreviation', 'anno', 'gp'],
                    how='left' # Mantiene tutti i risultati di gara, aggiunge info griglia se disponibili
                )
                # Se TeamName non era in df_r ma era in df_q (ora TeamName_Q), e vogliamo usarlo
                if 'TeamName' not in df_all_results_for_stats.columns and 'TeamName_Q' in df_all_results_for_stats.columns:
                    df_all_results_for_stats.rename(columns={'TeamName_Q': 'TeamName'}, inplace=True)
                elif 'TeamName_Q' in df_all_results_for_stats.columns: # Se entrambi ci sono, TeamName (da R) ha la precedenza
                    df_all_results_for_stats.drop(columns=['TeamName_Q'], inplace=True, errors='ignore')

            else: # Solo dati di gara, non di qualifica
                df_all_results_for_stats = df_r.copy()
                df_all_results_for_stats['GridPosition'] = np.nan
            
            # Rinomina 'Abbreviation' in 'Driver' alla fine per le funzioni di stats
            df_all_results_for_stats.rename(columns={'Abbreviation': 'Driver'}, inplace=True)
        else:
            df_all_results_for_stats = pd.DataFrame() # Nessun dato di gara, non possiamo procedere
            
        if df_all_results_for_stats.empty: logging.error("Stats DataFrame empty after merge.")
        else: logging.info(f"Combined {len(df_all_results_for_stats)} entries for historical stats.")
        # *** FINE SEZIONE MODIFICATA ***

    pre_race_records = []
    # Itera sulle gare definite in schedule_for_data per creare il nostro training set
    for _, race_event in tqdm(schedule_for_data.iterrows(), total=len(schedule_for_data), desc="Building Pre-Race Records"):
        year, gp_name, rnd = int(race_event["anno"]), str(race_event["nome_gran_premio"]), int(race_event["round"])
        logging.debug(f"Event: {year} {gp_name} R{rnd}")

        current_event_quali = pd.DataFrame()
        if not df_all_results_for_stats.empty and 'GridPosition' in df_all_results_for_stats.columns and 'Driver' in df_all_results_for_stats.columns:
            current_event_quali = df_all_results_for_stats[
                (df_all_results_for_stats['anno'] == year) &
                (df_all_results_for_stats['gp'] == gp_name) &
                (df_all_results_for_stats['GridPosition'].notna())
            ].copy() # Contiene già 'Driver' e 'TeamName' (se presente)
        
        if current_event_quali.empty: logging.warning(f"No Quali data in stats_df for {year} {gp_name}. Skipping race."); continue

        current_event_race_target = pd.DataFrame()
        if not df_all_results_for_stats.empty and 'Position' in df_all_results_for_stats.columns and 'Driver' in df_all_results_for_stats.columns:
             current_event_race_target = df_all_results_for_stats[
                 (df_all_results_for_stats['anno'] == year) &
                 (df_all_results_for_stats['gp'] == gp_name)
             ][['Driver', 'Position']].set_index('Driver')['Position']
        if current_event_race_target.empty: logging.warning(f"No Race (target) data in stats_df for {year} {gp_name}. Skipping race."); continue
            
        elo_snapshot = df_elo_history[(df_elo_history['anno'] == year) & (df_elo_history['round'] == rnd)]
        if elo_snapshot.empty: logging.warning(f"No ELO for {year} R{rnd} {gp_name}. Using defaults."); elo_drv, elo_team = {}, {}
        else: elo_drv, elo_team = elo_snapshot.iloc[0]['elo_driver_pre_race'], elo_snapshot.iloc[0]['elo_team_pre_race']

        for _, q_row in current_event_quali.iterrows():
            driver_abbr = q_row.get("Driver") # Ora usiamo 'Driver' perché è stato rinominato in df_all_results_for_stats
            team_name = q_row.get("TeamName") 
            grid_pos = pd.to_numeric(q_row.get("GridPosition"), errors='coerce')

            if not driver_abbr or pd.isna(grid_pos): logging.debug(f"Skipping driver {driver_abbr} - missing info."); continue
            if not team_name: team_name = "Unknown" # Fallback se TeamName non è stato propagato correttamente

            final_pos = pd.to_numeric(current_event_race_target.get(driver_abbr), errors='coerce')
            if pd.isna(final_pos): final_pos = 99

            driver_stats = get_circuit_stats_for_driver(driver_abbr, gp_name, year, df_all_results_for_stats, YEARS_LOOKBACK_CIRCUIT_STATS)
            team_stats = get_circuit_stats_for_team(team_name, gp_name, year, df_all_results_for_stats, YEARS_LOOKBACK_CIRCUIT_STATS)
            is_new_to_circuit = (driver_stats.get(f"races_on_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y", 0) == 0)

            record = {
                "anno": year, "gp": gp_name, "round": rnd, "driver": driver_abbr, "team": team_name,
                "starting_grid_position": int(grid_pos),
                "elo_driver_pre_race": elo_drv.get(driver_abbr, 1500.0),
                "elo_team_pre_race": elo_team.get(team_name, 1500.0),
                "is_driver_new_to_circuit": is_new_to_circuit,
                "final_position": int(final_pos)
            }
            record.update(driver_stats); record.update(team_stats)
            pre_race_records.append(record)

    if not pre_race_records: logging.warning("No pre-race records generated. Output empty."); df_out = pd.DataFrame()
    else: df_out = pd.DataFrame(pre_race_records)

    cols_to_impute_pre_race = [f"avg_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                               f"avg_grid_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                               f"avg_pos_gained_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                               f"best_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                               f"laps_completed_pct_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                               f"team_avg_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                               f"team_avg_pos_gained_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y"]
    for col in cols_to_impute_pre_race:
        if col in df_out.columns:
            fill_val = 25 if ("pos_circuit" in col or "grid_pos" in col) else (0 if "gained" in col or "races_on_circuit" in col else (0.5 if "pct" in col else df_out[col].mean()))
            df_out[col] = df_out[col].fillna(fill_val if pd.notna(fill_val) else (25 if "pos" in col else 0) )
    if "is_driver_new_to_circuit" in df_out.columns: df_out["is_driver_new_to_circuit"] = df_out["is_driver_new_to_circuit"].astype(int)

    logging.info(f"Generated {len(df_out)} pre-race records.")
    try:
        expected_cols_output = ["anno", "gp", "round", "driver", "team", "starting_grid_position",
                                "elo_driver_pre_race", "elo_team_pre_race",
                                f"avg_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                                f"avg_grid_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                                f"avg_pos_gained_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                                f"races_on_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                                f"best_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                                f"laps_completed_pct_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                                f"team_avg_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                                f"team_avg_pos_gained_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
                                "is_driver_new_to_circuit", "final_position"]
        
        df_final_out = pd.DataFrame(columns=expected_cols_output)
        if not df_out.empty:
            for col in expected_cols_output:
                if col in df_out.columns: df_final_out[col] = df_out[col]
        df_final_out.to_csv(OUTPUT_PRE_RACE_DATASET_PATH, index=False)
        logging.info(f"✅ Pre-race dataset saved to {OUTPUT_PRE_RACE_DATASET_PATH} ({len(df_final_out)} righe)")
    except Exception as e: logging.error(f"Failed to save pre-race dataset: {e}")

if __name__ == "__main__":
    main()