# utils/elo.py (MODIFICATO - Corretto session.load)

import fastf1
import pandas as pd
from collections import defaultdict
import logging
import numpy as np

# Configura logging per questo modulo se vuoi messaggi specifici da elo.py
logger_elo = logging.getLogger(__name__)



# --- Iper-parametri ELO ---
K_BASE       = 32
K_MIN, K_MAX = 10, 40
DECAY        = 0.02
GAP_SCALER   = 400.0
DNF_PENALTY  = 25

# Funzione di utilità per calcolare il punteggio atteso
def expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

# Funzione per calcolare il fattore K
# Il fattore K determina quanto il punteggio ELO di un giocatore può cambiare dopo una partita

def k_factor(rating: float, race_idx: int, total_races_in_season: int = 22) -> float:
    ratio = (race_idx / total_races_in_season) if total_races_in_season > 0 else 0
    base = K_BASE * (1.0 - 0.5 * ratio)
    return max(K_MIN, min(K_MAX, base))

# Funzione per aggiornare il punteggio ELO
def update_rating(r_a: float, r_b: float, score: float, gap: float, race_idx: int, total_races_in_season: int) -> float:
    exp = expected(r_a, r_b)
    k   = k_factor(r_a, race_idx, total_races_in_season)
    g_factor = 1.0
    if pd.notna(gap) and GAP_SCALER != 0: # Aggiunto controllo per GAP_SCALER !=0
        g_factor = 1.0 + (abs(gap) / GAP_SCALER)
    return r_a + k * g_factor * (score - exp)

# Funzione per calcolare la storia ELO
# Questa funzione calcola la storia ELO per i piloti e le squadre in base ai risultati delle gare
def compute_elo_history(schedule_df: pd.DataFrame, initial_elo_driver=None, initial_elo_team=None):
    elo_driver = defaultdict(lambda: 1500.0)
    if initial_elo_driver:
        for driver, rating in initial_elo_driver.items(): elo_driver[driver] = rating
            
    elo_team = defaultdict(lambda: 1500.0)
    if initial_elo_team:
        for team, rating in initial_elo_team.items(): elo_team[team] = rating

    elo_history = []
    sched = schedule_df.sort_values(["anno", "round"]).copy()
    max_rounds_per_year = sched.groupby("anno")["round"].max().to_dict()
    processed_years = set()

    for _, row in sched.iterrows():
        year = int(row["anno"])
        gp   = str(row["nome_gran_premio"])
        rnd  = int(row["round"])
        total_races_this_season = max_rounds_per_year.get(year, 22)

        if year not in processed_years and processed_years:
            prev_year = max(processed_years)
            logger_elo.info(f"Applying end-of-season ELO decay for year {prev_year} before starting {year}")
            for d_key in list(elo_driver.keys()): elo_driver[d_key] *= (1.0 - DECAY)
            for t_key in list(elo_team.keys()): elo_team[t_key] *= (1.0 - DECAY)
        processed_years.add(year)

        elo_history.append({
            "anno": year, "round": rnd, "gp": gp,
            "elo_driver_pre_race": dict(elo_driver),
            "elo_team_pre_race": dict(elo_team)
        })

        logger_elo.info(f"Processing ELO for: {year} {gp} (Round {rnd}/{total_races_this_season})")
        
        try:
            sess = fastf1.get_session(year, gp, "R")

            sess.load(laps=True, telemetry=False, weather=False, messages=False) # laps=True per aiutare a caricare i risultati
        except Exception as e:
            logger_elo.warning(f"FastF1: Could not load session for {year} {gp}: {e}. Skipping ELO update.")
            continue

        if sess.results is None or sess.results.empty:
            logger_elo.warning(f"FastF1: No results for {year} {gp}. Skipping ELO update.")
            continue

        results = sess.results.sort_values(by="Position").reset_index(drop=True)
        if results.empty:
            logger_elo.warning(f"Results empty for {year} {gp}. Skipping ELO update.")
            continue

        win_sec = np.nan
        if 'Time' in results.columns and not results.empty and pd.notna(results.iloc[0].get("Time")): # Usato .get per sicurezza
            win_sec = results.iloc[0]["Time"].total_seconds()
        else:
            logger_elo.warning(f"Winner's time missing for {year} {gp}. Gap calculations will be affected.")

        n = len(results)
        for i in range(n):
            ri = results.iloc[i]
            di, ti = ri.get("Abbreviation"), ri.get("TeamName")
            pi = pd.to_numeric(ri.get("Position"), errors='coerce')

            if not di or not ti or pd.isna(pi):
                logger_elo.debug(f"Skipping ELO update for entry index {i} in {year} {gp} (Driver: {di}, Team: {ti}, Pos: {pi})")
                continue
            
            gap_i_val = np.nan
            if pd.notna(win_sec) and 'Time' in ri and pd.notna(ri.get("Time")):
                 gap_i_val = ri.get("Time").total_seconds() - win_sec
            elif pd.notna(win_sec):
                 gap_i_val = 90.0 + float(pi) * 2
            
            for j in range(i + 1, n):
                rj = results.iloc[j]
                dj, tj = rj.get("Abbreviation"), rj.get("TeamName")
                pj = pd.to_numeric(rj.get("Position"), errors='coerce')

                if not dj or not tj or pd.isna(pj):
                    logger_elo.debug(f"Skipping ELO pair for entry index {j} in {year} {gp}.")
                    continue
                
                gap_j_val = np.nan
                if pd.notna(win_sec) and 'Time' in rj and pd.notna(rj.get("Time")):
                    gap_j_val = rj.get("Time").total_seconds() - win_sec
                elif pd.notna(win_sec):
                    gap_j_val = 90.0 + float(pj) * 2
                
                diff_gap_ij, diff_gap_ji = 0.0, 0.0
                if pd.notna(gap_i_val) and pd.notna(gap_j_val):
                     diff_gap_ij = gap_i_val - gap_j_val
                     diff_gap_ji = gap_j_val - gap_i_val
                
                s_i = 1.0 if pi < pj else (0.0 if pi > pj else 0.5)
                s_j = 1.0 - s_i

                elo_driver[di] = update_rating(elo_driver[di], elo_driver[dj], s_i, diff_gap_ij, rnd, total_races_this_season)
                elo_driver[dj] = update_rating(elo_driver[dj], elo_driver[di], s_j, diff_gap_ji, rnd, total_races_this_season)
                elo_team[ti] = update_rating(elo_team[ti], elo_team[tj], s_i, diff_gap_ij, rnd, total_races_this_season)
                elo_team[tj] = update_rating(elo_team[tj], elo_team[ti], s_j, diff_gap_ji, rnd, total_races_this_season)

            status = str(ri.get("Status", "")).lower()
            mechanical_dnf_keywords = ["engine", "power unit", "gearbox", "hydraulics", "suspension", "electronics", "transmission", "clutch", "fuel system", "oil leak", "water leak", "electrical"]
            is_mechanical_dnf = any(keyword in status for keyword in mechanical_dnf_keywords)
            is_incident_dnf = "accident" in status or "collision" in status or "spun off" in status
            
            if "retired" in status and not is_incident_dnf or is_mechanical_dnf :
                if di: elo_driver[di] -= DNF_PENALTY
                if ti: elo_team[ti]   -= DNF_PENALTY / 2
    
    if processed_years:
        last_processed_year = max(processed_years)
        logger_elo.info(f"Applying final ELO decay for year {last_processed_year}.")
        for d_key in list(elo_driver.keys()): elo_driver[d_key] *= (1.0 - DECAY)
        for t_key in list(elo_team.keys()): elo_team[t_key] *= (1.0 - DECAY)

    final_elo_drivers = dict(elo_driver)
    final_elo_teams = dict(elo_team)

    return final_elo_drivers, final_elo_teams, pd.DataFrame(elo_history) if elo_history else pd.DataFrame()

# Funzione per calcolare l'ELO dinamico
def compute_dynamic_elo(schedule_df: pd.DataFrame):
    final_elo_drivers, final_elo_teams, _ = compute_elo_history(schedule_df)
    return final_elo_drivers, final_elo_teams