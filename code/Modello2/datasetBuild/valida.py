#!/usr/bin/env python3
# validate_pre_race_dataset.py

import sys
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Definisci qui quanti anni di lookback hai usato per le stats, per la validazione dei nomi colonna
YEARS_LOOKBACK_CIRCUIT_STATS = 2 # DEVE CORRISPONDERE A QUELLO USATO IN build_dataset_pre_race.py

def validate_pre_race(path: str):
    logging.info(f"🔍 Caricando dataset pre-gara da: {path}")
    try:
        df = pd.read_csv(path)
        logging.info(f"🗃️ Shape: {df.shape}")
        if df.empty:
            logging.warning("⚠️ Il dataset è vuoto. La validazione avrà un output limitato.")
            # Non terminare, ma molti controlli non avranno senso.
    except FileNotFoundError:
        logging.error(f"❌ File non trovato: {path}")
        return
    except Exception as e:
        logging.error(f"❌ Errore durante il caricamento del CSV: {e}")
        return

    # --- 1) Colonne Obbligatorie per il Dataset Pre-Gara ---
    required_core = {
        'anno', 'gp', 'round', 'driver', 'team', 'starting_grid_position',
        'elo_driver_pre_race', 'elo_team_pre_race',
        'is_driver_new_to_circuit',
        'final_position' # Target
    }
    # Colonne statistiche attese (i nomi dipendono da YEARS_LOOKBACK_CIRCUIT_STATS)
    required_stats = {
        f"avg_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
        f"avg_grid_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
        f"avg_pos_gained_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
        f"races_on_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
        f"best_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
        f"laps_completed_pct_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
        f"team_avg_finish_pos_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
        f"team_avg_pos_gained_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y",
    }
    required = required_core.union(required_stats)
    
    actual_cols = set(df.columns)
    miss = required - actual_cols
    extra = actual_cols - required # Utile per debug

    print("\n--- 1. Controllo Colonne Obbligatorie (Pre-Gara) ---")
    if miss:
        logging.warning(f"⚠️ Colonne obbligatorie MANCANTI: {sorted(list(miss))}")
    else:
        logging.info("✅ Tutte le colonne obbligatorie pre-gara presenti.")
    if extra:
         logging.info(f"ℹ️ Colonne EXTRA trovate (non in quelle obbligatorie definite): {sorted(list(extra))}")


    # --- 2) Valori Nulli (Generale) ---
    # Per un dataset pronto per il training, idealmente non ci sono NaN
    print("\n--- 2. Valori Nulli ---")
    if df.empty:
        logging.info("ℹ️ Dataset vuoto, controllo NaN saltato.")
    else:
        nan_counts = df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        if nan_counts.empty:
            logging.info("✅ Nessun valore NaN trovato nel dataset pre-gara.")
        else:
            logging.warning("⚠️ Trovati valori NaN nel dataset pre-gara:")
            print(nan_counts.to_string())


    # --- 3) Range Posizioni di Partenza e Finali ---
    print("\n--- 3. Range Posizioni di Partenza e Finali ---")
    if df.empty:
        logging.info("ℹ️ Dataset vuoto, controllo range posizioni saltato.")
    else:
        all_positions_ok = True
        if 'starting_grid_position' in df.columns:
            # Posizione di partenza tipicamente 1-20, a volte più alta per penalità o pitlane start
            # Permettiamo un range leggermente più ampio, ma valori negativi o > 30 sono sospetti
            invalid_grid_pos = df[~df['starting_grid_position'].between(1, 30, inclusive='both')] 
            if not invalid_grid_pos.empty:
                logging.warning(f"⚠️ {len(invalid_grid_pos)} righe con 'starting_grid_position' fuori [1, 30]. Esempi: {invalid_grid_pos['starting_grid_position'].unique()[:5]}")
                all_positions_ok = False
        else: logging.warning("⚠️ Colonna 'starting_grid_position' mancante."); all_positions_ok = False

        if 'final_position' in df.columns:
            # La posizione finale (target) potrebbe includere 99 per DNF, come l'abbiamo definita.
            # Valori < 1 (a parte 0 se usato per DNF specifici) o > 99 sono strani.
            # Se 99 è un DNF valido, il range è 1-2X e 99.
            # Per ora, controlliamo che non ci siano valori negativi o estremamente alti (es. >100)
            invalid_final_pos = df[(df['final_position'] < 1) | (df['final_position'] > 99)]
            if not invalid_final_pos.empty:
                logging.warning(f"⚠️ {len(invalid_final_pos)} righe con 'final_position' < 1 o > 99. Esempi: {invalid_final_pos['final_position'].unique()[:5]}")
                all_positions_ok = False
        else: logging.warning("⚠️ Colonna 'final_position' (TARGET) mancante."); all_positions_ok = False
        
        if all_positions_ok: logging.info("✅ Range posizioni di partenza e finali sembrano OK.")


    # --- 4) Statistiche Elo Pre-Gara ---
    print("\n--- 4. Statistiche ELO Pre-Gara ---")
    if df.empty:
        logging.info("ℹ️ Dataset vuoto, controllo Elo saltato.")
    else:
        for col in ['elo_driver_pre_race', 'elo_team_pre_race']:
            if col in df.columns:
                logging.info(f"📐 {col} summary:")
                print(df[col].describe().to_string())
                if (df[col] == 1500.0).all(): # Controlla se tutti gli Elo sono rimasti al default
                    logging.warning(f"    ⚠️ ATTENZIONE: Tutti i valori in '{col}' sono 1500.0 (default). L'Elo non sembra essere stato calcolato/applicato correttamente.")
            else:
                logging.warning(f"⚠️ Colonna ELO '{col}' non trovata.")

    # --- 5) Statistiche Feature Storiche Circuito ---
    print("\n--- 5. Statistiche Feature Storiche Circuito ---")
    if df.empty:
        logging.info("ℹ️ Dataset vuoto, controllo stats circuito saltato.")
    else:
        stats_cols = [col for col in df.columns if "circuit_prev" in col]
        if not stats_cols:
            logging.warning("⚠️ Nessuna colonna di statistiche storiche del circuito trovata.")
        else:
            logging.info(f"Descrittive per {len(stats_cols)} colonne statistiche circuito:")
            print(df[stats_cols].describe().T[['min', 'mean', 'max', 'std', 'count']].to_string())
            # Controlla se alcune colonne di stats sono interamente NaN o 0 (potrebbe indicare problemi nel calcolo)
            for scol in stats_cols:
                if df[scol].isna().all():
                    logging.warning(f"    ⚠️ Colonna '{scol}' contiene solo NaN.")
                elif (df[scol] == 0).all() and "races_on_circuit" not in scol and "gained" not in scol : # 0 può essere valido per races o pos_gained
                    logging.warning(f"    ⚠️ Colonna '{scol}' contiene solo 0.")
    
    # --- 6) Controllo 'is_driver_new_to_circuit' ---
    print("\n--- 6. Controllo Flag 'is_driver_new_to_circuit' ---")
    if df.empty:
        logging.info("ℹ️ Dataset vuoto, controllo flag saltato.")
    elif 'is_driver_new_to_circuit' in df.columns:
        logging.info(f"📊 Distribuzione 'is_driver_new_to_circuit':")
        print(df['is_driver_new_to_circuit'].value_counts(dropna=False).to_string())
        # Verifica coerenza con 'races_on_circuit_prevXy'
        races_col = f"races_on_circuit_prev{YEARS_LOOKBACK_CIRCUIT_STATS}y"
        if races_col in df.columns:
            inconsistencies = df[
                (df['is_driver_new_to_circuit'] == True) & (df[races_col] > 0) |
                (df['is_driver_new_to_circuit'] == False) & (df[races_col] == 0)
            ]
            if not inconsistencies.empty:
                logging.warning(f"⚠️ {len(inconsistencies)} inconsistenze trovate tra 'is_driver_new_to_circuit' e '{races_col}'.")
            else:
                logging.info(f"✅ Coerenza OK tra 'is_driver_new_to_circuit' e '{races_col}'.")
        else:
            logging.warning(f"⚠️ Colonna '{races_col}' non trovata per controllo coerenza flag 'new_to_circuit'.")
    else:
        logging.warning("⚠️ Colonna 'is_driver_new_to_circuit' non trovata.")

    # --- 7) Controllo Anni e Round ---
    print("\n--- 7. Controllo Anni e Round ---")
    if df.empty:
        logging.info("ℹ️ Dataset vuoto, controllo anni/round saltato.")
    else:
        if 'anno' in df.columns: logging.info(f"Anni presenti: {sorted(df['anno'].unique())}")
        else: logging.warning("⚠️ Colonna 'anno' mancante.")
        if 'round' in df.columns: logging.info(f"Round unici (per debug, potrebbero ripetersi tra anni): {sorted(df['round'].unique())}")
        else: logging.warning("⚠️ Colonna 'round' mancante.")

    print("\n\n✅ Validazione dataset pre-gara completata.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python validate_pre_race_dataset.py <path_dataset.csv>")
        sys.exit(1)
    
    # Imposta YEARS_LOOKBACK_CIRCUIT_STATS globalmente se necessario,
    # o passalo come argomento a validate_pre_race se preferisci
    # Per ora è una costante globale nello script.
    validate_pre_race(sys.argv[1])