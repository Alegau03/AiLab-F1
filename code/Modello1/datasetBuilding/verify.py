#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Columns expected to be fully imputed by build_dataset.py
# REMOVED 'hard_brakes'
COLS_IMPUTED_IN_BUILD = [
    'lap_time', 'sector1', 'sector2', 'sector3',
    'speed_avg', 'throttle_pct', 'brake_pct', 'drs_pct', 'gear_avg'
]

def validate(path):
    logging.info(f"🔍 Caricando dataset da: {path}")
    try:
        df = pd.read_csv(path)
        logging.info(f"🗃️ Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"❌ File non trovato: {path}")
        return
    except Exception as e:
        logging.error(f"❌ Errore durante il caricamento del CSV: {e}")
        return

    # --- 1) Colonne obbligatorie ---
    # REMOVED 'hard_brakes' and 'hard_brakes_diff'
    required = {
        'anno', 'gp', 'round', 'lap_number', 'driver', 'team',
        'compound', 'position', 'final_position', 'csv_pit_stop_count',
        'elo_driver', 'elo_team',
        'stint', 'lap_in_stint', 'pit_in', 'pit_out',
        'sector1', 'sector2', 'sector3',
        'lap_time', 'gap_to_leader_diff',
        'speed_avg', 'throttle_pct', 'brake_pct', 'drs_pct', 'gear_avg',
        'lap_time_diff',
        'speed_avg_diff', 'throttle_pct_diff', 'brake_pct_diff', 'drs_pct_diff',
        'gear_avg_diff',
        'lap_time_pct', 'gap_to_leader_pct',
        'track_status'
    }
    actual_cols = set(df.columns)
    miss = required - actual_cols
    extra = actual_cols - required

    print("\n--- 1. Controllo Colonne Obbligatorie ---")
    if miss:
        logging.warning(f"⚠️ Colonne obbligatorie MANCANTI: {sorted(miss)}")
    else:
        logging.info("✅ Tutte le colonne obbligatorie presenti.")
    if extra:
         logging.info(f"ℹ️ Colonne EXTRA trovate (non in quelle obbligatorie definite): {sorted(extra)}")


    # --- 2) Valori Nulli (Generale) ---
    print("\n--- 2. Valori Nulli (Top 10) ---")
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if nan_counts.empty:
        logging.info("✅ Nessun valore NaN trovato nel dataset.")
    else:
        logging.warning("⚠️ Trovati valori NaN:")
        print(nan_counts.head(10).to_string()) # Usare to_string per evitare troncamenti


    # --- 3) Controllo Imputazione Specifica ---
    print("\n--- 3. Controllo NaN post-imputazione ---")
    cols_to_check_imputation = [c for c in COLS_IMPUTED_IN_BUILD if c in df.columns]
    if not cols_to_check_imputation:
         logging.warning(f"⚠️ Nessuna colonna da controllare per l'imputazione (lista COLS_IMPUTED_IN_BUILD o colonne DataFrame vuote).")
    else:
        nan_in_imputed = df[cols_to_check_imputation].isnull().sum()
        nan_in_imputed = nan_in_imputed[nan_in_imputed > 0]
        if nan_in_imputed.empty:
            logging.info(f"✅ Nessun NaN trovato nelle colonne che dovevano essere imputate ({len(cols_to_check_imputation)} colonne).")
        else:
            logging.error(f"❌ NaN trovati in colonne che dovevano essere imputate:")
            print(nan_in_imputed.to_string())


    # --- 4) Range lap_number ---
    print("\n--- 4. Range Lap Number ---")
    if 'lap_number' in df.columns:
        bad_laps = df[~df['lap_number'].between(1, 30, inclusive='both')]
        if not bad_laps.empty:
             logging.warning(f"⚠️ {len(bad_laps)} righe con lap_number fuori dall'intervallo [1, 30].")
        else:
             logging.info("✅ Tutte le righe hanno lap_number nell'intervallo [1, 30].")
    else:
        logging.warning("⚠️ Colonna 'lap_number' non trovata per controllo range.")


    # --- 5) Duplicati (driver-lap per GP) ---
    print("\n--- 5. Duplicati Driver-Lap per GP ---")
    dup_cols = ['anno', 'gp', 'lap_number', 'driver']
    if all(c in df.columns for c in dup_cols):
        duplicates = df[df.duplicated(subset=dup_cols, keep=False)]
        num_dup_rows = len(duplicates)
        num_dup_sets = df.duplicated(subset=dup_cols).sum()
        if num_dup_rows > 0:
            logging.warning(f"⚠️ Trovati {num_dup_rows} righe duplicate (relative a {num_dup_sets} set duplicati) per {dup_cols}.")
        else:
            logging.info(f"✅ Nessun duplicato trovato per {dup_cols}.")
    else:
        logging.warning(f"⚠️ Impossibile controllare duplicati, colonne mancanti: {[c for c in dup_cols if c not in df.columns]}")


    # --- 6) Leader per ogni gp+lap ---
    print("\n--- 6. Presenza Leader (Position=1) per GP-Lap ---")
    group_cols = ['anno', 'gp', 'lap_number']
    if all(c in df.columns for c in group_cols + ['position']):
        # Assicurati che 'position' sia numerica per il confronto con 1
        df['position_numeric_val'] = pd.to_numeric(df['position'], errors='coerce')
        grp = df.groupby(group_cols)
        missing_leader_groups = grp.filter(lambda g: not (g['position_numeric_val'] == 1).any())
        if not missing_leader_groups.empty:
             unique_missing = missing_leader_groups[group_cols].drop_duplicates().values.tolist()
             logging.warning(f"⚠️ {len(unique_missing)} combinazioni GP-Lap non hanno un pilota con position==1.")
        else:
             logging.info("✅ Ogni combinazione GP-Lap ha almeno un pilota con position==1.")
        df.drop(columns=['position_numeric_val'], inplace=True, errors='ignore') # Pulisci colonna temporanea
    else:
        logging.warning(f"⚠️ Impossibile controllare presenza leader, colonne mancanti per il raggruppamento o 'position'.")


    # --- 7) Diff==0 per leader ---
    print("\n--- 7. Controllo Differenze per Leader ---")
    diff_cols = sorted([c for c in df.columns if c.endswith('_diff')])
    if 'position' in df.columns and diff_cols:
        # Assicurati che 'position' sia numerica per il confronto
        df['position_numeric_val'] = pd.to_numeric(df['position'], errors='coerce')
        leaders = df[df['position_numeric_val'] == 1].copy()
        
        if not leaders.empty:
            non_zero_diffs = leaders[diff_cols].abs() > 1e-6
            leaders['any_non_zero_diff'] = non_zero_diffs.any(axis=1)
            lead_with_nonzero_diff_count = leaders['any_non_zero_diff'].sum()

            if lead_with_nonzero_diff_count > 0:
                logging.warning(f"⚠️ {lead_with_nonzero_diff_count} righe leader hanno almeno una colonna '_diff' != 0 (tolleranza 1e-6).")
            else:
                logging.info("✅ Tutti i leader hanno colonne '_diff' ≈ 0.")
        else:
            logging.info("ℹ️ Nessuna riga leader (position==1) trovata per controllare le differenze.")
        df.drop(columns=['position_numeric_val'], inplace=True, errors='ignore')
    else:
         logging.warning(f"⚠️ Impossibile controllare differenze leader ('position' o colonne '_diff' mancanti).")


    # --- 8) Percentili in [0,1] ---
    print("\n--- 8. Range Percentili ---")
    pct_cols = [c for c in df.columns if c.endswith('_pct')]
    all_pct_ok = True
    if not pct_cols:
        logging.info("ℹ️ Nessuna colonna percentile (*_pct) trovata.")
    else:
        for pct in pct_cols:
            if pct not in df.columns: logging.warning(f"⚠️ Colonna percentile {pct} non trovata."); all_pct_ok = False; continue
            
            nan_pct = df[pct].isna().sum()
            if nan_pct > 0: logging.warning(f"⚠️ {nan_pct} valori NaN trovati in {pct}."); all_pct_ok = False
            
            # Controlla il range escludendo i NaN
            not_na_mask = df[pct].notna()
            if not_na_mask.any(): # Solo se ci sono valori non-NaN
                oob = df.loc[not_na_mask, pct].between(0, 1, inclusive='both')
                num_oob = len(oob) - oob.sum()
                if num_oob > 0: logging.warning(f"⚠️ {num_oob} valori fuori [0, 1] in {pct}."); all_pct_ok = False
        if all_pct_ok: logging.info(f"✅ Valori colonne percentili ({len(pct_cols)}) in [0, 1] (o NaN).")


    # --- 9) Statistiche Elo ---
    print("\n--- 9. Statistiche ELO ---")
    for col in ['elo_driver', 'elo_team']:
        if col in df.columns:
            logging.info(f"📐 {col} summary:")
            print(df[col].describe().to_string())
        else:
            logging.warning(f"⚠️ Colonna ELO '{col}' non trovata.")


    # --- 10) Statistiche Continue Principali ---
    print("\n--- 10. Statistiche Feature Continue ---")
    # Usa la lista aggiornata COLS_IMPUTED_IN_BUILD
    cont_cols_present = [c for c in COLS_IMPUTED_IN_BUILD if c in df.columns]
    if cont_cols_present:
        print(df[cont_cols_present].describe().T[['min', 'mean', 'max', 'std']].to_string())
    else:
        logging.warning(f"⚠️ Nessuna colonna continua da COLS_IMPUTED_IN_BUILD trovata per statistiche.")


    # --- 11) Settori >= 0 ---
    print("\n--- 11. Controllo Settori Non Negativi ---")
    sector_cols = ['sector1', 'sector2', 'sector3']
    all_sectors_ok = True
    for s_col in sector_cols:
        if s_col in df.columns:
            # Assicurati che la colonna sia numerica prima del confronto
            if pd.api.types.is_numeric_dtype(df[s_col]):
                neg_val_count = (df[s_col] < 0).sum() # < 0; 0 è un tempo valido sebbene improbabile
                if neg_val_count > 0:
                    logging.warning(f"⚠️ {neg_val_count} righe con {s_col} < 0.")
                    all_sectors_ok = False
                nan_count = df[s_col].isna().sum()
                if nan_count > 0: # Non dovrebbe accadere dopo l'imputazione
                     logging.error(f"❌ {nan_count} righe con {s_col} NaN dopo l'imputazione!")
                     all_sectors_ok = False
            else:
                logging.warning(f"⚠️ Colonna settore {s_col} non è numerica.")
                all_sectors_ok = False
        else:
            logging.warning(f"⚠️ Colonna settore {s_col} non trovata.")
            all_sectors_ok = False # Considera errore se manca colonna settore attesa
    if all_sectors_ok: logging.info("✅ Valori settori >= 0 (o colonna mancante/non numerica).")


    # --- 12) Controllo Categorie ---
    print("\n--- 12. Statistiche Feature Categoriche ---")
    cat_checks = {'driver': 'nunique', 'team': 'nunique', 'compound': 'value_counts', 'track_status': 'value_counts'}
    for col, check_type in cat_checks.items():
        if col in df.columns:
            if check_type == 'nunique': logging.info(f"🏷️ {col} unici: {df[col].nunique()}")
            elif check_type == 'value_counts':
                logging.info(f"📊 Distribuzione {col}:")
                print(df[col].value_counts(dropna=False).to_string())
        else: logging.warning(f"⚠️ Colonna categorica '{col}' non trovata.")


    # --- 13) Controllo Pit Flags ---
    print("\n--- 13. Statistiche Flag Pit Stop ---")
    for col in ['pit_in', 'pit_out']:
         if col in df.columns:
              # Converte in booleano prima di value_counts per uniformità
              try:
                  # Tentativo di conversione robusta a booleano
                  bool_series = df[col].copy()
                  if pd.api.types.is_object_dtype(bool_series.dtype) or pd.api.types.is_string_dtype(bool_series.dtype):
                      bool_series = bool_series.str.lower().map({'true': True, 'false': False, '1':True, '0':False, 'sì':True, 'no':False})
                  bool_series = bool_series.astype(bool)
                  counts = bool_series.value_counts(dropna=False)
              except Exception: # Fallback se la conversione fallisce
                  counts = df[col].value_counts(dropna=False)

              logging.info(f"🏁 Conteggio {col}:")
              print(counts.to_string())
         else: logging.warning(f"⚠️ Colonna pit flag '{col}' non trovata.")

    print("\n\n✅ Controllo completato.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python validate_dataset.py <path_dataset.csv>")
        sys.exit(1)
    validate(sys.argv[1])