#!/usr/bin/env python
# generate_diagnostic_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.pipeline import Pipeline 
import traceback # Aggiunto per debug

# --- CONFIGURAZIONE GLOBALE (DEVE CORRISPONDERE AGLI SCRIPT DI TRAINING) ---
MODELS_DIR = Path("models")
DATA_DIR = Path("data") # Assicurati che DATA_DIR sia definito
DIAGRAMS_DIR = Path("diagrams")
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42 
TARGET_COL = "final_position"
GROUP_COL_M1 = "gp" 
GROUP_COLS_M2 = ['anno', 'gp'] 

# --- Percorsi ai dataset (DEVONO ESSERE DEFINITI QUI) ---
HISTORICAL_DATA_PATH = DATA_DIR / "training_dataset_COMPLETO.csv" # Per Modello 1
YEAR_CURRENT_SEASON_M1 = 2025 # Assumendo che questo sia l'anno per i dati M1 aggiuntivi
CURRENT_SEASON_DISPUTED_M1_DATA_PATH = DATA_DIR / f"training_dataset_{YEAR_CURRENT_SEASON_M1}.csv" # Per Modello 1

DATASET_M2_PATH = DATA_DIR / "pre_race_prediction_dataset.csv" # Per Modello 2


# --- Configurazioni per MODELLO 1 (Post-Giri) ---
# PREDICTIONS_M1_FILE_PATH non è più necessario se generiamo le predizioni qui
LGBM_M1_PIPELINE_PATH = MODELS_DIR / "lgbm_pipeline.joblib"
CATBOOST_M1_PIPELINE_PATH = MODELS_DIR / "catboost_pipeline.joblib"
MODEL_M1_NAME = 'Ensemble (LGBM+CB) - Modello 1' # Nome per i titoli dei grafici
NUM_COLS_M1 = [ "lap_time", "lap_time_diff", "lap_time_pct", "gap_to_leader_diff", "gap_to_leader_pct", "speed_avg", "throttle_pct", "brake_pct", "drs_pct", "gear_avg", "sector1", "sector2", "sector3", "stint", "lap_in_stint", "elo_driver", "elo_team", "position", "lap_number" ]
CAT_COLS_M1 = [ "team", "driver", "compound", "pit_in", "pit_out", "track_status" ]
ORIGINAL_FEATURES_M1 = NUM_COLS_M1 + CAT_COLS_M1

# --- Configurazioni per MODELLO 2 (Pre-Gara) ---
# PREDICTIONS_M2_FILE_PATH non è più necessario se generiamo le predizioni qui
CATBOOST_M2_PIPELINE_PATH = MODELS_DIR / "pre_race_catboost_pipeline.joblib"
MODEL_M2_NAME = 'CatBoost (Modello 2 Pre-Gara)' # Nome per i titoli
YEARS_LOOKBACK_M2 = 2 
NUM_COLS_M2 = [ "starting_grid_position", "elo_driver_pre_race", "elo_team_pre_race", f"avg_finish_pos_circuit_prev{YEARS_LOOKBACK_M2}y", f"avg_grid_pos_circuit_prev{YEARS_LOOKBACK_M2}y", f"avg_pos_gained_circuit_prev{YEARS_LOOKBACK_M2}y", f"races_on_circuit_prev{YEARS_LOOKBACK_M2}y", f"best_finish_pos_circuit_prev{YEARS_LOOKBACK_M2}y", f"laps_completed_pct_circuit_prev{YEARS_LOOKBACK_M2}y", f"team_avg_finish_pos_circuit_prev{YEARS_LOOKBACK_M2}y", f"team_avg_pos_gained_circuit_prev{YEARS_LOOKBACK_M2}y" ]
CAT_COLS_M2 = [ "driver", "team", "gp", "is_driver_new_to_circuit" ]
ORIGINAL_FEATURES_M2 = NUM_COLS_M2 + CAT_COLS_M2

# Importa to_str_type per caricare le pipeline CatBoost
try:
    from utils_gradio import to_str_type
    print("✅ Funzione 'to_str_type' importata da utils_gradio.py")
except ImportError:
    print("⚠️ WARNING: utils_gradio.py non trovato. Definizione locale di to_str_type.")
    def to_str_type(X_input): # Fallback
        if isinstance(X_input, pd.Series): return X_input.astype(str)
        elif isinstance(X_input, pd.DataFrame):
            X_output = X_input.copy();
            for col in X_output.columns: X_output[col] = X_output[col].astype(str)
            return X_output
        elif isinstance(X_input, np.ndarray): return X_input.astype(str)
        else: print(f"WARNING: to_str_type tipo non atteso: {type(X_input)}"); return X_input

# --- Funzioni di Plot (rimangono invariate, come te le ho fornite prima) ---
def plot_scatter_predicted_vs_actual(df_preds, y_true_col, y_pred_col, model_name, output_path):
    # ... (codice della funzione come prima) ...
    if y_true_col not in df_preds.columns or y_pred_col not in df_preds.columns:
        print(f"Errore Scatter: Colonne '{y_true_col}' o '{y_pred_col}' non trovate per {model_name}.")
        return
    df_plot = df_preds[[y_true_col, y_pred_col]].dropna()
    if df_plot.empty: print(f"Nessun dato per scatter plot ({model_name})."); return
    plt.figure(figsize=(8, 8)); sns.set_theme(style="whitegrid")
    plt.scatter(df_plot[y_true_col], df_plot[y_pred_col], alpha=0.6, edgecolors="w", linewidth=0.5)
    min_val_plot = min(df_plot[y_true_col].min(), df_plot[y_pred_col].min()) -1
    max_val_plot = max(df_plot[y_true_col].max(), df_plot[y_pred_col].max()) +1
    plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', lw=2, label="Predizione Perfetta")
    plt.title(f'Predetto vs. Reale - {model_name}', fontsize=16)
    plt.xlabel('Posizione Finale Reale', fontsize=12); plt.ylabel('Posizione Finale Predetta', fontsize=12)
    
    # Gestione ticks per evitare sovraffollamento
    tick_min = int(np.floor(min_val_plot))
    tick_max = int(np.ceil(max_val_plot))
    ticks = np.arange(tick_min, tick_max + 1, 1)
    if len(ticks) > 20: # Se ci sono troppi tick (es. >20 posizioni), usa step 2
        ticks = np.arange(tick_min, tick_max + 1, 2)
    if len(ticks) > 30: # Se ancora troppi, usa step 5
         ticks = np.arange(tick_min, tick_max + 1, 5)

    plt.xticks(ticks); plt.yticks(ticks)
    plt.xlim(min_val_plot + 0.5, max_val_plot - 0.5); plt.ylim(min_val_plot + 0.5, max_val_plot - 0.5)
    
    # Inverti assi per visualizzazione classifica
    current_xlim = plt.xlim()
    current_ylim = plt.ylim()
    if current_xlim[0] < current_xlim[1] : plt.xlim(current_xlim[1], current_xlim[0]) # Inverti se necessario
    if current_ylim[0] < current_ylim[1] : plt.ylim(current_ylim[1], current_ylim[0]) # Inverti se necessario

    plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
    try: plt.savefig(output_path, dpi=300); print(f"Grafico scatter salvato: {output_path}"); plt.close()
    except Exception as e: print(f"Errore salvataggio scatter: {e}")


def plot_residuals_histogram(df_preds, y_true_col, y_pred_col, model_name, output_path):
    # ... (codice della funzione come prima) ...
    if y_true_col not in df_preds.columns or y_pred_col not in df_preds.columns:
        print(f"Errore Residui: Colonne '{y_true_col}' o '{y_pred_col}' non trovate per {model_name}.")
        return
    df_plot = df_preds[[y_true_col, y_pred_col]].dropna()
    if df_plot.empty: print(f"Nessun dato per istogramma residui ({model_name})."); return
    residuals = df_plot[y_pred_col] - df_plot[y_true_col]
    plt.figure(figsize=(10, 6)); sns.set_theme(style="whitegrid")
    # Calcola il numero di bin in modo più robusto
    num_unique_residuals = residuals.nunique()
    bins = 'auto' # Lascia che seaborn/numpy scelgano, o imposta un numero ragionevole
    if num_unique_residuals < 10: bins = num_unique_residuals # Se pochi valori unici, usa quelli
    elif num_unique_residuals > 50 : bins = 50 # Limita il numero massimo di bin

    sns.histplot(residuals, kde=True, bins=bins)
    mean_error = residuals.mean(); std_error = residuals.std()
    plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=1.5, label=f'Errore Medio: {mean_error:.2f} (Std: {std_error:.2f})')
    plt.axvline(0, color='k', linestyle='solid', linewidth=1)
    plt.title(f'Distribuzione Errori (Residui) - {model_name}', fontsize=16)
    plt.xlabel('Errore di Predizione (Predetto - Reale)', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12); plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    try: plt.savefig(output_path, dpi=300); print(f"Grafico residui salvato: {output_path}"); plt.close()
    except Exception as e: print(f"Errore salvataggio istogramma residui: {e}")

def plot_feature_importance(model_pipeline, original_feature_names, model_name_title, output_path, top_n=15):
    # ... (codice della funzione come prima) ...
    if model_pipeline is None: print(f"Modello '{model_name_title}' non fornito."); return
    try:
        model = model_pipeline.named_steps['regressor'] if isinstance(model_pipeline, Pipeline) and 'regressor' in model_pipeline.named_steps else model_pipeline
        if not hasattr(model, 'feature_importances_'): print(f"Modello '{model_name_title}' non ha 'feature_importances_'."); return
        importances = model.feature_importances_
        
        transformed_feature_names = None
        # Prova a ottenere i nomi dal preprocessor, se fallisce usa altri metodi
        if isinstance(model_pipeline, Pipeline) and 'preprocess' in model_pipeline.named_steps:
            preprocessor = model_pipeline.named_steps['preprocess']
            if hasattr(preprocessor, 'get_feature_names_out'):
                try: transformed_feature_names = preprocessor.get_feature_names_out(input_features=original_feature_names)
                except Exception as e_fn: print(f"  Warning (get_feature_names_out) per {model_name_title}: {e_fn}")
        
        if transformed_feature_names is None or len(transformed_feature_names) != len(importances):
            if hasattr(model, 'feature_names_') and len(model.feature_names_) == len(importances): # CatBoost spesso ha questo
                transformed_feature_names = model.feature_names_
            elif len(original_feature_names) == len(importances) and not (isinstance(model_pipeline, Pipeline) and 'preprocess' in model_pipeline.named_steps and hasattr(model_pipeline.named_steps['preprocess'].named_transformers_.get('cat'), 'categories_')):
                 transformed_feature_names = original_feature_names # Se non c'è OHE che espande
            else: # Fallback
                print(f"  Warning: Mismatch/mancanza nomi feature per {model_name_title} ({len(importances)} importances). Uso generici.");
                transformed_feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        fi_df = pd.DataFrame({'feature': transformed_feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).head(top_n)
        plt.figure(figsize=(12, max(6, top_n * 0.5))); sns.set_theme(style="whitegrid") # Aumentato width
        sns.barplot(x='importance', y='feature', data=fi_df, palette="viridis_r", hue='feature', dodge=False, legend=False)
        plt.title(f'Top {top_n} Feature Importance - {model_name_title}', fontsize=16)
        plt.xlabel('Importanza', fontsize=12); plt.ylabel('Feature', fontsize=12); plt.tight_layout()
        try: plt.savefig(output_path, dpi=300); print(f"Grafico FI salvato: {output_path}"); plt.close()
        except Exception as e: print(f"Errore salvataggio FI: {e}")
    except Exception as e: print(f"Errore generale FI per {model_name_title}: {e}"); traceback.print_exc()


# === MAIN SCRIPT LOGIC ===
if __name__ == "__main__":
    # --- Preparazione Dati e Modelli per MODELLO 1 ---
    print("\n--- Preparazione per Grafici Modello 1 ---")
    df_list_m1 = []
    if HISTORICAL_DATA_PATH.exists():
        try: df_list_m1.append(pd.read_csv(HISTORICAL_DATA_PATH))
        except Exception as e: print(f"Errore caricamento {HISTORICAL_DATA_PATH}: {e}")
    else: print(f"File storico M1 non trovato: {HISTORICAL_DATA_PATH}")

    if CURRENT_SEASON_DISPUTED_M1_DATA_PATH.exists():
        try:
            df_curr = pd.read_csv(CURRENT_SEASON_DISPUTED_M1_DATA_PATH)
            if not df_curr.empty: df_list_m1.append(df_curr)
        except Exception as e: print(f"Errore caricamento {CURRENT_SEASON_DISPUTED_M1_DATA_PATH}: {e}")
    else: print(f"File stagione corrente M1 non trovato: {CURRENT_SEASON_DISPUTED_M1_DATA_PATH}")
    
    if not df_list_m1:
        print(f"Nessun dataset M1 caricato. Impossibile generare grafici per M1.")
    else:
        df_m1_full = pd.concat(df_list_m1, ignore_index=True)
        print(f"Dataset M1 combinato per grafici: {df_m1_full.shape}")
        
        # Coercizione tipi minima (assumendo che il grosso sia fatto negli script di build)
        for col in CAT_COLS_M1 + ['gp', TARGET_COL]: # Assicurati che gp e target siano stringa per split/lookup
            if col in df_m1_full.columns: df_m1_full[col] = df_m1_full[col].astype(str)
        for col in NUM_COLS_M1 + ['anno']:
            if col in df_m1_full.columns: df_m1_full[col] = pd.to_numeric(df_m1_full[col], errors='coerce')

        df_m1_full[TARGET_COL] = pd.to_numeric(df_m1_full[TARGET_COL], errors='coerce').fillna(0).astype(int)
        df_m1_full.dropna(subset=[TARGET_COL], inplace=True)
        df_m1_full.dropna(subset=[GROUP_COL_M1], inplace=True) # Per GroupShuffleSplit
        
        if df_m1_full.empty:
            print("Dataset M1 vuoto dopo pulizia. Impossibile generare grafici.")
        else:
            X_m1 = df_m1_full[ORIGINAL_FEATURES_M1]
            y_m1 = df_m1_full[TARGET_COL]
            groups_m1 = df_m1_full[GROUP_COL_M1]

            gss_m1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=GLOBAL_SEED)
            # Assicurati che ci siano abbastanza gruppi per lo split
            if len(groups_m1.unique()) < 2:
                print("WARNING: Non abbastanza gruppi unici nel dataset M1 per GroupShuffleSplit. Potrebbe fallire o dare risultati non rappresentativi.")
                # Potresti voler usare un semplice train_test_split come fallback qui se lo split di gruppo fallisce.
                # Per ora, procediamo e vediamo se gss.split solleva un errore.
            
            try:
                _, test_idx_m1 = next(gss_m1.split(X_m1, y_m1, groups_m1))
                X_test_m1, y_test_m1 = X_m1.iloc[test_idx_m1], y_m1.iloc[test_idx_m1]
                print(f"Set di test per Modello 1 ricreato. Shape: {X_test_m1.shape}")

                lgbm_m1 = load(LGBM_M1_PIPELINE_PATH) if LGBM_M1_PIPELINE_PATH.exists() else None
                cb_m1 = load(CATBOOST_M1_PIPELINE_PATH) if CATBOOST_M1_PIPELINE_PATH.exists() else None

                preds_m1_data = {'y_true': y_test_m1.values}
                if lgbm_m1: preds_m1_data['lgbm_pred'] = lgbm_m1.predict(X_test_m1)
                else: print("LGBM M1 non caricato.")
                if cb_m1: preds_m1_data['catboost_pred'] = cb_m1.predict(X_test_m1)
                else: print("CatBoost M1 non caricato.")
                
                # Calcola l'ensemble (assumendo media semplice se i pesi non sono qui)
                # Per usare l'ensemble pesato, dovresti replicare quella logica qui o caricarla
                temp_valid_preds_m1 = [preds_m1_data[k] for k in ['lgbm_pred', 'catboost_pred'] if k in preds_m1_data and not np.all(np.isnan(preds_m1_data[k]))]
                if temp_valid_preds_m1:
                    preds_m1_data['ensemble_pred'] = np.mean(np.array(temp_valid_preds_m1), axis=0)
                else:
                    preds_m1_data['ensemble_pred'] = np.full(len(y_test_m1), np.nan)
                    print("Nessuna predizione valida per l'ensemble M1.")


                if 'ensemble_pred' in preds_m1_data and not np.all(np.isnan(preds_m1_data['ensemble_pred'])):
                    df_predictions_m1 = pd.DataFrame(preds_m1_data)
                    plot_scatter_predicted_vs_actual(df_predictions_m1, 'y_true', 'ensemble_pred', MODEL_M1_NAME, DIAGRAMS_DIR / "scatter_m1_ensemble.png")
                    plot_residuals_histogram(df_predictions_m1, 'y_true', 'ensemble_pred', MODEL_M1_NAME, DIAGRAMS_DIR / "residuals_m1_ensemble.png")
                
                if lgbm_m1: plot_feature_importance(lgbm_m1, ORIGINAL_FEATURES_M1, "LightGBM (M1)", DIAGRAMS_DIR / "fi_lgbm_m1.png")
                if cb_m1: plot_feature_importance(cb_m1, ORIGINAL_FEATURES_M1, "CatBoost (M1)", DIAGRAMS_DIR / "fi_catboost_m1.png")

            except Exception as e:
                print(f"Errore durante la generazione dei grafici per Modello 1: {e}"); traceback.print_exc()

    # --- Preparazione Dati e Modelli per MODELLO 2 ---
    print("\n--- Preparazione per Grafici Modello 2 ---")
    if not DATASET_M2_PATH.exists():
        print(f"Dataset Modello 2 non trovato: {DATASET_M2_PATH}.")
    else:
        try:
            df_m2_full = pd.read_csv(DATASET_M2_PATH)
            print(f"Dataset M2 per grafici: {df_m2_full.shape}")
            if df_m2_full.empty: raise ValueError("Dataset M2 è vuoto.")

            for col in CAT_COLS_M2 + ['gp', TARGET_COL]: # 'gp' è categorica per M2
                 if col in df_m2_full.columns: df_m2_full[col] = df_m2_full[col].astype(str)
            for col in NUM_COLS_M2 + ['anno','round']:
                 if col in df_m2_full.columns: df_m2_full[col] = pd.to_numeric(df_m2_full[col], errors='coerce')

            df_m2_full[TARGET_COL] = pd.to_numeric(df_m2_full[TARGET_COL], errors='coerce').fillna(0).astype(int)
            df_m2_full.dropna(subset=[TARGET_COL], inplace=True)
            df_m2_full.dropna(subset=GROUP_COLS_M2, inplace=True)
            df_m2_full['race_id_m2'] = df_m2_full['anno'].astype(str) + "_" + df_m2_full['gp'].astype(str)

            if df_m2_full.empty:
                print("Dataset M2 vuoto dopo pulizia.")
            else:
                X_m2 = df_m2_full[ORIGINAL_FEATURES_M2]
                y_m2 = df_m2_full[TARGET_COL]
                groups_m2 = df_m2_full['race_id_m2']

                if len(X_m2) > 0 and len(groups_m2.unique()) > 1:
                    gss_m2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=GLOBAL_SEED)
                    _, test_idx_m2 = next(gss_m2.split(X_m2, y_m2, groups_m2))
                    X_test_m2, y_test_m2 = X_m2.iloc[test_idx_m2], y_m2.iloc[test_idx_m2]
                    print(f"Set di test per Modello 2 ricreato. Shape: {X_test_m2.shape}")

                    cb_m2 = load(CATBOOST_M2_PIPELINE_PATH) if CATBOOST_M2_PIPELINE_PATH.exists() else None

                    if cb_m2 and not X_test_m2.empty:
                        preds_m2_vals = cb_m2.predict(X_test_m2)
                        df_predictions_m2 = pd.DataFrame({'y_true': y_test_m2.values, 'pred_m2': preds_m2_vals})
                        
                        plot_scatter_predicted_vs_actual(df_predictions_m2, 'y_true', 'pred_m2', MODEL_M2_NAME, DIAGRAMS_DIR / "scatter_m2.png")
                        plot_residuals_histogram(df_predictions_m2, 'y_true', 'pred_m2', MODEL_M2_NAME, DIAGRAMS_DIR / "residuals_m2.png")
                        plot_feature_importance(cb_m2, ORIGINAL_FEATURES_M2, MODEL_M2_NAME, DIAGRAMS_DIR / "fi_catboost_m2.png")
                    elif X_test_m2.empty: print("Set di test M2 vuoto.")
                    else: print("Modello 2 non caricato.")
                else: print("Dati/gruppi M2 insufficienti per split.")
        except Exception as e:
            print(f"Errore durante la generazione dei grafici per Modello 2: {e}"); traceback.print_exc()
            
    print(f"\n✅ Script di generazione grafici completato. Controlla: {DIAGRAMS_DIR}")