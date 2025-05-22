#!/usr/bin/env python
# train_model2.py (Per predizioni Pre-Gara)

import warnings
import math
import random
from pathlib import Path
import gc
import json # Per salvare/caricare i pesi dell'ensemble (se li usassimo qui)

import numpy as np
import pandas as pd
import torch # Anche se non usiamo modelli PyTorch, CatBoost/LGBM GPU potrebbero averne bisogno
from joblib import dump, load
from tabulate import tabulate

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit # Per splittare per gara
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted
import typing 
from typing import List, Optional, Any # Aggiungi Optional qui
# Importa il modello scelto
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("WARNING: catboost not found. CatBoost model will be skipped.")
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

# Importa la funzione helper se necessaria per il preprocessor di CatBoost
try:
    from utils.utils_gradio import to_str_type
    TO_STR_TYPE_AVAILABLE = True
    print("✅ Funzione 'to_str_type' importata da utils_gradio.py")
except ImportError:
    print("⚠️ WARNING: utils_gradio.py o to_str_type non trovati. Verrà usata una definizione locale.")
    TO_STR_TYPE_AVAILABLE = False
    def to_str_type(X_input):
        if isinstance(X_input, pd.Series): return X_input.astype(str)
        elif isinstance(X_input, pd.DataFrame):
            X_output = X_input.copy();
            for col in X_output.columns: X_output[col] = X_output[col].astype(str)
            return X_output
        elif isinstance(X_input, np.ndarray): return X_input.astype(str)
        else: print(f"WARNING: to_str_type tipo non atteso: {type(X_input)}"); return X_input

# ─── Seed Everything ─────────────────────────────────────────────────────────
GLOBAL_SEED = 42
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(GLOBAL_SEED)

if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}. Precision: medium.")
    torch.set_float32_matmul_precision("medium")
else: print("CUDA not available.")
warnings.filterwarnings("ignore", category=UserWarning); warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Configuration ──────────────────────────────────────────
DATA_DIR = Path("data")
PRE_RACE_DATASET_PATH = DATA_DIR / "pre_race_prediction_dataset.csv"
OUTPUT_DIR = Path("models"); OUTPUT_DIR.mkdir(exist_ok=True)
PRE_RACE_MODEL_PIPELINE_PATH = OUTPUT_DIR / "pre_race_catboost_pipeline.joblib" # Nome specifico

TARGET = "final_position"
# Definiamo le feature per il Modello 2 basandoci su pre_race_prediction_dataset.csv
# Anno, round, gp potrebbero essere usate per lo splitting o come feature se si fa OHE su 'gp'
# Per ora, usiamo 'gp' come feature categorica diretta per CatBoost.

YEARS_LOOKBACK = 2 # Deve corrispondere a quello usato in build_dataset_pre_race.py

NUM_COLS_M2 = [
    "starting_grid_position",
    "elo_driver_pre_race", "elo_team_pre_race",
    f"avg_finish_pos_circuit_prev{YEARS_LOOKBACK}y",
    f"avg_grid_pos_circuit_prev{YEARS_LOOKBACK}y",
    f"avg_pos_gained_circuit_prev{YEARS_LOOKBACK}y",
    f"races_on_circuit_prev{YEARS_LOOKBACK}y",
    f"best_finish_pos_circuit_prev{YEARS_LOOKBACK}y",
    f"laps_completed_pct_circuit_prev{YEARS_LOOKBACK}y",
    f"team_avg_finish_pos_circuit_prev{YEARS_LOOKBACK}y",
    f"team_avg_pos_gained_circuit_prev{YEARS_LOOKBACK}y",
]
CAT_COLS_M2 = [
    "driver", "team", "gp", # 'gp' (nome del circuito) come feature categorica
    "is_driver_new_to_circuit" # Questo è booleano (0/1), CatBoost può gestirlo come numerico o categorico. Trattiamolo come categorico per ora.
]
# Colonna per raggruppare nello split (una gara è definita da anno + gp)
GROUP_COLS_M2 = ['anno', 'gp']


# ─── Load Dataset ────────────────────────────────────────────────────────────
print(f"\n💾 Loading pre-race dataset from: {PRE_RACE_DATASET_PATH}")
if not PRE_RACE_DATASET_PATH.exists():
    raise FileNotFoundError(f"Pre-race dataset not found at {PRE_RACE_DATASET_PATH}.")
try:
    df = pd.read_csv(PRE_RACE_DATASET_PATH)
    print(f"Pre-race dataset loaded. Shape: {df.shape}")
    if df.empty: raise ValueError("Pre-race dataset is empty.")
except Exception as e:
    raise RuntimeError(f"Failed to load pre-race dataset: {e}")

# Verifica NaN prima della coercizione (già fatta da validate_pre_race_dataset.py)
if df.isnull().values.any(): print("⚠️ WARNING: NaNs detected BEFORE type coercion!")
else: print("✅ No NaNs detected in loaded dataset.")

# ─── Data Type Verification and Coercion (Semplificata, dataset già dovrebbe essere pulito) ───
print("\n🔍 Verifying and coercing data types for Model 2...")
all_expected_features = NUM_COLS_M2 + CAT_COLS_M2
missing_cols = [c for c in all_expected_features + [TARGET] + GROUP_COLS_M2 if c not in df.columns]
if missing_cols: raise ValueError(f"Essential columns missing: {missing_cols}")

for col in NUM_COLS_M2:
    if col in df.columns: # Assicura che le numeriche siano numeriche
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            print(f"  Coercing NUM '{col}' from {df[col].dtype} to numeric.")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    else: print(f"WARNING: Numeric column {col} not found in dataset.")

# Per CatBoost, le categoriche devono essere stringhe o oggetti (o int se si passano indici)
# La funzione to_str_type nel preprocessor se le aspetta come input che può processare.
# Il preprocessor si occuperà della conversione finale a stringa per CatBoost.
# Qui ci assicuriamo solo che non siano tipi completamente sbagliati.
for col in CAT_COLS_M2:
    if col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col].dtype):
             df[col] = df[col].astype(str) # Assicura stringa per il preprocessor
        elif pd.api.types.is_bool_dtype(df[col].dtype) or pd.api.types.is_integer_dtype(df[col].dtype):
             print(f"  Coercing CAT '{col}' from {df[col].dtype} to string for CatBoost.")
             df[col] = df[col].astype(str) # Converti bool/int a stringa
    else: print(f"WARNING: Categorical column {col} not found in dataset.")


df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce').fillna(99).astype(int) # 99 per DNF/target mancante
df.dropna(subset=[TARGET], inplace=True) # Rimuovi se target è ancora NaN dopo coercizione (improbabile)

print("✅ Data type verification/coercion complete for Model 2.")

# ─── Data Splitting (Group Shuffle by Race Event) ───────────────────────────
print("\n🔪 Splitting data for Model 2...")
# Rimuovi righe dove mancano i gruppi per lo split
df.dropna(subset=GROUP_COLS_M2, inplace=True)

# Crea un ID univoco per ogni gara (anno + gp) per GroupShuffleSplit
df['race_id'] = df['anno'].astype(str) + "_" + df['gp'].astype(str)
groups_m2 = df['race_id']

X = df[NUM_COLS_M2 + CAT_COLS_M2]
y = df[TARGET]

gss_m2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=GLOBAL_SEED)
if len(X) > 0 and len(groups_m2.unique()) > 1 : # Necessari almeno 2 gruppi per splittare
    try:
        train_idx, test_idx = next(gss_m2.split(X, y, groups_m2))
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    except ValueError as ve: # Se non ci sono abbastanza gruppi
        print(f"ValueError during split (not enough groups?): {ve}. Using random split as fallback.")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=y if y.nunique() > 1 else None)
else: # Non abbastanza dati o gruppi per GroupShuffleSplit, usa un semplice split o tutto per training
    print("Not enough data/groups for GroupShuffleSplit. Using a simple train_test_split or all data for training.")
    if len(X) < 10: # Molto pochi dati, usa tutto per training, nessuna valutazione qui
        X_train, y_train = X.copy(), y.copy()
        X_test, y_test = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype) # Test set vuoti
        print("  Too few samples, using all data for training, no test set evaluation.")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=y if y.nunique() > 1 else None)

print(f"Model 2 - Train size: {len(X_train)}, Test size: {len(X_test)}")
del df # Libera memoria
gc.collect()


# ─── Preprocessing Pipeline for Model 2 (CatBoost) ──────────────────────────
print("\n🛠️ Defining preprocessing pipeline for Model 2 (CatBoost)...")
numeric_transformer_m2 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Le categoriche sono già state convertite in stringa, l'imputer le gestirà.
# Il FunctionTransformer con to_str_type assicura che siano stringhe.
categorical_transformer_m2 = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='_STR_MISSING_')), # String fill value
    ('astype_str', FunctionTransformer(to_str_type)) # Assicura che l'output sia stringa per CatBoost
])

preprocessor_m2 = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_m2, NUM_COLS_M2 if NUM_COLS_M2 else []),
        ('cat', categorical_transformer_m2, CAT_COLS_M2 if CAT_COLS_M2 else [])
    ],
    remainder='drop', # Scarta altre colonne (es. anno, round, race_id)
    n_jobs=1
)

# ─── Model 2 Training (CatBoost GPU) ────────────────────────────────────────
pre_race_catboost_pipeline: Optional[Pipeline] = None
if CATBOOST_AVAILABLE:
    print("\n--- Training/Loading Model 2: Pre-Race CatBoost (GPU) ---")
    if PRE_RACE_MODEL_PIPELINE_PATH.exists():
        try:
            print(f"⏳ Loading Pre-Race CatBoost pipeline from {PRE_RACE_MODEL_PIPELINE_PATH}...")
            pre_race_catboost_pipeline = load(PRE_RACE_MODEL_PIPELINE_PATH)
            if not isinstance(pre_race_catboost_pipeline, Pipeline) or 'regressor' not in pre_race_catboost_pipeline.named_steps or \
               not isinstance(pre_race_catboost_pipeline.named_steps['regressor'], CatBoostRegressor):
                raise ValueError("Loaded object is not a valid Pre-Race CatBoost pipeline.")
            print(f"⚡ Pre-Race CatBoost pipeline loaded successfully.\n")
        except Exception as e:
            print(f"⚠️ Failed loading Pre-Race CatBoost: {e}. Retraining...")
            pre_race_catboost_pipeline = None

    if pre_race_catboost_pipeline is None and not X_train.empty:
        try:
            print("⏳ Fitting Pre-Race CatBoost pipeline...")
            # Determina gli indici delle feature categoriche DOPO il preprocessor
            # Questo è cruciale se il preprocessor cambia l'ordine o il numero delle colonne
            # Fitta il preprocessor per vedere l'output
            _ = preprocessor_m2.fit_transform(X_train) # Fitta per ottenere i nomi/indici trasformati
            
            # Ottieni i nomi delle feature trasformate (se il transformer li supporta)
            # O calcola gli indici basandoti sull'ordine dei transformer
            num_transformed_cols_count_m2 = 0
            if NUM_COLS_M2 and 'num' in preprocessor_m2.named_transformers_ and preprocessor_m2.named_transformers_['num'] != 'drop':
                num_transformed_cols_count_m2 = preprocessor_m2.named_transformers_['num'].fit_transform(X_train[NUM_COLS_M2]).shape[1]

            cat_features_indices_m2 = []
            if CAT_COLS_M2 and 'cat' in preprocessor_m2.named_transformers_ and preprocessor_m2.named_transformers_['cat'] != 'drop':
                 # Assumendo che il trasf. categorico non cambi il numero di colonne (SimpleImputer + FunctionTransformer(to_str))
                cat_features_indices_m2 = list(range(num_transformed_cols_count_m2, num_transformed_cols_count_m2 + len(CAT_COLS_M2)))
            
            print(f"  CatBoost M2 - Categorical feature indices (post-transform): {cat_features_indices_m2}")

            cb_params_m2 = {
                'iterations': 1000, 'learning_rate': 0.05, 'depth': 8,
                'l2_leaf_reg': 3, 'loss_function': 'MAE', 'eval_metric': 'MAE',
                'task_type': 'GPU', 'devices': '0',
                'random_seed': GLOBAL_SEED, 'verbose': 200,
                'early_stopping_rounds': 50
            }
            model_instance_m2 = CatBoostRegressor(**cb_params_m2)
            pre_race_catboost_pipeline = Pipeline([
                ('preprocess', preprocessor_m2),
                ('regressor', model_instance_m2)
            ])

            print("  Fitting Pre-Race CatBoost pipeline...")
            # Per l'early stopping, CatBoost sklearn wrapper ha bisogno di un eval_set passato a fit
            # Se X_test è piccolo, potremmo non usarlo per early stopping.
            # Per ora, fitto su tutto X_train.
            if cat_features_indices_m2:
                 pre_race_catboost_pipeline.fit(X_train, y_train, regressor__cat_features=cat_features_indices_m2)
            else:
                 pre_race_catboost_pipeline.fit(X_train, y_train)

            dump(pre_race_catboost_pipeline, PRE_RACE_MODEL_PIPELINE_PATH)
            print(f"✅ Pre-Race CatBoost pipeline saved to {PRE_RACE_MODEL_PIPELINE_PATH}.\n")
        except Exception as e:
            print(f"❌ Error during Pre-Race CatBoost training/saving: {e}");
            import traceback; traceback.print_exc()
            pre_race_catboost_pipeline = None
    elif X_train.empty :
        print("No training data for Model 2, skipping training.")

elif not CATBOOST_AVAILABLE:
     print("\n--- Skipping Model 2: CatBoost (catboost not available) ---")

# ─── Evaluation for Model 2 ────────────────────────────────────────────────
print("\n--- Evaluating Model 2 (Pre-Race CatBoost) ---")
results_m2 = []
if pre_race_catboost_pipeline is not None and not X_test.empty:
    print("📊 Evaluating Pre-Race CatBoost...")
    try:
         preds_m2 = pre_race_catboost_pipeline.predict(X_test)
         mae_m2 = mean_absolute_error(y_test, preds_m2)
         rmse_m2 = math.sqrt(mean_squared_error(y_test, preds_m2))
         r2_m2 = r2_score(y_test, preds_m2)
         results_m2.append(["Pre-Race CatBoost", mae_m2, rmse_m2, r2_m2])
         print(f"  Pre-Race CatBoost - MAE: {mae_m2:.4f}, RMSE: {rmse_m2:.4f}, R²: {r2_m2:.4f}")
    except Exception as e: print(f"❌ Error evaluating Pre-Race CatBoost: {e}")
elif X_test.empty:
    print("⚠️ No test data for Model 2 evaluation.")
else:
    print("⚠️ Pre-Race CatBoost model not available for evaluation.")

# --- Display Results for Model 2 ---
if results_m2:
    print("\n🎯 MODEL 2 TEST METRICS")
    print(tabulate([[name, f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}"] for name, mae, rmse, r2 in results_m2],
                   headers=["Model", "MAE", "RMSE", "R²"], tablefmt="github"))
else:
    print("No Model 2 results to display.")

print(f"\n🏁 Training script for Model 2 finished. Model saved in: {PRE_RACE_MODEL_PIPELINE_PATH if pre_race_catboost_pipeline else 'Not Saved'}")