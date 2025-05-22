#!/usr/bin/env python
# train_model.py (v6_catboost_fix_catembed_skip)
# ------------------------------------------------------------------------------
import json
import warnings
import math
import random
import typing
import collections
from pathlib import Path
import shutil
from dataclasses import fields
import gc

import numpy as np
import pandas as pd
import torch
import optuna
from joblib import dump, load
from tabulate import tabulate

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer # Added FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted

# --- Helper function for CatBoost preprocessor ---
def to_str_type(X_input):
    # Ensure input is DataFrame to handle potential Series after imputer
    if isinstance(X_input, pd.Series):
        X_input = X_input.to_frame()
    # Apply astype(str) to each column if it's a DataFrame
    return X_input.astype(str)
# --- Model Imports with Availability Check ---
try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    print("WARNING: lightgbm not found. LightGBM model will be skipped.")
    LGBM_AVAILABLE = False
    LGBMRegressor = None # Dummy



# *** CatBoost Import ***
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("WARNING: catboost not found. CatBoost model will be skipped.")
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None # Dummy


# --- Utility Function ---
def safe_cfg(cls, **kw):
    if cls is None: return None
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in kw.items() if k in valid})

# ─── Seed Everything ─────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
        print(f"PyTorch Lightning seed set to {seed}")
    except ImportError:
        print(f"Manual seeding set to {seed}. PyTorch Lightning not found or not used for seeding here.")

set_seed(GLOBAL_SEED)

if torch.cuda.is_available():
    print(f"CUDA available. Setting float32 matmul precision to 'medium'. Device: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("medium")
else:
    print("CUDA not available. Models requiring GPU will run on CPU or fail.")

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabular")
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Configuration: Paths & Columns ──────────────────────────────────────────
DATA_PATH = Path("data/training_dataset_COMPLETO.csv")
OUTPUT_DIR = Path("models"); OUTPUT_DIR.mkdir(exist_ok=True)
TARGET = "final_position"
GROUP_COL = "gp"

NUM_COLS = [
    "lap_time", "lap_time_diff", "lap_time_pct", "gap_to_leader_diff",
    "gap_to_leader_pct", "speed_avg", "throttle_pct", "brake_pct", "drs_pct",
    "gear_avg", "sector1", "sector2", "sector3", "stint", "lap_in_stint",
    "elo_driver", "elo_team", "position", "lap_number"
]
CAT_COLS = [
    "team", "driver", "compound", "pit_in", "pit_out", "track_status"
]

# ─── Load Dataset ────────────────────────────────────────────────────────────
print(f"\n💾 Loading dataset from: {DATA_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run build_dataset.py first.")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except Exception as e:
    raise RuntimeError(f"Failed to load dataset from {DATA_PATH}: {e}")

if df.isnull().values.any():
     print("⚠️ WARNING: NaNs detected in the loaded dataset BEFORE type coercion!")
else:
     print("✅ No NaNs detected in loaded dataset.")

# ─── Data Type Verification and Coercion ─────────────────────────────────────
print("\n🔍 Verifying and coercing data types...")
essential_cols = NUM_COLS + CAT_COLS + [TARGET, GROUP_COL]
missing_essential = [c for c in essential_cols if c not in df.columns]
if missing_essential:
    raise ValueError(f"Essential columns missing from dataset: {missing_essential}")


for col in NUM_COLS:
    if not pd.api.types.is_numeric_dtype(df[col].dtype) or df[col].dtype == 'object':
        
        try:
            original_nan_count = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            new_nan_count = df[col].isna().sum()
            if new_nan_count > original_nan_count:
                print(f"      Converted {new_nan_count - original_nan_count} non-numeric values to NaN in '{col}'.")
            if df[col].isna().all() and original_nan_count < len(df):
                print(f"      ⚠️ WARNING: Column '{col}' is now entirely NaN!")
        except Exception as e:
            raise TypeError(f"Failed to coerce column '{col}' to numeric: {e}")
    elif pd.api.types.is_bool_dtype(df[col].dtype):

        df[col] = df[col].astype(int)

NUM_COLS = [c for c in NUM_COLS if c in df.columns]
if not NUM_COLS:
     print("    ⚠️ WARNING: No columns left in NUM_COLS list.")
else:
    numeric_dtypes = df[NUM_COLS].dtypes
    non_numeric_mask = ~numeric_dtypes.apply(pd.api.types.is_numeric_dtype)
    non_numeric_in_num = numeric_dtypes[non_numeric_mask]
    if not non_numeric_in_num.empty:
        print("❌ ERROR: Non-numeric dtypes still present in NUM_COLS after coercion!")
        print(non_numeric_in_num)
        raise TypeError("Preprocessing cannot continue with non-numeric columns in NUM_COLS")
    else:
        print("    ✅ All NUM_COLS are numeric.")

print(f"  Coercing TARGET column '{TARGET}'...")
if TARGET in df.columns:
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    initial_len = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) < initial_len:
        print(f"    Removed {initial_len - len(df)} rows due to NaN in target column.")
else:
    raise ValueError(f"Target column '{TARGET}' not found in DataFrame.")

print("  Coercing CAT_COLS to category...")
CAT_COLS = [c for c in CAT_COLS if c in df.columns]
for c in CAT_COLS:
    if not pd.api.types.is_categorical_dtype(df[c].dtype):
        # print(f"    Column '{c}' ({df[c].dtype}) -> category") # Less verbose
        df[c] = df[c].astype(str).astype("category")

print("✅ Data type verification and coercion complete.")

# ─── Data Splitting (Group Shuffle) ──────────────────────────────────────────
print("\n🔪 Splitting data using GroupShuffleSplit (keeping GPs together)...")
if not all(c in df.columns for c in NUM_COLS + CAT_COLS):
     raise ValueError("One or more feature columns not found in DataFrame after cleaning.")
if GROUP_COL not in df.columns:
     raise ValueError(f"Group column '{GROUP_COL}' not found.")

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET]
groups = df[GROUP_COL]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=GLOBAL_SEED)
try:
    train_idx, test_idx = next(gss.split(X, y, groups))
except Exception as e:
     raise RuntimeError(f"Failed to split data: {e}")

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
groups_train = groups.iloc[train_idx].copy()

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
del df
gc.collect()


print("\n🛠️ Defining preprocessing pipeline for LightGBM...")
try:
    onehot_kwargs = dict(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=10)
except TypeError:
    onehot_kwargs = dict(sparse=False, handle_unknown='ignore')

numeric_transformer_lgbm = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer_lgbm = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='_MISSING_')),
    ('onehot', OneHotEncoder(**onehot_kwargs)) # OHE for LightGBM
])
if not NUM_COLS: print("⚠️ WARNING: NUM_COLS list is empty for ColumnTransformer.")
if not CAT_COLS: print("⚠️ WARNING: CAT_COLS list is empty for ColumnTransformer.")

preprocessor_lgbm = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_lgbm, NUM_COLS if NUM_COLS else []),
        ('cat', categorical_transformer_lgbm, CAT_COLS if CAT_COLS else [])
    ],
    remainder='drop',
    n_jobs=1 # Keep n_jobs=1 for stability
)

# Preprocessor for CatBoost (handles categoricals internally, so no OHE)
# It still needs imputation for numerical and categorical, and scaling for numerical
print("\n🛠️ Defining preprocessing pipeline for CatBoost...")
numeric_transformer_cb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer_cb = Pipeline([ # Only imputes, CatBoost handles encoding
    ('imputer', SimpleImputer(strategy='constant', fill_value='_MISSING_')),
    #('astype', FunctionTransformer(lambda x: x.astype(str))) # CatBoost expects string/object for cat features
    ('astype', FunctionTransformer(to_str_type))
])
preprocessor_catboost = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_cb, NUM_COLS if NUM_COLS else []),
        ('cat', categorical_transformer_cb, CAT_COLS if CAT_COLS else [])
    ],
    remainder='drop',
    n_jobs=1
)


# ─── Model Training Initialization ───────────────────────────────────────────
lgbm_pipeline = None
#cat_model = None
catboost_pipeline = None # For CatBoost with its preprocessor

# ─── 1. LightGBM (GPU) ───────────────────────────────────────────────────────
if LGBM_AVAILABLE:
    print("\n--- Training/Loading Model 1: LightGBM (GPU) ---")
    lgbm_path = OUTPUT_DIR / "lgbm_pipeline.joblib"

    if lgbm_path.exists():
        try:
            print(f"⏳ Attempting to load LightGBM pipeline from {lgbm_path}...")
            lgbm_pipeline = load(lgbm_path)
            if not isinstance(lgbm_pipeline, Pipeline) or 'preprocess' not in lgbm_pipeline.named_steps:
                 raise ValueError("Loaded object is not a valid LightGBM pipeline.")
            # Check if the 'preprocess' step (ColumnTransformer) is fitted
            check_is_fitted(lgbm_pipeline.named_steps['preprocess'])
            print(f"⚡ LightGBM pipeline loaded successfully.\n")
        except Exception as e:
            print(f"⚠️ Failed to load valid LightGBM pipeline: {e}. Retraining...")
            lgbm_pipeline = None

    if lgbm_pipeline is None:
        def lgbm_objective(trial):
            params = {
                'objective': 'regression_l1', 'metric': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1, 'random_state': GLOBAL_SEED, 'device': 'gpu',
                'gpu_platform_id': 0, 'gpu_device_id': 0
            }
            # Use the LGBM-specific preprocessor
            temp_pipeline = Pipeline([('preprocess', preprocessor_lgbm), ('regressor', LGBMRegressor(**params))])
            gkf = GroupKFold(n_splits=5)
            scores = []
            for fold, (train_fold_idx, val_fold_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
                try:
                    temp_pipeline.fit(X_train.iloc[train_fold_idx], y_train.iloc[train_fold_idx])
                    preds = temp_pipeline.predict(X_train.iloc[val_fold_idx])
                    scores.append(mean_absolute_error(y_train.iloc[val_fold_idx], preds))
                except Exception as e:
                    print(f"Error during Optuna fold {fold+1} T{trial.number}: {e}")
                    return float('inf')
            return np.mean(scores) if scores else float('inf')

        print("🔎 Tuning LightGBM with Optuna...")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
        try:
            study.optimize(lgbm_objective, n_trials=40, timeout=1800, gc_after_trial=True)
            print(f"✅ Optuna best score (MAE): {study.best_value:.4f}")
            print(f"✅ Optuna best params: {study.best_params}")
            best_lgbm_params = study.best_params
        except Exception as e:
            print(f"❌ Optuna optimization failed: {e}. Using default LGBM parameters.")
            best_lgbm_params = {}

        best_lgbm_params.update({
             'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
             'objective': 'regression_l1', 'metric': 'mae',
             'random_state': GLOBAL_SEED, 'n_jobs': -1
        })
        lgbm_pipeline = Pipeline([
            ('preprocess', preprocessor_lgbm), # Use LGBM-specific preprocessor
            ('regressor', LGBMRegressor(**best_lgbm_params))
        ])
        try:
            print("⏳ Fitting final LightGBM pipeline...")
            lgbm_pipeline.fit(X_train, y_train)
            dump(lgbm_pipeline, lgbm_path)
            print(f"✅ LightGBM pipeline saved to {lgbm_path}.\n")
        except Exception as e:
            print(f"❌ Error fitting/saving final LightGBM: {e}")
            lgbm_pipeline = None
        if 'study' in locals(): del study; gc.collect()
else:
    print("\n--- Skipping Model 1: LightGBM (lightgbm not available) ---")


# ─── 2. CatBoost (GPU)──────────────────────────────────
catboost_pipeline_path = OUTPUT_DIR / "catboost_pipeline.joblib" # CatBoost pipeline path
if CATBOOST_AVAILABLE:
    print("\n--- Training/Loading Model 3: CatBoost (GPU) ---")
    catboost_pipeline = None # Initialize

    # *** Skip Training Logic for CatBoost Pipeline ***
    if catboost_pipeline_path.exists():
        try:
            print(f"⏳ Loading CatBoost pipeline from {catboost_pipeline_path}...")
            catboost_pipeline = load(catboost_pipeline_path)
            if not isinstance(catboost_pipeline, Pipeline) or 'regressor' not in catboost_pipeline.named_steps:
                 raise ValueError("Loaded object is not a valid CatBoost pipeline.")
            # Simple check if the regressor step is a CatBoostRegressor
            if not isinstance(catboost_pipeline.named_steps['regressor'], CatBoostRegressor):
                 raise ValueError("Regressor in pipeline is not a CatBoostRegressor.")
            print(f"⚡ CatBoost pipeline loaded successfully.\n")
        except Exception as e:
            print(f"⚠️ Failed loading CatBoost pipeline: {e}. Retraining...")
            catboost_pipeline = None

    if catboost_pipeline is None: # Train only if needed
        try:
            print("⏳ Fitting CatBoost pipeline...")
            cat_features_indices = [X_train.columns.get_loc(col) for col in CAT_COLS if col in X_train.columns]

            cb_params = {
                'iterations': 1500, 'learning_rate': 0.05, 'depth': 10,
                'l2_leaf_reg': 3, 'loss_function': 'MAE', 'eval_metric': 'MAE',
                'task_type': 'GPU', 'devices': '0',
                'random_seed': GLOBAL_SEED, 'verbose': 200, # Less verbose
                'early_stopping_rounds': 50
            }
            # Create a temporary CatBoostRegressor instance for fitting
            temp_cb_model = CatBoostRegressor(**cb_params)

            # Create the full pipeline including its specific preprocessor
            catboost_pipeline = Pipeline([
                ('preprocess', preprocessor_catboost), # Use CatBoost-specific preprocessor
                ('regressor', temp_cb_model)
            ])

            # CatBoost's fit method in sklearn wrapper might not directly support eval_set from pipeline
            # So we fit the pipeline, which will fit CatBoost with its data.
            # Early stopping for CatBoost within a pipeline is tricky.
            # For simplicity, we'll fit without eval_set here. Can be added with more complex setup.
            catboost_pipeline.fit(X_train, y_train, regressor__cat_features=cat_features_indices)

            dump(catboost_pipeline, catboost_pipeline_path)
            print(f"✅ CatBoost pipeline saved to {catboost_pipeline_path}.\n")
            # No need to assign temp_cb_model to catboost_model, pipeline is the main object

        except Exception as e:
            print(f"❌ Error during CatBoost training/saving: {e}");
            # import traceback; traceback.print_exc() # For detailed error
            catboost_pipeline = None # Ensure it's None on failure
            gc.collect()
elif not CATBOOST_AVAILABLE:
     print("\n--- Skipping Model 3: CatBoost (catboost not available) ---")


# ─── 4. Evaluation & Ensemble ────────────────────────────────────────────────
print("\n--- Evaluating Models and Ensemble ---")
results = []; predictions = {'y_true': y_test.values}; y_test_eval = y_test
if 'X_test' not in locals(): raise RuntimeError("X_test not defined for evaluation.")

# Model 1: LightGBM
if lgbm_pipeline is not None:
    print("📊 Evaluating LightGBM...")
    try: lgbm_preds = lgbm_pipeline.predict(X_test)
    except Exception as e: print(f"❌ Error evaluating LightGBM: {e}"); lgbm_preds = np.full(len(y_test_eval), np.nan)
    if isinstance(lgbm_preds, np.ndarray):
        predictions['lgbm_pred'] = lgbm_preds
        results.append(["LightGBM", mean_absolute_error(y_test_eval, lgbm_preds), math.sqrt(mean_squared_error(y_test_eval, lgbm_preds)), r2_score(y_test_eval, lgbm_preds)])
    else: predictions['lgbm_pred'] = np.full(len(y_test_eval), np.nan)
else: print("⚠️ LightGBM not available."); predictions['lgbm_pred'] = np.full(len(y_test_eval), np.nan)


# Model 2: CatBoost (replaces TabPFN)
if catboost_pipeline is not None: # Check the pipeline object
    print("📊 Evaluating CatBoost...")
    try:
         cb_preds = catboost_pipeline.predict(X_test)
         predictions['catboost_pred'] = cb_preds # Use new key
         results.append(["CatBoost", mean_absolute_error(y_test_eval, cb_preds), math.sqrt(mean_squared_error(y_test_eval, cb_preds)), r2_score(y_test_eval, cb_preds)])
         gc.collect()
    except Exception as e: print(f"❌ Error evaluating CatBoost: {e}"); predictions['catboost_pred'] = np.full(len(y_test_eval), np.nan)
else: print("⚠️ CatBoost not available."); predictions['catboost_pred'] = np.full(len(y_test_eval), np.nan)


# --- NUOVA SEZIONE: GESTIONE DEI PESI POSIZIONALI ---
WEIGHTS_FILE = OUTPUT_DIR / "positional_weights.json"
POSITIONS_OF_INTEREST = list(range(1, 21)) # Assumendo posizioni da 1 a 20

# !!! CORREZIONE QUI !!!
# Definiamo i nomi base dei modelli che useremo per i pesi
MODEL_BASE_NAMES_FOR_WEIGHTS = ['lgbm', 'catboost']
DEFAULT_WEIGHTS = {name: 1.0 / len(MODEL_BASE_NAMES_FOR_WEIGHTS) for name in MODEL_BASE_NAMES_FOR_WEIGHTS} # Pesi uguali se non specificato

def calculate_positional_weights(y_true_train, predictions_train_dict, positions_of_interest, model_pred_keys):
    """
    Calcola i pesi per ciascun modello basati sulla loro performance storica
    nel predire specifiche posizioni sul set di training.
    model_pred_keys è una lista come ['lgbm_pred', 'catboost_pred']
    """
    print("⚖️ Calculating positional weights based on training performance...")
    positional_weights = {}
    epsilon = 1e-6 # Per evitare divisione per zero

    y_true_train_rounded = np.round(y_true_train).astype(int)

    for pos in positions_of_interest:
        indices = np.where(y_true_train_rounded == pos)[0]
        if len(indices) == 0:
            # Usiamo MODEL_BASE_NAMES_FOR_WEIGHTS per i default weights qui
            positional_weights[pos] = {name: DEFAULT_WEIGHTS.get(name, 0.5) for name in MODEL_BASE_NAMES_FOR_WEIGHTS}
            continue

        raw_skills = {}
        total_skill_for_pos = 0

        for model_key in model_pred_keys: # es. 'lgbm_pred', 'catboost_pred'
            model_base_name = model_key.replace('_pred', '') # Ottiene 'lgbm' o 'catboost'
            model_pred_train = predictions_train_dict.get(model_key) # Usa .get() per sicurezza

            if model_pred_train is None or not isinstance(model_pred_train, np.ndarray) or len(model_pred_train) != len(y_true_train):
                print(f"  ⚠️ Skipping {model_key} for position {pos} weight calculation due to invalid training predictions.")
                raw_skills[model_base_name] = 0
                continue

            errors = np.abs(model_pred_train[indices] - y_true_train[indices])
            mae = np.mean(errors) if len(errors) > 0 else float('inf')
            skill = 1.0 / (mae + epsilon)
            raw_skills[model_base_name] = skill
            total_skill_for_pos += skill

        current_pos_weights = {}
        if total_skill_for_pos > 0:
            for model_name_simple, skill_value in raw_skills.items():
                current_pos_weights[model_name_simple] = skill_value / total_skill_for_pos
        else:
            current_pos_weights = {name: DEFAULT_WEIGHTS.get(name, 0.5) for name in MODEL_BASE_NAMES_FOR_WEIGHTS}

        positional_weights[pos] = current_pos_weights

    print(f"Positional weights calculated. Example for P1: {positional_weights.get(1)}")


    try:
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump({str(k): v for k, v in positional_weights.items()}, f, indent=4)
        print(f"✅ Positional weights saved to {WEIGHTS_FILE}")
    except Exception as e:
        print(f"❌ Error saving positional weights: {e}")
    return positional_weights

def load_positional_weights():
    if WEIGHTS_FILE.exists():
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                loaded_w = json.load(f)
                weights = {int(k): v for k, v in loaded_w.items()}
            print(f"✅ Positional weights loaded from {WEIGHTS_FILE}")
            return weights
        except Exception as e:
            print(f"⚠️ Error loading positional weights: {e}. Will attempt to recalculate.")
    return None

# --- FINE NUOVA SEZIONE ---

# ─── 4. Evaluation & Ensemble ────────────────────────────────────────────────
print("\n--- Evaluating Models and Ensemble ---")
results = []
predictions = {'y_true': y_test.values} # y_test qui è una Series Pandas
y_test_eval = y_test.values # Usiamo y_test.values per consistenza con le predizioni numpy

if 'X_test' not in locals(): raise RuntimeError("X_test not defined for evaluation.")

# Model 1: LightGBM
if lgbm_pipeline is not None:
    print("📊 Evaluating LightGBM...")
    try: lgbm_preds = lgbm_pipeline.predict(X_test)
    except Exception as e: print(f"❌ Error evaluating LightGBM: {e}"); lgbm_preds = np.full(len(y_test_eval), np.nan)
    if isinstance(lgbm_preds, np.ndarray):
        predictions['lgbm_pred'] = lgbm_preds
        results.append(["LightGBM", mean_absolute_error(y_test_eval, lgbm_preds), math.sqrt(mean_squared_error(y_test_eval, lgbm_preds)), r2_score(y_test_eval, lgbm_preds)])
    else: predictions['lgbm_pred'] = np.full(len(y_test_eval), np.nan)
else: print("⚠️ LightGBM not available."); predictions['lgbm_pred'] = np.full(len(y_test_eval), np.nan)


# Model 2: CatBoost
if catboost_pipeline is not None:
    print("📊 Evaluating CatBoost...")
    try:
         cb_preds = catboost_pipeline.predict(X_test)
         predictions['catboost_pred'] = cb_preds
         results.append(["CatBoost", mean_absolute_error(y_test_eval, cb_preds), math.sqrt(mean_squared_error(y_test_eval, cb_preds)), r2_score(y_test_eval, cb_preds)])
         gc.collect()
    except Exception as e: print(f"❌ Error evaluating CatBoost: {e}"); predictions['catboost_pred'] = np.full(len(y_test_eval), np.nan)
else: print("⚠️ CatBoost not available."); predictions['catboost_pred'] = np.full(len(y_test_eval), np.nan)


# --- Calcolo/Caricamento Pesi Posizionali ---
# Abbiamo bisogno delle predizioni dei modelli sul set di training per calcolare i pesi
# Se i modelli sono appena stati addestrati, potremmo farli ri-predire su X_train
# Oppure, se Optuna fa cross-validation, potremmo salvare le predizioni out-of-fold.
# Per semplicità qui, ri-prediciamo su X_train se i pesi non sono caricabili.

position_specific_weights = load_positional_weights()
model_keys_for_weights = ['lgbm_pred', 'catboost_pred'] # Chiavi usate in predictions_train_dict

if position_specific_weights is None:
    print("Could not load weights, calculating them from training data...")
    predictions_train_dict_for_weights = {}
    # Assicurati che X_train e y_train siano disponibili
    if 'X_train' in locals() and 'y_train' in locals():
        if lgbm_pipeline is not None:
            try:
                predictions_train_dict_for_weights['lgbm_pred'] = lgbm_pipeline.predict(X_train)
            except Exception as e:
                print(f"Error getting lgbm training predictions for weight calculation: {e}")
                predictions_train_dict_for_weights['lgbm_pred'] = None
        else:
             predictions_train_dict_for_weights['lgbm_pred'] = None

        if catboost_pipeline is not None:
            try:
                predictions_train_dict_for_weights['catboost_pred'] = catboost_pipeline.predict(X_train)
            except Exception as e:
                print(f"Error getting catboost training predictions for weight calculation: {e}")
                predictions_train_dict_for_weights['catboost_pred'] = None
        else:
            predictions_train_dict_for_weights['catboost_pred'] = None

        # Verifica che ci siano predizioni valide per il calcolo dei pesi
        valid_train_preds_for_weights = all(
            isinstance(predictions_train_dict_for_weights.get(k), np.ndarray) and \
            len(predictions_train_dict_for_weights[k]) == len(y_train)
            for k in model_keys_for_weights if predictions_train_dict_for_weights.get(k) is not None
        )

        if valid_train_preds_for_weights and len([k for k in model_keys_for_weights if predictions_train_dict_for_weights.get(k) is not None]) == len(model_keys_for_weights):
             position_specific_weights = calculate_positional_weights(
                y_train.values, # Passa y_train.values (numpy array)
                predictions_train_dict_for_weights,
                POSITIONS_OF_INTEREST,
                model_keys_for_weights
            )
        else:
            print("⚠️ Not enough valid training predictions to calculate positional weights. Ensemble will use default/equal weights.")
            position_specific_weights = {} # Saranno usati pesi di default
    else:
        print("⚠️ X_train/y_train not available for weight calculation. Ensemble will use default/equal weights.")
        position_specific_weights = {}


# Ensemble
print("\n📊 Calculating Weighted Ensemble...")
valid_preds_for_ensemble = []
model_simple_names_map = {'lgbm_pred': 'lgbm', 'catboost_pred': 'catboost'} # Mappa le chiavi di predizione ai nomi usati nei pesi
pred_keys_for_ensemble = ['lgbm_pred', 'catboost_pred'] # Modelli da includere nell'ensemble

# Raccogli predizioni valide sul TEST SET
active_model_predictions = {}
num_valid_models_for_ensemble = 0

for key in pred_keys_for_ensemble:
    pred_array = predictions.get(key) # Predizioni sul TEST set
    is_valid = isinstance(pred_array, np.ndarray) and \
               len(pred_array) == len(y_test_eval) and \
               not np.isnan(pred_array).all()
    if is_valid:
        print(f"  Including '{key}' in weighted ensemble.")
        active_model_predictions[key] = pred_array
        num_valid_models_for_ensemble +=1
    else:
        print(f"  Skipping '{key}' (invalid/missing) for weighted ensemble.")
        active_model_predictions[key] = None # o np.full(len(y_test_eval), np.nan)

if num_valid_models_for_ensemble > 0:
    ensemble_preds_weighted = np.zeros_like(y_test_eval, dtype=float)
    num_samples = len(y_test_eval)

    for i in range(num_samples):
        current_sample_preds = {}
        sum_weights_for_sample = 0
        weighted_sum_for_sample = 0

        # Determina la posizione target per recuperare i pesi
        # Usa la media delle predizioni dei modelli ATTIVI per questo campione
        active_preds_for_avg_pos = [active_model_predictions[k][i] for k in active_model_predictions if active_model_predictions[k] is not None]
        if not active_preds_for_avg_pos: # Nessun modello attivo per questo campione (improbabile se num_valid_models > 0)
            ensemble_preds_weighted[i] = np.nan # o un valore di fallback
            continue

        avg_predicted_pos_for_weights = np.round(np.mean(active_preds_for_avg_pos)).astype(int)
        avg_predicted_pos_for_weights = max(min(POSITIONS_OF_INTEREST), min(max(POSITIONS_OF_INTEREST), avg_predicted_pos_for_weights)) # Clamping

        # Recupera i pesi per questa posizione media predetta
        weights_for_target_pos = position_specific_weights.get(avg_predicted_pos_for_weights)

        for model_key in active_model_predictions: # Itera sui modelli che hanno predizioni valide per il test set
            if active_model_predictions[model_key] is None:
                continue

            model_simple_name = model_simple_names_map.get(model_key)
            weight = DEFAULT_WEIGHTS.get(model_simple_name, 1.0/num_valid_models_for_ensemble) # Peso di default

            if weights_for_target_pos and model_simple_name in weights_for_target_pos:
                weight = weights_for_target_pos[model_simple_name]
            elif not weights_for_target_pos: # Se non ci sono pesi specifici per la posizione (es. non calcolati)
                 pass # usa il default
            # else: il modello non è nei pesi per questa posizione, usa default

            current_pred_value = active_model_predictions[model_key][i]
            weighted_sum_for_sample += current_pred_value * weight
            sum_weights_for_sample += weight

        if sum_weights_for_sample > 0:
            ensemble_preds_weighted[i] = weighted_sum_for_sample / sum_weights_for_sample # Normalizza se i pesi non sommano a 1
        else:
            # Fallback se la somma dei pesi è 0 (molto improbabile qui)
            ensemble_preds_weighted[i] = np.mean(active_preds_for_avg_pos) if active_preds_for_avg_pos else np.nan


    predictions['ensemble_pred_weighted'] = ensemble_preds_weighted
    results.append([f"Weighted Ensemble ({num_valid_models_for_ensemble} models)",
                    mean_absolute_error(y_test_eval, ensemble_preds_weighted),
                    math.sqrt(mean_squared_error(y_test_eval, ensemble_preds_weighted)),
                    r2_score(y_test_eval, ensemble_preds_weighted)])
else:
    print("⚠️ No valid preds for weighted ensembling.")
    predictions['ensemble_pred_weighted'] = np.full(len(y_test_eval), np.nan)


# --- Display Results ---
# Modifica questa parte per includere 'ensemble_pred_weighted' se vuoi
print("\n🎯 FINAL TEST METRICS")
headers = ["Model", "MAE", "RMSE", "R²"]
table_data = []
for name, mae, rmse, r2 in results:
    table_data.append([name, f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}"])

print(tabulate(table_data if table_data else [["No results", "N/A", "N/A", "N/A"]], headers=headers, tablefmt="github"))

# --- Save Predictions ---
# Modifica questa parte per salvare 'ensemble_pred_weighted'
final_pred_len = len(y_test_eval)
valid_prediction_data = {'y_true': predictions['y_true']}
for key, value in predictions.items():
     if key != 'y_true': # Includi tutte le predizioni, inclusa quella pesata
          if isinstance(value, np.ndarray) and len(value) == final_pred_len:
              valid_prediction_data[key] = value
          else:
              print(f"  Filling '{key}' with NaNs in CSV due to invalid format/length.")
              valid_prediction_data[key] = np.full(final_pred_len, np.nan)
try:
    pred_df = pd.DataFrame(valid_prediction_data)
    pred_path = OUTPUT_DIR / "ensemble_predictions_with_weighted.csv" # Nuovo nome file
    pred_df.to_csv(pred_path, index=False)
    print(f"\n✅ Ensemble predictions saved to {pred_path}")
except Exception as e:
    print(f"\n❌ Error saving predictions CSV: {e}")

print(f"\n🏁 Training script finished. Models saved in: {OUTPUT_DIR}")