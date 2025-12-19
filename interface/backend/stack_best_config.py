from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Preprocessing helpers (reused from the notebooks)
from .preproc_helpers import (
    cat_feat,
    correct_invalid_brands_in_df,
    preprocess_categorical,
    fill_unknown,
    column_string_transformer,
    basic_string_transformer,
    fit_year_median,
    transform_year_with_model_median,
    fit_mileage_imputer,
    transform_mileage_imputer,
    fit_engine_size_imputer,
    transform_engine_size_imputer,
    transform_tax_custom_rules,
    fit_mpg_imputer,
    transform_mpg_imputer,
    fit_previous_owners_imputer,
    transform_previous_owners_imputer,
    fit_ambiguous_brand_resolver,
    transform_ambiguous_brands,
    fit_invalid_model_resolver,
    transform_invalid_models,
    fit_transmission_resolver,
    transform_transmission_resolver,
    fit_fueltype_resolver,
    transform_fueltype_resolver,
    MyTargetEncoder,
    MyOneHotEncoder,
)

# Dataset loading (done once at import time)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "project_data" / "train.csv"
# Load train once
full_train_dataset = pd.read_csv(TRAIN_PATH)
full_train_dataset = full_train_dataset.drop(columns=['carID', 'hasDamage', 'paintQuality%'])

RANDOM_STATE = 42
TARGET_COL = "price"

valid_brands = ['FORD', 'MERCEDES', 'VW', 'OPEL', 'BMW', 'AUDI', 'TOYOTA', 'SKODA', 'HYUNDAI']
full_train_dataset = preprocess_categorical(
    full_train_dataset,
    cat_feat,
    remove_middle_spaces=True,
    allow_extra_chars=""
)
invalids = sorted(
    [b for b in full_train_dataset['Brand'].unique() if b not in valid_brands],
    key=len
)

full_train_dataset, corrections, remaining_invalids = correct_invalid_brands_in_df(
    full_train_dataset,
    col='Brand',
    valid_brands=valid_brands,
    invalids=invalids
)

valid_models_by_brand= {'FORD': ['FOCUS', 'FIESTA', 'KUGA', 'ECOSPORT', 'C-MAX', 'KA+', 'MANDEO' ],
'MERCEDES': ['C CLASS', 'A CLASS', 'E CLASS','GLC CLASS', 'GLA CLASS', 'B CLASS', 'CL CLASS', 'GLE CLASS'],
'VW': ['GOLF', 'POLO', 'TIGUAN', 'PASSAT', 'UP', 'T-ROC', 'TOUAREG', 'TOURAN', 'T-CROSS'],
'OPEL': ['CORSA', 'ASTRA', 'MOKKA X', 'INSIGNIA', 'MOKKA', 'CROSSLAND X', 'ZAFIRA', 'GRANDLAND X', 'ADAM', 'VIVA'],
'BMW': ['1 SERIES','2 SERIES','3 SERIES','4 SERIES','5 SERIES', 'X1', 'X3', 'X5', 'X2', 'X4', 'M4', '6 SERIES', 'Z4', 'X6', '7 SERIES', 'X7'],
'AUDI': ['A3', 'Q3', 'A4', 'A1', 'Q5', 'A5', 'Q2', 'A6', 'Q7', 'TT'],
'TOYOTA': ['YARIS', 'AYGO', 'AURIS', 'C-HR', 'RAV4', 'COROLLA', 'PRIUS', 'VERSO'],
'SKODA': ['FABIA', 'OCTAVIA', 'SUPERB', 'YETI OUTDOOR', 'CITIGO', 'KODIAQ', 'KAROQ', 'SCALA','KAMIQ', 'RAPID', 'YETI'],
'HYUNDAI': ['TUCSON', 'I10', 'I30', 'I20', 'KONA', 'IONIQ', 'SANTA FE', 'IX20', 'I40', 'IX35', 'I800']
}


valid_models_by_brand = {
    brand: [
        basic_string_transformer(
            model,
            remove_middle_spaces=True, # default
            allow_extra_chars=""  # default
        )
        for model in models
    ]
    for brand, models in valid_models_by_brand.items()
}

valid_transmissions = ['MANUAL', 'AUTOMATIC', 'SEMIAUTO']
valid_fueltypes = ['PETROL', 'DIESEL', 'HYBRID']


# Separate features and target
y = full_train_dataset[TARGET_COL].copy()
X = full_train_dataset.drop(columns=[TARGET_COL]).copy()

# Explicit feature schema (the same order used in stacking notebook)
categorical_features = ["Brand", "model", "transmission", "fuelType"]
numeric_features = ["mileage", "engineSize", "tax", "mpg", "year"]

COLS_TO_NORMALIZE = ["Brand", "model", "transmission", "fuelType"]
HIGH_CARD = ["Brand", "model"]
DROP_FROM_MODEL = ["previousOwners"]

STACK_PARAMS = {
    "hgb": {
        "max_iter": 1000,
        "learning_rate": 0.1,
        "max_depth": 20,
        "max_leaf_nodes": 191,
        "min_samples_leaf": 20,
        "l2_regularization": 3.0,
        "random_state": RANDOM_STATE,
    },
    "rf": {
        "n_estimators": 1000,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": 0.33,
        "max_depth": 20,
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "et": {
        "n_estimators": 800,
        "max_depth": 20,
        "min_samples_split": 4,
        "min_samples_leaf": 1,
        "max_features": 0.7,
        "bootstrap": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}


def fit_stacking_best(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    valid_brands: list,
    valid_models_by_brand: dict,
    valid_transmissions: list,
    valid_fueltypes: list,
    categorical_features: list = categorical_features,
    numeric_features: list = numeric_features,
    params: dict = STACK_PARAMS,
) -> dict:
    """
    Fit the full preprocessing + encoding + scaling pipeline and a stacking model.

    This is intentionally aligned with the stacking submission script:
    - string normalization with column_string_transformer(df, col)
    - previousOwners imputation, then drop previousOwners for modeling
    - target encoding (high-card) + one-hot (low-card)
    - StandardScaler
    - StackingRegressor(cv=5, passthrough=False) with Ridge(alpha=4.0)
    - model trained on log1p(price); predictions inverted with expm1
    """
    X_full = X.copy()
    y_full = y.copy().astype(float)
    y_full_log = np.log1p(y_full)

    # Low-card features are the remaining categoricals
    low_card_features = [c for c in categorical_features if c not in HIGH_CARD]

    # String normalization (train)
    for col in COLS_TO_NORMALIZE:
        if col in X_full.columns:
            X_full[col] = fill_unknown(X_full[col])
            X_full = column_string_transformer(X_full, col)

    # Fit all transforms on full train (states reused at inference)
    year_state = fit_year_median(X_full, "year", "model")
    X_full = transform_year_with_model_median(X_full, year_state)

    mileage_state = fit_mileage_imputer(X_full, "mileage", do_abs=True)
    X_full = transform_mileage_imputer(X_full, mileage_state)

    engine_state = fit_engine_size_imputer(X_full, "engineSize")
    X_full = transform_engine_size_imputer(X_full, engine_state)

    X_full = transform_tax_custom_rules(X_full, "tax", "year", "fuelType", "engineSize")

    mpg_state = fit_mpg_imputer(X_full, "mpg", do_abs=True)
    X_full = transform_mpg_imputer(X_full, mpg_state)

    owners_state = fit_previous_owners_imputer(X_full, "previousOwners", "year", "mileage")
    X_full = transform_previous_owners_imputer(X_full, owners_state)

    brand_state = fit_ambiguous_brand_resolver(X_full, valid_brands)
    X_full, _, _ = transform_ambiguous_brands(X_full, brand_state)

    model_state = fit_invalid_model_resolver(X_full, valid_models_by_brand)
    X_full, _, _ = transform_invalid_models(X_full, model_state)

    transm_state = fit_transmission_resolver(X_full, valid_transmissions)
    X_full, _, _ = transform_transmission_resolver(X_full, transm_state)

    fuel_state = fit_fueltype_resolver(X_full, valid_fueltypes)
    X_full, _, _ = transform_fueltype_resolver(X_full, fuel_state)

    # Drop previousOwners (ablation)
    X_full = X_full.drop(columns=DROP_FROM_MODEL, errors="ignore")

    # Encoding
    te = MyTargetEncoder(smoothing=5)
    te.fit(X_full[HIGH_CARD], y_full)
    X_high = te.transform(X_full[HIGH_CARD])

    ohe = MyOneHotEncoder()
    ohe.fit(X_full[low_card_features])
    X_low = ohe.transform(X_full[low_card_features])

    X_train_final = pd.concat([X_full[numeric_features], X_high, X_low], axis=1)
    train_columns = list(X_train_final.columns)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)

    # Stacking model
    estimators = [
        ("hgb", HistGradientBoostingRegressor(**params["hgb"])),
        ("rf", RandomForestRegressor(**params["rf"])),
        ("et", ExtraTreesRegressor(**params["et"])),
    ]

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=4.0),
        cv=5,
        passthrough=False,
        n_jobs=1,
    )

    stack.fit(X_train_scaled, y_full_log)

    return {
        "expected_cols": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "low_card_features": low_card_features,
        "train_columns": train_columns,
        "year_state": year_state,
        "mileage_state": mileage_state,
        "engine_state": engine_state,
        "mpg_state": mpg_state,
        "owners_state": owners_state,
        "brand_state": brand_state,
        "model_state": model_state,
        "transm_state": transm_state,
        "fuel_state": fuel_state,
        "te": te,
        "ohe": ohe,
        "scaler": scaler,
        "model": stack,
    }


def predict_stacking_best(df_input: pd.DataFrame, fitted: dict, *, id_col: str = "carID") -> pd.DataFrame:
    """
    Apply the fitted pipeline and return price predictions in the original scale.
    """
    df = df_input.copy()

    # Preserve IDs 
    ids = df[id_col].copy() if id_col in df.columns else None

    # String normalization (test)
    for col in COLS_TO_NORMALIZE:
        if col in df.columns:
            df[col] = fill_unknown(df[col])
            df = column_string_transformer(df, col)

    # Align to expected input columns
    X_test = df.reindex(columns=fitted["expected_cols"], fill_value=np.nan).copy()

    # Apply transforms with fitted states
    X_test = transform_year_with_model_median(X_test, fitted["year_state"])
    X_test = transform_mileage_imputer(X_test, fitted["mileage_state"])
    X_test = transform_engine_size_imputer(X_test, fitted["engine_state"])
    X_test = transform_tax_custom_rules(X_test, "tax", "year", "fuelType", "engineSize")
    X_test = transform_mpg_imputer(X_test, fitted["mpg_state"])
    X_test = transform_previous_owners_imputer(X_test, fitted["owners_state"])

    X_test, _, _ = transform_ambiguous_brands(X_test, fitted["brand_state"])
    X_test, _, _ = transform_invalid_models(X_test, fitted["model_state"])
    X_test, _, _ = transform_transmission_resolver(X_test, fitted["transm_state"])
    X_test, _, _ = transform_fueltype_resolver(X_test, fitted["fuel_state"])

    # Drop previousOwners (ablation)
    X_test = X_test.drop(columns=DROP_FROM_MODEL, errors="ignore")

    # Encoding
    X_test_high = fitted["te"].transform(X_test[HIGH_CARD])
    X_test_low = fitted["ohe"].transform(X_test[fitted["low_card_features"]])

    X_test_final = pd.concat(
        [X_test[fitted["numeric_features"]], X_test_high, X_test_low],
        axis=1,
    )

    # Align columns to training matrix
    X_test_final = X_test_final.reindex(columns=fitted["train_columns"], fill_value=0)

    # Scale and predict
    X_test_scaled = fitted["scaler"].transform(X_test_final)

    pred_log = fitted["model"].predict(X_test_scaled)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

    out = pd.DataFrame({"price": pred})
    if ids is not None:
        out.insert(0, id_col, ids.values)

    return out
