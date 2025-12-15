# interface/backend/hgb_best_simple.py
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel

# ============================================================
# TODO: GARANTE que estas funções/classes estão importáveis aqui
# (ou cola-as neste ficheiro)
#
# fill_unknown
# column_string_transformer
# fit_year_median / transform_year_with_model_median
# fit_mileage_imputer / transform_mileage_imputer
# fit_engine_size_imputer / transform_engine_size_imputer
# transform_tax_custom_rules
# fit_mpg_imputer / transform_mpg_imputer
# fit_ambiguous_brand_resolver / transform_ambiguous_brands
# fit_invalid_model_resolver / transform_invalid_models
# fit_transmission_resolver / transform_transmission_resolver
# fit_fueltype_resolver / transform_fueltype_resolver
# create_age_and_drop_year
# MyTargetEncoder / MyOneHotEncoder
# ============================================================
from .preproc_helpers import (
    cat_feat,
    correct_invalid_brands_in_df,
    preprocess_categorical,
    fill_unknown,
    column_string_transformer,
    basic_string_transformer,
    fit_year_median, transform_year_with_model_median,
    fit_mileage_imputer, transform_mileage_imputer,
    fit_engine_size_imputer, transform_engine_size_imputer,
    transform_tax_custom_rules,
    fit_mpg_imputer, transform_mpg_imputer,
    fit_ambiguous_brand_resolver, transform_ambiguous_brands,
    fit_invalid_model_resolver, transform_invalid_models,
    fit_transmission_resolver, transform_transmission_resolver,
    fit_fueltype_resolver, transform_fueltype_resolver,
    create_age_and_drop_year,
    MyTargetEncoder, MyOneHotEncoder,
)

### the changes made in preproc_helpers.py regarding full_train_dataset
valid_brands = ['FORD', 'MERCEDES', 'VW', 'OPEL', 'BMW', 'AUDI', 'TOYOTA', 'SKODA', 'HYUNDAI']
full_train_dataset = pd.read_csv("project_data/train.csv")
full_train_dataset = full_train_dataset.drop(columns=['carID', 'hasDamage', 'paintQuality%'])
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

valid_transmissions = ['MANUAL', 'AUTOMATIC', 'SEMIAUTO']
valid_fueltypes = ['PETROL', 'DIESEL', 'HYBRID'] 

valid_models_by_brand = {
    brand: [
        basic_string_transformer(
            model,
            remove_middle_spaces=True, # default
            allow_extra_chars=""       # default
        )
        for model in models
    ]
    for brand, models in valid_models_by_brand.items()
}



####
# this is what we run in the model's files
TARGET_COL = "price"  
# separate features and target variable from the full training datase
y = full_train_dataset[TARGET_COL].copy()
X = full_train_dataset.drop(columns=[TARGET_COL]).copy()

# at this point, this are all the features in use 
categorical_features = ['Brand', 'model', 'transmission', 'fuelType']              
numeric_features = ['year', 'mileage', 'engineSize', 'tax', 'mpg', 'previousOwners']

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Num features:", numeric_features)
print("Cat features:", categorical_features)

N_SPLITS = 8 # the number of folds for K-Fold cross-validation
RANDOM_STATE = 42 # seed to control randomness


N_SPLITS = 8
N_ITER = 10
RANDOM_STATE = 42

###

# CONFIG 
RANDOM_STATE = 42

NUMERIC_FEATURES = ["year", "mileage", "engineSize", "tax", "mpg"]

FINAL_PARAMS = {
    "min_samples_leaf": 16,
    "max_leaf_nodes": 191,
    "max_iter": 1200,
    "max_depth": 20,
    "loss": "squared_error",
    "learning_rate": 0.07,
    "l2_regularization": 3.0,
}

FS_KEEP_RATIO = 1.0  # como no teu teste
RF_FS_PARAMS = {
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
}

COLS_TO_NORMALIZE = ["Brand", "model", "transmission", "fuelType"]
HIGH_CARD = ["Brand", "model"]
BASE_YEAR = 2020


def fit_hgb_best(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    categorical_features: list,
    valid_brands,
    valid_models_by_brand,
    valid_transmissions,
    valid_fueltypes,
) -> dict:
    """
    Faz o teu 'FIT ON FULL TRAIN' + encoding + FS + treino final.
    Devolve um dict com tudo o que é preciso para inferência.
    """
    X_full = X.copy()
    y_full = y.copy()
    y_full_log = np.log1p(y_full.astype(float))

    low_card_features = [c for c in categorical_features if c not in HIGH_CARD]
    low_card_curr = low_card_features  # sem engine_bin

    # 3) STRING NORMALIZATION (TRAIN)
    for col in COLS_TO_NORMALIZE:
        if col in X_full.columns:
            X_full[col] = fill_unknown(X_full[col])
            X_full = column_string_transformer(
                X_full, column=col, remove_middle_spaces=True, allow_extra_chars=""
            )

    # schema esperado (como no teu teste)
    expected_cols = [c for c in (NUMERIC_FEATURES + categorical_features) if c in X_full.columns]
    X_full = X_full[expected_cols].copy()

    # 4) FIT + TRANSFORM ON FULL TRAIN
    year_state = fit_year_median(X_full, year_col="year", model_col="model")
    X_full = transform_year_with_model_median(X_full, state=year_state)

    mileage_state = fit_mileage_imputer(X_full, mileage_col="mileage", do_abs=True)
    X_full = transform_mileage_imputer(X_full, state=mileage_state)

    engine_state = fit_engine_size_imputer(X_full, engine_col="engineSize")
    X_full = transform_engine_size_imputer(X_full, state=engine_state)

    X_full = transform_tax_custom_rules(X_full, "tax", "year", "fuelType", "engineSize")

    mpg_state = fit_mpg_imputer(X_full, mpg_col="mpg", do_abs=True)
    X_full = transform_mpg_imputer(X_full, state=mpg_state)

    brand_state = fit_ambiguous_brand_resolver(X_full, valid_brands)
    X_full, _, _ = transform_ambiguous_brands(X_full, brand_state)

    model_state = fit_invalid_model_resolver(X_full, valid_models_by_brand)
    X_full, _, _ = transform_invalid_models(X_full, model_state)

    transm_state = fit_transmission_resolver(X_full, valid_transmissions)
    X_full, _, _ = transform_transmission_resolver(X_full, transm_state)

    fuel_state = fit_fueltype_resolver(X_full, valid_fueltypes)
    X_full, _, _ = transform_fueltype_resolver(X_full, fuel_state)

    # 5) FE: ONLY AGE (drop year)
    X_full = create_age_and_drop_year(X_full, year_col="year", base_year=BASE_YEAR)

    # 6) ENCODING (fit on full train) using LOG TARGET
    te = MyTargetEncoder(smoothing=5)
    te.fit(X_full[HIGH_CARD], y_full_log)
    X_full_high = te.transform(X_full[HIGH_CARD])

    ohe = MyOneHotEncoder()
    ohe.fit(X_full[low_card_curr])
    X_full_low = ohe.transform(X_full[low_card_curr])

    X_full_cat = pd.concat([X_full_high, X_full_low], axis=1)

    drop_for_numeric = set(HIGH_CARD + low_card_curr)
    numeric_features_curr = [c for c in X_full.columns if c not in drop_for_numeric]

    X_full_final = pd.concat([X_full[numeric_features_curr], X_full_cat], axis=1)
    train_columns = list(X_full_final.columns)

    # 7) FEATURE SELECTION (kept)
    n_feats = X_full_final.shape[1]
    k = int(np.ceil(FS_KEEP_RATIO * n_feats))
    k = max(1, min(k, n_feats))

    rf_fs = RandomForestRegressor(**RF_FS_PARAMS)
    rf_fs.fit(X_full_final, y_full_log)

    selector = SelectFromModel(
        estimator=rf_fs,
        threshold=-np.inf,
        max_features=k,
        prefit=True
    )

    selected_cols = list(X_full_final.columns[selector.get_support()])
    X_full_sel = X_full_final[selected_cols]

    # 8) TRAIN FINAL HGB (LOG TARGET)
    hgb_final = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **FINAL_PARAMS)
    hgb_final.fit(X_full_sel, y_full_log)

    return {
        "expected_cols": expected_cols,
        "categorical_features": categorical_features,
        "low_card_curr": low_card_curr,
        "numeric_features_curr": numeric_features_curr,
        "train_columns": train_columns,
        "selected_cols": selected_cols,

        "year_state": year_state,
        "mileage_state": mileage_state,
        "engine_state": engine_state,
        "mpg_state": mpg_state,
        "brand_state": brand_state,
        "model_state": model_state,
        "transm_state": transm_state,
        "fuel_state": fuel_state,

        "te": te,
        "ohe": ohe,
        "model": hgb_final,
    }


def predict_hgb_best(df_input: pd.DataFrame, fitted: dict, *, id_col: str = "carID") -> pd.DataFrame:
    """
    Faz o teu 'TRANSFORM TEST + PREDICT', mas usando df_input em vez de test.csv.
    """
    df = df_input.copy()

    # Normalização strings (igual)
    for col in COLS_TO_NORMALIZE:
        if col in df.columns:
            df[col] = fill_unknown(df[col])
            df = column_string_transformer(
                df, column=col, remove_middle_spaces=True, allow_extra_chars=""
            )

    # schema base (igual)
    X_test = df.reindex(columns=fitted["expected_cols"], fill_value=np.nan).copy()

    # transforms (transform-only)
    X_test = transform_year_with_model_median(X_test, state=fitted["year_state"])
    X_test = transform_mileage_imputer(X_test, state=fitted["mileage_state"])
    X_test = transform_engine_size_imputer(X_test, state=fitted["engine_state"])
    X_test = transform_tax_custom_rules(X_test, "tax", "year", "fuelType", "engineSize")
    X_test = transform_mpg_imputer(X_test, state=fitted["mpg_state"])

    X_test, _, _ = transform_ambiguous_brands(X_test, fitted["brand_state"])
    X_test, _, _ = transform_invalid_models(X_test, fitted["model_state"])
    X_test, _, _ = transform_transmission_resolver(X_test, fitted["transm_state"])
    X_test, _, _ = transform_fueltype_resolver(X_test, fitted["fuel_state"])

    # FE: ONLY AGE
    X_test = create_age_and_drop_year(X_test, year_col="year", base_year=BASE_YEAR)

    # encoding transform-only
    te = fitted["te"]
    ohe = fitted["ohe"]
    low_card_curr = fitted["low_card_curr"]

    X_test_high = te.transform(X_test[HIGH_CARD])
    X_test_low  = ohe.transform(X_test[low_card_curr])
    X_test_cat  = pd.concat([X_test_high, X_test_low], axis=1)

    drop_for_numeric = set(HIGH_CARD + low_card_curr)
    numeric_features_curr_test = [c for c in X_test.columns if c not in drop_for_numeric]

    X_test_final = pd.concat([X_test[numeric_features_curr_test], X_test_cat], axis=1)

    # align to train columns before selecting
    X_test_final = X_test_final.reindex(columns=fitted["train_columns"], fill_value=0)

    # apply selected cols
    X_test_sel = X_test_final.reindex(columns=fitted["selected_cols"], fill_value=0)

    # predict + invert log1p
    pred_log = fitted["model"].predict(X_test_sel)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0)

    out = pd.DataFrame({"price": pred})
    if id_col in df.columns:
        out.insert(0, id_col, df[id_col].values)

    return out
