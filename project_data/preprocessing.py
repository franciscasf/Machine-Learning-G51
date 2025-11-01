import numpy as np
import pandas as pd 
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def abs_transform(X):
    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce").abs()
    return np.abs(np.asarray(X, dtype=float))

def floor_transform(X):
    if isinstance(X, pd.DataFrame):
        return np.floor(X.apply(pd.to_numeric, errors="coerce"))
    return np.floor(np.asarray(X, dtype=float))

abs_tf   = FunctionTransformer(abs_transform,   feature_names_out="one-to-one")
floor_tf = FunctionTransformer(floor_transform, feature_names_out="one-to-one")


def _ensure_df(X):
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

def _clip_df_or_array(X, low=None, high=None):
    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce").clip(lower=low, upper=high)
    X = np.asarray(X, dtype=float)
    return np.clip(X, low if low is not None else -np.inf, high if high is not None else np.inf)



# to be used in year columns
# to be used in year columns
def fix_max_year(X, max_year=2020):
    return _clip_df_or_array(X, low=None, high=max_year)

fix_year_transformer = FunctionTransformer(
    fix_max_year,
    kw_args={"max_year": 2020},
    feature_names_out="one-to-one"  
)

#-----


# para o year vou por um dropper na pipeline 


# previous owners
class FixPreviousOwners(BaseEstimator, TransformerMixin):
    def __init__(self, max_year=2020, mileage_threshold=15000, age_threshold=2):
        self.max_year = max_year
        self.mileage_threshold = mileage_threshold
        self.age_threshold = age_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _ensure_df(X).copy()
        required = {"year", "mileage", "previousOwners"}
        if required.issubset(set(X.columns)):
            # garantir numéricos
            X["year"] = pd.to_numeric(X["year"], errors="coerce")
            X["mileage"] = pd.to_numeric(X["mileage"], errors="coerce")
            X["previousOwners"] = pd.to_numeric(X["previousOwners"], errors="coerce")

            cond = (
                ((self.max_year - X["year"]) > self.age_threshold) |
                (X["mileage"] > self.mileage_threshold)
            ) & (X["previousOwners"] == 0)

            X.loc[cond, "previousOwners"] = 1
            # volta a int se possível
            try:
                X["previousOwners"] = X["previousOwners"].astype("Int64").astype("int64")
            except Exception:
                pass
        return X

# paint quality 
def fix_paint_quality(X, min_val=0, max_val=100):
    return _clip_df_or_array(X, low=min_val, high=max_val)

fix_paint_transformer = FunctionTransformer(
    fix_paint_quality,
    kw_args={'min_val': 0, 'max_val': 100},
    feature_names_out='one-to-one'
)


# mpg value range corrections
def fix_mpg(X, min_val=10, max_val=200):
    return _clip_df_or_array(X, low=min_val, high=max_val)

fix_mpg_transformer = FunctionTransformer(
    fix_mpg,
    kw_args={'min_val': 10, 'max_val': 200},
    feature_names_out='one-to-one'
)

