#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# __TABLE OF CONTENTS__
# 1. [Introduction](#intro)
# 2. [Setup and Data Loading](#setup)
# 3. [Global Pre-Processing](#global)
# 4. [Fit and Transform Functions](#fit-transf)
# 5. [Encoding](#encoding)
# 6. [Feature Selection](#fs)
# 7. [General Pre-Processing Application](#general-preproc)
# 8. [Feature Engineering](#fe)

# <a id="intro"></a>
# ## 1. Introduction

# This notebook contains all preprocessing functions that will be applied to the training and validation datasets during the cross-validation stage, as well as to the final test dataset.
# 
# - Some of these functions follow a fit-transform logic, meaning they learn parameters from the training fold (fit) and then apply the corresponding transformation to the validation fold (transform), ensuring that no data leakage occurs during k-fold cross-validation;
# 
# - Other functions are deterministic and dataset-independent, so they are executed before starting cross-validation because their output does not depend on the data split.

# <a id="setup"></a>
# ## 2. Setup and Data Loading

# We load the raw datasets and prepare the initial environment for preprocessing.  
# No transformations are applied here: this section only establishes the working
# data structures used throughout the notebook.

# In[ ]:


import pandas as pd
import numpy as np
import unicodedata
import re
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestRegressor


# In[2]:


# full_train_dataset = pd.read_csv('../../project_data/train.csv')


# <a id="global"></a>
# ## 3. Global preprocessing (pre-CV)

# These transformations are deterministic and dataset-independent.
# 
# They standardize categorical strings, replace missing values with `"UNKNOWN"`,
# and ensure all text fields follow the same normalization rules before any
# fit-based logic is applied.
# 
# Since nothing here depends on training statistics, this step is performed
# before cross-validation to avoid leakage.

# ### 3.1. "fill_unknown" 

# In[ ]:


def fill_unknown(series):
    """
    Replace missing values in a pandas Series with the string literal 'UNKNOWN'.

    This function is applied before cross-validation, ensuring that categorical-like variables consistently
    encode missingness as an explicit category rather than dropping rows or imputing statistically.

    - Parameters
    series : pd.Series
        A pandas Series that may contain missing values (NaN). Ideally used for categorical columns.

    - Behavior
        1) All NaN entries are replaced with the string 'UNKNOWN'.
        2) Existing non-null values are kept unchanged.
        3) The returned Series preserves the original index.

      Representing missing values with an explicit token is beneficial because:
        1) It avoids mixing missingness with the mode/most frequent category, which could introduce bias.
        2) It preserves information that the value was not originally provided, allowing the model or encoder to learn from the missingness pattern.
        3) It avoids row deletion, ensuring dataset size remains consistent.
        
    - Returns
    pd.Series
        A new Series where missing values have been substituted with 'UNKNOWN'.
    """
    return series.fillna("UNKNOWN")


# ### 3.2. "basic_string_transformer" 
#  

# In[ ]:


def basic_string_transformer(
    word, 
    remove_middle_spaces: bool = True,
    allow_extra_chars: str = ""
    ):
    """
    Apply controlled string normalization to a single value.  
    This function standardizes textual entries before model training, reducing noise from formatting differences 
    (capitalization, accents, spacing, punctuation) and ensuring consistent behavior in later encoding steps.

    - Parameters
    word : any
        Input value to normalize. Expected to be a string or convertible to string. 
        Missing values (NaN) are detected and returned unchanged.
    remove_middle_spaces : bool, default True
        If True, all internal spaces are removed.  
        If False, multiple consecutive spaces are compressed to a single space and preserved.
    allow_extra_chars : str, default ""
        Additional characters that should NOT be removed during punctuation filtering.
        For example, passing "-/" keeps '-' and '/' in the output.

    - Behavior
        1) If the input is NaN, return it unchanged so later imputers can handle it.
        2) Convert to string and remove leading/trailing whitespace.
        3) Convert to uppercase for consistent categorical representation.
        4) Remove accents using Unicode normalization ("NFD") to avoid treating accented and non-accented
           versions of the same word as different categories.
        5) Normalize spacing:
             If remove_middle_spaces=True, remove all spaces;
             Otherwise, compress multiple spaces into one and trim edges.
        6) Remove symbols/punctuation not included in the allowed set:
             By default, keep only uppercase letters A-Z and digits 0-9;
             If allow_extra_chars is provided, those characters are preserved;
             If middle spaces are kept, space is also added to the allowed set.
        7) After cleaning, if the resulting string becomes empty (original value was only symbols), return NaN 
        so missingness can be consistently tracked.

    - Returns
    str or np.nan
        The normalized string, or NaN if the transformation removes all meaningful content.
    """

    if pd.isna(word):
        return word
    
    s = str(word)

    # strip the leading and trailing whitespaces 
    s = s.strip()

    # uppercase the string
    s = s.upper()

    # removal of accents
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

    # remove middle spaces if required
    if remove_middle_spaces:
        s = s.replace(" ", "")
    else:
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
    
    # remove punctuation/symbols etc.
    allowed = allow_extra_chars or "" 
    if not remove_middle_spaces:
        allowed += " "
    pattern = rf"[^A-Z0-9{re.escape(allowed)}]"
    s = re.sub(pattern, "", s)

    # if string becomes empty or only has spaces, return NaN
    if s.strip() == "":
        return np.nan
    
    return s


# ### 3.3. "column_string_transformer" 

# In[ ]:


def column_string_transformer(
    df: pd.DataFrame,
    column: str,
    remove_middle_spaces: bool = True,
    allow_extra_chars: str = ""
    ) -> pd.DataFrame: #returns a dataframe
    """ 
    Apply basic string normalization to a given column of a DataFrame.
    This function processes all entries in the specified column using the `basic_string_transformer`,
    ensuring consistent formatting across the dataset before model training.

    - Parameters: 
    df : pd.DataFrame
        Input DataFrame containing the column to be transformed. The DataFrame is not
        modified in place; a copy is produced.
    column : str
        Name of the column to transform.
        Expected to contain textual or categorical-like entries.
    remove_middle_spaces : bool, default True
        If True, remove all spaces (including in the middle);
        If False, keep inner spaces (normalizing multiple spaces to just one).
    allow_extra_chars : str, default ""
        String of extra characters to keep (e.g., "-/" to keep '-' and '/').

    - Behavior
        1) Creates a copy of the DataFrame to avoid modifying the original object.
        2) Applies `basic_string_transformer` to every value in the specified column.
        3) Converts the resulting column to pandas' StringDtype for consistency in
           encoding steps and to avoid mixed Python-object/string types.

    - Returns
    pd.DataFrame
        A new DataFrame identical to the input except for the specified column,
        which is replaced by its normalized version.
    """
    
    df_out = df.copy()
    df_out[column] = df_out[column].apply(
        lambda x: basic_string_transformer(
            x,      
            remove_middle_spaces=remove_middle_spaces,
            allow_extra_chars=allow_extra_chars,
        )
    )   
    df_out[column] = df_out[column].astype("string")

    return df_out


# ### 3.4. "correct_invalid_brands_in_df" 
# 

# In[6]:


valid_brands = ['FORD', 'MERCEDES', 'VW', 'OPEL', 'BMW', 'AUDI', 'TOYOTA', 'SKODA', 'HYUNDAI']


# In[ ]:


def correct_invalid_brands_in_df(df, col, valid_brands, invalids):
    """
    Identify and correct invalid brand names in a DataFrame column by matching them
    to valid brand names using substring containment. This function standardizes
    categorical values before modeling, ensuring that misspellings or inconsistent
    variants are reconciled with the known valid set.

    - Parameters
    df : pd.DataFrame
        The DataFrame containing the column to be corrected. The function modifies
        this DataFrame directly (in-place) for efficiency.
    col : str
        The name of the column containing brand labels to validate and correct.
    valid_brands : list of str
        A list of all accepted brand names. Any values not included here are treated
        as invalid unless corrected.
    invalids : list of str
        A list of detected invalid brand strings (typically after cleaning steps).
        These are the entries subject to correction.

    - Behavior
        1) Initializes an empty dictionary `corrections` to record which invalid
           labels get mapped to which valid brand.
        2) Iterates through each item in `invalids`:
             Skips the placeholder "UNKNOWN", which is intentionally not corrected;
             For each invalid brand, checks how many valid brands contain it
             as a substring;
             If it matches exactly one valid brand, it replaces occurrences
             of the invalid label with that valid brand and records the correction.
        3) After processing all invalids, identifies any remaining invalid entries
           still present in the column. These are values that:
             Matched multiple valid brands (ambiguous), OR
             Had no match at all.
        4) Returns both the corrected DataFrame and diagnostic information about
           what was corrected and what remains unresolved.

    - Returns
    tuple
        (df, corrections, remaining_invalids)

        df : pd.DataFrame  
            The DataFrame with corrected brand names.
        corrections : dict  
            A mapping {invalid_brand -> corrected_valid_brand} showing all automatic fixes applied.
        remaining_invalids : list  
            Values in the column that are still invalid after correction, indicating
            ambiguous or unmatched cases that require manual inspection.
    """
    
    # Dictionary to store corrections applied
    corrections = {}
    
    for invalid in invalids:
        if invalid == 'UNKNOWN':
            continue  # Skip UNKNOWN values
        
        # Check if the invalid brand is contained in exactly one valid brand
        matches = [vb for vb in valid_brands if invalid in vb]
        
        if len(matches) == 1:
            valid = matches[0]
            df.loc[df[col] == invalid, col] = valid  # Replace invalid with the valid brand
            corrections[invalid] = valid

    # Identify any remaining invalid values that were not corrected
    remaining_invalids = [
        b for b in df[col].unique() 
        if b not in valid_brands and b not in corrections.keys()
    ]
    
    return df, corrections, remaining_invalids


# <a id="fit-tranf"></a>
# ## 4. Fit and Transform Functions

# These preprocessing components follow the standard `.fit()` / `.transform()` 
# pattern:  
# - **fit()** learns correction or imputation rules using the training fold;
# - **transform()** applies those rules to validation and test data.
# 
# This structure guarantees that no information from validation or test leaks
# into the training process during cross-validation.

# ### 4.1. Brand

# In[8]:


valid_brands # previously defined list of valid brands


# In[ ]:


def _choose_brand_from_counts(counts: pd.Series, invalid_brand: str | None) -> str | None:
    """
    Select the most appropriate brand label from a frequency count vector,
    resolving ties using substring logic. This helper is used to infer a corrected brand
    for an invalid or ambiguous entry based on contextual frequency information.

    - Parameters
    counts : pd.Series
        A Series typically produced by value_counts(), where the index contains 
        candidate brand names and the values represent frequencies within a specific group
        (e.g., all cars with the same model, same fuelType, etc.)
        The Series is expected to be sorted in descending order, which is the
        default behavior of value_counts().
    invalid_brand : str or None
        The original invalid or ambiguous brand string we are trying to resolve.
        Used only for tie-breaking via substring matching. If None, substring
        resolution is skipped.

    - Behavior
        1) If `counts` is empty or None, return None because no inference is possible.
        2) Identify the highest frequency (the mode) and determine all brands tied
           for that maximum count.
        3) If there is only one top brand, return it immediately.
        4) If multiple brands tie:
             4.1. Convert invalid_brand to uppercase for normalization.
             4.2. Check whether exactly one of the tied brands contains the invalid brand
                  substring (case-insensitive).
             4.3. If exactly one tied brand matches, return that one.
             4.4. Otherwise, return the first brand in the tied list (stable fallback).
        5) If no rule applies (substring ambiguous), the deterministic fallback
           ensures reproducible results.

    - Returns
    str or None
        The chosen brand name, or None when no decision can be made due to empty input.
    """

    if counts is None or len(counts) == 0:
        return None

    # value_counts is already sorted by frequency (descending) by default
    max_count = counts.iloc[0]
    top_brands = counts[counts == max_count].index.tolist()

    if len(top_brands) == 1:
        # Unique mode
        return top_brands[0]

    # There is a tie: try to break it using substring of the invalid brand
    if invalid_brand is None:
        invalid_upper = ""
    else:
        invalid_upper = str(invalid_brand).upper()

    substring_matches = [b for b in top_brands if invalid_upper and invalid_upper in str(b).upper()]

    if len(substring_matches) == 1:
        # Exactly one candidate contains the invalid brand string
        return substring_matches[0]

    # Either no substring match, or more than one -> fall back to the first top brand
    return top_brands[0]


# In[ ]:


def fit_ambiguous_brand_resolver(
    train_df: pd.DataFrame,
    valid_brands: list[str],
    brand_col: str = "Brand",
    model_col: str = "model",
    year_col: str = "year",
) -> Dict[str, Any]:
    """
    Fit step for resolving ambiguous or invalid brand names.

    This function computes all statistical information needed to later correct brand 
    inconsistencies during the transform phase. 
    It uses only the training data to avoid data leakage and builds several frequency
    distributions that help infer the most likely valid brand for any ambiguous entry 
    encountered later.

    - Parameters
    train_df : pd.DataFrame
        The training dataset. Only this data is allowed for learning brand resolution statistics 
        (never validation or test sets!)
    valid_brands : list of str
        Canonical list of all accepted brand names. Any value not in this list is considered 
        invalid and must be resolved during transform.
    brand_col : str, default "Brand"
        Name of the column containing brand labels. It must already be cleaned
        (uppercased, trimmed, accents removed).
    model_col : str, default "model"
        Column indicating the car model. Used to learn per-model brand frequencies.
    year_col : str, default "year"
        Column indicating production year. Used to learn per-year brand frequencies.

    - Behavior
        1) Create a copy of the training data and identify which rows contain
           valid brand labels.
        2) Compute the global most frequent brand (overall mode) among all valid
           brands. This acts as a last-resort fallback during transform.
        3) Compute, for each year, a `value_counts()` distribution of brand
           frequencies restricted to valid brands. These distributions reflect
           which brands are most common in each production year.
        4) Compute, for each model, a similar `value_counts()` distribution.
           These reveal which brands typically appear for a given model name.
        5) Package all learned statistics into a dictionary (`state`) that will
           be fed into the transform function to resolve brand ambiguities.

    - Returns
    dict
        A structured dictionary containing:
            - brand_col, model_col, year_col  
            - valid_brands (as a set for fast lookup)  
            - global_most_common_brand  
            - brand_counts_by_year : {year -> Series of brand frequencies}  
            - brand_counts_by_model : {model -> Series of brand frequencies}  
        This dictionary defines the learned rules for resolving invalid or ambiguous brand 
        labels during the transform phase.
    """
    tmp = train_df.copy()

    valid_set = set(valid_brands)

    # Filter to rows where brand is valid
    valid_mask = tmp[brand_col].isin(valid_set)

    # Global most frequent valid brand
    brand_freq = tmp.loc[valid_mask, brand_col].value_counts()
    global_most_common_brand = brand_freq.index[0] if not brand_freq.empty else None

    # Brand frequency per year (only using valid brands)
    brand_counts_by_year: Dict[Any, pd.Series] = {}
    if year_col in tmp.columns:
        grouped_year = tmp.loc[valid_mask].groupby(year_col)[brand_col]
        for year_val, series in grouped_year:
            # series is the brand column for that year
            counts = series.value_counts()
            brand_counts_by_year[year_val] = counts

    # Brand frequency per model (only using valid brands)
    brand_counts_by_model: Dict[Any, pd.Series] = {}
    if model_col in tmp.columns:
        grouped_model = tmp.loc[valid_mask].groupby(model_col)[brand_col]
        for model_val, series in grouped_model:
            counts = series.value_counts()
            brand_counts_by_model[model_val] = counts

    state = {
        "brand_col": brand_col,
        "model_col": model_col,
        "year_col": year_col,
        "valid_brands": valid_set,
        "global_most_common_brand": global_most_common_brand,
        "brand_counts_by_year": brand_counts_by_year,
        "brand_counts_by_model": brand_counts_by_model,
    }

    return state


# In[ ]:


def transform_ambiguous_brands(
    df: pd.DataFrame,
    state: Dict[str, Any],
) -> tuple[pd.DataFrame, dict, list]:
    """
    Transform step for resolving ambiguous or invalid brand labels using the statistics learned 
    during `fit_ambiguous_brand_resolver`.  
    This function applies the learned rules row by row to assign the most probable valid brand 
    to each inconsistent entry.

    - Parameters
    df : pd.DataFrame
        The dataset to transform. This may be training, validation, or test data. 
        The function will operate on a copy so the original DataFrame remains unchanged.
    state : dict
        The dictionary of learned statistics returned by fit_ambiguous_brand_resolver.
        Contains:
            - brand_col, model_col, year_col  
            - valid_brands (set)
            - global_most_common_brand 
            - brand_counts_by_year
            - brand_counts_by_model 

    - Behavior
        1) Create a copy of the DataFrame and ensure missing brands become "UNKNOWN".
        2) Iterate over each row:
              If the brand is already valid, leave it unchanged.
              Otherwise, determine correction based on the available context:
                 A) If model is missing or equals "UNKNOWN":
                       - If both brand=="UNKNOWN" and year is missing/empty:
                           assign global fallback brand.
                       - Else if year has known statistics:
                           choose brand using _choose_brand_from_counts.
                       - Else:
                           fallback to global most frequent brand.
                 B) If model is present:
                       - If model has known brand-frequency statistics:
                           choose brand using _choose_brand_from_counts.
                       - If no info exists for this model:
                           - If brand=="UNKNOWN", use global fallback.
                           - Otherwise, try to be consistent with past substitutions
                             (past_replacements), or fallback to global most common.
        3) Record every correction in the `corrections` dictionary using the key
           (original_brand, model, year).
        4) Track past replacements so identical invalid values map consistently.
        5) After all rows are processed, identify any remaining invalid brands.

    - Returns
    df_out : pd.DataFrame
        A copy of df with corrected brand values.
    corrections : dict
        A mapping of (original_brand, model, year) -> corrected_valid_brand used for auditing.
    still_invalid : list
        Any brand values that remain invalid after transformation (excluding "UNKNOWN").
    """

    brand_col = state["brand_col"]
    model_col = state["model_col"]
    year_col = state["year_col"]
    valid_brands = state["valid_brands"]
    global_most_common_brand = state["global_most_common_brand"]
    brand_counts_by_year = state["brand_counts_by_year"]
    brand_counts_by_model = state["brand_counts_by_model"]

    df_out = df.copy()

    # Ensure missing brand values do not cause issues downstream
    df_out[brand_col] = df_out[brand_col].fillna("UNKNOWN")

    corrections: dict = {}
    past_replacements: dict = {}

    for idx, row in df_out.iterrows():
        original_brand = row[brand_col]
        brand_upper = str(original_brand).upper() if pd.notna(original_brand) else "UNKNOWN"

        # Skip if already valid
        if brand_upper in valid_brands:
            continue

        model = row.get(model_col, None)
        year = row.get(year_col, None)

        corrected = None

        # Case 1: Model is unknown or missing 
        if pd.isna(model) or str(model).strip().upper() == "UNKNOWN":

            # Special case: both brand and year unknown -> assign global fallback
            if (pd.isna(year) or str(year).strip() == "") and brand_upper == "UNKNOWN":
                corrected = global_most_common_brand

            else:
                # Try to resolve using year-based frequency info
                if pd.notna(year) and year in brand_counts_by_year:
                    counts = brand_counts_by_year[year]
                    corrected = _choose_brand_from_counts(counts, brand_upper)

                # If still unresolved -> global fallback
                if corrected is None:
                    corrected = global_most_common_brand

        # Case 2: Model is known
        else:
            if model in brand_counts_by_model:
                counts = brand_counts_by_model[model]
                corrected = _choose_brand_from_counts(counts, brand_upper)

            # If no info exists for this model
            if corrected is None:
                if brand_upper == "UNKNOWN":
                    corrected = global_most_common_brand
                else:
                    # Maintain consistency across repeated invalid brands
                    corrected = past_replacements.get(brand_upper, global_most_common_brand)

        # Apply the correction and record it
        df_out.at[idx, brand_col] = corrected
        corrections[(original_brand, model, year)] = corrected
        past_replacements[brand_upper] = corrected

    # Identify still-invalid values (excluding the placeholder 'UNKNOWN')
    still_invalid = [
        b for b in df_out[brand_col].unique()
        if b not in valid_brands and b != "UNKNOWN"
    ]

    return df_out, corrections, still_invalid


# ### 4.2. Model

# In[ ]:


def _choose_model_from_counts(
        counts: pd.Series, 
        invalid_model: str | None) -> str | None:
    """
    Select the most appropriate model label from a frequency count vector, using the same 
    decision logic described in `_choose_brand_from_counts`. This function resolves ambiguous 
    or invalid model names by examining model-frequency distributions learned during fitting.

    - Parameters
    counts : pd.Series
        Frequency counts of candidate models, typically obtained via value_counts().
        - Index: possible model names.
        - Values: their frequencies within a given context.
        As with the analogous brand function, the Series is expected to be sorted
        in descending frequency.
    invalid_model : str or None
        The original invalid or ambiguous model string. Used for tie-breaking based
        on substring containment. If None, substring resolution cannot be applied.

    - Behavior
        This function mirrors the tie-resolution strategy used for brands:
        1) If counts is empty -> return None.
        2) Identify all models with the highest frequency.
        3) If there is only one such model -> return it directly.
        4) Otherwise:
           - Convert invalid_model to uppercase;
           - Check whether exactly one of the tied candidates contains invalid_model
             as a substring;
           - If exactly one match exists -> choose it;
           - If not -> fall back to the first model among the tied ones, ensuring
             a deterministic and reproducible choice.

    - Returns
    str or None
        The selected model name, or None if no decision can be made due to empty input.
    """

    if counts is None or len(counts) == 0:
        return None

    max_count = counts.iloc[0]
    top_models = counts[counts == max_count].index.tolist()

    if len(top_models) == 1:
        return top_models[0]

    invalid_upper = (
        str(invalid_model).upper()
        if invalid_model is not None else ""
    )

    substring_matches = [
        m for m in top_models
        if invalid_upper and invalid_upper in str(m).upper()
    ]

    if len(substring_matches) == 1:
        return substring_matches[0]

    # Either no substring match or multiple matches -> fall back to first top model
    return top_models[0]


# In[ ]:


def fit_invalid_model_resolver(
    train_df: pd.DataFrame,
    valid_models_by_brand: Dict[str, list],
    brand_col: str = "Brand",
    model_col: str = "model",
    year_col: str = "year",
    fuel_col: str = "fuelType",
    mpg_col: str = "mpg",
) -> Dict[str, Any]:
    
    """
    Fit step for resolving invalid or ambiguous model names.

    This function mirrors the philosophy of the brand-resolver fit step, but with
    additional contextual logic because models are more numerous, more varied,
    and more prone to ambiguity. It learns all the information necessary to later
    correct model names in a deterministic and leakage-free way.

    - Assumptions
      - model_col and brand_col have already been normalized using the previously
        defined string-cleaning utilities;
      - Missing model values have already been replaced with the placeholder
        'UNKNOWN';
      - valid_models_by_brand contains the canonical brand -> valid models mapping
        obtained from metadata or domain knowledge.

    - Training Only (no leakage)
      The resolver uses exclusively training data to learn:
      1) A refined dictionary of valid models per brand:
         - Starts from valid_models_by_brand.
         - May expand when the training set contains previously unseen, but
           legitimate, model strings.
      2) The global set of all valid models (union of all brand lists).
      3) A mapping invalid_model -> chosen_model for invalid strings observed
         in train, excluding 'UNKNOWN'.
      4) Mean mpg per model for mpg-based disambiguation:
         - Useful when substring matches are ambiguous.
      5) Context maps for resolving 'UNKNOWN' models:
         - (Brand, year, fuelType) -> mode(model)
         - (Brand, year) -> mode(model)
         - (Brand, fuelType) -> mode(model)
         - Brand -> mode(model)
         These mimic the hierarchical logic used in the hierarchical imputers
         previously defined.

    - Behavior
      1) Standardize brand and model fields in train_df to ensure all logic is
         applied to consistent uppercase strings.
      2) Make a deep copy of valid_models_by_brand to avoid mutating the input.
      3) Construct sets for fast membership checks and preserve deterministic ordering
         for tie resolution.
      4) Compute mean mpg per model using only rows with valid models.
      5) Build context maps by grouping over combinations of brand, year, and fuel:
         - Only use rows where model != 'UNKNOWN'.
         - For each group, compute the mode(model).
      6) Identify invalid models seen in training:
         - Sorted longest-first to ensure substring matching behaves deterministically.
      7) For each invalid model:
         - Determine its dominant brand in the training set.
         - Look for substring matches within the valid models of that brand.
         - Cases:
             a) Exactly one substring match -> treat as a typo.
             b) No substring matches -> treat invalid as a new valid model.
             c) Multiple matches -> use mpg heuristic to pick the closest model.
            If still ambiguous -> fall back to the first model (ordered).
      8) Pack all learned structures into a single state dictionary to use in the
         transform step.

    - Returns
    dict
        A state dictionary containing:
        - brand_col, model_col, year_col, fuel_col, mpg_col  
        - valid_models_by_brand (possibly expanded)  
        - all_valid_models  
        - invalid_to_model (train-observed corrections)  
        - mpg_means  
        - Context maps:  
          * ctx_brand_year_fuel  
          * ctx_brand_year  
          * ctx_brand_fuel  
          * ctx_brand  
    """
    tmp = train_df.copy()

    # Ensure model is a clean string with 'UNKNOWN' where needed
    tmp[model_col] = (
        tmp[model_col]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    tmp[brand_col] = (
        tmp[brand_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Deep copy of valid_models_by_brand to avoid mutating the original dict
    valid_models_by_brand_fit: Dict[str, list] = {
        b: list(models) for b, models in valid_models_by_brand.items()
    }
    # Keep sets for fast lookup; sorted versions used for deterministic iteration
    brand_valid_sets: Dict[str, set] = {
        b: set(models) for b, models in valid_models_by_brand_fit.items()
    }

    # Global set of all valid models (for membership checks only)
    all_valid_models: set = set(
        m for models in valid_models_by_brand_fit.values() for m in models
    )

    # Mean mpg per model (only valid models)
    if mpg_col in tmp.columns:
        mpg_means = (
            tmp[tmp[model_col].isin(all_valid_models)]
            .groupby(model_col)[mpg_col]
            .mean()
        )
    else:
        mpg_means = pd.Series(dtype=float)

    # ----------------------------------------------------
    # Context maps for UNKNOWN-model inference (TRAIN ONLY) 
    known_mask = tmp[model_col] != "UNKNOWN"
    known = tmp[known_mask]

    ctx_brand_year_fuel: Dict[tuple, str] = {}
    ctx_brand_year: Dict[tuple, str] = {}
    ctx_brand_fuel: Dict[tuple, str] = {}
    ctx_brand: Dict[str, str] = {}

    # (Brand, year, fuelType)
    if all(c in known.columns for c in [brand_col, year_col, fuel_col]):
        grp = known.groupby([brand_col, year_col, fuel_col])[model_col]
        for key, series in grp:
            mode = series.mode()
            if not mode.empty:
                ctx_brand_year_fuel[key] = mode.iloc[0]

    # (Brand, year)
    if all(c in known.columns for c in [brand_col, year_col]):
        grp = known.groupby([brand_col, year_col])[model_col]
        for key, series in grp:
            mode = series.mode()
            if not mode.empty:
                ctx_brand_year[key] = mode.iloc[0]

    # (Brand, fuelType)
    if all(c in known.columns for c in [brand_col, fuel_col]):
        grp = known.groupby([brand_col, fuel_col])[model_col]
        for key, series in grp:
            mode = series.mode()
            if not mode.empty:
                ctx_brand_fuel[key] = mode.iloc[0]

    # Brand only
    grp = known.groupby(brand_col)[model_col]
    for b, series in grp:
        mode = series.mode()
        if not mode.empty:
            ctx_brand[b] = mode.iloc[0]


    # -------------------------------------
    # Mapping invalid_model -> chosen_model
    invalid_models = sorted(
        [
            m for m in tmp[model_col].unique()
            if m not in all_valid_models and m != "UNKNOWN"
        ],
        key=len,
        reverse=True,   # deterministic: longest first for substring checks
    )

    invalid_to_model: Dict[str, str] = {}

    for invalid in invalid_models:
        subset = tmp[tmp[model_col] == invalid]
        if subset.empty:
            continue

        # Determine dominant brand for this invalid model
        brand_mode = subset[brand_col].mode()
        if brand_mode.empty:
            continue
        brand = brand_mode.iloc[0]

        brand_valids_set = brand_valid_sets.get(brand, set())
        brand_valids_sorted = sorted(brand_valids_set)

        # Substring matches with known valid models of this brand
        matches = [vm for vm in brand_valids_sorted if invalid in vm]

        # Case 1: exactly one substring match -> treat as a typo
        if len(matches) == 1:
            chosen = matches[0]

        # Case 2: no matches -> treat invalid as a new valid model
        elif len(matches) == 0:
            chosen = invalid
            valid_models_by_brand_fit.setdefault(brand, []).append(invalid)
            brand_valid_sets.setdefault(brand, set()).add(invalid)
            all_valid_models.add(invalid)

        # Case 3: multiple matches -> use mpg heuristic, then fallback
        else:
            chosen = None
            if mpg_col in subset.columns and subset[mpg_col].notna().any() and not mpg_means.empty:
                invalid_mean_mpg = subset[mpg_col].mean(skipna=True)
                candidate_means = {
                    m: mpg_means[m]
                    for m in matches
                    if m in mpg_means.index
                }
                if candidate_means:
                    # deterministic tie-break: first by mpg distance, then lexicographically
                    chosen = min(
                        candidate_means.keys(),
                        key=lambda m: (abs(candidate_means[m] - invalid_mean_mpg), m)
                    )

            if chosen is None:
                chosen = matches[0]   # deterministic fallback

        invalid_to_model[invalid] = chosen

    state = {
        "brand_col": brand_col,
        "model_col": model_col,
        "year_col": year_col,
        "fuel_col": fuel_col,
        "mpg_col": mpg_col,
        "valid_models_by_brand": valid_models_by_brand_fit,
        "all_valid_models": all_valid_models,
        "invalid_to_model": invalid_to_model,
        "mpg_means": mpg_means,
        "ctx_brand_year_fuel": ctx_brand_year_fuel,
        "ctx_brand_year": ctx_brand_year,
        "ctx_brand_fuel": ctx_brand_fuel,
        "ctx_brand": ctx_brand,
    }

    return state


# In[ ]:


def transform_invalid_models(
    df: pd.DataFrame,
    state: Dict[str, Any],
) -> tuple[pd.DataFrame, dict, list]:
    """
    Transform step for resolving invalid or ambiguous model strings using the statistics 
    learned in `fit_invalid_model_resolver`. The logic closely parallels the brand-transformer,
    but is extended with model-specific context (Brand, year, fuel) and an mpg-based heuristic 
    when substring matches are ambiguous.

    - Parameters
    df : pd.DataFrame
        Any dataset requiring correction. A copy is made so the original is not modified.
    state : dict
        The dictionary produced by fit_invalid_model_resolver. Contains:
        - Columns names (brand_col, model_col, year_col, fuel_col, mpg_col)
        - valid_models_by_brand and all_valid_models
        - invalid_to_model (direct mapping learned on TRAIN)
        - mpg_means (for tie-breaking)
        - Context inference maps:
          * ctx_brand_year_fuel  
          * ctx_brand_year  
          * ctx_brand_fuel  
          * ctx_brand  

    - Behavior
      For each row, apply the following workflow:

      1) If the model is already valid (in all_valid_models):
         - Keep as is.

      2) If the model is 'UNKNOWN':
         - Attempt hierarchical inference using context maps, in order:
           a) (Brand, year, fuel) -> inferred model
           b) (Brand, year) -> inferred model
           c) (Brand, fuel) -> inferred model
           d) (Brand) -> inferred model
         - If no context is available, leave as 'UNKNOWN'.
         This approach mirrors the hierarchical-style imputers defined earlier.

      3) If the model is invalid but not 'UNKNOWN':
         - If it appeared during TRAIN:
           * Replace it with the choice stored in invalid_to_model.
         - If it never appeared in TRAIN:
           * Attempt substring matching against the brand's known valid models:
             - If exactly one match -> use it.
             - If multiple matches:
               + If mpg_value is available and mpg_means exist:
                 -> choose the model whose mean mpg is closest.
               + Else:
                 -> deterministic fallback: first match.
             - If no match -> keep the original invalid string.

      4) Record every correction in the corrections dictionary using the key  
         (original_model, brand, year, fuel, row_index)

      5) After processing all rows, compute a list of still-invalid models
         (excluding 'UNKNOWN')

    - Returns
    df_out : pd.DataFrame
        A corrected copy of df.
    corrections : dict
        Detailed mapping of context -> corrected model for auditing.
    still_invalid : list
        Models that remain outside the valid set and are not 'UNKNOWN'.
    """

    brand_col = state["brand_col"]
    model_col = state["model_col"]
    year_col = state["year_col"]
    fuel_col = state["fuel_col"]
    mpg_col = state["mpg_col"]

    valid_models_by_brand = state["valid_models_by_brand"]
    all_valid_models = state["all_valid_models"]
    invalid_to_model = state["invalid_to_model"]
    mpg_means = state["mpg_means"]

    ctx_brand_year_fuel = state["ctx_brand_year_fuel"]
    ctx_brand_year = state["ctx_brand_year"]
    ctx_brand_fuel = state["ctx_brand_fuel"]
    ctx_brand = state["ctx_brand"]

    df_out = df.copy()

    # Safety normalization (same rationale as previous transform functions)
    df_out[model_col] = (
        df_out[model_col]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df_out[brand_col] = (
        df_out[brand_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    corrections: dict = {}

    for idx, row in df_out.iterrows():
        original_model = row[model_col]
        brand = row.get(brand_col, None)
        year = row.get(year_col, None)
        fuel = row.get(fuel_col, None)
        mpg_value = row.get(mpg_col, None)

        model_upper = (
            str(original_model).upper()
            if pd.notna(original_model)
            else "UNKNOWN"
        )

        # Case 0: valid model -> nothing to do
        if model_upper in all_valid_models:
            continue

        corrected = model_upper

        # ---------- Case 1: 'UNKNOWN' ----------
        if model_upper == "UNKNOWN":
            key3 = (brand, year, fuel)
            key2_by = (brand, year)
            key2_bf = (brand, fuel)

            if key3 in ctx_brand_year_fuel:
                corrected = ctx_brand_year_fuel[key3]
            elif key2_by in ctx_brand_year:
                corrected = ctx_brand_year[key2_by]
            elif key2_bf in ctx_brand_fuel:
                corrected = ctx_brand_fuel[key2_bf]
            elif brand in ctx_brand:
                corrected = ctx_brand[brand]
            else:
                corrected = "UNKNOWN"

        # ---------- Case 2: seen invalid model ----------
        elif model_upper in invalid_to_model:
            corrected = invalid_to_model[model_upper]

        # ---------- Case 3: unseen invalid ----------
        else:
            brand_valids = set(valid_models_by_brand.get(brand, []))
            matches = [vm for vm in brand_valids if model_upper in vm]

            if len(matches) == 1:
                corrected = matches[0]

            elif len(matches) > 1:
                chosen = None
                if pd.notna(mpg_value) and not mpg_means.empty:
                    candidate_means = {
                        m: mpg_means[m]
                        for m in matches
                        if m in mpg_means.index
                    }
                    if candidate_means:
                        chosen = min(
                            candidate_means.keys(),
                            key=lambda m: abs(candidate_means[m] - mpg_value)
                        )
                if chosen is None:
                    chosen = matches[0]
                corrected = chosen

            else:
                # No match within the brand's known valid models -> keep the invalid string
                corrected = model_upper

        # Apply correction if changed
        if corrected != model_upper:
            df_out.at[idx, model_col] = corrected

        corrections[(original_model, brand, year, fuel, idx)] = corrected

    # Remaining invalid models
    still_invalid = [
        m for m in df_out[model_col].unique()
        if m not in all_valid_models and m != "UNKNOWN"
    ]

    return df_out, corrections, still_invalid


# ### 4.3. Fuel Type

# In[ ]:


def fit_fueltype_resolver(
    train_df: pd.DataFrame,
    valid_fueltypes: List[str],
    fuel_col: str = "fuelType",
    brand_col: str = "Brand",
    model_col: str = "model",
    transm_col: str = "transmission",
) -> Dict[str, Any]:
    """
    Fit step for resolving invalid or ambiguous values in fuelType.

    This function follows the same overall strategy used in the brand/model resolvers, but 
    adapted to the smaller and more structured universe of fuel types. It learns deterministic 
    correction rules (fit) that will later be applied during transform without leaking 
    information from validation/test.

    Pre-conditions:
      - fuel_col, brand_col, model_col, and transm_col have already undergone
        basic string normalization.
      - Missing values have been converted to the placeholder 'UNKNOWN'.
      - valid_fueltypes is the base whitelist of allowed fuel labels
        (e.g., ["PETROL", "DIESEL", "HYBRID"]).

    - Training logic
      1) Clean the base list of valid fuel types (upper, strip).
      2) Identify invalid values observed in train (including 'UNKNOWN', but
         treat 'UNKNOWN' only in context inference, not in direct correction).
      3) For each invalid fuel value (excluding 'UNKNOWN'):
         - Rule 1: If it matches exactly one valid fuel via substring logic
           (inv in valid or valid in inv) -> map to that one.
         - Rule 2: If it matches none -> promote invalid as a new valid fuel.
         - Rule 3: If it matches multiple valid fuels -> choose the one that is
           most frequent in TRAIN among the matched candidates.
         These rules mirror the structure of the model-resolver logic, but
         simplified for the lower cardinality and predictable nature of fuel types.
      4) Replace invalids in a temporary column ("fuel_clean") to build context maps.
      5) Compute context-based inference maps for resolving UNKNOWN values later:
         - (Brand, Model, Transmission) -> mode(fuel_clean)
         - (Brand, Model)               -> mode(fuel_clean)
         - Brand                        -> mode(fuel_clean)
         These context maps work similarly to the ones previously used in the
         model and brand resolvers.

    - Returns
    A state dictionary containing:
      - fuel_col, brand_col, model_col, transm_col
      - valid_set: full set of valid fuels after potential promotion (Rule 2)
      - invalid_to_valid: mapping invalid_fuel -> corrected fuel
      - ctx_bmt: (Brand, Model, Transmission) -> inferred fuel
      - ctx_bm : (Brand, Model) -> inferred fuel
      - ctx_b  : Brand -> inferred fuel
    """

    tmp = train_df.copy()

    # Safety normalization (same approach as brand/model resolvers)
    for col in [fuel_col, brand_col, model_col, transm_col]:
        if col in tmp.columns:
            tmp[col] = (
                tmp[col]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # Clean base valid fueltypes
    base_valid = []
    for f in valid_fueltypes:
        if f is None:
            continue
        s = str(f).strip().upper()
        if s != "":
            base_valid.append(s)

    valid_fueltypes_clean = base_valid.copy()
    valid_set = set(valid_fueltypes_clean)

    # Identify invalid fuels observed in TRAIN (including UNKNOWN)
    unique_vals = tmp[fuel_col].unique()
    invalid_fuels = sorted(
        [v for v in unique_vals if v not in valid_set],
        key=len,
        reverse=True,
    )

    invalid_to_valid: Dict[str, str] = {}

    # Dynamic set of valid fuels (can expand via Rule 2)
    dynamic_valids = set(valid_set)

    # Global frequencies (used only for tie-breaking among matched candidates)
    freq_all = tmp[fuel_col].value_counts(dropna=False)

    for invalid in invalid_fuels:
        # UNKNOWN is handled in the context phase, not direct mapping
        if invalid == "UNKNOWN":
            continue

        subset = tmp[tmp[fuel_col] == invalid]
        if subset.empty:
            continue

        inv_u = str(invalid).strip().upper()

        # Substring-based matching against the current valid set
        matches = [
            v for v in dynamic_valids
            if inv_u in v or v in inv_u
        ]

        # Rule 1: exactly one match -> treat as typo
        if len(matches) == 1:
            chosen = matches[0]

        # Rule 2: no match -> promote invalid to new valid
        elif len(matches) == 0:
            chosen = inv_u
            dynamic_valids.add(inv_u)
            valid_fueltypes_clean.append(inv_u)

        # Rule 3: multiple matches -> choose most frequent candidate in TRAIN
        else:
            df_f = tmp[tmp[fuel_col].isin(matches)]
            if not df_f.empty:
                chosen = df_f[fuel_col].mode().iloc[0]
            else:
                chosen = matches[0]

        invalid_to_valid[invalid] = chosen

    # Build a cleaned fuel column to compute context maps
    tmp["fuel_clean"] = tmp[fuel_col].replace(invalid_to_valid)

    # Only consider lines with known fuel for context inference
    known = tmp[tmp["fuel_clean"] != "UNKNOWN"].copy()

    # Context: (Brand, Model, Transmission)
    ctx_bmt: Dict[Tuple[str, str, str], str] = {}
    if all(c in known.columns for c in [brand_col, model_col, transm_col]):
        grouped = known.groupby([brand_col, model_col, transm_col])["fuel_clean"]
        for key, series in grouped:
            m = series.mode()
            if not m.empty:
                ctx_bmt[key] = m.iloc[0]

    # Context: (Brand, Model)
    ctx_bm: Dict[Tuple[str, str], str] = {}
    if all(c in known.columns for c in [brand_col, model_col]):
        grouped = known.groupby([brand_col, model_col])["fuel_clean"]
        for key, series in grouped:
            m = series.mode()
            if not m.empty:
                ctx_bm[key] = m.iloc[0]

    # Context: (Brand)
    ctx_b: Dict[str, str] = {}
    if brand_col in known.columns:
        grouped = known.groupby(brand_col)["fuel_clean"]
        for b, series in grouped:
            m = series.mode()
            if not m.empty:
                ctx_b[b] = m.iloc[0]

    state: Dict[str, Any] = {
        "fuel_col": fuel_col,
        "brand_col": brand_col,
        "model_col": model_col,
        "transm_col": transm_col,
        "valid_set": dynamic_valids,         # base + promoted (Rule 2)
        "invalid_to_valid": invalid_to_valid,
        "ctx_bmt": ctx_bmt,
        "ctx_bm": ctx_bm,
        "ctx_b": ctx_b,
    }

    return state


# In[ ]:


def transform_fueltype_resolver(
    df: pd.DataFrame,
    state: Dict[str, Any],
) -> tuple[pd.DataFrame, dict, list]:
    
    """
    Transform step for resolving invalid or ambiguous fuelType values,
    using the correction rules and context maps learned in `fit_fueltype_resolver`.

    The logic mirrors the transform steps used for brands and models, but
    simplified due to the smaller set of possible fuel types.

    - Parameters
      df : pd.DataFrame
          Dataset requiring correction. A copy is made, so df is not modified.
      state : dict
          Output of fit_fueltype_resolver, containing:
          - valid_set: all accepted fuel types (base + promoted)
          - invalid_to_valid: mapping from invalid fuels observed in TRAIN
            to their corrected values
          - ctx_bmt, ctx_bm, ctx_b: context maps for inferring UNKNOWN values
            using hierarchical keys (Brand, Model, Transmission) -> (Brand, Model) -> Brand.

    - Behavior (row-level logic)
      A) If fuel in valid_set:
         - Keep as is (already valid).

      B) If fuel == 'UNKNOWN':
         - Attempt hierarchical inference using context maps (same order as fit):
           1) (Brand, Model, Transmission)
           2) (Brand, Model)
           3) Brand
         - If no rule applies, keep 'UNKNOWN'.

      C) If fuel is invalid but appeared in TRAIN:
         - Replace using invalid_to_valid (deterministic mapping).

      D) If fuel is invalid and not seen in TRAIN:
         - Attempt substring matching with valid_set:
           * If ≥1 match -> choose the first (consistent deterministic behavior).
           * If no match -> keep original (no basis for correction).

      Throughout, store corrections in a dictionary for auditing.

    - Returns
      df_out : pd.DataFrame
          Corrected DataFrame.
      corrections : dict
          Mapping {(idx, original_fuel) -> corrected_fuel}.
      still_invalid : list
          Fuel types that remain outside valid_set and are not 'UNKNOWN'.
    """

    df_out = df.copy()

    fuel_col   = state["fuel_col"]
    brand_col  = state["brand_col"]
    model_col  = state["model_col"]
    transm_col = state["transm_col"]

    valid_set        = state["valid_set"]
    invalid_to_valid = state["invalid_to_valid"]
    ctx_bmt          = state["ctx_bmt"]
    ctx_bm           = state["ctx_bm"]
    ctx_b            = state["ctx_b"]

    corrections: Dict[Tuple[int, str], str] = {}

    # Safety normalization (same pattern used in previous transform functions)
    for col in [fuel_col, brand_col, model_col, transm_col]:
        if col in df_out.columns:
            df_out[col] = (
                df_out[col]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
                .str.upper()
            )

    for idx, row in df_out.iterrows():
        original = row[fuel_col]

        # A) Already valid
        if original in valid_set:
            continue

        # B) UNKNOWN fuel -> contextual inference
        if original == "UNKNOWN":
            b = row[brand_col]
            m = row[model_col]
            t = row[transm_col]

            key_bmt = (b, m, t)
            key_bm  = (b, m)
            key_b   = b

            if key_bmt in ctx_bmt:
                corrected = ctx_bmt[key_bmt]
            elif key_bm in ctx_bm:
                corrected = ctx_bm[key_bm]
            elif key_b in ctx_b:
                corrected = ctx_b[key_b]
            else:
                corrected = "UNKNOWN"

            df_out.at[idx, fuel_col] = corrected
            corrections[(idx, original)] = corrected
            continue

        # C) Invalid but observed in TRAIN -> direct mapping
        if original in invalid_to_valid:
            corrected = invalid_to_valid[original]
            df_out.at[idx, fuel_col] = corrected
            corrections[(idx, original)] = corrected
            continue

        # D) Invalid and unseen -> substring matching against valid_set
        orig_u = str(original).strip().upper()
        matches = [v for v in valid_set if orig_u in v or v in orig_u]

        if len(matches) >= 1:
            corrected = matches[0]   # deterministic choice
        else:
            corrected = original     # insufficient information

        df_out.at[idx, fuel_col] = corrected
        corrections[(idx, original)] = corrected

    # Fuel types still invalid (excluding UNKNOWN)
    still_invalid = sorted(
        {
            f for f in df_out[fuel_col].unique()
            if f not in valid_set and f != "UNKNOWN"
        }
    )

    return df_out, corrections, still_invalid


# ### 4.4. Transmission

# In[ ]:


def _normalize_unknown_like_transm(v: Any) -> str:
    """
    Normalize any transmission string that resembles an UNKNOWN value.

    - The goal is to collapse all noisy variants such as:
      UNKNOW, NKNOWN, UNK, UNKNOWN1, etc.
      into the canonical 'UNKNOWN'.

    - Logic:
      1) If the value is None -> return 'UNKNOWN'.
      2) Convert to uppercase string and trim spaces.
      3) If the resulting string contains 'UNK' anywhere,
         treat it as an UNKNOWN placeholder.
      4) Otherwise, return the cleaned string unchanged.

    This helper ensures that the main resolver does not have to deal with
    multiple inconsistent mis-typed UNKNOWN values.
    """
    if v is None:
        return "UNKNOWN"
    s = str(v).strip().upper()
    if "UNK" in s:
        return "UNKNOWN"
    return s


# In[ ]:


def fit_transmission_resolver(
    train_df: pd.DataFrame,
    valid_transmissions: List[str],
    transm_col: str = "transmission",
    brand_col: str = "Brand",
    model_col: str = "model",
    fuel_col: str = "fuelType",
) -> Dict[str, Any]:
    
    """
    Fit step for resolving ambiguous or invalid transmission labels.
    This function parallels the logic of the previous resolvers.

    - Pre-normalization:
      The transmission, brand, model, and fuel columns are normalized:
      1) Fill missing values as 'UNKNOWN'.
      2) Convert to trimmed uppercase strings.
      3) Collapse unknown-like variants ('UNKNOW', 'NKNOWN', etc.) into 'UNKNOWN'.

    - Training objectives:
      1) Build valid_set:
         - Only contains the manually provided valid_transmissions.
         - Unlike model/fuel resolvers, no new transmissions are ever promoted.
      2) Learn invalid_to_valid mapping:
         For each invalid string seen in TRAIN:
         - If substring matches exactly one valid transmission -> map to it.
         - If multiple matches -> choose the most frequent valid transmission
           among the matched candidates.
         - If no matches -> fallback to global_most_common (most frequent valid).
      3) Learn context maps for resolving UNKNOWN values:
         - (Brand, Model, Fuel) -> mode(transmission)
         - (Brand, Model)       -> mode(transmission)
         - Brand                -> mode(transmission)
         These context structures mirror the ones used in previous resolvers.
      4) Compute global_most_common:
         - The most frequent valid transmission in TRAIN.
         - Used as fallback for difficult corrections.

    - Returns
      A dictionary with:
      - Column names
      - valid_set
      - invalid_to_valid
      - ctx_bmf, ctx_bm, ctx_b
      - global_most_common
    """

    tmp = train_df.copy()

    # Basic normalization
    for col in [transm_col, brand_col, model_col, fuel_col]:
        if col in tmp.columns:
            tmp[col] = (
                tmp[col]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # Collapse variants of UNKNOWN (UNKNOW, NKNOWN, etc.)
    tmp[transm_col] = tmp[transm_col].apply(_normalize_unknown_like_transm)

    # Normalized  valid transmissions
    base_valid = []
    for t in valid_transmissions:
        if t is None:
            continue
        s = str(t).strip().upper()
        if s != "":
            base_valid.append(s)

    # Final set of valid transmissions: only the base ones
    valid_set = set(base_valid)

    # Global frequencies in TRAIN
    freq_all = tmp[transm_col].value_counts(dropna=False)

    # Global most common only between valid transmission values
    if not freq_all.empty:
        # keep order to have deterministic choice
        # only among valid transmissions
        valid_counts = freq_all[freq_all.index.isin(valid_set)]
        if not valid_counts.empty:
            global_most_common = valid_counts.index[0]
        else:
            # extreme case: none of the valids appear in train
            global_most_common = None
    else:
        global_most_common = None

    # Discoveer invalid values: not in valid_set and different from UNKNOWN
    unique_vals = tmp[transm_col].unique()
    invalid_vals = [
        v for v in unique_vals
        if v not in valid_set and v != "UNKNOWN"
    ]

    invalid_to_valid: Dict[str, str] = {}

    # Learn mapping invalid -> valid
    for invalid in invalid_vals:
        inv_u = str(invalid).strip().upper()

        # Matches per substring within valid_set
        # (ex.: "MANU" -> "MANUAL"; "SEMI AUTO" -> "SEMIAUTO")
        matches = [v for v in valid_set if inv_u in v or v in inv_u]

        if len(matches) == 1:
            chosen = matches[0]

        elif len(matches) > 1:
            # Multiple matches -> choose the most frequent in TRAIN
            best = None
            best_count = -1
            for cand in matches:
                c = int(freq_all.get(cand, 0))
                if c > best_count:
                    best_count = c
                    best = cand
            chosen = best if best is not None else matches[0]

        else:
            # len(matches) == 0
            # we do NOT promote new valid transmissions here
            # use fallback logic to get a valid choice
            if global_most_common is not None:
                chosen = global_most_common
            elif len(valid_set) > 0:
                # extreme fallback: valid_set's 1st element
                chosen = next(iter(valid_set))
            else:
                #very extreme, undesired case
                chosen = inv_u  

        invalid_to_valid[invalid] = chosen

    # build a cleaned transmission column to learn context maps
    tmp["transm_clean"] = tmp[transm_col].replace(invalid_to_valid)

    known = tmp[tmp["transm_clean"] != "UNKNOWN"].copy()

    # Context (Brand, Model, Fuel)
    ctx_bmf: Dict[Tuple[str, str, str], str] = {}
    if all(c in known.columns for c in [brand_col, model_col, fuel_col]):
        grouped = known.groupby([brand_col, model_col, fuel_col])["transm_clean"]
        for key, series in grouped:
            m = series.mode()
            if not m.empty:
                ctx_bmf[key] = m.iloc[0]

    # Context (Brand, Model)
    ctx_bm: Dict[Tuple[str, str], str] = {}
    if all(c in known.columns for c in [brand_col, model_col]):
        grouped = known.groupby([brand_col, model_col])["transm_clean"]
        for key, series in grouped:
            m = series.mode()
            if not m.empty:
                ctx_bm[key] = m.iloc[0]

    # Context (Brand)
    ctx_b: Dict[str, str] = {}
    if brand_col in known.columns:
        grouped = known.groupby(brand_col)["transm_clean"]
        for b, series in grouped:
            m = series.mode()
            if not m.empty:
                ctx_b[b] = m.iloc[0]

    state: Dict[str, Any] = {
        "transm_col": transm_col,
        "brand_col": brand_col,
        "model_col": model_col,
        "fuel_col": fuel_col,
        "valid_set": valid_set,
        "invalid_to_valid": invalid_to_valid,
        "ctx_bmf": ctx_bmf,
        "ctx_bm": ctx_bm,
        "ctx_b": ctx_b,
        "global_most_common": global_most_common,
    }

    return state


# In[ ]:


def transform_transmission_resolver(
    df: pd.DataFrame,
    state: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], str], List[str]]:
    """
    Transform step for resolving invalid or ambiguous transmission labels,
    using the learned state from fit_transmission_resolver.

    The structure mirrors the transform steps for brands, models, and fuel types.

    - Inputs
      df : pd.DataFrame
          Any dataset needing correction.
      state : dict
          Contains:
          - valid_set
          - invalid_to_valid
          - context maps (ctx_bmf, ctx_bm, ctx_b)
          - global_most_common

    - Behavior (row-wise):
      A) If transmission is already in valid_set:
         - Keep unchanged.

      B) If transmission == 'UNKNOWN':
         - Attempt hierarchical inference:
           1) (Brand, Model, Fuel)
           2) (Brand, Model)
           3) Brand
         - If all fail, fallback to global_most_common if it exists.
         - Otherwise keep 'UNKNOWN'.

      C) If transmission is invalid but seen in TRAIN:
         - Replace using invalid_to_valid[original].

      D) If transmission is invalid and unseen in TRAIN:
         - Attempt substring matching with valid_set:
             - If one or more matches -> take the first match.
             - Else -> fallback to global_most_common or 'UNKNOWN'.
         - This is analogous to the logic used in the unseen-model resolver.

    - Output values
      df_out : pd.DataFrame
          The corrected DataFrame.
      corrections : dict
          Mapping (row_index, original_value) -> corrected_value.
      still_problematic : list
          Transmission labels that remain outside valid_set.
    """

    df_out = df.copy()

    transm_col = state["transm_col"]
    brand_col  = state["brand_col"]
    model_col  = state["model_col"]
    fuel_col   = state["fuel_col"]

    valid_set        = state["valid_set"]
    invalid_to_valid = state["invalid_to_valid"]
    ctx_bmf          = state["ctx_bmf"]
    ctx_bm           = state["ctx_bm"]
    ctx_b            = state["ctx_b"]
    global_most      = state["global_most_common"]

    corrections: Dict[Tuple[int, str], str] = {}

    # Safety normalization (same idea as fuel/model resolvers)
    for col in [transm_col, brand_col, model_col, fuel_col]:
        if col in df_out.columns:
            df_out[col] = (
                df_out[col]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # Also normalize UNKNOWN-like transmission values
    df_out[transm_col] = df_out[transm_col].apply(_normalize_unknown_like_transm)

    for idx, row in df_out.iterrows():
        original = row[transm_col]

        # A) Already valid
        if original in valid_set:
            continue

        # B) UNKNOWN -> try context
        if original == "UNKNOWN":
            brand = row[brand_col] if brand_col in df_out.columns else "UNKNOWN"
            model = row[model_col] if model_col in df_out.columns else "UNKNOWN"
            fuel  = row[fuel_col]  if fuel_col  in df_out.columns else "UNKNOWN"

            key_bmf = (brand, model, fuel)
            key_bm  = (brand, model)
            key_b   = brand

            if key_bmf in ctx_bmf:
                corrected = ctx_bmf[key_bmf]
            elif key_bm in ctx_bm:
                corrected = ctx_bm[key_bm]
            elif key_b in ctx_b:
                corrected = ctx_b[key_b]
            elif global_most is not None:
                corrected = global_most
            else:
                corrected = "UNKNOWN"

            df_out.at[idx, transm_col] = corrected
            corrections[(idx, original)] = corrected
            continue

        # C) Invalid but observed in TRAIN
        if original in invalid_to_valid:
            corrected = invalid_to_valid[original]
            df_out.at[idx, transm_col] = corrected
            corrections[(idx, original)] = corrected
            continue

        # D) Invalid and unseen -> substring matching
        orig_u = str(original).strip().upper()
        matches = [v for v in valid_set if orig_u in v or v in orig_u]

        if len(matches) >= 1:
            corrected = matches[0]
        elif global_most is not None:
            corrected = global_most
        else:
            corrected = "UNKNOWN"

        df_out.at[idx, transm_col] = corrected
        corrections[(idx, original)] = corrected

    still_problematic = sorted(
        {v for v in df_out[transm_col].unique() if v not in valid_set}
    )

    return df_out, corrections, still_problematic


# In[19]:


# dropping carid and has damage
#....


# ### 4.5. Year

# In[ ]:


def fit_year_median(
        train_df: pd.DataFrame,
        year_col: str = "year",
        model_col: str = "model",
        max_valid_year: int = 2020,
):
    """
    Fit step for learning median year values for imputation.

    This resolver is intentionally simpler than the brand/model/fuel/transmission
    systems because year is numeric and has a well-defined ordering. The goal is to
    compute robust central estimates (medians) per model and globally, so that
    later, during transform, missing or invalid year values can be filled with a
    meaningful and consistent value.

    - Preprocessing (safety normalization):
      1) Convert year_col to numeric (coercing non-numeric values to NaN).
      2) Cap any year values exceeding max_valid_year to that limit.
         This preserves physical plausibility and avoids unrealistic model years.
      3) Floor values to integers to remove decimals that may appear after cleaning.
         This step mirrors earlier normalization, but is repeated for safety.

    - Learned statistics:
      - global_year_median:
        The median year across the entire training dataset.
        Used as a fallback when model-level information is missing or unreliable.

      - model_year_median:
        A Series mapping each model -> median year of all cars of that model.
        This provides a model-specific imputation rule that is typically more
        informative than the global statistic.

    - Assumptions:
      - String cleaning and normalization for model_col have already been applied.
      - max_valid_year was defined earlier as the highest permissible model year
        in the dataset (e.g., 2020), ensuring consistency with earlier year cleaning.

    - Returns
      state : dict
        A dictionary containing:
        - year_col, model_col, max_valid_year
        - global_year_median
        - model_year_median (model-wise medians for imputation)
    """
    
    tmp = train_df.copy()

    # Ensure numeric
    tmp[year_col] = pd.to_numeric(tmp[year_col], errors="coerce")

    # Cap to maximum allowed year (safety)
    tmp.loc[tmp[year_col] > max_valid_year, year_col] = max_valid_year

    # Floor to integer (safety)
    tmp[year_col] = np.floor(tmp[year_col])

    # Compute medians
    global_year_median = tmp[year_col].median()
    model_year_median = tmp.groupby(model_col)[year_col].median()

    state = {
        "year_col": year_col,
        "model_col": model_col,
        "max_valid_year":  max_valid_year,
        "global_year_median": global_year_median,
        "model_year_median": model_year_median
    }

    return state


# In[ ]:


def transform_year_with_model_median(
        df: pd.DataFrame,
        state: dict,
) -> pd.DataFrame:
    """
    Transform step for imputing and normalizing the year column using
    the medians learned in `fit_year_median`.

    This transform mirrors the structure of other resolvers but is simpler
    because year is numeric and follows a clear ordering. The goal is to ensure
    that every row ends with a plausible, valid, and model-consistent year value.

    - Inputs
      df : pd.DataFrame
          The dataset to be corrected.
      state : dict
          Output of fit_year_median, containing:
          - year_col, model_col, max_valid_year
          - global_year_median
          - model_year_median (per-model medians)

    - Behavior
      1) Convert the year column to numeric.
         - Any non-numeric values (e.g., 'NA', 'unknown') become NaN.
         - Ensures consistency and avoids unexpected string contamination.

      2) Cap any values above max_valid_year.
         - Even though earlier cleaning should already enforce this restriction,
           this step adds an extra layer of robustness.

      3) Floor values to integers.
         - Removes unintended decimals introduced earlier in cleaning.

      4) Impute missing or invalid years:
         - First attempt model-specific imputation:
             year = model_year_median[model]
         - Then fill remaining NaN values with the global_year_median.
         - This hierarchical imputation mirrors the structure used in other
           resolvers: specific info first, general fallback second.

    - Returns
      df_out : pd.DataFrame
          A corrected copy of df, with year values imputed and normalized.
    """

    year_col = state["year_col"]
    model_col = state["model_col"]
    max_valid_year = state["max_valid_year"]
    global_median = state["global_year_median"]
    model_median = state["model_year_median"]

    df_out = df.copy()

    # 1) Ensure numeric
    df_out[year_col] = pd.to_numeric(df_out[year_col], errors="coerce")

    # 2) Cap to max_valid_year
    df_out.loc[df_out[year_col] > max_valid_year, year_col] = max_valid_year

    # 3) Floor to integer
    df_out[year_col] = np.floor(df_out[year_col])

    # 4) Impute missing years (hierarchical)
    #    a) Model-specific medians
    df_out[year_col] = df_out[year_col].fillna(
        df_out[model_col].map(model_median)
    )
    #    b) Global fallback
    df_out[year_col] = df_out[year_col].fillna(global_median)


    return df_out


# ### 4.6. previousOwners

# In[ ]:


def fit_previous_owners_imputer(
    train_df: pd.DataFrame,
    owners_col: str = "previousOwners",
    year_col: str = "year",
    mileage_col: str = "mileage",
    mileage_threshold: float = 15000.0,
    min_years_since_registration: float = 2.0,
):
    """
    Fit step for preprocessing the previousOwners column.

    This resolver is designed to correct noisy owner counts and learn a stable
    imputation rule from TRAIN ONLY. It follows a deterministic numerical
    cleaning sequence, then computes a robust central statistic (median) to be
    used during transform.

    - Assumptions
      - The year and mileage columns have already been cleaned.
      - previousOwners may contain invalid strings, negative values, and
        suspicious zeros that require correction.

    - Behavior on train_df
      1) Convert previousOwners to numeric (non-numeric values become NaN).
      2) Take absolute value (negative owner counts, which are physically impossible, 
      become positive).
      3) Round to nearest integer (owner counts should be whole numbers).
      4) Apply zero-imprecision correction:
         - A value of 0 owners is suspicious for a resale company, specially for older 
         or heavily used vehicles.
           Condition:
             ((max_year - year) > min_years_since_registration)
             OR (mileage > mileage_threshold)
             AND previousOwners == 0
         - Rows satisfying this condition are set to previousOwners = 1.
         - This heuristic corrects underreported owner counts.
      5) Compute imputation median:
         - Median of cleaned previousOwners (train only).
         - Used to fill missing or invalid values during transform.

    - Returns
      state : dict with keys:
        - owners_col, year_col, mileage_col
        - max_year (from TRAIN)
        - mileage_threshold, min_years_since_registration
        - imputation_median (numeric median from TRAIN)
    """
    tmp = train_df.copy()

    # 1) Convert to numeric
    tmp[owners_col] = pd.to_numeric(tmp[owners_col], errors="coerce")

    # 2) Absolute values
    tmp[owners_col] = tmp[owners_col].abs()

    # 3) Round to nearest integer
    tmp[owners_col] = tmp[owners_col].round()

    # Compute max_year based on train only
    max_year = tmp[year_col].max()

    # 4) Zero imprecision correction
    inac_own_condition_train = (
        ((max_year - tmp[year_col]) > min_years_since_registration)
        | (tmp[mileage_col] > mileage_threshold)
    ) & (tmp[owners_col] == 0)

    tmp.loc[inac_own_condition_train, owners_col] = 1

    # 5) Compute median after cleaning (NaNs are skipped)
    imputation_median = tmp[owners_col].median(skipna=True)


    state = {
        "owners_col": owners_col,
        "year_col": year_col,
        "mileage_col": mileage_col,
        "max_year": max_year,
        "mileage_threshold": mileage_threshold,
        "min_years_since_registration": min_years_since_registration,
        "imputation_median": imputation_median,
    }

    return state


# In[ ]:


def transform_previous_owners_imputer(
    df: pd.DataFrame,
    state: dict,
) -> pd.DataFrame:
    
    """
    Transform step applying the previousOwners cleaning rules learned in
    fit_previous_owners_imputer.

    This function reproduces the same correction sequence used during training,
    ensuring consistency across train, validation, and test datasets.
    The median used for imputation comes from TRAIN (to avoid leakage).

    - Parameters
      df : pd.DataFrame
          Any dataset to be transformed.
      state : dict
          Dictionary produced in the fit step, containing:
          - owners_col, year_col, mileage_col
          - max_year (TRAIN-only)
          - mileage_threshold, min_years_since_registration
          - imputation_median

    - Behavior
      1) Convert previousOwners to numeric (invalid -> NaN).
      2) Take absolute value.
         - Removes effects of accidental negative entry.
      3) Round to nearest integer.
      4) Apply zero-imprecision correction:
         - Uses max_year learned from TRAIN, keeping behavior consistent.
         - Same condition as fit:
             ((max_year - year) > min_years_since_registration)
             OR (mileage > mileage_threshold)
             AND previousOwners == 0
         -> Set such entries to 1.
      5) Impute remaining NaNs with the train-based median.

      This mirrors the exact logic applied in fit, ensuring the transform
      behaves deterministically and without leakage.

    - Returns
      df_out : pd.DataFrame
          Copy of df with cleaned and imputed previousOwners.
    """

    owners_col = state["owners_col"]
    year_col = state["year_col"]
    mileage_col = state["mileage_col"]
    max_year = state["max_year"]
    mileage_threshold = state["mileage_threshold"]
    min_years_since_registration = state["min_years_since_registration"]
    imputation_median = state["imputation_median"]

    df_out = df.copy()

    # 1) Convert to numeric
    df_out[owners_col] = pd.to_numeric(df_out[owners_col], errors="coerce")

    # 2) Absolute values
    df_out[owners_col] = df_out[owners_col].abs()

    # 3) Round to nearest integer
    df_out[owners_col] = df_out[owners_col].round()

    # 4) Zero imprecision correction (using max_year from TRAIN ONLY)
    inac_own_condition = (
        ((max_year - df_out[year_col]) > min_years_since_registration)
        | (df_out[mileage_col] > mileage_threshold)
    ) & (df_out[owners_col] == 0)

    df_out.loc[inac_own_condition, owners_col] = 1

    # 5) Impute NaNs with train-based median
    df_out[owners_col] = df_out[owners_col].fillna(imputation_median)

    return df_out


# ### 4.7. mileage

# In[ ]:


def fit_mileage_imputer(
    train_df: pd.DataFrame,
    mileage_col: str = "mileage",
    do_abs: bool = True,
):
    """
    Fit step for mileage preprocessing.

    This imputer mirrors the structure of the previous numeric resolvers
    (year, previousOwners) but is intentionally simpler because mileage is a
    continuous variable with no hierarchical dependencies. The goal is to
    clean obvious inconsistencies and learn a robust central value (median)
    from TRAIN ONLY for later imputation.

    - Parameters
      train_df : pd.DataFrame
          Training dataset from which the imputation statistic is learned.
      mileage_col : str, default "mileage"
          Column containing mileage values.
      do_abs : bool, default True
          Whether to apply absolute-value correction to the mileage column.

    - Behavior 
      1) Convert the mileage column of train_df to numeric.
         - Non-numeric entries become NaN.
         - This step ensures consistent numeric handling across the dataset.

      2) Optionally apply abs() to mileage values.
         - Negative mileage values are physically impossible and typically
           arise from data entry errors.
         - If do_abs=True, convert all mileage values to their absolute form.
         - This mirrors the negative-value correction applied in
           previousOwners cleaning.

      3) Compute the imputation median.
         - The median is calculated after cleaning, skipping NaNs.
         - This median is stored and used during transform for imputing 
           missing or invalid mileage values.
         - Using the train-based median avoids data leakage and produces a
           stable, robust estimator less influenced by extreme values.

    - Returns
      state : dict
          The learned parameters needed for transformation:
          - 'mileage_col' : name of the mileage column
          - 'do_abs'      : whether abs() should be applied in transform
          - 'imputation_median' : median mileage value derived from TRAIN
    """
    
    tmp = train_df.copy()

    # 1) Convert to numeric (invalid values -> NaN)
    tmp[mileage_col] = pd.to_numeric(tmp[mileage_col], errors="coerce")

    # 2) Absolute values (if requested)
    if do_abs:
        tmp[mileage_col] = tmp[mileage_col].abs()

    # 3) Median after cleaning (NaNs skipped)
    imputation_median = tmp[mileage_col].median(skipna=True)

    state = {
        "mileage_col": mileage_col,
        "do_abs": do_abs,
        "imputation_median": imputation_median,
    }

    return state


# In[ ]:


def transform_mileage_imputer(
    df: pd.DataFrame,
    state: dict,
) -> pd.DataFrame:
    """
    Transform step for mileage preprocessing, applying the same cleaning
    sequence used in the fit step and imputing missing values with the
    median learned from TRAIN.

    This function ensures that mileage is treated consistently across
    train, validation, and test splits, without re-learning statistics.

    - Inputs
      df : pd.DataFrame
          Any dataset requiring mileage correction.
      state : dict
          Dictionary produced by fit_mileage_imputer, containing:
          - mileage_col : name of the mileage field
          - do_abs : whether absolute-value correction should be applied
          - imputation_median : train-derived median mileage

    - Behavior
      1) Convert mileage values to numeric.
         - Invalid values become NaN.
         - Ensures uniform numeric processing across datasets.

      2) Apply absolute-value correction (if used during fit).
         - Removes physically impossible negative mileage entries.
         - Keeps transform consistent with the fit logic.

      3) Impute NaNs using the TRAIN median.
         - Uses imputation_median from state.
         - Avoids leakage by not recomputing statistics on validation/test.

    - Returns
      df_out : pd.DataFrame
          A cleaned and imputed version of df, suitable for downstream modeling.
    """

    mileage_col = state["mileage_col"]
    do_abs = state["do_abs"]
    imputation_median = state["imputation_median"]

    df_out = df.copy()

    # 1) Convert to numeric
    df_out[mileage_col] = pd.to_numeric(df_out[mileage_col], errors="coerce")

    # 2) Absolute values (if used in fit)
    if do_abs:
        df_out[mileage_col] = df_out[mileage_col].abs()

    # 3) Impute NaNs with train-based median
    df_out[mileage_col] = df_out[mileage_col].fillna(imputation_median)

    return df_out


# ### 4.8. Tax

# In[ ]:


def fit_tax_imputer(
    train_df: pd.DataFrame,
    tax_col: str = "tax",
    do_abs: bool = True,
):
    """
    Fit step for tax preprocessing.

    This imputer follows the same structure as the mileage and previousOwners
    numeric resolvers, but is intentionally simpler because tax is a numerical
    field without hierarchical or contextual dependencies. The goal is to clean
    obvious inconsistencies and compute a stable imputation value using TRAIN
    only, avoiding leakage.

    - Behavior on train_df
      1) Convert the tax column to numeric (any non-numeric entries become NaN), 
      ensuring consistent downstream processing.

      2) Apply absolute value to tax (optional)
         - Negative tax values are physically impossible and typically arise
           from mistakes in the original dataset.
         - If do_abs=True, convert all tax values to abs(tax).

      3) Compute the median (after cleaning and skipping NaNs).
         - This median is the imputation statistic to be used during transform.

    - Parameters
      train_df : pd.DataFrame
          Training dataset from which the imputation statistic is derived.
      tax_col : str, default 'tax'
          Name of the tax column.
      do_abs : bool, default True
          Whether negative values should be converted to their absolute form.

    - Returns
      state : dict containing:
        - 'tax_col' : name of the tax column
        - 'do_abs' : whether abs() should be applied during transform
        - 'imputation_median' : median of cleaned tax values from TRAIN
    """

    tmp = train_df.copy()

    # 1) Convert to numeric (invalid values -> NaN)
    tmp[tax_col] = pd.to_numeric(tmp[tax_col], errors="coerce")

    # 2) Absolute values (if requested)
    if do_abs:
        tmp[tax_col] = tmp[tax_col].abs()

    # 3) Median after cleaning (NaNs skipped)
    imputation_median = tmp[tax_col].median(skipna=True)

    state = {
        "tax_col": tax_col,
        "do_abs": do_abs,
        "imputation_median": imputation_median,
    }

    return state


# In[ ]:


def transform_tax_imputer(
    df: pd.DataFrame,
    state: dict,
) -> pd.DataFrame:
    """
    Transform step for tax preprocessing, using the cleaning rules and
    imputation median learned in fit_tax_imputer.

    This function mirrors the sequence applied during fit, ensuring that
    tax data is treated consistently across training, validation, and test
    sets. No statistics are re-learned here, preventing leakage.

    - Inputs
      df : pd.DataFrame
          Any dataset whose tax column requires correction.
      state : dict
          Output of fit_tax_imputer containing:
          - 'tax_col' : name of the tax column
          - 'do_abs' : whether to apply absolute-value correction
          - 'imputation_median' : train-derived median for imputation

    - Behavior
      1) Convert the tax column to numeric(invalid strings become NaN),
       ensuring uniform numeric representation.

      2) Apply absolute value (if used in fit).
         - Maintains consistency with the cleaning logic learned from TRAIN.
         - Prevents negative tax values from propagating.

      3) Impute remaining NaN values using the TRAIN median.
         - No recomputation is performed here.
         - Ensures deterministic, leakage-free imputation.

    - Returns
      df_out : pd.DataFrame
          A transformed copy of df with cleaned and imputed tax values.
    """

    tax_col = state["tax_col"]
    do_abs = state["do_abs"]
    imputation_median = state["imputation_median"]

    df_out = df.copy()

    # 1) Convert to numeric
    df_out[tax_col] = pd.to_numeric(df_out[tax_col], errors="coerce")

    # 2) Absolute values (if used in fit)
    if do_abs:
        df_out[tax_col] = df_out[tax_col].abs()

    # 3) Impute NaNs with train-based median
    df_out[tax_col] = df_out[tax_col].fillna(imputation_median)

    return df_out


# ### 4.9. mpg

# In[ ]:


def fit_mpg_imputer(
    train_df: pd.DataFrame,
    mpg_col: str = "mpg",
    do_abs: bool = True,
    clip_lower: float = 10.0,
    clip_upper: float = 200.0,
):
    """
    Fit step for mpg preprocessing.

    This resolver parallels the tax and mileage imputers but includes an
    additional concept: CLIPPING. 

    Clipping is a post-processing step where numerical values are forced to 
    lie within a chosen minimum and maximum range. In other words:
    - If a value is below the lower bound, it is replaced with the lower bound
    - If a value is above the upper bound, it is replaced with the upper bound
    (Values inside the range stay unchanged)

    As we said, it is a post-processing step. Therefore clipping is not applied 
    during fit, but the chosen bounds are stored so that the transform step can 
    enforce physically plausible mpg values. The median used for imputation is 
    always computed before clipping to avoid introducing bias.

    - Steps performed on TRAIN:
      1) Convert the mpg column to numeric (invalid entries become NaN), 
      ensuring consistent downstream numeric processing.

      2) Optionally apply absolute value to mpg (mpg < 0 is physically impossible)
         - If do_abs=True, mpg values become abs(mpg).
         - Mirrors the logic of mileage and tax resolvers.

      3) Compute the imputation median (TRAIN only).
         - Crucially, this is done BEFORE any clipping.
         - The median provides a robust central tendency for imputing missing values.
         - Using TRAIN-only statistics prevents leakage.

      No clipping is applied here; only the bounds clip_lower and clip_upper
      are stored so they can be used consistently in the transform step.

    - Parameters
      train_df : pd.DataFrame
          The training dataset.
      mpg_col : str
          Column containing mpg values.
      do_abs : bool, default True
          Whether negative values should be made positive.
      clip_lower : float, default 10.0
          Lower bound to be applied later during transform.
      clip_upper : float, default 200.0
          Upper bound to be applied later during transform.

    - Returns
      state : dict containing:
        - 'mpg_col' : name of the mpg column
        - 'do_abs' : whether abs() should be applied during transform
        - 'clip_lower' : lower bound for clipping (not applied here)
        - 'clip_upper' : upper bound for clipping (not applied here)
        - 'imputation_median' : train-based median mpg (pre-clip)
    """

    tmp = train_df.copy()

    # 1) Convert to numeric
    tmp[mpg_col] = pd.to_numeric(tmp[mpg_col], errors="coerce")

    # 2) Absolute values (if requested)
    if do_abs:
        tmp[mpg_col] = tmp[mpg_col].abs()

    # 3) Median before clipping
    imputation_median = tmp[mpg_col].median(skipna=True)

    state = {
        "mpg_col": mpg_col,
        "do_abs": do_abs,
        "clip_lower": clip_lower,
        "clip_upper": clip_upper,
        "imputation_median": imputation_median,
    }

    return state


# In[ ]:


def transform_mpg_imputer(
    df: pd.DataFrame,
    state: dict,
) -> pd.DataFrame:
    """
    Apply the mpg preprocessing rules learned in `fit_mpg_imputer`.

    This transform replicates the exact cleaning steps applied during fitting,
    ensuring that mpg values are numerically consistent, corrected, imputed,
    and restricted to realistic physical limits. 
    All statistics come strictly from the train dataset to avoid leakage.

    - Inputs
      df : pd.DataFrame
          Dataset to be corrected (train, validation, or test).
      state : dict
          Output of fit_mpg_imputer, containing:
          - 'mpg_col' : name of the mpg column
          - 'do_abs' : whether abs() correction should be applied
          - 'clip_lower', 'clip_upper' : clipping bounds determined in fit
          - 'imputation_median' : train-based median (always pre-clipping)

    - Behavior
      1) Convert mpg to numeric (invalid strings become NaN), ensuring the 
      column is strictly numeric before further processing.

      2) Apply absolute-value correction (if used in fit).
         - Removes any physically impossible negative mpg values.
         - Keeps transform behavior aligned with the fitted configuration.

      3) Impute remaining NaNs with the TRAIN median.
         - Uses imputation_median; no statistics are computed on df.
         - Guarantees consistency and prevents validation/test leakage.

      4) Clip all mpg values to the [clip_lower, clip_upper] interval.
         - Enforces realistic bounds (e.g., mpg cannot be below 10 or above 200).
         - Uses the fit-defined thresholds for consistency across all splits.

    - Returns
      df_out : pd.DataFrame
          A corrected, imputed, and clipped copy of df suitable for modeling.
    """

    mpg_col = state["mpg_col"]
    do_abs = state["do_abs"]
    clip_lower = state["clip_lower"]
    clip_upper = state["clip_upper"]
    imputation_median = state["imputation_median"]

    df_out = df.copy()

    # 1) Convert to numeric
    df_out[mpg_col] = pd.to_numeric(df_out[mpg_col], errors="coerce")

    # 2) Absolute values (if used in fit)
    if do_abs:
        df_out[mpg_col] = df_out[mpg_col].abs()

    # 3) Impute NaNs with train-based median
    df_out[mpg_col] = df_out[mpg_col].fillna(imputation_median)

    # 4) Clip to the specified range
    df_out[mpg_col] = df_out[mpg_col].clip(lower=clip_lower, upper=clip_upper)

    return df_out


# ### 4.10. engine size

# In[ ]:


def fit_engine_size_imputer(
    train_df: pd.DataFrame,
    engine_col: str = "engineSize",
    do_abs: bool = True,
    treat_zero_as_nan: bool = True,
):
    """
    Fit step for preprocessing the engineSize column.

    This resolver mirrors the pattern used for mileage, tax, and mpg, but
    includes an additional domain-informed rule: engine size values equal to 0
    are typically invalid and should be treated as missing. The goal is to clean
    structurally incorrect values and learn a stable imputation statistic
    (median) from TRAIN ONLY.

    - Preprocessing on TRAIN:
      1) Convert the engine size column to numeric.
         - Non-numeric values become NaN.
         - Ensures uniform numeric handling.

      2) Optionally apply absolute-value correction.
         - Negative engine sizes are physically impossible and usually caused
           by data entry errors.
         - If do_abs=True, convert values to abs(engineSize).

      3) Optionally treat zeros as missing.
         - If treat_zero_as_nan=True:
             - Replace engineSize == 0 with NaN.
             - Justified because a real vehicle cannot have 0L displacement.
             - Ensures the imputer treats zeros the same as other invalid values.

      4) Compute the imputation median.
         - Median is computed after all cleaning steps.
         - NaNs are ignored.
         - This median will be used during transform to fill missing values in
           any dataset split, ensuring no leakage.

    - Parameters
      train_df : pd.DataFrame
          Training dataset used to learn the imputation statistic.
      engine_col : str
          Name of the engine size column.
      do_abs : bool, default True
          Whether negative values should be corrected using abs().
      treat_zero_as_nan : bool, default True
          Whether exact zeros should be considered missing.

    - Returns
      state : dict containing:
        - 'engine_col' : name of the engineSize column
        - 'do_abs' : whether abs() correction should be applied
        - 'treat_zero_as_nan' : whether zeros should be treated as NaN
        - 'imputation_median' : train-based median engineSize after cleaning
    """

    tmp = train_df.copy()

    # 1) Convert to numeric
    tmp[engine_col] = pd.to_numeric(tmp[engine_col], errors="coerce")

    # 2) Absolute values (if requested)
    if do_abs:
        tmp[engine_col] = tmp[engine_col].abs()

    # 3) Treat zeros as NaN (if requested)
    if treat_zero_as_nan:
        tmp.loc[tmp[engine_col] == 0, engine_col] = np.nan

    # 4) Median after cleaning (NaNs skipped)
    imputation_median = tmp[engine_col].median(skipna=True)

    state = {
        "engine_col": engine_col,
        "do_abs": do_abs,
        "treat_zero_as_nan": treat_zero_as_nan,
        "imputation_median": imputation_median,
    }

    return state


# In[ ]:


def transform_engine_size_imputer(
    df: pd.DataFrame,
    state: dict,
) -> pd.DataFrame:
    
    """
    Transform step for engineSize preprocessing, applying the same cleaning
    logic and imputation strategy learned during `fit_engine_size_imputer`.

    This transformation ensures that every dataset split (train, validation,
    test) receives a consistent treatment of numerical anomalies in engineSize,
    using only the train-derived statistics stored in `state`.

    - Inputs
      df : pd.DataFrame
          Dataset requiring engineSize correction.
      state : dict
          Output of fit_engine_size_imputer, containing:
          - 'engine_col'          : column name for engine size
          - 'do_abs'              : whether abs() correction must be applied
          - 'treat_zero_as_nan'   : whether exact zeros should be flagged as missing
          - 'imputation_median'   : median engineSize from train after cleaning

    - Behavior
      1) Convert engine size values to numeric (invalid entries become NaN), 
      ensuring uniform numeric processing before applying corrections.

      2) Apply absolute-value correction (if used during fit).
         - Removes physically impossible negative engine sizes.
         - Guarantees consistency between fit and transform steps.

      3) Treat zeros as missing (if the fit step used this rule).
         - Replaces exact zero with NaN when treat_zero_as_nan=True.
         - Encodes the assumption that an engine cannot have 0L capacity.

      4) Impute missing values using the train-based median.
         - No new statistics are computed here.
         - Prevents leakage and ensures all splits follow the same rule.

    - Returns
      df_out : pd.DataFrame
          A corrected and imputed copy of df, ready for downstream pipelines.
    """

    engine_col = state["engine_col"]
    do_abs = state["do_abs"]
    treat_zero_as_nan = state["treat_zero_as_nan"]
    imputation_median = state["imputation_median"]

    df_out = df.copy()

    # 1) Convert to numeric
    df_out[engine_col] = pd.to_numeric(df_out[engine_col], errors="coerce")

    # 2) Absolute values (if used in fit)
    if do_abs:
        df_out[engine_col] = df_out[engine_col].abs()

    # 3) Treat zeros as NaN (if used in fit)
    if treat_zero_as_nan:
        df_out.loc[df_out[engine_col] == 0, engine_col] = np.nan

    # 4) Impute NaNs with train-based median
    df_out[engine_col] = df_out[engine_col].fillna(imputation_median)

    return df_out


# <a id="encoding"></a>
# ## 5. Encoding 

# We encode categorical variables using custom encoders that provide full control
# over category handling.  
# - **One-Hot Encoding:** deterministic, consistent feature schema across all folds; 
# - **Target Encoding:** smoothed encoding that reduces noise for rare categories.
# 
# These encoders integrate naturally into our fit-transform preprocessing design.

# In[ ]:


class MyOneHotEncoder:
    def __init__(self):
        self.categories_ = {}

    def fit(self, X: pd.DataFrame):
        """
        Fit step for a custom one-hot encoder.

        This encoder mimics the behavior of sklearn's OneHotEncoder but is
        implemented manually for clarity and full control. The purpose of the
        fit step is to discover and store all unique categories present in
        each column of the training data, so that the transform step can be
        applied consistently across train, validation, and test.

        - Inputs
          X : pd.DataFrame
              DataFrame containing only categorical columns.

        - Behavior
          1) For each column in X, extract all distinct non-null values.
          2) Sort the categories to ensure deterministic column ordering.
          3) Store the resulting list in self.categories_ under the column's name.

        - Returns
          self : allows chaining via encoder.fit(X).transform(X_val)
        """
        for col in X.columns:
            self.categories_[col] = sorted(X[col].dropna().unique())
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform step applying one-hot encoding using the categories learned
        during fit. No new categories are created here; this avoids leakage
        and ensures consistent feature dimensions across dataset splits.

        - Inputs
          X : pd.DataFrame
              DataFrame to be encoded. Must contain the same columns used in fit,
              but values may include categories not seen during training.

        - Behavior
          1) For each column:
             - Retrieve its category list from self.categories_.
             - Apply pandas.get_dummies to obtain one-hot columns for the
               categories present in X.
          2) For categories that existed in TRAIN but do not appear in X:
             - Add the corresponding column with zeros.
             - This ensures identical feature schemas in train/val/test.
          3) Reorder columns to exactly match the TRAIN category order.
          4) Concatenate encoded blocks for all columns.

        - NOTES:
          - Categories unseen during fit simply do not have a column and are
            ignored (handle_unknown='ignore');
          - This encoder always produces dense (not sparse) DataFrames.

        - Returns
          pd.DataFrame : one-hot encoded representation of X.
        """
        X_new = []

        for col in X.columns:
            cats = self.categories_[col]

            # Standard dummy encoding
            encoded = pd.get_dummies(X[col], prefix=col)

            # Ensure all TRAIN-learned categories appear
            for c in cats:
                col_name = f"{col}_{c}"
                if col_name not in encoded.columns:
                    encoded[col_name] = 0

            # Reorder to match TRAIN category order
            encoded = encoded[[f"{col}_{c}" for c in cats]]

            X_new.append(encoded)

        return pd.concat(X_new, axis=1)


# We use a custom OHE instead of sklearn’s version for three practical reasons:
# 1. **Transparency:** The project requires full visibility into how categories are learned and transformed. Our encoder makes every step explicit.
# 2. **Deterministic schema:** We enforce a strict category order and guarantee identical output columns across train/validation/test.
# 3. **Pipeline compatibility:** Our preprocessing is built from custom fit/transform functions rather than sklearn Pipelines, so a minimal, self-contained encoder integrates more cleanly.
# 
# This keeps the encoding logic simple, predictable, and fully aligned with our custom preprocessing framework.

# In[ ]:


class MyTargetEncoder:
    def __init__(self, smoothing=5):
        self.smoothing = smoothing
        self.target_means_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit step for a custom target encoder.

        This encoder replaces each categorical value with a smoothed estimate
        of the mean target for that category. The smoothing parameter controls
        how strongly rare categories are pulled toward the global mean, reducing
        the risk of overfitting.

        - Behavior
          1) Compute the global target mean (used as fallback and smoothing base).
          2) For each categorical column:
             - Group by category and compute:
               * mean(target)
               * count of rows in the category
             - Compute smoothing weight = 1 / (1 + exp(-(count - smoothing_param)))
               This pushes low-frequency categories toward the global mean.
             - Combine category mean and global mean using the smoothing weight.
          3) Store the resulting mappings in `self.target_means_`.

        - Returns
          self
        """
        df = X.copy()
        df["target"] = y
        self.global_mean_ = y.mean()

        for col in X.columns:
            stats = df.groupby(col)["target"].agg(["mean", "count"])
            smoothing = 1 / (1 + np.exp(-(stats["count"] - self.smoothing)))

            enc = self.global_mean_ * (1 - smoothing) + stats["mean"] * smoothing
            self.target_means_[col] = enc.to_dict()

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform step that replaces categories with the smoothed target means
        computed during fit.

        - Behavior
          1) For each column, map each category to its encoded value.
          2) Categories unseen in TRAIN map to NaN; these are replaced with
             the global mean.
          3) Output is a numerical DataFrame suitable for model training.

        - Returns
          X_new : pd.DataFrame with encoded categorical columns.
          
        """
        X_new = X.copy()

        for col in X.columns:
            mapping = self.target_means_[col]
            X_new[col] = X_new[col].map(mapping)
            X_new[col] = X_new[col].fillna(self.global_mean_)

        return X_new


# Similarly to what happened with OHE, we felt that custom target encoder would give us **full control over the smoothing formula, the fallback behavior for unseen categories, and the fit/transform mechanics**.
# 
# It integrates naturally with our custom preprocessing pipeline (which does not rely on sklearn Pipelines) and ensures deterministic, transparent behavior across train/validation/test.
# 
# In contrast, library implementations hide many details (regularization choices, handling of unknown categories, column-wise logic), making it harder to document each step and match the exact methodological requirements of the project.

# <a id="fs"></a>
# ## 6. Feature Selection

# We use a Random-Forest-based selector to rank features by importance and retain
# only the most predictive ones.
# 
# This reduces dimensionality, simplifies model training, and improves the
# signal-to-noise ratio before fitting final models.

# In[ ]:


class MyRandomForestSelector:
    def __init__(self, n_features=10, n_estimators=300, random_state=42):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_ = None
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit step for a Random-Forest-based feature selector.

        This selector uses a RandomForestRegressor to estimate feature
        importances and keeps only the top `n_features` predictors.
        It follows the same fit/transform design as the rest of the
        preprocessing pipeline, allowing for clean separation of
        training-based learning and deterministic transformation.

        - Behavior
          1) Train a RandomForestRegressor on (X, y).
             - Random forests naturally compute impurity-based feature
               importance, which quantifies each feature's contribution
               to reducing prediction error.

          2) Extract feature importances and sort them in descending order.

          3) Select the indices of the top `n_features` most important
             predictors and store their names in `self.selected_features_`.

          4) Store the trained model in `self.model_` (useful for inspection,
             not used during transform)

        - NOTES:
          - Using RandomForestRegressor makes the selector robust to
            nonlinearities and interactions between features.

        - Returns
          self
        """
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X, y)

        importances = model.feature_importances_
        idx = importances.argsort()[::-1][:self.n_features]

        self.selected_features_ = X.columns[idx].tolist()
        self.model_ = model

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform step that reduces the dataset to the features selected
        during fit.

        - Behavior
          Returns a DataFrame containing only the columns stored in
          `self.selected_features_`.

        - Assumptions
          - X contains at least these columns.

        - Returns
          pd.DataFrame with reduced feature set.
        """
        return X[self.selected_features_]


# __Why Random Forest for feature selection?__
# 
# Random Forest is a strong choice because it:
# - Captures nonlinear relationships and interactions
# - Provides stable, built-in importance scores
# - Works well with mixed feature types
# - Ranks features without depending on any specific downstream model
# 
# This makes it a reliable, model-agnostic method for selecting the most informative predictors.

# <a id="general-preproc"></a>
# ## 7. General Preprocessing Application

# We now sequentially apply all preprocessing components to the full training dataset.
# 
# This includes categorical normalization, deterministic corrections, numerical imputation, encoding, and feature selection. The result is a clean and model-ready dataset.

# In[36]:


# full_train_dataset = full_train_dataset.drop(columns=['carID', 'hasDamage', 'paintQuality%'])


# In[37]:


# full_train_dataset.head()


# In[ ]:


# categorical features 
cat_feat = ['Brand', 'model', 'transmission', 'fuelType']

# numerical features 
num_feat = ['year', 'mileage', 'engineSize', 'tax', 'mpg', 'previousOwners']


# ### 7.1. Categorical pre-processing

# In[ ]:


def preprocess_categorical(df, columns, remove_middle_spaces=True, allow_extra_chars=""):
    """
    Preprocess categorical columns by applying consistent string normalization
    and explicit missing-value encoding.

    This function performs global, deterministic preprocessing. It does not
    depend on train statistics, so it can be applied equally to train,
    validation, and test splits without introducing leakage.

    - Inputs
      df : pd.DataFrame
          Dataset containing categorical columns.
      columns : list of str
          Column names to be processed.
      remove_middle_spaces : bool, default True
          Whether to remove internal spaces ("C CLASS" -> "CCLASS").
          This matches the behavior of all earlier string-normalization steps.
      allow_extra_chars : str, default ""
          Extra non-alphanumeric characters to preserve in the cleaned output.

    - Behavior
      For each specified column:
        1) Replace missing values (NaN) with the literal token "UNKNOWN", ensuring 
        explicit handling of missingness and avoiding category leakage through imputation.

        2) Apply `column_string_transformer`, which performs:
           - whitespace stripping
           - uppercase conversion
           - accent removal
           - optional removal of internal spaces
           - removal of punctuation / symbols
           This enforces a strict canonical representation for all categories.

      The function always returns a new DataFrame, leaving the original untouched.

    - Returns
      pd.DataFrame
          A copy of df where all selected categorical columns have been
          cleaned and normalized.
    """

    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue  # ignore missing columns gracefully

        # Step 1: explicit missing-value token
        df[col] = fill_unknown(df[col])

        # Step 2: apply the unified string normalizer
        df = column_string_transformer(
            df=df,
            column=col,
            remove_middle_spaces=remove_middle_spaces,
            allow_extra_chars=allow_extra_chars,
        )

    return df


# In[40]:

""" 
full_train_dataset = preprocess_categorical(
    full_train_dataset,
    cat_feat,
    remove_middle_spaces=True,
    allow_extra_chars=""
)
"""


# ### 7.2. Brand deterministic preprocessing 

# In[ ]:

""" 
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
"""

# This identifies every brand string not appearing in valid_brands.
# Then correct_invalid_brands_in_df applies deterministic rules:
# - substring-based correction when the invalid string matches exactly one valid brand,
# - storage of corrections for transparency,
# - return of any residual invalid values for inspection.
# 
# This ensures that all brand names become valid and standardized before model-mapping.
# 

# ### 7.3. Model deterministic normalization

# In[ ]:


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
            allow_extra_chars=""       # default
        )
        for model in models
    ]
    for brand, models in valid_models_by_brand.items()
}

valid_models_by_brand


# Then we pass each model name through `basic_string_transformer` to guarantee the same normalization rules used for real dataset entries.
# 
# This ensures that model resolution later on (via fit/transform) uses perfectly normalized valid model names.

# ### 7.4. Deterministic transmission and fuelType

# In[ ]:


valid_transmissions = ['MANUAL', 'AUTOMATIC', 'SEMIAUTO']
valid_fueltypes = ['PETROL', 'DIESEL', 'HYBRID']



# In[ ]:


def transform_tax_custom_rules(
    df: pd.DataFrame,
    tax_col: str = "tax",
    year_col: str = "year",
    fuel_col: str = "fuelType",
    engine_col: str = "engineSize"
) -> pd.DataFrame:
    
    

    df_out = df.copy()

    # Step 1: enforce numeric representation
    df_out[tax_col] = pd.to_numeric(df_out[tax_col], errors="coerce")

    # Encapsulated business logic for UK VED-based corrections
    def apply_rules(row):
        year = row[year_col]
        engine = row[engine_col]
        val = row[tax_col]

        # Normalize fuel type for comparability
        fuel = str(row[fuel_col]).strip().upper()

        # Helper function for seamlessly applying min/max bounds
        def apply_cap(value, min_v, max_v):
            if pd.isna(value):
                return (min_v + max_v) / 2  # midpoint imputation for unknown tax
            if value < min_v: return min_v
            if value > max_v: return max_v
            return value

        # --- Diesel / Petrol rules ---
        if fuel in ['DIESEL', 'PETROL']:
            if year < 2001:
                return apply_cap(val, 100, 400)
            elif 2001 <= year < 2017:
                if engine < 1.3:
                    return apply_cap(val, 0, 40)
                elif 1.3 <= engine < 1.7:
                    return apply_cap(val, 100, 160)
                elif 1.7 <= engine < 2.3:
                    return apply_cap(val, 165, 250)
                else:
                    return apply_cap(val, 300, 500)
            else:
                return apply_cap(val, 140, 600)

        # --- Electric vehicles ---
        elif fuel == 'ELECTRIC':
            return 0

        # --- Hybrids & misc fuels ---
        elif fuel in ['HYBRID', 'OTHER']:
            if year < 2017:
                if engine < 2.0:
                    return apply_cap(val, 0, 20)
                elif 2.0 <= engine <= 3.0:
                    return apply_cap(val, 30, 250)
                else:
                    # Extreme block
                    if pd.isna(val) or val < 200: return 200
                    if val > 500: return 500
                    return val
            else:
                if engine > 2.5:
                    return apply_cap(val, 150, 500)
                else:
                    return apply_cap(val, 150, 200)

        # --- General fallback ---
        if year < 2001:
            return apply_cap(val, 100, 350)
        elif year < 2017:
            return apply_cap(val, 0, 580)
        else:
            return apply_cap(val, 140, 600)

    # Step 2: apply rule-based correction row-by-row
    df_out[tax_col] = df_out.apply(apply_rules, axis=1)

    return df_out


# <a id="fe"></a>
# ## 8. Feature Engineering

# We generate additional derived features that capture domain-specific structure.
# 
# These features improve model expressiveness and help tree-based and linear 
# models capture relevant relationships more effectively.

# ### 8.1. Car Age

# In[ ]:


def create_age_and_drop_year(df, year_col="year", base_year=2020, clip_future=True):
    """
    Create an 'age' feature from the car's registration year and optionally
    drop the original year column.

    - Purpose
      Convert a raw 'year' variable into a more meaningful feature: the vehicle's
      age. Age captures depreciation and usage patterns more directly than the
      raw production year.

    - Inputs
      df : pd.DataFrame
          Dataset containing the year column.
      year_col : str
          Column storing the car's registration/production year.
      base_year : int
          Reference year used to compute age as (base_year - year).
          Is 2020, since that's what we've been told in the project guidelines.
      clip_future : bool
          If True, ages below 0 (cars supposedly in the “future”) are clipped
          to 0 to avoid implausible negative ages.

    - Behavior
      1) Convert year to numeric and coerce invalid values to NaN.
      2) Compute age = base_year - year.
      3) Clip negative values (future years) to 0 if clip_future=True.
      4) Add new column 'age'.
      5) Drop the original year column for cleaner feature space.

    - Returns
      pd.DataFrame with the new 'age' feature and without the original year column.
    """

    df = df.copy()

    year = pd.to_numeric(df[year_col], errors="coerce")
    age = base_year - year

    if clip_future:
        age = age.clip(lower=0)

    df["age"] = age
    df = df.drop(columns=[year_col])
    
    return df


# ### 8.2. Multiple Previous Owners

# In[ ]:


def add_owners_flagged(df, owners_col="previousOwners", new_col="owners_flagged",
                       drop_original=True, na_as_zero=True):
    """
    Create a binary flag indicating whether a vehicle had multiple owners.

    - Purpose
      Cars with more than one previous owner often exhibit different resale
      patterns and maintenance histories. A binary feature captures this signal
      in a compact form.

    - Inputs
      df : pd.DataFrame
          Dataset containing the previous owners column.
      owners_col : str
          Column storing the count of previous owners.
      new_col : str
          Name of the new binary feature.
      drop_original : bool
          Whether to remove the raw owners column after creating the flag.
      na_as_zero : bool
          Whether missing values should be treated as zero owners.

    - Behavior
      1) Convert owners to numeric.
      2) Replace NaN with 0 if na_as_zero=True.
      3) Create binary flag: 1 if owners > 1, else 0.
      4) Drop original owners column if requested.

    - Returns
      pd.DataFrame with new 'owners_flagged' feature (int8).
    """
    df = df.copy()

    owners = pd.to_numeric(df[owners_col], errors="coerce")

    if na_as_zero:
        owners = owners.fillna(0)

    df[new_col] = (owners > 1).fillna(False).astype("int8")

    if drop_original and owners_col in df.columns:
        df = df.drop(columns=[owners_col])
        
    return df


# ### 8.3. Log-Mileage and Mileage per Year

# In[ ]:


def add_mileage_features(df, mileage_col="mileage", age_col="age",
                         drop_original=True, drop_ratio=True):
    """
    Add log-transformed mileage features and mileage-per-year features.

    - Purpose
      Mileage is one of the strongest predictors of car price, but:
      - its distribution is heavily right-skewed,
      - raw mileage does not account for vehicle age.

      These engineered features stabilize scale (via log transform) and
      capture usage intensity (through miles per year).

    - Inputs
      df : pd.DataFrame
          Dataset containing mileage and age.
      mileage_col : str
          Column storing total mileage.
      age_col : str
          Column storing vehicle age (must be computed earlier).
      drop_original : bool
          Whether to drop raw mileage after computing new features.
      drop_ratio : bool
          Whether to drop the raw miles_per_year ratio and keep only its log form.

    - Behavior
      1) Convert mileage to numeric; negative values become NaN.
      2) Create log_mileage, useful for skewed distributions.
      3) Compute miles_per_year = mileage / max(age, 1)  
         - age clipped at 1 to avoid division by zero.
      4) Create log_miles_per_year.
      5) Optionally drop raw ratio and raw mileage.

    - Returns
      pd.DataFrame with new engineered mileage features.
    """

    df = df.copy()

    mileage = pd.to_numeric(df[mileage_col], errors="coerce")
    mileage = mileage.where(mileage >= 0, np.nan)

    df["log_mileage"] = np.log1p(mileage)

    age = pd.to_numeric(df[age_col], errors="coerce")
    age_safe = age.clip(lower=1)

    df["miles_per_year"] = mileage / age_safe
    df["log_miles_per_year"] = np.log1p(df["miles_per_year"])

    if drop_ratio and "miles_per_year" in df.columns:
        df = df.drop(columns=["miles_per_year"])

    if drop_original and mileage_col in df.columns:
        df = df.drop(columns=[mileage_col])

    return df


# ### 8.4. EngineSize Bins

# In[ ]:


def add_engine_bins(df, engine_col="engineSize", new_col="engine_bin", bins=None):
    """
    Create discretized bins for engine displacement.

    - Purpose
      Engine size often has nonlinear relationships with price (for example, jumps
      between 1.0L, 1.2L, 1.6L, 2.0L, etc.). 
      Binning captures these category-like thresholds and helps tree-based models 
      identify meaningful splits.

    - Inputs
      df : pd.DataFrame
          Dataset containing engine size.
      engine_col : str
          Name of the original engine column.
      new_col : str
          Name of the new binned feature.
      bins : list or None
          Custom bin edges. If None, a default set of automotive-appropriate
          displacement breakpoints is used.

    - Behavior
      1) Convert engine size to numeric (negative values become NaN).
      2) If bins is None, use sensible defaults:
         [0, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, inf]
      3) Use pd.cut to assign each engine size to a bin.
      4) Return labeled integer bins as a new column.

    - Returns
      pd.DataFrame including an integer-coded engine_bin feature.
    """
    df = df.copy()
    engine = pd.to_numeric(df[engine_col], errors="coerce")
    engine = engine.where(engine >= 0, np.nan)

    if bins is None:
        bins = [0, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, np.inf]

    df[new_col] = pd.cut(engine, bins=bins, include_lowest=True, labels=False).astype("Int64")
    return df

