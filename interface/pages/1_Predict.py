import numpy as np
import pandas as pd
import streamlit as st

# We keep the modeling/training logic in a separate backend module
# This Streamlit file is only responsible for:
# - collecting inputs (CSV or single form)
# - validating schema
# - calling the backend "fit" (cached) and "predict"
from backend.stack_best_config import (
    fit_stacking_best, predict_stacking_best,
    X, y,
    valid_brands, valid_models_by_brand, valid_transmissions, valid_fueltypes
)

# Configure the Streamlit page - title and layout
# "wide" for better display
st.set_page_config(page_title="Predict | CARS4YOU", layout="wide")

# Global CSS customizations.
# important for styling
# - st.button(type="primary")
# - st.form_submit_button(type="primary")
#
# We define a single CSS variable (--c4y-orange) so the color is consistent.
# this variable represents the hexadecimal of the orange color
st.markdown(
    """
    <style>
      :root { --c4y-orange: #FFC619; }

      /* st.button(type="primary") */
      div.stButton > button[kind="primary"] {
        background-color: var(--c4y-orange) !important;
        border-color: var(--c4y-orange) !important;
        color: #000000 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
      }
      div.stButton > button[kind="primary"]:hover {
        filter: brightness(0.95);
      }

      /* st.form_submit_button(type="primary") (wrapper diferente) */
      div[data-testid="stFormSubmitButton"] > button,
      div[data-testid="stFormSubmitButton"] button[kind="primary"],
      button[kind="primaryFormSubmit"] {
        background-color: var(--c4y-orange) !important;
        border-color: var(--c4y-orange) !important;
        color: #000000 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
      }
      div[data-testid="stFormSubmitButton"] > button:hover,
      div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
      button[kind="primaryFormSubmit"]:hover {
        filter: brightness(0.95);
      }
    </style>
    """,
    unsafe_allow_html=True,  # Streamlit blocks raw HTML by default; we explicitly allow it for CSS styling
)

# Page title
st.markdown('<div class="page-title">Predict</div>', unsafe_allow_html=True)

# Input mode selector:
# - CSV mode supports batch predictions
# - Single form mode supports an interactive single-car prediction
mode = st.radio(
    "You can choose one of these input modes:",
    options=["Multiple cars (CSV)", "Single car (form)"],
    horizontal=True,
)

# These are the columns that the batch CSV upload must contain
# Values may be missing, but the column names must exist to build the expected schema
CSV_REQUIRED_COLS = [
    "carID",
    "Brand",
    "model",
    "year",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize",
    "paintQuality%",
    "previousOwners",
    "hasDamage",
]

# These are the fields requested in the single-car form.
SINGLE_COLS = [
    "Brand",
    "model",
    "year",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize",
    "previousOwners",
]


# We cache the fitted pipeline/model to avoid re-training on every interaction
# This is important in Streamlit to avoid having to do it again for each script rerun
@st.cache_resource
def get_fitted(_cache_buster: str = "v7"):
    return fit_stacking_best(
        X, y,
        valid_brands=valid_brands,
        valid_models_by_brand=valid_models_by_brand,
        valid_transmissions=valid_transmissions,
        valid_fueltypes=valid_fueltypes,
    )

st.divider()

# ---> here we do the csv upload
if mode == "Multiple cars (CSV)":
    st.subheader("Multiple cars (CSV)")

    # File uploader accepts exactly one CSV at a time.
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

    # Show required schema to reduce user errors.
    st.caption("Required columns (must exist, values may be missing):")
    st.code(", ".join(CSV_REQUIRED_COLS), language="text")

    if uploaded is not None:
        try:
            # Read uploaded CSV into a DataFrame.
            df = pd.read_csv(uploaded)

            # Preview the first rows so we can visually check columns and values.
            st.write("File preview:")
            st.dataframe(df.head(20), use_container_width=True)

            # schema check: all required columns must be present
            missing = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            st.success("All required columns are present.")

            # generate prediction button
            if st.button("Generate predictions", type="primary", use_container_width=True):
                with st.spinner("Training/loading model (cached) + predicting..."):
                    fitted = get_fitted()
                    # Backend handles preprocessing + model inference
                    # We also pass id_col so the output can keep identifiers
                    out = predict_stacking_best(df, fitted, id_col="carID")

                st.subheader("Predictions preview")
                st.dataframe(out.head(50), use_container_width=True)

                # Provide a direct CSV download for the predictions
                st.download_button(
                    "Download predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # in case there is a problem we display an error message, instead of crashing
        except Exception as e:
            st.error(f"Could not read/predict from the CSV: {e}")

# this else represents the "selection" of the single car form
else:
    st.subheader("Single car (form)")
    st.caption("Fields may be left blank; blank numeric fields are sent as NaN to the backend.")

    # We restyle form buttons here too
    st.markdown(
        """
        <style>
          :root { --c4y-orange: #FFC619; }

          /* botões primary (geral) */
          button[kind="primary"],
          button[data-testid="baseButton-primary"] {
            background-color: var(--c4y-orange) !important;
            border-color: var(--c4y-orange) !important;
            color: #000000 !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
          }
          button[kind="primary"]:hover,
          button[data-testid="baseButton-primary"]:hover {
            filter: brightness(0.95);
          }

          /* submit dentro de forms (wrapper diferente) */
          div[data-testid="stFormSubmitButton"] button {
            background-color: var(--c4y-orange) !important;
            border-color: var(--c4y-orange) !important;
            color: #000000 !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
          }
          div[data-testid="stFormSubmitButton"] button:hover {
            filter: brightness(0.95);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # We use a form so that predictions only run when we submit,
    # not whenever a single widget changes
    with st.form("single_car_form", clear_on_submit=False):

        # We group inputs into rows of 3 columns for readability
        # row 1: brand, model, year
        c1, c2, c3 = st.columns(3)
        with c1:
            Brand = st.text_input(
                "Brand",
                value="",
                help="Manufacturer / brand (e.g., FORD, BMW)."
            )
        with c2:
            model = st.text_input(
                "Model",
                value="",
                help="Car model (e.g., FOCUS, GOLF)."
            )
        with c3:
            year = st.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                step=1,
                value=None,  # <-- allow blank (returns None)
                help="Registration/manufacture year."
            )

        # Row 2: transmission, fuel type and previous owners
        c1, c2, c3 = st.columns(3)
        with c1:
            transmission = st.text_input(
                "Transmission",
                value="",
                help="Transmission type (e.g., MANUAL, AUTOMATIC, SEMIAUTO)."
            )
        with c2:
            fuelType = st.text_input(
                "Fuel type",
                value="",
                help="Fuel category (e.g., PETROL, DIESEL, HYBRID, ELECTRIC)."
            )
        with c3:
            previousOwners = st.number_input(
                "Previous owners",
                min_value=0,
                step=1,
                value=None,  # <-- allow blank
                help="Number of previous owners."
            )

        # row 3: mileage, engine size and mpg
        c1, c2, c3 = st.columns(3)
        with c1:
            mileage = st.number_input(
                "Mileage",
                min_value=0.0,
                step=1000.0,
                value=None,  # <-- allow blank
                help="Total mileage of the car (in miles)."
            )
        with c2:
            engineSize = st.number_input(
                "Engine size",
                min_value=0.0,
                step=0.1,
                value=None,  # <-- allow blank
                help="Engine displacement (in liters, e.g., 1.6)."
            )
        with c3:
            mpg = st.number_input(
                "MPG",
                min_value=0.0,
                step=0.1,
                value=None,  # <-- allow blank
                help="Miles per gallon."
            )

        # row 4: tax (alone, but aligned on the left)
        c1, _, _ = st.columns(3)
        with c1:
            tax = st.number_input(
                "Tax (£)",
                min_value=0.0,
                step=1.0,
                value=None,  # <-- allow blank
                help="Vehicle tax value."
            )

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        # Helpers: keep NaN when the user leaves a numeric field blank
        def opt_int(v):
            return np.nan if v is None else int(v)

        def opt_float(v):
            return np.nan if v is None else float(v)

        # Build a single-row record (match backend schema)
        row = {
            "Brand": str(Brand).strip(),
            "model": str(model).strip(),
            "year": opt_int(year),
            "transmission": str(transmission).strip(),
            "mileage": opt_float(mileage),
            "fuelType": str(fuelType).strip(),
            "tax": opt_float(tax),
            "mpg": opt_float(mpg),
            "engineSize": opt_float(engineSize),
            "previousOwners": opt_int(previousOwners),
        }

        df_single = pd.DataFrame([row], columns=SINGLE_COLS)

        with st.spinner("Training/loading model (cached) + predicting..."):
            fitted = get_fitted()
            out = predict_stacking_best(df_single, fitted, id_col="carID")

        st.write("Input:")
        st.dataframe(df_single, use_container_width=True)

        st.subheader("Prediction (£)")
        st.write(float(out["price"].iloc[0]))
