import pandas as pd
import streamlit as st

from interface.backend.hgb_best_config import (
    fit_hgb_best, predict_hgb_best,
    X, y,
    categorical_features,
    valid_brands, valid_models_by_brand, valid_transmissions, valid_fueltypes
)

st.set_page_config(page_title="Predict | CARS4YOU", layout="wide")
st.title("Predict")

mode = st.radio(
    "Choose input mode:",
    options=["Multiple cars (CSV)", "Single car (form)"],
    horizontal=True,
)

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

@st.cache_resource
def get_fitted():
    return fit_hgb_best(
        X, y,
        categorical_features=categorical_features,
        valid_brands=valid_brands,
        valid_models_by_brand=valid_models_by_brand,
        valid_transmissions=valid_transmissions,
        valid_fueltypes=valid_fueltypes,
    )

st.divider()

if mode == "Multiple cars (CSV)":
    st.subheader("Multiple cars (CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

    st.caption("Required columns (must exist, values may be missing):")
    st.code(", ".join(CSV_REQUIRED_COLS), language="text")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("File preview:")
            st.dataframe(df.head(20), use_container_width=True)

            missing = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            st.success("All required columns are present.")

            if st.button("Generate predictions", type="primary", use_container_width=True):
                with st.spinner("Training/loading model (cached) + predicting..."):
                    fitted = get_fitted()
                    out = predict_hgb_best(df, fitted, id_col="carID")

                st.subheader("Predictions preview")
                st.dataframe(out.head(50), use_container_width=True)

                st.download_button(
                    "Download predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Could not read/predict from the CSV: {e}")

else:
    st.subheader("Single car (form)")
    st.caption("carID is not required here.")

    with st.form("single_car_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            Brand = st.text_input("Brand", value="")
            model_ = st.text_input("model", value="")
            year = st.number_input("year", min_value=1900, max_value=2100, step=1, value=2017)

        with c2:
            transmission = st.text_input("transmission", value="")
            mileage = st.number_input("mileage", min_value=0.0, step=1000.0, value=0.0)
            fuelType = st.text_input("fuelType", value="")
            tax = st.number_input("tax", step=1.0, value=0.0)

        with c3:
            mpg = st.number_input("mpg", step=0.1, value=0.0)
            engineSize = st.number_input("engineSize", step=0.1, value=0.0)
            previousOwners = st.number_input("previousOwners", min_value=0, step=1, value=0)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = {
            "Brand": str(Brand),
            "model": str(model_),
            "year": int(year),
            "transmission": str(transmission),
            "mileage": float(mileage),
            "fuelType": str(fuelType),
            "tax": float(tax),
            "mpg": float(mpg),
            "engineSize": float(engineSize),
            "previousOwners": int(previousOwners),
        }
        df_single = pd.DataFrame([row], columns=SINGLE_COLS)

        with st.spinner("Training/loading model (cached) + predicting..."):
            fitted = get_fitted()
            out = predict_hgb_best(df_single, fitted, id_col="carID")

        st.write("Input:")
        st.dataframe(df_single, use_container_width=True)

        st.subheader("Prediction (€)")
        st.write(float(out["price"].iloc[0]))
