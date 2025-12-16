import streamlit as st

# Configure the Streamlit app at the very beginning.
# - page_title appears in the browser tab
# - wide layout gives more horizontal space 
# - collapsed sidebar 
st.set_page_config(
    page_title="CARS4YOU",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# custom CSS 
st.markdown(
    """
    <style>
      /* Page background (light gray) */
      [data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 600px at 50% 0%, rgba(255, 198, 25, 0.18), transparent 60%),
                    linear-gradient(180deg, #F5F6F8 0%, #EEF0F3 100%);
      }

      .block-container {
        padding-top: 1rem;
      }

      .topbar {
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        padding: 0.75rem 0;
        margin-bottom: 1.25rem;
      }

      .brand {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
      }

      div[data-testid="column"] button {
        width: 100%;
      }

      /* Hero as a card (white-ish) */
      .hero-wrap {
        max-width: 900px;
        margin: 9rem auto 0 auto;
        margin-bottom: 3rem;   /* <-- ISTO sobe o botão (mais negativo = mais sobe) */
        padding: 2.5rem 2rem;
        border-radius: 18px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        background: rgba(255, 255, 255, 0.75);
        box-shadow: 0 12px 28px rgba(0,0,0,0.10);
        text-align: center;
        backdrop-filter: blur(6px);
      }

      .accent {
        width: 70px;
        height: 6px;
        border-radius: 999px;
        margin: 0 auto 1.25rem auto;
        background: #FFC619;
      }

      .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
      }

      .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.85;
        margin: 0 0 1.2rem 0;
      }

      /* Primary button = orange (#FFC619) */
      div.stButton > button[kind="primary"] {
        background-color: #FFC619 !important;
        border-color: #FFC619 !important;
        color: #000000 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
      }
      div.stButton > button[kind="primary"]:hover {
        background-color: #FFC619 !important;
        border-color: #FFC619 !important;
        color: #000000 !important;
        filter: brightness(0.95);
      }
      .predict-wrap {
        max-width: 900px;
        margin: -1.0rem auto 0 auto;  /* <-- sobe (negativo) / desce (positivo) */
        padding: 0 2rem;              /* alinha com o padding do card */
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# modal 
# st.dialog creates a pop up UI 
@st.dialog("How to use CARS4YOU")
def show_info_dialog():
    st.markdown(
        """
**Quick instructions**

1. Go to **Predict**.
2. Choose:
   - **Multiple cars (CSV)**: upload a CSV with the required columns; you will be able to download a CSV with the predictions.
   - **Single car (form)**: fill the form and get one prediction.

**Notes**
- Missing values are allowed as long as the required columns exist (in case you submit the csv).
- Predictions are returned in pounds sterling (£).
        """
    )
    st.caption("Close this dialog with the X in the top-right corner.")


# Top bar layout:
# - left side shows the brand name
# - right side shows an Info button that opens the dialog
st.markdown('<div class="topbar">', unsafe_allow_html=True)

# columns to control spacing
c1, c2 = st.columns([9, 1], vertical_alignment="center")

with c1:
    st.markdown('<div class="brand">CARS4YOU</div>', unsafe_allow_html=True)

with c2:
    # clicking "Info" is the trigger to the modal function call
    if st.button("Info", use_container_width=True):
        show_info_dialog()

st.markdown("</div>", unsafe_allow_html=True)

# This is static landing content that introduces the app
st.markdown(
    """
    <div class="hero-wrap">
      <div class="accent"></div>
      <div class="hero-title">CARS4YOU</div>
      <div class="hero-subtitle">Welcome to Cars4You Prediction Interface</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# wrapper that controls the Predict button alignment and spacing
st.markdown('<div class="predict-wrap">', unsafe_allow_html=True)

# Use 3 columns to center the Predict button:
# - left spacer (4)
# - button column (2)
# - right spacer (4)
_, cbtn, _ = st.columns([4, 2, 4])

with cbtn:
    # Primary button uses the orange styling defined in CSS above
    # st.switch_page navigates to the streamlit multipage app file
    if st.button("Predict", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Predict.py")

st.markdown("</div>", unsafe_allow_html=True)
