import streamlit as st

st.set_page_config(
    page_title="CARS4YOU",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CSS para dar aspeto de "barra" no topo ---
st.markdown(
    """
    <style>
      .topbar {
        border-bottom: 1px solid rgba(49, 51, 63, 0.18);
        padding: 0.75rem 0;
        margin-bottom: 1.25rem;
      }
      .brand {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
      }
      /* Ajuste ligeiro dos botões no topo */
      div[data-testid="column"] button {
        width: 100%;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Top bar (CARS4YOU à esquerda, botões à direita) ---
st.markdown('<div class="topbar">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([8, 1, 1], vertical_alignment="center")

with c1:
    st.markdown('<div class="brand">CARS4YOU</div>', unsafe_allow_html=True)

with c2:
    if st.button("Predict", use_container_width=True):
        # precisa de existir: interface/pages/1_Predict.py
        st.switch_page("pages/1_Predict.py")

with c3:
    if st.button("Chat", use_container_width=True):
        # precisa de existir: interface/pages/2_Chat.py
        st.switch_page("pages/2_Chat.py")

st.markdown("</div>", unsafe_allow_html=True)

# --- Conteúdo da homepage ---
st.title("Homepage")
st.write("Placeholder")
