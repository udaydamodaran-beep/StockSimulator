import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Stock Price Simulator", layout="wide")

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <h1 style="text-align: center; color: #2E86C1; margin-bottom: 0;">
        📈 Stock Price Simulator
    </h1>
    <p style="text-align: center; color: gray; font-size: 16px; margin-top: 0;">
        Based on the Ito drift-diffusion process
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# ----------------------------
# Mobile detection with JS
# ----------------------------
is_mobile = st.session_state.get("is_mobile", False)

detect_mobile = """
<script>
var isMobile = window.innerWidth < 768;
var streamlitDoc = window.parent.document;
streamlitDoc.dispatchEvent(new CustomEvent("streamlit:setComponentValue", {detail: {key: "is_mobile", value: isMobile}}));
</script>
"""
st.components.v1.html(detect_mobile, height=0)

if "is_mobile" not in st.session_state:
    st.session_state.is_mobile = False

# ----------------------------
# Inputs
# ----------------------------
st.subheader("Simulation Inputs")

if st.session_state.is_mobile:
    # Mobile → stacked inputs
    initial_price = st.number_input("Initial Stock Price", min_value=1.0, value=100.0, step=1.0)
    mu = st.number_input("Expected Return (% per annum)", value=10.0, step=0.1)
    sigma = st.number_input("Volatility (% per annum)", value=20.0, step=0.1)
    T = st.number_input("Time Horizon (years, max 10)", min_value=1, max_value=10, value=1)
    n_sims = st.slider("Number of Simulations", min_value=10, max_value=500, value=100, step=10)
else:
    # Desktop → 2-column layout
    col1, col2 = st.columns(2)
    with col1:
        initial_price = st.number_input("Initial Stock Price", min_value=1.0, value=100.0, step=1.0)
        mu = st.number_input("Expected Return (% per annum)", value=10.0, step=0.1)
    with col2:
        sigma = st.number_input("Volatility (% per annum)", value=20.0, step=0.1)
        T = st.number_input("Time Horizon (years, max 10)", min_value=1, max_value=10, value=1)
    n_sims = st.slider("Number of Simulations", min_value=10, max_value=500, value=100, step=10)

# ----------------------------
# Simulation
# ----------------------------
dt = 1 / (252 * 6.5)  # 1 trading hour
n_steps = int(T / dt)

def simulate_stock_paths(S0, mu, sigma, T, n_sims, dt):
    n_steps = int(T / dt)
    paths = np.zeros((n_steps + 1, n_sims))
    paths[0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_sims)
        paths[t] = paths[t - 1] * np.exp((mu/100 - 0.5*(sigma/100)**2)*dt + (sigma/100)*np.sqrt(dt)*Z)
    return paths

if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    paths = simulate_stock_paths(initial_price, mu, sigma, T, n_sims, dt)

    # Detect theme (dark or light)
    theme_bg = st.get_option("theme.base")  # "light" or "dark"
    dark_mode = theme_bg == "dark"

    # Plot stock price simulation
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(paths, linewidth=0.7, alpha=0.6, color="#2E86C1")
    ax.set_title(f"Simulated Stock Price Paths over {T} year(s)", fontsize=14, color="white" if dark_mode else "black")
    ax.set_xlabel("Time (Years)", color="white" if dark_mode else "black")
    ax.set_ylabel("Stock Price", color="white" if dark_mode else "black")
    ax.grid(True, alpha=0.3, color="white" if dark_mode else "gray")

    # Set x-axis ticks in yearly increments
    x_ticks = np.linspace(0, n_steps, T + 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i) for i in range(T + 1)], color="white" if dark_mode else "black")
    ax.tick_params(axis="y", colors="white" if dark_mode else "black")

    st.pyplot(fig)

    # Plot histogram of annualized returns
    annualized_returns = (paths[-1] / initial_price) ** (1/T) - 1
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(annualized_returns * 100, bins=50, color="#2E86C1", edgecolor="white", alpha=0.9)
    ax2.set_title("Distribution of Annualized Returns", fontsize=14, color="white" if dark_mode else "black")
    ax2.set_xlabel("Annualized Return (%)", color="white" if dark_mode else "black")
    ax2.set_ylabel("Frequency", color="white" if dark_mode else "black")
    ax2.grid(True, alpha=0.4, color="white" if dark_mode else "gray")
    ax2.tick_params(axis="x", colors="white" if dark_mode else "black")
    ax2.tick_params(axis="y", colors="white" if dark_mode else "black")
    st.pyplot(fig2)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <p style="text-align: center; color: #888888; font-size: 13px; margin-top: 40px;">
        Developed by Uday Damodaran for pedagogical purposes only
    </p>
    """,
    unsafe_allow_html=True,
)
