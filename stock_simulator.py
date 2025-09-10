import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Ito Drift-Diffusion Stock Simulator", layout="wide")

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-header {text-align: center; font-size: 2.2em; font-weight: bold; margin-bottom: 0.2em;}
    .sub-header {text-align: center; font-size: 1.2em; color: gray; margin-bottom: 1em;}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 8px; padding: 0.6em 1.2em; font-weight: bold; border: none;}
    .stButton>button:hover {background-color: #105289; color: white;}
    .download-button {background-color: #1f77b4; color: white; border-radius: 8px; padding: 0.4em 1em; font-weight: bold; border: none;}
    .download-button:hover {background-color: #105289; color: white;}
    .metric-label {font-size: 1.1em; font-weight: 600;}
    .metric-value {font-size: 1.4em; font-weight: bold; color: #1f77b4;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">Ito Drift-Diffusion Stock Price Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explore stock price paths under geometric Brownian motion</div>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Simulation Inputs")
S0 = st.sidebar.number_input("Initial Stock Price", value=100.0, min_value=1.0)
mu_pct = st.sidebar.number_input("Expected Return (% per annum)", value=10.0, min_value=-100.0, max_value=100.0)
sigma_pct = st.sidebar.number_input("Volatility (% per annum)", value=20.0, min_value=0.0, max_value=500.0)
years = st.sidebar.number_input("Period (years, max 10)", value=1.0, min_value=0.1, max_value=10.0)
trading_days = st.sidebar.number_input("Trading Days per Year", value=252, min_value=1)
trading_hours_per_day = st.sidebar.number_input("Trading Hours per Day", value=6, min_value=1)
n_paths = st.sidebar.slider("Number of Paths", 1, 50, 5)
seed = st.sidebar.number_input("Random Seed (0 = none)", value=0, min_value=0)

# Convert to decimals
mu = mu_pct / 100
sigma = sigma_pct / 100

# Run simulation button
if st.button("Run Simulation"):
    # Set random seed
    if seed != 0:
        np.random.seed(int(seed))
    else:
        np.random.seed(None)

    hours_per_year = trading_days * trading_hours_per_day
    n_steps = int(years * hours_per_year)
    dt = 1 / hours_per_year

    # Simulation
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for i in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, i] = paths[:, i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Time grid in years
    time_grid = np.linspace(0, years, n_steps + 1)

    # Plot simulated paths
    fig = go.Figure()
    for i in range(n_paths):
        fig.add_scatter(x=time_grid, y=paths[i], mode="lines", line=dict(width=1), name=f"Path {i+1}")

    fig.update_layout(
        title={"text": "Simulated Stock Price Paths", "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        xaxis_title="Time (years)",
        yaxis_title="Stock Price",
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            range=[0, years],
            showgrid=True,
            gridcolor='gray',
            gridwidth=2
        ),
        yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Final prices
    final_prices = paths[:, -1]
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    path1_final = final_prices[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-label'>Mean Final Price</div><div class='metric-value'>{mean_price:.2f}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-label'>Median Final Price</div><div class='metric-value'>{median_price:.2f}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-label'>Final Price (Path 1)</div><div class='metric-value'>{path1_final:.2f}</div>", unsafe_allow_html=True)

    # Summary box
    st.markdown(
        """
        ---
        **Summary:**
        - The **mean final price** across all simulated paths is shown.
        - The **median final price** indicates the 50th percentile outcome.
        - The **final price of Path 1** is displayed as an example single trajectory.
        """
    )

    # Calculate annualized returns
    annualized_returns = (final_prices / S0) ** (1 / years) - 1
    annualized_returns_pct = annualized_returns * 100

    # Histogram of annualized returns with finer bins and clearer gridlines
    hist_return_fig = go.Figure()
    hist_return_fig.add_histogram(
        x=annualized_returns_pct,
        nbinsx=50,
        name="Annualized Returns (%)",
        marker_color='#1f77b4',
        opacity=0.8
    )
    hist_return_fig.update_layout(
        title={"text": "Distribution of Annualized Returns (% per annum)", "x": 0.5, "xanchor": "center", "font": {"size": 20}},
        xaxis_title="Annualized Return (%)",
        yaxis_title="Frequency",
        yaxis=dict(showgrid=True, gridcolor='gray', gridwidth=1)
    )
    st.plotly_chart(hist_return_fig, use_container_width=True)

    # Download CSV
    df = pd.DataFrame(paths.T, columns=[f"Path {i+1}" for i in range(n_paths)])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Data (CSV)", data=csv, file_name="simulation_paths.csv", mime="text/csv", key="download_csv", help="Download simulated paths", type="primary")

    # Simulate Again button
    if st.button("Simulate Again"):
        st.rerun()

# Footer credit
st.markdown(
    "<div style='text-align: center; font-size: 0.85em; color: gray; margin-top: 2em;'>"
    "Developed by Uday Damodaran for pedagogical purposes only"
    "</div>",
    unsafe_allow_html=True,
)
