import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import log, sqrt, exp

# ============================================================
# Page config (best practice: set once, early)
# ============================================================
st.set_page_config(
    page_title="Black–Scholes Option Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Light CSS: keep minimal (avoid heavy custom HTML unless needed)
# ============================================================
st.markdown(
    """
<style>
/* Slightly tighter top padding */
.block-container { padding-top: 1.2rem; }

/* Make tables a bit cleaner */
[data-testid="stTable"] { width: 100%; }

/* Subtle callout styles */
.small-note { font-size: 0.9rem; opacity: 0.85; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Black-Scholes (keep pure: return dict, avoid storing state)
# ============================================================
def black_scholes(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
):
    """
    Returns prices + Greeks for European options under Black–Scholes.

    Notes:
    - Assumes no dividends (q=0).
    - T in years, sigma in annualized decimals, r in annualized decimals.
    """
    # Guard rails (best practice: avoid crashing on edge cases)
    eps = 1e-12
    T = max(T, eps)
    sigma = max(sigma, eps)
    S = max(S, eps)
    K = max(K, eps)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    call = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    put = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Greeks
    pdf_d1 = norm.pdf(d1)

    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0

    gamma = pdf_d1 / (S * sigma * sqrt(T))
    vega = S * pdf_d1 * sqrt(T)  # per 1.00 in vol (not 1%)
    call_theta = (-S * pdf_d1 * sigma / (2 * sqrt(T))) - r * K * exp(-r * T) * norm.cdf(d2)
    put_theta = (-S * pdf_d1 * sigma / (2 * sqrt(T))) + r * K * exp(-r * T) * norm.cdf(-d2)

    call_rho = K * T * exp(-r * T) * norm.cdf(d2)
    put_rho = -K * T * exp(-r * T) * norm.cdf(-d2)

    # Probabilities (risk-neutral)
    call_prob_itm = norm.cdf(d2)     # P(S_T > K)
    put_prob_itm = norm.cdf(-d2)     # P(S_T < K)

    return {
        "d1": d1,
        "d2": d2,
        "call": call,
        "put": put,
        "call_delta": call_delta,
        "put_delta": put_delta,
        "gamma": gamma,
        "vega": vega,
        "call_theta": call_theta,
        "put_theta": put_theta,
        "call_rho": call_rho,
        "put_rho": put_rho,
        "call_prob_itm": call_prob_itm,
        "put_prob_itm": put_prob_itm,
    }

# ============================================================
# Heatmap plotting (matplotlib only; avoid seaborn in Streamlit)
#   - Less clutter: no per-cell annotations by default
#   - Use st.tabs instead of two giant side-by-side plots
# ============================================================
@st.cache_data(show_spinner=False)
def compute_price_grids(S_vals, sig_vals, K, T, r):
    call_grid = np.zeros((len(sig_vals), len(S_vals)))
    put_grid  = np.zeros((len(sig_vals), len(S_vals)))

    for i, sigma in enumerate(sig_vals):
        for j, S in enumerate(S_vals):
            res = black_scholes(S=S, K=K, T=T, sigma=sigma, r=r)
            call_grid[i, j] = res["call"]
            put_grid[i, j]  = res["put"]

    return call_grid, put_grid

def plot_heatmap_matplotlib(grid, x_vals, y_vals, title, xlab, ylab):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # Fewer ticks = less visual noise
    xticks = np.linspace(0, len(x_vals) - 1, min(6, len(x_vals))).astype(int)
    yticks = np.linspace(0, len(y_vals) - 1, min(6, len(y_vals))).astype(int)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x_vals[i]:.2f}" for i in xticks])
    ax.set_yticklabels([f"{y_vals[i]:.2f}" for i in yticks])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Option Price")

    fig.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def compute_prob_table(S_vals, K, T, sigma, r):
    call_prob_otm = []
    put_prob_otm = []
    call_prob_itm = []
    put_prob_itm = []

    for S in S_vals:
        res = black_scholes(S=S, K=K, T=T, sigma=sigma, r=r)
        call_prob_itm.append(res["call_prob_itm"])
        put_prob_itm.append(res["put_prob_itm"])
        call_prob_otm.append(1 - res["call_prob_itm"])
        put_prob_otm.append(1 - res["put_prob_itm"])

    df = pd.DataFrame(
        {
            "Spot": np.round(S_vals, 2),
            "Call P(ITM)": np.round(call_prob_itm, 3),
            "Call P(OTM)": np.round(call_prob_otm, 3),
            "Put  P(ITM)": np.round(put_prob_itm, 3),
            "Put  P(OTM)": np.round(put_prob_otm, 3),
        }
    ).set_index("Spot")

    return df

def plot_prob_lines(df_probs):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df_probs.index.values, df_probs["Call P(ITM)"].values, label="Call P(ITM)")
    ax.plot(df_probs.index.values, df_probs["Put  P(ITM)"].values, label="Put P(ITM)")
    ax.set_title("Risk-neutral probability of expiring ITM (via N(d2))")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig

# ============================================================
# Sidebar (best practice: group inputs; use sliders for “range”)
# ============================================================
with st.sidebar:
    st.header("📌 Inputs")
    st.caption("Set core model inputs. Use the Expander for sensitivity analysis.")

    S = st.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K = st.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")

    # Better UX for time + rates
    T = st.number_input("Time to maturity (years, T)", value=1.0, min_value=0.001, step=0.25, format="%.3f")
    sigma = st.number_input("Volatility (σ)", value=0.20, min_value=0.0001, step=0.01, format="%.4f")
    r = st.number_input("Risk-free rate (r)", value=0.05, step=0.005, format="%.4f")

    st.divider()

    with st.expander("Sensitivity (Heatmap) settings", expanded=False):
        st.caption("Adjust the grid range and resolution. Lower resolution = faster & cleaner.")
        spot_span = st.slider("Spot range (±% around S)", 5, 80, 20, step=5)
        vol_span  = st.slider("Vol range (±% around σ)", 5, 120, 50, step=5)
        grid_n    = st.slider("Grid resolution", 8, 30, 15, step=1)

    with st.expander("About", expanded=False):
        st.markdown(
            """
<div class="small-note">
European options, no dividends.  
Probabilities shown are risk-neutral (from N(d2)).
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("LinkedIn: `Ikgalaletse Keatlegile Neo, Sebola`")
        st.markdown("🔗 https://www.linkedin.com/in/neo-sebola-499b72313/")

# ============================================================
# Main layout (best practice: clear hierarchy + tabs)
# ============================================================
st.title("Black–Scholes Option Pricer")

res = black_scholes(S=S, K=K, T=T, sigma=sigma, r=r)

# --- Top KPI row (clean: use st.metric, not HTML blocks) ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Call price", f"{res['call']:.4f}")
k2.metric("Put price", f"{res['put']:.4f}")
k3.metric("d1", f"{res['d1']:.4f}")
k4.metric("d2", f"{res['d2']:.4f}")

# --- Input summary in an expander (reduce clutter) ---
with st.expander("Show inputs", expanded=False):
    st.dataframe(
        pd.DataFrame(
            {
                "Spot (S)": [S],
                "Strike (K)": [K],
                "T (years)": [T],
                "Vol (σ)": [sigma],
                "Rate (r)": [r],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# --- Tabs: Greeks / Heatmaps / Probabilities ---
tab_greeks, tab_heatmaps, tab_probs = st.tabs(["Greeks", "Heatmaps", "Probabilities"])

with tab_greeks:
    st.subheader("Option sensitivities (Greeks)")
    greeks_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Call": [
                res["call_delta"],
                res["gamma"],
                res["vega"],
                res["call_theta"],
                res["call_rho"],
            ],
            "Put": [
                res["put_delta"],
                res["gamma"],
                res["vega"],
                res["put_theta"],
                res["put_rho"],
            ],
        }
    )

    # Formatting for readability
    st.dataframe(
        greeks_df.style.format({"Call": "{:.6f}", "Put": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Note: Vega is per 1.00 change in volatility (e.g., σ: 0.20 → 0.21 is +0.01).")

with tab_heatmaps:
    st.subheader("Price sensitivity heatmaps")

    # Build ranges cleanly around current values
    spot_min = S * (1 - spot_span / 100)
    spot_max = S * (1 + spot_span / 100)
    vol_min  = max(0.0001, sigma * (1 - vol_span / 100))
    vol_max  = sigma * (1 + vol_span / 100)

    spot_vals = np.linspace(spot_min, spot_max, grid_n)
    vol_vals  = np.linspace(vol_min, vol_max, grid_n)

    call_grid, put_grid = compute_price_grids(spot_vals, vol_vals, K=K, T=T, r=r)

    t1, t2 = st.tabs(["Call heatmap", "Put heatmap"])

    with t1:
        fig = plot_heatmap_matplotlib(
            call_grid,
            x_vals=spot_vals,
            y_vals=vol_vals,
            title="Call price across Spot & Volatility",
            xlab="Spot (S)",
            ylab="Volatility (σ)",
        )
        st.pyplot(fig, use_container_width=True)

    with t2:
        fig = plot_heatmap_matplotlib(
            put_grid,
            x_vals=spot_vals,
            y_vals=vol_vals,
            title="Put price across Spot & Volatility",
            xlab="Spot (S)",
            ylab="Volatility (σ)",
        )
        st.pyplot(fig, use_container_width=True)

    st.caption("Tip: Increase grid resolution only when you need more detail—otherwise it gets noisy.")

with tab_probs:
    st.subheader("Probability view (risk-neutral)")

    st.write("These are **risk-neutral** probabilities from the model (via **N(d2)**).")
    df_probs = compute_prob_table(spot_vals, K=K, T=T, sigma=sigma, r=r)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(df_probs, use_container_width=True)
    with c2:
        figp = plot_prob_lines(df_probs)
        st.pyplot(figp, use_container_width=True)