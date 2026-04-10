import streamlit as st
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG & CUSTOM STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ReactorIQ",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap');

  :root {
    --bg:        #080b12;
    --surface:   #0e1220;
    --surface2:  #141926;
    --border:    #1c2235;
    --border2:   #252d44;
    --accent:    #e8a44a;
    --accent2:   #f0c070;
    --teal:      #4ecdc4;
    --red:       #ff6b6b;
    --purple:    #a78bfa;
    --text:      #d4d8e8;
    --muted:     #5c647e;
    --muted2:    #8890a8;
  }

  html, body, [class*="css"] {
    font-family: 'Crimson Pro', Georgia, serif;
    background-color: var(--bg) !important;
    color: var(--text);
  }

  /* ── Background grid texture ── */
  .main .block-container {
    background-image:
      linear-gradient(rgba(232,164,74,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(232,164,74,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    padding-top: 2rem !important;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.2rem;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stCheckbox label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
  }

  /* Sidebar inputs */
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] select,
  section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: var(--surface2) !important;
    border-color: var(--border2) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
  }

  /* ── Header ── */
  .riq-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.3rem;
  }
  .riq-wordmark {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    color: var(--accent);
    letter-spacing: -0.04em;
    line-height: 1;
  }
  .riq-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 3px 8px;
    position: relative;
    top: -4px;
  }
  .riq-tagline {
    font-family: 'Crimson Pro', Georgia, serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--muted2);
    letter-spacing: 0.01em;
    margin-bottom: 1.5rem;
  }

  /* ── Divider ── */
  .riq-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--border) 40%, transparent 100%);
    margin-bottom: 1.8rem;
    opacity: 0.4;
  }

  /* ── Section labels ── */
  .section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
  }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    color: var(--accent2);
    line-height: 1;
    letter-spacing: -0.02em;
  }
  .metric-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.15rem;
  }
  .metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 5px !important;
    gap: 3px !important;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    padding: 9px 22px !important;
    transition: all 0.15s !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(232,164,74,0.15), rgba(232,164,74,0.05)) !important;
    color: var(--accent2) !important;
    border: 1px solid rgba(232,164,74,0.25) !important;
  }

  /* ── Solve button ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #d4883a 100%) !important;
    color: #080b12 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(232,164,74,0.25) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(232,164,74,0.4) !important;
  }

  /* ── Info / result box ── */
  .info-box {
    background: linear-gradient(135deg, rgba(78,205,196,0.04), rgba(78,205,196,0.01));
    border: 1px solid rgba(78,205,196,0.18);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    font-family: 'Crimson Pro', serif;
    font-size: 1rem;
    color: var(--muted2);
    line-height: 1.7;
  }
  .info-box strong {
    color: var(--teal);
    font-weight: 600;
  }

  /* ── Welcome panel ── */
  .welcome-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    position: relative;
    overflow: hidden;
  }
  .welcome-panel::after {
    content: '⚗';
    position: absolute;
    right: 2rem;
    top: 1.5rem;
    font-size: 5rem;
    opacity: 0.04;
    line-height: 1;
  }
  .welcome-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: var(--accent);
    margin-bottom: 0.8rem;
    letter-spacing: -0.01em;
  }
  .welcome-body {
    font-family: 'Crimson Pro', serif;
    font-size: 1.05rem;
    color: var(--muted2);
    line-height: 1.75;
  }
  .welcome-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.2rem;
  }
  .chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    background: rgba(232,164,74,0.08);
    border: 1px solid rgba(232,164,74,0.2);
    border-radius: 20px;
    padding: 4px 12px;
  }

  /* ── Slider track ── */
  .stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
  }

  /* ── Checkbox ── */
  .stCheckbox > label > div:first-child {
    background: var(--surface2) !important;
    border-color: var(--border2) !important;
  }

  hr { border-color: var(--border) !important; }

  /* ── Plotly chart container ── */
  .js-plotly-plot { border-radius: 12px; overflow: hidden; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CHEMISTRY / SOLVER FUNCTIONS
# ─────────────────────────────────────────────

def rate_law(CA, CB, k, order, k_rev=0, Keq=None, reaction_type="irreversible"):
    """Compute reaction rate based on order and type."""
    if reaction_type == "irreversible":
        if order == 1:
            return k * CA
        elif order == 2:
            return k * CA ** 2
        elif order == "1+1":  # bimolecular A+B
            return k * CA * CB
    elif reaction_type == "reversible":
        if order == 1:
            r_fwd = k * CA
            r_rev = k_rev * CB if CB is not None else (k / Keq) * CB
            return r_fwd - r_rev
        elif order == 2:
            r_fwd = k * CA ** 2
            r_rev = k_rev * CB ** 2
            return r_fwd - r_rev
    return k * CA


def solve_batch(CA0, CB0, k, k_rev, order, reaction_type, t_final, T, Ea, T_ref, adiabatic, rho_cp, dH_rxn, n_points=300):
    """Solve batch reactor ODEs."""
    t_span = np.linspace(0, t_final, n_points)

    def batch_odes(t, y):
        CA, CB, Temp = y
        # Arrhenius temperature correction
        k_T = k * np.exp(-Ea / 8.314 * (1 / Temp - 1 / T_ref)) if Ea > 0 else k
        k_rev_T = k_rev * np.exp(-Ea / 8.314 * (1 / Temp - 1 / T_ref)) if (Ea > 0 and reaction_type == "reversible") else k_rev

        r = rate_law(CA, CB, k_T, order, k_rev_T, reaction_type=reaction_type)
        r = max(r, 0)

        dCA_dt = -r
        dCB_dt = r
        if adiabatic and rho_cp > 0:
            dT_dt = (-dH_rxn * r) / rho_cp
        else:
            dT_dt = 0
        return [dCA_dt, dCB_dt, dT_dt]

    sol = solve_ivp(batch_odes, [0, t_final], [CA0, CB0, T],
                    t_eval=t_span, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def solve_pfr(CA0, CB0, k, k_rev, order, reaction_type, V_total, T, Ea, T_ref, adiabatic, rho_cp, dH_rxn, n_points=300):
    """Solve PFR using ODEs along reactor volume."""
    V_span = np.linspace(0, V_total, n_points)

    def pfr_odes(V, y):
        CA, CB, Temp = y
        k_T = k * np.exp(-Ea / 8.314 * (1 / Temp - 1 / T_ref)) if Ea > 0 else k
        k_rev_T = k_rev * np.exp(-Ea / 8.314 * (1 / Temp - 1 / T_ref)) if (Ea > 0 and reaction_type == "reversible") else k_rev

        r = rate_law(CA, CB, k_T, order, k_rev_T, reaction_type=reaction_type)
        r = max(r, 0)

        dCA_dV = -r
        dCB_dV = r
        if adiabatic and rho_cp > 0:
            dT_dV = (-dH_rxn * r) / rho_cp
        else:
            dT_dV = 0
        return [dCA_dV, dCB_dV, dT_dV]

    sol = solve_ivp(pfr_odes, [0, V_total], [CA0, CB0, T],
                    t_eval=V_span, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def solve_cstr(CA0, k, k_rev, order, reaction_type, tau, T, Ea, T_ref):
    """Solve CSTR at steady state via mole balance."""
    k_T = k * np.exp(-Ea / 8.314 * (1 / T - 1 / T_ref)) if Ea > 0 else k
    k_rev_T = k_rev * np.exp(-Ea / 8.314 * (1 / T - 1 / T_ref)) if (Ea > 0 and reaction_type == "reversible") else k_rev

    def cstr_eq(CA):
        CA = max(CA[0], 1e-12)
        CB = CA0 - CA
        r = rate_law(CA, CB, k_T, order, k_rev_T, reaction_type=reaction_type)
        return [(CA0 - CA) / tau - r]

    CA_guess = CA0 * 0.5
    CA_sol = fsolve(cstr_eq, [CA_guess], full_output=True)
    CA_exit = max(CA_sol[0][0], 0)
    X = (CA0 - CA_exit) / CA0
    return CA_exit, X


def solve_pbr(CA0, CB0, k, k_rev, order, reaction_type, W_total, T, Ea, T_ref, adiabatic, rho_cp, dH_rxn, bulk_density, n_points=300):
    """Solve PBR (uses catalyst weight as independent variable)."""
    W_span = np.linspace(0, W_total, n_points)

    def pbr_odes(W, y):
        CA, CB, Temp = y
        k_T = k * np.exp(-Ea / 8.314 * (1 / Temp - 1 / T_ref)) if Ea > 0 else k
        k_rev_T = k_rev * np.exp(-Ea / 8.314 * (1 / Temp - 1 / T_ref)) if (Ea > 0 and reaction_type == "reversible") else k_rev

        r = rate_law(CA, CB, k_T, order, k_rev_T, reaction_type=reaction_type)
        r = max(r, 0)

        dCA_dW = -r / bulk_density
        dCB_dW = r / bulk_density
        if adiabatic and rho_cp > 0:
            dT_dW = (-dH_rxn * r) / (rho_cp * bulk_density)
        else:
            dT_dW = 0
        return [dCA_dW, dCB_dW, dT_dW]

    sol = solve_ivp(pbr_odes, [0, W_total], [CA0, CB0, T],
                    t_eval=W_span, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────

COLORS = {
    "CA": "#4ecdc4",
    "CB": "#ff6b6b",
    "X": "#e8a44a",
    "T": "#a78bfa",
    "grid": "#1c2235",
    "bg": "#080b12",
    "paper": "#0e1220",
}

def base_layout(x_title, y_title, title=""):
    return dict(
        title=dict(text=title, font=dict(family="Space Mono", size=13, color="#5a5f78")),
        xaxis=dict(title=x_title, gridcolor=COLORS["grid"], color="#5a5f78",
                   tickfont=dict(family="Space Mono", size=10), linecolor="#1e2230"),
        yaxis=dict(title=y_title, gridcolor=COLORS["grid"], color="#5a5f78",
                   tickfont=dict(family="Space Mono", size=10), linecolor="#1e2230"),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["paper"],
        font=dict(family="DM Sans", color="#8b90a8"),
        legend=dict(bgcolor="#13161e", bordercolor="#1e2230", borderwidth=1,
                    font=dict(family="Space Mono", size=10, color="#8b90a8")),
        margin=dict(l=50, r=20, t=40, b=50),
    )


def plot_concentration_and_conversion(x_vals, CA_vals, CB_vals, x_label, CA0):
    X_vals = (CA0 - CA_vals) / CA0

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Concentration Profiles", "Conversion"])
    fig.update_layout(
        plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["paper"],
        font=dict(family="DM Sans", color="#8b90a8"),
        legend=dict(bgcolor="#13161e", bordercolor="#1e2230", borderwidth=1,
                    font=dict(family="Space Mono", size=10, color="#8b90a8")),
        margin=dict(l=50, r=20, t=50, b=50),
        height=400,
    )

    # Concentration subplot
    fig.add_trace(go.Scatter(x=x_vals, y=CA_vals, name="C_A",
                              line=dict(color=COLORS["CA"], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=CB_vals, name="C_B",
                              line=dict(color=COLORS["CB"], width=2.5)), row=1, col=1)

    # Conversion subplot
    fig.add_trace(go.Scatter(x=x_vals, y=X_vals, name="X",
                              line=dict(color=COLORS["X"], width=2.5),
                              fill='tozeroy', fillcolor='rgba(255,209,102,0.08)'), row=1, col=2)

    # Axis styling
    for i, yl in enumerate(["Concentration (mol/L)", "Conversion X"]):
        fig.update_yaxes(gridcolor=COLORS["grid"], color="#5a5f78",
                         tickfont=dict(family="Space Mono", size=10),
                         title_text=yl, row=1, col=i+1)
        fig.update_xaxes(gridcolor=COLORS["grid"], color="#5a5f78",
                         tickfont=dict(family="Space Mono", size=10),
                         title_text=x_label, row=1, col=i+1)
    fig.update_annotations(font=dict(family="Space Mono", size=11, color="#5a5f78"))
    return fig


def plot_temperature(x_vals, T_vals, x_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=T_vals - 273.15, name="T (°C)",
                              line=dict(color=COLORS["T"], width=2.5),
                              fill='tozeroy', fillcolor='rgba(199,125,255,0.05)'))
    fig.update_layout(**base_layout(x_label, "Temperature (°C)", "Temperature Profile"),
                      height=300)
    return fig


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="riq-header">
  <span class="riq-wordmark">ReactorIQ</span>
  <span class="riq-badge">v1.0</span>
</div>
<div class="riq-tagline">Chemical Reactor Design &amp; Analysis Suite</div>
<div class="riq-divider"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR INPUTS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid #1c2235; margin-bottom: 1.5rem;">
      <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.4rem; color:#e8a44a; letter-spacing:-0.02em;">⚗ ReactorIQ</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#5c647e; letter-spacing:0.15em; text-transform:uppercase; margin-top:0.2rem;">Reactor Analysis Suite</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Reactor Configuration</div>', unsafe_allow_html=True)

    reactor_type = st.selectbox("Reactor Type", ["CSTR", "PFR", "Batch", "PBR (Packed Bed)"])
    reaction_type = st.selectbox("Reaction Type", ["Irreversible", "Reversible"])
    reaction_order = st.selectbox("Reaction Order", ["1st Order", "2nd Order (A→P)", "2nd Order (A+B→P)"])

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Feed Conditions</div>', unsafe_allow_html=True)

    CA0 = st.number_input("C_A0 (mol/L)", min_value=0.01, value=1.0, step=0.1, format="%.3f")
    CB0 = st.number_input("C_B0 (mol/L)", min_value=0.0, value=0.0, step=0.1, format="%.3f")
    T_feed = st.number_input("Feed Temp T₀ (K)", min_value=200.0, value=300.0, step=5.0)

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Kinetics</div>', unsafe_allow_html=True)

    k = st.number_input("Rate Constant k (1/s or L/mol·s)", min_value=1e-6, value=0.5, step=0.05, format="%.4f")
    k_rev = 0.0
    if reaction_type == "Reversible":
        k_rev = st.number_input("Reverse Rate Constant k_rev", min_value=0.0, value=0.1, step=0.01, format="%.4f")
    Ea = st.number_input("Activation Energy Ea (J/mol) — 0 to skip", min_value=0.0, value=0.0, step=1000.0)
    T_ref = st.number_input("Reference Temp T_ref (K)", min_value=200.0, value=300.0, step=5.0)

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Reactor Size</div>', unsafe_allow_html=True)

    if reactor_type == "CSTR":
        tau = st.number_input("Residence Time τ (s)", min_value=0.01, value=2.0, step=0.5)
        V_cstr_range = st.slider("V range for sizing (L)", 0.1, 50.0, (0.5, 20.0))
    elif reactor_type == "PFR":
        V_pfr = st.number_input("Reactor Volume V (L)", min_value=0.01, value=5.0, step=0.5)
    elif reactor_type == "Batch":
        t_final = st.number_input("Reaction Time t_final (s)", min_value=0.1, value=10.0, step=1.0)
    elif reactor_type == "PBR (Packed Bed)":
        W_total = st.number_input("Catalyst Weight W (kg)", min_value=0.01, value=10.0, step=1.0)
        bulk_density = st.number_input("Bulk Density ρ_b (kg/L)", min_value=0.1, value=1.5, step=0.1)

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Energy Balance</div>', unsafe_allow_html=True)

    adiabatic = st.checkbox("Adiabatic Operation", value=False)
    dH_rxn, rho_cp = 0.0, 0.0
    if adiabatic:
        dH_rxn = st.number_input("ΔH_rxn (J/mol) — negative=exothermic", value=-50000.0, step=1000.0)
        rho_cp = st.number_input("ρCp (J/L·K)", min_value=1.0, value=4184.0, step=100.0)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("▶  SOLVE REACTOR")


# ─────────────────────────────────────────────
#  MAP ORDER STRING
# ─────────────────────────────────────────────

order_map = {"1st Order": 1, "2nd Order (A→P)": 2, "2nd Order (A+B→P)": "1+1"}
order = order_map[reaction_order]
rxn_type = reaction_type.lower()


# ─────────────────────────────────────────────
#  MAIN RESULTS
# ─────────────────────────────────────────────

if not run:
    st.markdown("""
    <div class="welcome-panel">
      <div class="welcome-title">Configure your reactor in the sidebar</div>
      <div class="welcome-body">
        Select a reactor type, set your feed conditions and kinetics, then click
        <strong style="color:#e8a44a;">Solve Reactor</strong> to generate concentration profiles,
        conversion curves, temperature plots, and sizing analysis.<br><br>
        Supports Arrhenius temperature dependence and adiabatic energy balance.
      </div>
      <div class="welcome-chips">
        <span class="chip">CSTR</span>
        <span class="chip">PFR</span>
        <span class="chip">Batch</span>
        <span class="chip">PBR</span>
        <span class="chip">1st / 2nd Order</span>
        <span class="chip">Reversible</span>
        <span class="chip">Adiabatic</span>
        <span class="chip">Arrhenius</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    tabs = st.tabs(["📊  Concentration & Conversion", "🌡️  Temperature Profile", "📐  Reactor Sizing", "📋  Summary"])

    # ── CSTR ──────────────────────────────────
    if reactor_type == "CSTR":
        CA_exit, X_exit = solve_cstr(CA0, k, k_rev, order, rxn_type, tau, T_feed, Ea, T_ref)
        CB_exit = CA0 - CA_exit

        # Sizing curve: X vs V (using tau = V/v0, sweep V)
        V_range = np.linspace(V_cstr_range[0], V_cstr_range[1], 200)

        with tabs[0]:
            st.markdown('<div class="section-title">Steady-State CSTR Results</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                ("C_A exit", f"{CA_exit:.4f}", "mol/L"),
                ("C_B exit", f"{CB_exit:.4f}", "mol/L"),
                ("Conversion X", f"{X_exit*100:.2f}", "%"),
                ("τ (residence)", f"{tau:.2f}", "s"),
            ]
            for col, (label, val, unit) in zip([c1, c2, c3, c4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-unit">{unit}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div class="info-box">CSTR operates at steady-state exit conditions throughout the vessel.
            The single point shown represents the uniform composition inside the reactor.</div>""",
            unsafe_allow_html=True)

        with tabs[2]:
            st.markdown('<div class="section-title">CSTR Sizing — Conversion vs. Volume</div>', unsafe_allow_html=True)

            X_sizing = []
            for V in V_range:
                tau_v = V  # normalized: assume v0=1 L/s
                _, X_v = solve_cstr(CA0, k, k_rev, order, rxn_type, tau_v, T_feed, Ea, T_ref)
                X_sizing.append(X_v)

            fig_size = go.Figure()
            fig_size.add_trace(go.Scatter(x=V_range, y=np.array(X_sizing) * 100,
                                           line=dict(color=COLORS["X"], width=2.5),
                                           fill='tozeroy', fillcolor='rgba(255,209,102,0.06)',
                                           name="Conversion %"))
            fig_size.add_vline(x=tau, line_dash="dash", line_color="#00e5cc",
                                annotation_text=f"τ = {tau}s", annotation_font_color="#00e5cc")
            fig_size.update_layout(**base_layout("Residence Time / Volume (L or s)", "Conversion (%)",
                                                  "CSTR Sizing Curve"), height=380)
            st.plotly_chart(fig_size, use_container_width=True)

        with tabs[3]:
            st.markdown(f"""
            <div class="info-box">
            <strong>Reactor:</strong> CSTR &nbsp;|&nbsp; <strong>Order:</strong> {reaction_order} &nbsp;|&nbsp;
            <strong>Type:</strong> {reaction_type}<br><br>
            At residence time τ = {tau} s, the exit conversion is <strong>X = {X_exit*100:.2f}%</strong>.<br>
            Exit concentration C_A = {CA_exit:.4f} mol/L, C_B = {CB_exit:.4f} mol/L.
            </div>""", unsafe_allow_html=True)

    # ── PFR ──────────────────────────────────
    elif reactor_type == "PFR":
        V_arr, CA_arr, CB_arr, T_arr = solve_pfr(
            CA0, CB0, k, k_rev, order, rxn_type, V_pfr, T_feed, Ea, T_ref, adiabatic, rho_cp, dH_rxn
        )
        X_arr = (CA0 - CA_arr) / CA0
        X_final = X_arr[-1]
        CA_exit = CA_arr[-1]

        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                ("C_A exit", f"{CA_exit:.4f}", "mol/L"),
                ("C_B exit", f"{CB_arr[-1]:.4f}", "mol/L"),
                ("Conversion X", f"{X_final*100:.2f}", "%"),
                ("Volume", f"{V_pfr:.2f}", "L"),
            ]
            for col, (label, val, unit) in zip([c1, c2, c3, c4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-unit">{unit}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_conc = plot_concentration_and_conversion(V_arr, CA_arr, CB_arr, "Volume (L)", CA0)
            st.plotly_chart(fig_conc, use_container_width=True)

        with tabs[1]:
            if adiabatic:
                fig_T = plot_temperature(V_arr, T_arr, "Volume (L)")
                st.plotly_chart(fig_T, use_container_width=True)
                T_rise = T_arr[-1] - T_arr[0]
                st.markdown(f'<div class="info-box">Adiabatic temperature change: <strong>ΔT = {T_rise:.2f} K</strong></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">Enable Adiabatic Operation in the sidebar to see temperature profiles.</div>',
                            unsafe_allow_html=True)

        with tabs[2]:
            st.markdown('<div class="section-title">PFR Sizing — Required Volume for Target Conversion</div>',
                        unsafe_allow_html=True)
            X_targets = np.linspace(0.05, 0.99, 200)
            V_required = []
            for Xt in X_targets:
                # Levenspiel: integrate -rA from X=0 to Xt
                X_int = np.linspace(0, Xt, 500)
                CA_int = CA0 * (1 - X_int)
                CB_int = CA0 * X_int
                r_int = np.array([max(rate_law(ca, cb, k, order, k_rev, reaction_type=rxn_type), 1e-12)
                                   for ca, cb in zip(CA_int, CB_int)])
                V_lev = np.trapz(CA0 / r_int, X_int)
                V_required.append(V_lev)

            fig_lev = go.Figure()
            fig_lev.add_trace(go.Scatter(x=np.array(X_targets) * 100, y=V_required,
                                          line=dict(color=COLORS["CA"], width=2.5),
                                          fill='tozeroy', fillcolor='rgba(0,229,204,0.05)',
                                          name="Required Volume"))
            fig_lev.add_vline(x=X_final * 100, line_dash="dash", line_color=COLORS["Y"],
                               line_color_=COLORS["X"],
                               annotation_text=f"X={X_final*100:.1f}%", annotation_font_color=COLORS["X"])
            fig_lev.update_layout(**base_layout("Target Conversion (%)", "Required Volume (L)",
                                                 "Levenspiel Plot — PFR Sizing"), height=380)
            fig_lev.update_traces(line_color=COLORS["CA"])
            st.plotly_chart(fig_lev, use_container_width=True)

        with tabs[3]:
            st.markdown(f"""
            <div class="info-box">
            <strong>Reactor:</strong> PFR &nbsp;|&nbsp; <strong>Volume:</strong> {V_pfr} L<br><br>
            Final conversion: <strong>X = {X_final*100:.2f}%</strong><br>
            Exit: C_A = {CA_exit:.4f} mol/L, C_B = {CB_arr[-1]:.4f} mol/L
            {"<br>Exit temperature: " + f"{T_arr[-1]-273.15:.2f} °C" if adiabatic else ""}
            </div>""", unsafe_allow_html=True)

    # ── BATCH ──────────────────────────────────
    elif reactor_type == "Batch":
        t_arr, CA_arr, CB_arr, T_arr = solve_batch(
            CA0, CB0, k, k_rev, order, rxn_type, t_final, T_feed, Ea, T_ref, adiabatic, rho_cp, dH_rxn
        )
        X_arr = (CA0 - CA_arr) / CA0
        X_final = X_arr[-1]

        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                ("C_A final", f"{CA_arr[-1]:.4f}", "mol/L"),
                ("C_B final", f"{CB_arr[-1]:.4f}", "mol/L"),
                ("Conversion X", f"{X_final*100:.2f}", "%"),
                ("Time", f"{t_final:.2f}", "s"),
            ]
            for col, (label, val, unit) in zip([c1, c2, c3, c4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-unit">{unit}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_conc = plot_concentration_and_conversion(t_arr, CA_arr, CB_arr, "Time (s)", CA0)
            st.plotly_chart(fig_conc, use_container_width=True)

        with tabs[1]:
            if adiabatic:
                fig_T = plot_temperature(t_arr, T_arr, "Time (s)")
                st.plotly_chart(fig_T, use_container_width=True)
            else:
                st.markdown('<div class="info-box">Enable Adiabatic Operation in the sidebar to see temperature profiles.</div>',
                            unsafe_allow_html=True)

        with tabs[2]:
            st.markdown('<div class="section-title">Batch Sizing — Time Required for Target Conversion</div>',
                        unsafe_allow_html=True)
            t_targets = np.linspace(0.1, t_final * 2, 100)
            X_t_curve = []
            for t_tgt in t_targets:
                _, CA_t, _, _ = solve_batch(CA0, CB0, k, k_rev, order, rxn_type, t_tgt,
                                            T_feed, Ea, T_ref, adiabatic, rho_cp, dH_rxn, n_points=100)
                X_t_curve.append((CA0 - CA_t[-1]) / CA0 * 100)

            fig_batch_size = go.Figure()
            fig_batch_size.add_trace(go.Scatter(x=t_targets, y=X_t_curve,
                                                 line=dict(color=COLORS["CB"], width=2.5),
                                                 fill='tozeroy', fillcolor='rgba(255,107,107,0.05)',
                                                 name="Conversion %"))
            fig_batch_size.update_layout(**base_layout("Time (s)", "Conversion (%)",
                                                        "Conversion vs. Reaction Time"), height=380)
            st.plotly_chart(fig_batch_size, use_container_width=True)

        with tabs[3]:
            st.markdown(f"""
            <div class="info-box">
            <strong>Reactor:</strong> Batch &nbsp;|&nbsp; <strong>Time:</strong> {t_final} s<br><br>
            Final conversion: <strong>X = {X_final*100:.2f}%</strong><br>
            Final: C_A = {CA_arr[-1]:.4f} mol/L, C_B = {CB_arr[-1]:.4f} mol/L
            </div>""", unsafe_allow_html=True)

    # ── PBR ──────────────────────────────────
    elif reactor_type == "PBR (Packed Bed)":
        W_arr, CA_arr, CB_arr, T_arr = solve_pbr(
            CA0, CB0, k, k_rev, order, rxn_type, W_total, T_feed, Ea, T_ref,
            adiabatic, rho_cp, dH_rxn, bulk_density
        )
        X_arr = (CA0 - CA_arr) / CA0
        X_final = X_arr[-1]

        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                ("C_A exit", f"{CA_arr[-1]:.4f}", "mol/L"),
                ("C_B exit", f"{CB_arr[-1]:.4f}", "mol/L"),
                ("Conversion X", f"{X_final*100:.2f}", "%"),
                ("Catalyst W", f"{W_total:.2f}", "kg"),
            ]
            for col, (label, val, unit) in zip([c1, c2, c3, c4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-unit">{unit}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_conc = plot_concentration_and_conversion(W_arr, CA_arr, CB_arr, "Catalyst Weight (kg)", CA0)
            st.plotly_chart(fig_conc, use_container_width=True)

        with tabs[1]:
            if adiabatic:
                fig_T = plot_temperature(W_arr, T_arr, "Catalyst Weight (kg)")
                st.plotly_chart(fig_T, use_container_width=True)
            else:
                st.markdown('<div class="info-box">Enable Adiabatic Operation in the sidebar to see temperature profiles.</div>',
                            unsafe_allow_html=True)

        with tabs[2]:
            st.markdown('<div class="section-title">PBR Sizing — Conversion vs. Catalyst Weight</div>',
                        unsafe_allow_html=True)
            W_range_plot = np.linspace(0.5, W_total * 2, 150)
            X_W_curve = []
            for W_tgt in W_range_plot:
                _, CA_w, _, _ = solve_pbr(CA0, CB0, k, k_rev, order, rxn_type, W_tgt,
                                           T_feed, Ea, T_ref, False, 0, 0, bulk_density, n_points=100)
                X_W_curve.append((CA0 - CA_w[-1]) / CA0 * 100)

            fig_pbr_size = go.Figure()
            fig_pbr_size.add_trace(go.Scatter(x=W_range_plot, y=X_W_curve,
                                               line=dict(color=COLORS["T"], width=2.5),
                                               fill='tozeroy', fillcolor='rgba(199,125,255,0.05)',
                                               name="Conversion %"))
            fig_pbr_size.add_vline(x=W_total, line_dash="dash", line_color=COLORS["X"],
                                    annotation_text=f"W={W_total} kg", annotation_font_color=COLORS["X"])
            fig_pbr_size.update_layout(**base_layout("Catalyst Weight (kg)", "Conversion (%)",
                                                      "PBR Sizing — Conversion vs. Catalyst Weight"), height=380)
            st.plotly_chart(fig_pbr_size, use_container_width=True)

        with tabs[3]:
            st.markdown(f"""
            <div class="info-box">
            <strong>Reactor:</strong> PBR &nbsp;|&nbsp; <strong>Catalyst:</strong> {W_total} kg &nbsp;|&nbsp;
            <strong>Bulk density:</strong> {bulk_density} kg/L<br><br>
            Final conversion: <strong>X = {X_final*100:.2f}%</strong><br>
            Exit: C_A = {CA_arr[-1]:.4f} mol/L, C_B = {CB_arr[-1]:.4f} mol/L
            </div>""", unsafe_allow_html=True)
