# =============================================================
# streamlit_app.py  —  Combustion Tube FEA Analyser
#
# HOW TO DEPLOY (free, no installs on your laptop):
#   1. Create a free GitHub account at github.com
#   2. Create a new repository, upload this single file
#   3. Go to share.streamlit.io → "New app" → point to your repo
#   4. Streamlit Cloud builds and hosts it automatically
#   5. Share the URL — done!
#
# TO RUN LOCALLY (if you ever get Python approved):
#   pip install streamlit numpy scipy matplotlib
#   streamlit run streamlit_app.py
# =============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Combustion Tube FEA Analyser",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Combustion Tube / Process Coil — Stress Analyser")
st.markdown(
    "Thick-wall pressure + thermal FEA calculations · "
    "Lamé equations · API 530 allowable stress · Indicative creep life"
)

# ── Material data ───────────────────────────────────────────────────────────────
MATERIALS = {
    "Carbon steel A106":   dict(E=200_000, alpha=12e-6, nu=0.3, k=48,  key="cs"),
    "9Cr-1Mo P91":         dict(E=196_000, alpha=11e-6, nu=0.3, k=28,  key="p91"),
    "HP alloy 25Cr-35Ni":  dict(E=170_000, alpha=16e-6, nu=0.3, k=18,  key="hp"),
    "347H stainless":      dict(E=185_000, alpha=17e-6, nu=0.3, k=16,  key="347h"),
}

def allowable_stress(mat_key: str, T: float) -> float:
    """Simplified step-wise allowable stress (MPa) vs temperature (°C)."""
    if mat_key == "cs":
        return (138 if T<450 else 124 if T<480 else 110 if T<510 else
                90  if T<540 else 70  if T<570 else 45)
    elif mat_key == "p91":
        return (138 if T<550 else 117 if T<580 else 94 if T<610 else
                72  if T<640 else 52  if T<670 else 35 if T<700 else 20)
    elif mat_key == "hp":
        return (80 if T<750 else 55 if T<800 else 35 if T<850 else
                18 if T<900 else 10 if T<950 else 6)
    else:  # 347h
        return (130 if T<600 else 100 if T<650 else 72 if T<700 else
                50  if T<750 else 32  if T<800 else 18 if T<850 else 10)

# ── Sidebar inputs ──────────────────────────────────────────────────────────────
st.sidebar.header("Tube parameters")

mat_name = st.sidebar.selectbox("Material", list(MATERIALS.keys()), index=1)
mat = MATERIALS[mat_name]

OD_mm    = st.sidebar.slider("Outer diameter (mm)",      60,  400, 168)
t_mm     = st.sidebar.slider("Wall thickness (mm)",       4,   40,   9)
P_barg   = st.sidebar.slider("Internal pressure (barg)",  1,  150,  25)
T_fluid  = st.sidebar.slider("Process fluid temp (°C)", 200, 1000, 600)
Q_kW     = st.sidebar.slider("Fire-side heat flux (kW/m²)", 10, 250, 80)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Results are indicative engineering estimates. "
    "Always verify against full FEA and the applicable pressure vessel code "
    "(API 530, ASME Sec VIII, EN 13445)."
)

# ── Calculations ────────────────────────────────────────────────────────────────
P   = P_barg * 0.1           # MPa
Q   = Q_kW   * 1_000         # W/m²
t   = t_mm   / 1_000         # m
ro  = (OD_mm / 2) / 1_000   # m
ri  = ro - t

E     = mat["E"]
alpha = mat["alpha"]
nu    = mat["nu"]
k     = mat["k"]

# Temperature through wall
N         = 80
x_frac    = np.linspace(0, 1, N)
dT        = Q * t / k
T_inner   = float(T_fluid)
T_outer   = T_inner + dT
T_profile = T_inner + dT * x_frac
x_mm      = x_frac * t_mm     # position in mm from inner wall

# Lamé stresses (pressure)
sigma_hoop_inner = P * (ri**2 + ro**2) / (ro**2 - ri**2)
sigma_hoop_outer = P * (2 * ri**2)     / (ro**2 - ri**2)
sigma_axial      = P * ri**2           / (ro**2 - ri**2)

# Thermal stress
sigma_thermal = E * alpha * dT / (2 * (1 - nu))

# Von Mises at inner and outer surfaces
vm_outer = np.sqrt(
    (sigma_hoop_outer - sigma_thermal)**2
    + (sigma_axial    - sigma_thermal * nu)**2
    - (sigma_hoop_outer - sigma_thermal) * (sigma_axial - sigma_thermal * nu)
)
vm_inner = np.sqrt(
    (sigma_hoop_inner + sigma_thermal)**2
    + sigma_axial**2
    - (sigma_hoop_inner + sigma_thermal) * sigma_axial
)
vm_max = max(vm_outer, vm_inner)

# Hoop stress profile through wall (Lamé)
r_profile = ri + x_frac * t
hoop_profile = P * (ri**2 / (ro**2 - ri**2)) * (1 + ro**2 / r_profile**2)
thermal_profile = E * alpha * (T_profile - T_inner - dT/2) / (1 - nu)
combined_profile = hoop_profile + thermal_profile

# Allowable stress and utilisation
allowable   = allowable_stress(mat["key"], T_outer)
utilisation = vm_max / allowable
sf          = allowable / vm_max if vm_max > 0 else 99

# Creep life (Larson-Miller, indicative)
if vm_max > 0 and T_outer > 400:
    stress_ratio = max(vm_max / allowable, 0.05)
    life_hrs = min(100_000 * (1 / stress_ratio) ** 5, 500_000)
    lmp = (T_outer + 273.15) * (20 + np.log10(life_hrs)) / 1_000
else:
    life_hrs = 500_000
    lmp = 0.0

life_str = (">200k hrs" if life_hrs >= 200_000
            else f"{life_hrs/1000:.0f}k hrs" if life_hrs >= 1_000
            else f"{life_hrs:.0f} hrs")

# Status
if utilisation < 0.8:
    status_label = "✅  Within allowable limits"
    status_color = "green"
elif utilisation < 1.0:
    status_label = "⚠️  Caution — above 80% utilisation"
    status_color = "orange"
else:
    status_label = "❌  Overstress — exceeds allowable"
    status_color = "red"

# ── Results layout ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Hoop stress (inner)",     f"{sigma_hoop_inner:.1f} MPa", help="Lamé thick-wall, inner radius (maximum)")
    st.metric("Outer wall temperature",  f"{T_outer:.1f} °C",           help="T_fluid + ΔT through wall from heat flux")

with col2:
    st.metric("Thermal stress",          f"{sigma_thermal:.1f} MPa",    help="E·α·ΔT / 2(1-ν) — through-wall gradient")
    st.metric("Wall ΔT",                 f"{dT:.1f} °C",                help="Temperature difference inner to outer surface")

with col3:
    st.metric("Von Mises (max)",         f"{vm_max:.1f} MPa",           help="Maximum of inner and outer surface")
    st.metric("Allowable stress",        f"{allowable:.0f} MPa",        help=f"At outer wall temp {T_outer:.0f}°C, {mat_name}")

st.markdown("---")

# Utilisation bar
util_pct = min(utilisation * 100, 100)
st.markdown(f"### Stress utilisation: **{utilisation*100:.1f}%** &nbsp; · &nbsp; Safety factor: **{sf:.2f}×**")
st.progress(util_pct / 100)
st.markdown(f"**{status_label}**")

st.markdown("---")

# ── Charts side-by-side ─────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Temperature through wall")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(x_mm, T_profile, color='#D85A30', linewidth=2)
    ax.fill_between(x_mm, T_profile, alpha=0.12, color='#D85A30')
    ax.axvline(0,    color='#3B8BD4', linestyle='--', linewidth=1, alpha=0.6, label='Inner (process)')
    ax.axvline(t_mm, color='#E8593C', linestyle='--', linewidth=1, alpha=0.6, label='Outer (fire side)')
    ax.set_xlabel("Position from inner wall (mm)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    st.subheader("Hoop stress through wall")
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    ax2.plot(x_mm, hoop_profile,    color='#185FA5', linewidth=2,  label='Pressure hoop (Lamé)')
    ax2.plot(x_mm, combined_profile, color='#D85A30', linewidth=2, linestyle='--', label='Pressure + thermal')
    ax2.axhline(allowable, color='#3B6D11', linewidth=1.5, linestyle=':', label=f'Allowable ({allowable:.0f} MPa)')
    ax2.set_xlabel("Position from inner wall (mm)", fontsize=11)
    ax2.set_ylabel("Stress (MPa)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.15)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")

# ── Full breakdown table ─────────────────────────────────────────────────────────
st.subheader("Full stress breakdown")

breakdown_data = {
    "Quantity": [
        "Outer diameter", "Wall thickness", "Internal radius ri", "Outer radius ro",
        "Internal pressure", "Fire-side heat flux",
        "Inner wall temp", "Outer wall temp (fire side)", "Through-wall ΔT",
        "Hoop stress — inner surface (Lamé)", "Hoop stress — outer surface (Lamé)",
        "Axial stress (closed-end)",
        "Thermal stress magnitude", "Von Mises — outer surface", "Von Mises — inner surface",
        "Von Mises — maximum",
        "Allowable stress", "Utilisation", "Safety factor",
        "Indicative creep life (Larson-Miller)", "LMP (×10³)",
    ],
    "Value": [
        f"{OD_mm} mm", f"{t_mm} mm", f"{ri*1000:.1f} mm", f"{ro*1000:.1f} mm",
        f"{P_barg} barg ({P:.2f} MPa)", f"{Q_kW} kW/m²",
        f"{T_inner:.1f} °C", f"{T_outer:.1f} °C", f"{dT:.1f} °C",
        f"{sigma_hoop_inner:.2f} MPa", f"{sigma_hoop_outer:.2f} MPa",
        f"{sigma_axial:.2f} MPa",
        f"{sigma_thermal:.2f} MPa", f"{vm_outer:.2f} MPa", f"{vm_inner:.2f} MPa",
        f"{vm_max:.2f} MPa",
        f"{allowable:.1f} MPa", f"{utilisation*100:.1f}%", f"{sf:.2f}×",
        life_str, f"{lmp:.2f}",
    ],
}

import pandas as pd
df = pd.DataFrame(breakdown_data)
st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Disclaimer: Results are indicative engineering estimates based on simplified analytical models. "
    "They do not replace rigorous FEA using tools such as ANSYS, Abaqus, or FEniCSx, "
    "nor do they replace formal calculations per API 530, ASME Section VIII, or EN 13445. "
    "Always have designs verified by a qualified engineer."
)
