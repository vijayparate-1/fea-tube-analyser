# =============================================================
# wbh_heat_transfer.py  —  Direct-Fired Water Bath Heater
#                          Heat Transfer Analyser
#
# Covers:
#   - Firing tube: flue gas convection, tube wall conduction,
#     water bath natural convection → U_firing, skin temps
#   - Process coil: water bath convection, coil wall conduction,
#     natural gas convection → U_coil, LMTD, duty
#   - Full thermal resistance circuit for both sides
#
# Deploy: push to GitHub → Streamlit Cloud auto-deploys
# Run locally: python3 -m streamlit run wbh_heat_transfer.py
# =============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Bath Heater — Heat Transfer",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Direct-Fired Water Bath Heater — Heat Transfer Analyser")
st.markdown(
    "U-tube firing tube · Natural gas process coil · "
    "Overall U · Tube skin temperatures · Heat dissipation rates"
)

# ══════════════════════════════════════════════════════════════
# FLUID PROPERTY FUNCTIONS  (temperature in °C)
# ══════════════════════════════════════════════════════════════

def flue_gas_props(T_C: float) -> dict:
    """
    Approximate properties of combustion flue gas (products of
    natural gas combustion with ~15% excess air).
    Valid range: 200–1200 °C.
    """
    T = T_C
    Cp  = 1050 + 0.12 * T                        # J/kg·K
    k   = 0.0245 + 7.2e-5 * T                    # W/m·K
    mu  = (1.46e-5 + 4.0e-8 * T)                 # Pa·s
    rho = 1.25 * 273.15 / (273.15 + T)           # kg/m³ (ideal gas approx)
    Pr  = mu * Cp / k
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, Pr=Pr)


def natural_gas_props(T_C: float, P_barg: float = 5.0) -> dict:
    """
    Approximate properties of natural gas (methane-dominant, ~95% CH4).
    Valid range: 0–200 °C, 1–100 barg.
    """
    T   = T_C
    P   = (P_barg + 1.01325) * 1e5               # bar → Pa absolute
    M   = 16.04                                   # kg/kmol (methane)
    R   = 8314.0                                  # J/kmol·K
    Cp  = 2200 + 1.1 * T + 0.002 * T**2          # J/kg·K
    k   = 0.0302 + 8.5e-5 * T                    # W/m·K
    mu  = 1.05e-5 + 3.0e-8 * T                   # Pa·s
    rho = P * M / (R * (273.15 + T))             # kg/m³
    Pr  = mu * Cp / k
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, Pr=Pr)


def water_props(T_C: float) -> dict:
    """
    Liquid water properties.  Valid range: 20–95 °C.
    """
    T   = T_C
    Cp  = 4183 - 0.5  * (T - 35)**2 / 100        # J/kg·K  (≈ 4183 near 35°C)
    k   = 0.571 + 0.00175 * T - 6.0e-6 * T**2   # W/m·K
    mu  = 1e-3 * np.exp(-0.02 * (T - 20))        # Pa·s (approx exponential decay)
    rho = 1000 - 0.003 * T**2                    # kg/m³
    beta= 2.1e-4 + 5e-6 * T                      # thermal expansion 1/K
    Pr  = mu * Cp / k
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, beta=beta, Pr=Pr)


# ══════════════════════════════════════════════════════════════
# HEAT TRANSFER COEFFICIENT CORRELATIONS
# ══════════════════════════════════════════════════════════════

def h_internal_gnielinski(Re: float, Pr: float, k: float,
                           D: float, L: float) -> float:
    """
    Gnielinski correlation for internal turbulent pipe flow.
    Valid: 3000 < Re < 5e6, 0.5 < Pr < 2000.
    Returns h (W/m²·K).
    """
    Re = max(Re, 3001)
    f  = (0.790 * np.log(Re) - 1.64) ** -2       # Petukhov friction factor
    Nu = (f / 8) * (Re - 1000) * Pr / \
         (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2/3) - 1))
    Nu *= (1 + (D / L) ** (2/3))                  # entry length correction
    return Nu * k / D


def h_internal_dittus_boelter(Re: float, Pr: float, k: float,
                               D: float, heating: bool = True) -> float:
    """
    Dittus-Boelter for turbulent pipe flow.
    Simple fallback if Re < 3000.
    """
    n  = 0.4 if heating else 0.3
    Nu = 0.023 * Re**0.8 * Pr**n
    return Nu * k / D


def h_natural_conv_horiz_cylinder(D_o: float, T_surf: float,
                                  T_fluid: float, fluid: dict) -> float:
    """
    Churchill-Chu correlation for natural convection on a
    horizontal cylinder (outer surface of tube in water bath).
    Returns h (W/m²·K).
    """
    dT   = abs(T_surf - T_fluid)
    if dT < 0.1:
        return 500.0                              # avoid divide-by-zero
    g    = 9.81
    beta = fluid["beta"]
    nu   = fluid["mu"] / fluid["rho"]            # kinematic viscosity
    alpha= fluid["k"] / (fluid["rho"] * fluid["Cp"])   # thermal diffusivity
    Ra   = g * beta * dT * D_o**3 / (nu * alpha)
    Pr   = fluid["Pr"]
    term = Ra / (1 + (0.559 / Pr) ** (9/16)) ** (16/9)
    Nu   = (0.60 + 0.387 * term ** (1/6)) ** 2
    return Nu * fluid["k"] / D_o


# ══════════════════════════════════════════════════════════════
# OVERALL U — CYLINDRICAL WALL
# ══════════════════════════════════════════════════════════════

def overall_U_cylinder(h_i: float, h_o: float, k_wall: float,
                        D_i: float, D_o: float,
                        foul_i: float = 0.0,
                        foul_o: float = 0.0) -> dict:
    """
    Overall heat transfer coefficient referenced to OUTER area.

    1/U_o = (A_o/A_i)·(1/h_i + Rf_i)
            + A_o·ln(D_o/D_i)/(2π·k·L) → simplified to ro·ln(ro/ri)/k
            + 1/h_o + Rf_o

    Returns U_o (W/m²·K) and individual resistances (m²·K/W).
    """
    ro = D_o / 2
    ri = D_i / 2
    ratio = D_o / D_i                             # A_o / A_i

    R_i    = ratio * (1/h_i + foul_i)            # inner film + fouling
    R_wall = ro * np.log(ro / ri) / k_wall       # wall conduction
    R_o    = 1/h_o + foul_o                      # outer film + fouling

    R_total = R_i + R_wall + R_o
    U_o     = 1.0 / R_total

    return dict(
        U_o=U_o,
        R_inner=R_i,
        R_wall=R_wall,
        R_outer=R_o,
        R_total=R_total,
        pct_inner = R_i    / R_total * 100,
        pct_wall  = R_wall / R_total * 100,
        pct_outer = R_o    / R_total * 100,
    )


# ══════════════════════════════════════════════════════════════
# LMTD  (counter-current or co-current)
# ══════════════════════════════════════════════════════════════

def lmtd(T_hot_in, T_hot_out, T_cold_in, T_cold_out,
         flow="counter") -> float:
    if flow == "counter":
        dT1 = T_hot_in  - T_cold_out
        dT2 = T_hot_out - T_cold_in
    else:
        dT1 = T_hot_in  - T_cold_in
        dT2 = T_hot_out - T_cold_out
    dT1 = max(dT1, 0.01)
    dT2 = max(dT2, 0.01)
    if abs(dT1 - dT2) < 0.01:
        return dT1
    return (dT1 - dT2) / np.log(dT1 / dT2)


# ══════════════════════════════════════════════════════════════
# TUBE SKIN TEMPERATURES
# ══════════════════════════════════════════════════════════════

def skin_temperatures(Q_W: float, h_i: float, h_o: float,
                      k_wall: float, D_i: float, D_o: float,
                      A_o: float, T_hot: float, T_cold: float) -> dict:
    """
    Compute inner and outer tube skin temperatures from known Q, h values.

    T_outer_skin = T_cold  + Q / (h_o × A_o)
    T_inner_skin = T_outer + Q × ln(ro/ri) / (2π×k×L)
                           → simplified: T_outer + Q×(ro·ln(ro/ri)/k) / A_o
    """
    ro = D_o / 2
    ri = D_i / 2
    q  = Q_W / A_o                               # heat flux W/m²

    T_outer_skin = T_cold + q / h_o
    T_inner_skin = T_outer_skin + q * ro * np.log(ro/ri) / k_wall
    T_fluid_side = T_inner_skin + q * (D_o/D_i) / h_i

    return dict(
        heat_flux    = q,
        T_outer_skin = T_outer_skin,
        T_inner_skin = T_inner_skin,
        T_fluid_side = T_fluid_side,
    )


# ══════════════════════════════════════════════════════════════
# SIDEBAR INPUTS
# ══════════════════════════════════════════════════════════════

sb = st.sidebar
sb.header("Heater geometry & conditions")

sb.subheader("Firing tube (U-tube)")
ft_OD   = sb.slider("Firing tube OD (mm)",         80,  400, 168)
ft_t    = sb.slider("Firing tube wall thickness (mm)", 4, 25,  8)
ft_L    = sb.slider("Firing tube total length (m)",  2,  20,  8,
                    help="Total unrolled length of both legs + U-bend")
ft_mat  = sb.selectbox("Firing tube material",
                        ["Carbon steel (k=50)", "Alloy steel P11 (k=38)",
                         "Stainless 304 (k=16)"])

sb.subheader("Flue gas conditions")
fg_T_in  = sb.slider("Flue gas inlet temp (°C)",    600, 1400, 900)
fg_T_out = sb.slider("Flue gas outlet temp (°C)",   200,  700, 350)
fg_mdot  = sb.slider("Flue gas mass flow (kg/s)",   0.05, 5.0, 0.5, step=0.05)

sb.subheader("Water bath")
T_bath   = sb.slider("Water bath temperature (°C)",  60,  95,  82)

sb.subheader("Process coil")
pc_OD    = sb.slider("Process coil OD (mm)",         25, 150,  60)
pc_t     = sb.slider("Process coil wall (mm)",         3,  15,   5)
pc_L     = sb.slider("Process coil length (m)",        5,  60,  25)
pc_mat   = sb.selectbox("Process coil material",
                         ["Carbon steel (k=50)", "Alloy steel P11 (k=38)",
                          "Stainless 304 (k=16)"])

sb.subheader("Natural gas conditions")
ng_T_in  = sb.slider("Gas inlet temp (°C)",    5,   60,  15)
ng_T_out = sb.slider("Gas outlet temp (°C)",  30,  100,  65)
ng_P     = sb.slider("Gas pressure (barg)",    1,  100,  40)
ng_mdot  = sb.slider("Gas mass flow (kg/s)",  0.05, 10.0, 1.0, step=0.05)

sb.subheader("Fouling factors (m²·K/W)")
foul_fg  = sb.number_input("Flue gas side (inside firing tube)", 0.0, 0.01,
                            0.0002, format="%.4f")
foul_wb  = sb.number_input("Water bath side",                    0.0, 0.01,
                            0.0001, format="%.4f")
foul_ng  = sb.number_input("Gas side (inside process coil)",     0.0, 0.01,
                            0.0002, format="%.4f")

sb.markdown("---")
sb.caption("Results are indicative. Verify against HTRI/HTFS or detailed FEA.")

# ══════════════════════════════════════════════════════════════
# DERIVED GEOMETRY
# ══════════════════════════════════════════════════════════════

k_map = {"Carbon steel (k=50)": 50, "Alloy steel P11 (k=38)": 38,
         "Stainless 304 (k=16)": 16}
k_ft  = k_map[ft_mat]
k_pc  = k_map[pc_mat]

ft_Di = (ft_OD - 2*ft_t) / 1000   # m
ft_Do = ft_OD / 1000
ft_Ai = np.pi * ft_Di * ft_L
ft_Ao = np.pi * ft_Do * ft_L

pc_Di = (pc_OD - 2*pc_t) / 1000
pc_Do = pc_OD / 1000
pc_Ai = np.pi * pc_Di * pc_L
pc_Ao = np.pi * pc_Do * pc_L

# ══════════════════════════════════════════════════════════════
# CALCULATIONS
# ══════════════════════════════════════════════════════════════

# ── Mean temperatures
T_fg_mean = (fg_T_in + fg_T_out) / 2
T_ng_mean = (ng_T_in + ng_T_out) / 2

# ── Fluid properties at mean conditions
fp_fg = flue_gas_props(T_fg_mean)
fp_ng = natural_gas_props(T_ng_mean, ng_P)
fp_wb = water_props(T_bath)

# ── Reynolds numbers
vel_fg  = fg_mdot / (fp_fg["rho"] * np.pi/4 * ft_Di**2)
Re_fg   = fp_fg["rho"] * vel_fg * ft_Di / fp_fg["mu"]

vel_ng  = ng_mdot / (fp_ng["rho"] * np.pi/4 * pc_Di**2)
Re_ng   = fp_ng["rho"] * vel_ng * pc_Di / fp_ng["mu"]

# ── h_fi: flue gas inside firing tube
if Re_fg > 3000:
    h_fi = h_internal_gnielinski(Re_fg, fp_fg["Pr"], fp_fg["k"], ft_Di, ft_L)
else:
    h_fi = h_internal_dittus_boelter(Re_fg, fp_fg["Pr"], fp_fg["k"], ft_Di)

# ── h_fo: water bath on outside of firing tube
# First estimate outer skin temp for the natural convection iteration
T_fo_est = T_bath + 15   # initial guess for skin temp
for _ in range(6):       # iterate to converge
    h_fo = h_natural_conv_horiz_cylinder(ft_Do, T_fo_est, T_bath, fp_wb)
    U_ft_est = overall_U_cylinder(h_fi, h_fo, k_ft, ft_Di, ft_Do,
                                   foul_fg, foul_wb)["U_o"]
    # Approx total Q from energy balance on flue gas side
    Q_fg_total = fg_mdot * fp_fg["Cp"] * (fg_T_in - fg_T_out)
    q_est = Q_fg_total / ft_Ao
    T_fo_est = T_bath + q_est / h_fo

# ── Overall U — firing tube
U_ft = overall_U_cylinder(h_fi, h_fo, k_ft, ft_Di, ft_Do, foul_fg, foul_wb)

# ── Heat duty — firing tube (energy balance on flue gas)
Q_ft = fg_mdot * fp_fg["Cp"] * (fg_T_in - fg_T_out)  # W

# ── Firing tube skin temperatures
skin_ft = skin_temperatures(Q_ft, h_fi, h_fo, k_ft,
                             ft_Di, ft_Do, ft_Ao,
                             T_fg_mean, T_bath)

# ── h_ng: natural gas inside process coil
if Re_ng > 3000:
    h_gi = h_internal_gnielinski(Re_ng, fp_ng["Pr"], fp_ng["k"],
                                  pc_Di, pc_L)
else:
    h_gi = h_internal_dittus_boelter(Re_ng, fp_ng["Pr"], fp_ng["k"],
                                      pc_Di, heating=True)

# ── h_wo: water bath on outside of process coil (iterate)
T_pc_est = T_bath - 10
for _ in range(6):
    h_wo = h_natural_conv_horiz_cylinder(pc_Do, T_bath, T_pc_est, fp_wb)
    U_pc_est = overall_U_cylinder(h_gi, h_wo, k_pc, pc_Di, pc_Do,
                                   foul_ng, foul_wb)["U_o"]
    Q_ng = ng_mdot * fp_ng["Cp"] * (ng_T_out - ng_T_in)
    q_est2 = Q_ng / pc_Ao
    T_pc_est = T_bath - q_est2 / h_wo

# ── Overall U — process coil
U_pc = overall_U_cylinder(h_gi, h_wo, k_pc, pc_Di, pc_Do, foul_ng, foul_wb)

# ── Heat duty — process coil (energy balance on gas side)
Q_pc = ng_mdot * fp_ng["Cp"] * (ng_T_out - ng_T_in)   # W

# ── LMTD — process coil (water bath hot side, gas cold side, counter-current)
lmtd_pc = lmtd(T_bath, T_bath, ng_T_in, ng_T_out, flow="counter")
# Water bath is isothermal so LMTD simplifies:
# dT1 = T_bath - T_ng_out, dT2 = T_bath - T_ng_in
lmtd_pc = lmtd(T_bath, T_bath, ng_T_in, ng_T_out, flow="co")
# Isothermal hot side: use correct formula
dT1 = T_bath - ng_T_in
dT2 = T_bath - ng_T_out
if dT2 <= 0:
    dT2 = 0.5
if abs(dT1 - dT2) < 0.01:
    lmtd_pc_val = dT1
else:
    lmtd_pc_val = (dT1 - dT2) / np.log(dT1 / dT2)

# Required area for process coil duty
A_required_pc = Q_pc / (U_pc["U_o"] * lmtd_pc_val) if lmtd_pc_val > 0 else 0

# ── Process coil skin temperatures
skin_pc = skin_temperatures(Q_pc, h_gi, h_wo, k_pc,
                             pc_Di, pc_Do, pc_Ao,
                             T_bath, T_ng_mean)

# ══════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════

st.markdown("---")

# ── Top-level heat duties
col1, col2, col3, col4 = st.columns(4)
col1.metric("Firing tube duty",
            f"{Q_ft/1000:.1f} kW",
            help="From flue gas energy balance")
col2.metric("Process coil duty",
            f"{Q_pc/1000:.1f} kW",
            help="From natural gas energy balance")
col3.metric("Thermal efficiency",
            f"{Q_pc/Q_ft*100:.1f}%" if Q_ft > 0 else "—",
            help="Gas duty / Firing duty")
col4.metric("Excess heat to bath",
            f"{(Q_ft - Q_pc)/1000:.1f} kW",
            help="Heat remaining in water bath")

st.markdown("---")

# ── Two columns: Firing tube  |  Process coil
left, right = st.columns(2)

with left:
    st.subheader("Firing tube analysis")

    m1, m2 = st.columns(2)
    m1.metric("h_fi — flue gas (inside)",
              f"{h_fi:.0f} W/m²·K",
              help="Gnielinski correlation")
    m2.metric("h_fo — water bath (outside)",
              f"{h_fo:.0f} W/m²·K",
              help="Churchill-Chu natural convection")

    m3, m4 = st.columns(2)
    m3.metric("Overall U (firing tube)",
              f"{U_ft['U_o']:.1f} W/m²·K")
    m4.metric("Heat flux (outer surface)",
              f"{skin_ft['heat_flux']/1000:.1f} kW/m²")

    st.markdown("**Tube skin temperatures**")
    s1, s2, s3 = st.columns(3)
    s1.metric("Outer skin (water side)",
              f"{skin_ft['T_outer_skin']:.1f} °C",
              delta=f"+{skin_ft['T_outer_skin']-T_bath:.1f}°C vs bath")
    s2.metric("Inner skin (bore surface)",
              f"{skin_ft['T_inner_skin']:.1f} °C")
    s3.metric("Film temp (flue gas side)",
              f"{skin_ft['T_fluid_side']:.1f} °C")

    st.markdown("**Reynolds & flow regime**")
    st.info(
        f"Re_flue = {Re_fg:,.0f} — "
        f"{'Turbulent ✅' if Re_fg > 4000 else 'Transitional ⚠️' if Re_fg > 2300 else 'Laminar — check design ⚠️'}  |  "
        f"Pr = {fp_fg['Pr']:.2f}  |  "
        f"Velocity = {vel_fg:.1f} m/s"
    )

    # Resistance breakdown bar chart
    st.markdown("**Thermal resistance breakdown (% of total)**")
    fig1, ax1 = plt.subplots(figsize=(5, 1.4))
    bars = [U_ft["pct_inner"], U_ft["pct_wall"], U_ft["pct_outer"]]
    labels = [f"Flue gas film\n{bars[0]:.1f}%",
              f"Tube wall\n{bars[1]:.1f}%",
              f"Water bath\n{bars[2]:.1f}%"]
    colors = ["#D85A30", "#888780", "#185FA5"]
    left_pos = 0
    for b, c, lb in zip(bars, colors, labels):
        ax1.barh(0, b, left=left_pos, color=c, height=0.5)
        if b > 6:
            ax1.text(left_pos + b/2, 0, lb,
                     ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
        left_pos += b
    ax1.set_xlim(0, 100)
    ax1.axis("off")
    fig1.tight_layout(pad=0.2)
    st.pyplot(fig1)
    plt.close(fig1)

with right:
    st.subheader("Process coil analysis")

    m1, m2 = st.columns(2)
    m1.metric("h_gi — natural gas (inside)",
              f"{h_gi:.0f} W/m²·K",
              help="Gnielinski correlation")
    m2.metric("h_wo — water bath (outside)",
              f"{h_wo:.0f} W/m²·K",
              help="Churchill-Chu natural convection")

    m3, m4 = st.columns(2)
    m3.metric("Overall U (process coil)",
              f"{U_pc['U_o']:.1f} W/m²·K")
    m4.metric("LMTD",
              f"{lmtd_pc_val:.1f} °C")

    st.markdown("**Coil skin temperatures**")
    s1, s2, s3 = st.columns(3)
    s1.metric("Outer skin (water side)",
              f"{skin_pc['T_outer_skin']:.1f} °C")
    s2.metric("Inner skin (bore surface)",
              f"{skin_pc['T_inner_skin']:.1f} °C")
    s3.metric("Required area vs actual",
              f"{A_required_pc:.1f} m²",
              delta=f"Actual: {pc_Ao:.1f} m²",
              delta_color="normal")

    st.markdown("**Reynolds & flow regime**")
    st.info(
        f"Re_gas = {Re_ng:,.0f} — "
        f"{'Turbulent ✅' if Re_ng > 4000 else 'Transitional ⚠️' if Re_ng > 2300 else 'Laminar ⚠️'}  |  "
        f"Pr = {fp_ng['Pr']:.2f}  |  "
        f"Velocity = {vel_ng:.1f} m/s"
    )

    # Resistance breakdown
    st.markdown("**Thermal resistance breakdown (% of total)**")
    fig2, ax2 = plt.subplots(figsize=(5, 1.4))
    bars2 = [U_pc["pct_inner"], U_pc["pct_wall"], U_pc["pct_outer"]]
    labels2 = [f"Gas film\n{bars2[0]:.1f}%",
               f"Coil wall\n{bars2[1]:.1f}%",
               f"Water bath\n{bars2[2]:.1f}%"]
    left_pos2 = 0
    for b, c, lb in zip(bars2, ["#993C1D", "#888780", "#185FA5"], labels2):
        ax2.barh(0, b, left=left_pos2, color=c, height=0.5)
        if b > 6:
            ax2.text(left_pos2 + b/2, 0, lb,
                     ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
        left_pos2 += b
    ax2.set_xlim(0, 100)
    ax2.axis("off")
    fig2.tight_layout(pad=0.2)
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")

# ── Temperature profile charts
st.subheader("Temperature profiles along tube length")
chart_l, chart_r = st.columns(2)

N = 60
x = np.linspace(0, ft_L, N)

with chart_l:
    T_fg_profile = fg_T_in - (fg_T_in - fg_T_out) * x / ft_L
    T_outer_ft   = T_bath + (skin_ft["T_outer_skin"] - T_bath) * (1 - x/ft_L)
    T_inner_ft   = T_bath + (skin_ft["T_inner_skin"] - T_bath) * (1 - x/ft_L)

    fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
    ax3.plot(x, T_fg_profile,  color="#D85A30", lw=2,   label="Flue gas bulk")
    ax3.plot(x, T_inner_ft,    color="#BA7517", lw=1.5, linestyle="--",
             label="Firing tube inner skin")
    ax3.plot(x, T_outer_ft,    color="#185FA5", lw=1.5, linestyle="--",
             label="Firing tube outer skin")
    ax3.axhline(T_bath, color="#1D9E75", lw=1.2, linestyle=":",
                label=f"Water bath ({T_bath}°C)")
    ax3.set_xlabel("Position along tube (m)", fontsize=10)
    ax3.set_ylabel("Temperature (°C)", fontsize=10)
    ax3.set_title("Firing tube temperature profile", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.15)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

with chart_r:
    x2 = np.linspace(0, pc_L, N)
    T_ng_profile  = ng_T_in + (ng_T_out - ng_T_in) * x2 / pc_L
    T_outer_pc    = T_bath - (T_bath - skin_pc["T_outer_skin"]) * (1 - x2/pc_L)

    fig4, ax4 = plt.subplots(figsize=(5.5, 3.5))
    ax4.axhline(T_bath, color="#1D9E75", lw=1.5, linestyle=":",
                label=f"Water bath ({T_bath}°C)")
    ax4.plot(x2, T_outer_pc,   color="#185FA5", lw=1.5, linestyle="--",
             label="Coil outer skin")
    ax4.plot(x2, T_ng_profile, color="#993C1D", lw=2,   label="Natural gas bulk")
    ax4.set_xlabel("Position along coil (m)", fontsize=10)
    ax4.set_ylabel("Temperature (°C)", fontsize=10)
    ax4.set_title("Process coil temperature profile", fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.15)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

st.markdown("---")

# ── Full breakdown table
st.subheader("Complete calculation summary")

summary = {
    "Quantity": [
        # Firing tube geometry
        "Firing tube OD", "Firing tube ID", "Firing tube length",
        "Firing tube outer area",
        # Flue gas
        "Flue gas mean temp", "Flue gas velocity", "Re (flue gas)",
        "Pr (flue gas)", "h_fi — flue gas film",
        # Water bath / firing tube outside
        "h_fo — water bath (firing tube outside)",
        "Firing tube wall resistance",
        "Overall U — firing tube", "Firing tube heat duty",
        "Heat flux — firing tube outer",
        "Firing tube outer skin temp", "Firing tube inner skin temp",
        # Process coil geometry
        "Process coil OD", "Process coil ID", "Process coil length",
        "Process coil outer area",
        # Natural gas
        "Natural gas mean temp", "Gas pressure",
        "Gas velocity", "Re (natural gas)", "Pr (natural gas)",
        "h_gi — gas film (inside coil)",
        # Water bath / coil outside
        "h_wo — water bath (coil outside)",
        "Process coil wall resistance",
        "Overall U — process coil",
        "LMTD (water bath to gas)",
        "Process coil heat duty",
        "Required area vs actual",
        "Coil outer skin temp",
    ],
    "Value": [
        f"{ft_OD} mm",
        f"{ft_Di*1000:.1f} mm",
        f"{ft_L} m",
        f"{ft_Ao:.2f} m²",

        f"{T_fg_mean:.0f} °C",
        f"{vel_fg:.2f} m/s",
        f"{Re_fg:,.0f}",
        f"{fp_fg['Pr']:.3f}",
        f"{h_fi:.1f} W/m²·K",

        f"{h_fo:.1f} W/m²·K",
        f"{U_ft['R_wall']*1000:.3f} ×10⁻³ m²·K/W",
        f"{U_ft['U_o']:.1f} W/m²·K",
        f"{Q_ft/1000:.2f} kW",
        f"{skin_ft['heat_flux']/1000:.2f} kW/m²",
        f"{skin_ft['T_outer_skin']:.1f} °C",
        f"{skin_ft['T_inner_skin']:.1f} °C",

        f"{pc_OD} mm",
        f"{pc_Di*1000:.1f} mm",
        f"{pc_L} m",
        f"{pc_Ao:.2f} m²",

        f"{T_ng_mean:.0f} °C",
        f"{ng_P} barg",
        f"{vel_ng:.2f} m/s",
        f"{Re_ng:,.0f}",
        f"{fp_ng['Pr']:.3f}",
        f"{h_gi:.1f} W/m²·K",

        f"{h_wo:.1f} W/m²·K",
        f"{U_pc['R_wall']*1000:.3f} ×10⁻³ m²·K/W",
        f"{U_pc['U_o']:.1f} W/m²·K",
        f"{lmtd_pc_val:.1f} °C",
        f"{Q_pc/1000:.2f} kW",
        f"{A_required_pc:.1f} m² required / {pc_Ao:.1f} m² actual",
        f"{skin_pc['T_outer_skin']:.1f} °C",
    ],
}

df = pd.DataFrame(summary)
st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Correlations used: Gnielinski (turbulent internal flow, Re > 3000), "
    "Dittus-Boelter (fallback for lower Re), "
    "Churchill-Chu (natural convection on horizontal cylinder in water bath). "
    "Fluid properties are temperature-dependent polynomial fits. "
    "Results are indicative — verify against HTRI/HTFS for detailed design."
)
