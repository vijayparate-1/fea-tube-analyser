# ================================================================
# wbh_v3.py  —  Direct-Fired Water Bath Heater
#               Heat Transfer Analyser  v3
#
# KEY CORRECTIONS FROM v2:
#   - Flue gas mass flow DERIVED from burner duty + temps (not input)
#   - Re correctly derived: duty → ṁ → velocity → Re → h → U
#   - Process coil area = straight tubes + U-bends (both ends)
#   - Firing tube: single U-tube or N× U-tubes in series
#   - Natural gas flow rate is primary input (with unit conversion)
#   - End elevation schematic matching actual tube sheet view
#   - Overall U as headline metric throughout
#
# Run: python3 -m streamlit run wbh_v3.py
# ================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, PageBreak,
                                 KeepTogether)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="WBH Heat Transfer v3",
    page_icon="🔥",
    layout="wide",
)

st.markdown("""
<style>
    .metric-box {
        background: var(--background-color);
        border: 1px solid #d8d7cf;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 11px; color: #6b6b65; margin-bottom: 2px; }
    .metric-value { font-size: 22px; font-weight: 600; color: #1a1a18; }
    .metric-sub   { font-size: 11px; color: #888; margin-top: 2px; }
    div[data-testid="metric-container"] {
        border: 1px solid #e0dfd7;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# PIPE DATA — NPS → OD (mm), schedule → wall (mm)
# ================================================================
PIPE_DATA = {
    '2" NPS  (60.3mm OD)':  {"OD": 60.3,  "sch": {"Sch 10S":1.65,"Sch 40":3.91,"Sch 80":5.54,"Sch 160":8.74}},
    '3" NPS  (88.9mm OD)':  {"OD": 88.9,  "sch": {"Sch 10":3.05,"Sch 20":3.56,"Sch 40":5.49,"Sch 80":7.62,"Sch 160":11.13}},
    '4" NPS  (114.3mm OD)': {"OD": 114.3, "sch": {"Sch 10":3.05,"Sch 20":3.56,"Sch 40":6.02,"Sch 80":8.56,"Sch 160":13.49}},
    '6" NPS  (168.3mm OD)': {"OD": 168.3, "sch": {"Sch 10":3.40,"Sch 20":3.96,"Sch 40":7.11,"Sch 80":10.97,"Sch 160":18.26}},
    '8" NPS  (219.1mm OD)': {"OD": 219.1, "sch": {"Sch 10":3.76,"Sch 20":6.35,"Sch 40":8.18,"Sch 80":12.70,"Sch 160":23.01}},
    '10" NPS (273.0mm OD)': {"OD": 273.0, "sch": {"Sch 10":4.19,"Sch 20":6.35,"Sch 40":9.27,"Sch 80":12.70,"Sch 160":28.58}},
    '12" NPS (323.9mm OD)': {"OD": 323.9, "sch": {"Sch 10":4.57,"Sch 20":6.35,"Sch 40":9.53,"Sch 80":12.70,"Sch 160":33.32}},
}

# B31.3 allowable stress (MPa) vs temperature (°C)
B313 = {
    "A106 Gr B (Carbon steel)":  {38:138,93:138,149:131,204:128,260:124,316:117,371:110},
    "A333 Gr 6 (Low-temp CS)":   {38:138,93:138,149:131,204:128,260:124},
    "A312 TP316L (Stainless)":   {38:115,93:115,149:115,204:110,260:105,316:100,371:94},
    "A312 TP304 (Stainless)":    {38:115,93:115,149:110,204:105,260:100,316:95,371:90},
}

K_MAT = {
    "Carbon steel     (k = 50 W/m·K)": 50,
    "Alloy steel P11  (k = 38 W/m·K)": 38,
    "Stainless 316L   (k = 16 W/m·K)": 16,
    "Stainless 304    (k = 17 W/m·K)": 17,
}

# ================================================================
# FLUID PROPERTY FUNCTIONS
# ================================================================

def props_flue_gas(T_C):
    """Flue gas (natural gas combustion products) properties at T_C °C."""
    T   = max(T_C, 50.0)
    Cp  = 1050 + 0.12 * T                        # J/kg·K
    k   = 0.0245 + 7.2e-5 * T                    # W/m·K
    mu  = 1.46e-5 + 4.0e-8 * T                   # Pa·s
    rho = 1.25 * 273.15 / (273.15 + T)           # kg/m³
    Pr  = mu * Cp / k
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, Pr=Pr)

def props_natural_gas(T_C, P_barg):
    """Natural gas (~95% CH4) properties at T_C °C and P_barg barg."""
    T   = max(T_C, 5.0)
    P   = (P_barg + 1.01325) * 1e5               # Pa absolute
    M   = 16.04                                   # kg/kmol methane
    R   = 8314.0                                  # J/kmol·K
    Cp  = 2200 + 1.1*T + 0.002*T**2              # J/kg·K
    k   = 0.0302 + 8.5e-5 * T                    # W/m·K
    mu  = 1.05e-5 + 3.0e-8 * T                   # Pa·s
    rho = P * M / (R * (273.15 + T))             # kg/m³ (real gas approx)
    Pr  = mu * Cp / k
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, Pr=Pr)

def props_water(T_C):
    """Liquid water properties at T_C °C (20–95°C)."""
    T    = max(min(T_C, 95.0), 20.0)
    Cp   = 4210 - 1.4 * T + 0.006 * T**2        # J/kg·K
    k    = 0.571 + 0.00175*T - 6e-6*T**2        # W/m·K
    mu   = 1e-3 * np.exp(-0.02*(T - 20))         # Pa·s
    rho  = 1000 - 0.003*T**2                     # kg/m³
    beta = 2.1e-4 + 5e-6*T                       # 1/K
    Pr   = mu * Cp / k
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, beta=beta, Pr=Pr)

# ================================================================
# HEAT TRANSFER CORRELATIONS
# ================================================================

def h_gnielinski(Re, Pr, k, D_i, L):
    """Gnielinski — turbulent internal flow. Valid Re > 3000."""
    Re  = max(Re, 3001)
    f   = (0.790 * np.log(Re) - 1.64) ** -2      # Petukhov friction
    Nu  = (f/8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1))
    Nu *= (1 + (D_i / max(L, D_i)) ** (2/3))      # entry correction
    return max(Nu * k / D_i, 1.0)

def h_dittus_boelter(Re, Pr, k, D_i, heating=True):
    """Dittus-Boelter — simpler fallback."""
    n   = 0.4 if heating else 0.3
    Nu  = 0.023 * max(Re, 100)**0.8 * max(Pr, 0.1)**n
    return max(Nu * k / D_i, 1.0)

def h_churchill_chu(D_o, T_surf, T_bath, fluid):
    """
    Churchill-Chu — natural convection on horizontal cylinder.
    Returns h (W/m²·K) for outside of tube submerged in water bath.
    """
    dT = abs(T_surf - T_bath)
    if dT < 0.05:
        return 600.0
    g    = 9.81
    nu   = fluid["mu"] / fluid["rho"]
    alp  = fluid["k"] / (fluid["rho"] * fluid["Cp"])
    Ra   = g * fluid["beta"] * dT * D_o**3 / (nu * alp)
    Ra   = max(Ra, 1e3)
    denom = (1 + (0.559 / fluid["Pr"])**(9/16))**(16/9)
    Nu   = (0.60 + 0.387 * (Ra / denom)**(1/6))**2
    return max(Nu * fluid["k"] / D_o, 50.0)

# ================================================================
# OVERALL U  — cylindrical wall, referenced to OUTER area
# ================================================================

def calc_U(h_i, h_o, k_wall, D_i_m, D_o_m, Rf_i=0.0, Rf_o=0.0):
    """
    1/U_o = (D_o/D_i)·(1/h_i + Rf_i)
            + D_o·ln(D_o/D_i)/(2·k)   [= ro·ln(ro/ri)/k]
            + 1/h_o + Rf_o
    Returns dict with U_o (W/m²·K) and resistance breakdown.
    """
    ro      = D_o_m / 2
    ri      = D_i_m / 2
    ratio   = D_o_m / D_i_m
    R_i     = ratio * (1.0/h_i + Rf_i)           # inner film + fouling
    R_wall  = ro * np.log(ro / ri) / k_wall       # wall conduction
    R_o     = 1.0/h_o + Rf_o                     # outer film + fouling
    R_tot   = R_i + R_wall + R_o
    U_o     = 1.0 / R_tot
    return dict(
        U_o     = U_o,
        R_inner = R_i,
        R_wall  = R_wall,
        R_outer = R_o,
        R_total = R_tot,
        pct_i   = R_i    / R_tot * 100,
        pct_w   = R_wall / R_tot * 100,
        pct_o   = R_o    / R_tot * 100,
    )

# ================================================================
# LMTD — isothermal hot side (water bath)
# ================================================================

def calc_lmtd(T_hot, T_cold_in, T_cold_out):
    """Water bath isothermal → simple LMTD."""
    dT1 = T_hot - T_cold_in
    dT2 = T_hot - T_cold_out
    dT1 = max(dT1, 0.01); dT2 = max(dT2, 0.01)
    if abs(dT1 - dT2) < 0.01:
        return dT1
    return (dT1 - dT2) / np.log(dT1 / dT2)

# ================================================================
# PRESSURE DROP — Darcy-Weisbach with Swamee-Jain
# ================================================================

def swamee_jain(Re, eps_D):
    """Friction factor — Swamee-Jain (explicit Colebrook approximation)."""
    if Re < 2300:
        return 64.0 / max(Re, 1.0)
    return 0.25 / (np.log10(eps_D/3.7 + 5.74/Re**0.9))**2

def calc_pressure_drop(mdot_total, rho, mu, D_i_m,
                        L_straight_m, N_ubends, r_ubend_m,
                        N_parallel=1, roughness=4.6e-5):
    """
    Pressure drop for one parallel circuit.
    mdot_total splits into N_parallel circuits.
    Each circuit has L_straight_m total straight length
    and N_ubends × 180° bends.

    U-bend loss coefficient K ≈ 0.35 (r/D ≈ 2-3).
    Returns ΔP in Pa, kPa, mbar.
    """
    mdot_per = mdot_total / N_parallel
    A_cs     = np.pi / 4 * D_i_m**2
    v        = mdot_per / (rho * A_cs)
    Re       = rho * v * D_i_m / mu
    Re       = max(Re, 100)
    eps_D    = roughness / D_i_m
    f        = swamee_jain(Re, eps_D)
    dyn      = 0.5 * rho * v**2                  # dynamic pressure Pa

    dP_straight = f * (L_straight_m / D_i_m) * dyn
    K_ubend     = 0.35                            # 180° bend loss coeff
    dP_bends    = N_ubends * K_ubend * dyn

    dP_total = dP_straight + dP_bends
    regime = ("Laminar"      if Re < 2300 else
              "Transitional" if Re < 4000 else
              "Turbulent")
    return dict(
        mdot_per   = mdot_per,
        velocity   = v,
        Re         = Re,
        f          = f,
        regime     = regime,
        dP_straight= dP_straight,
        dP_bends   = dP_bends,
        dP_total   = dP_total,
        dP_kPa     = dP_total / 1000,
        dP_mbar    = dP_total / 100,
    )

# ================================================================
# B31.3 WALL THICKNESS CHECK
# ================================================================

def b313_allowable(material, T_C):
    tbl = B313[material]
    temps = sorted(tbl.keys())
    if T_C <= temps[0]:  return tbl[temps[0]]
    if T_C >= temps[-1]: return tbl[temps[-1]]
    for i in range(len(temps)-1):
        if temps[i] <= T_C <= temps[i+1]:
            fr = (T_C - temps[i]) / (temps[i+1] - temps[i])
            return tbl[temps[i]] + fr * (tbl[temps[i+1]] - tbl[temps[i]])
    return tbl[temps[-1]]

def b313_check(P_barg, D_o_mm, T_C, material, t_sel_mm,
               E=1.0, Y=0.4, mill_tol=0.125):
    P   = P_barg * 0.1
    S   = b313_allowable(material, T_C)
    t_min = (P * D_o_mm) / (2 * (S*E + P*Y))
    t_req = t_min / (1 - mill_tol)
    margin = t_sel_mm - t_req
    return dict(
        S=round(S,1), t_min=round(t_min,2),
        t_req=round(t_req,2), t_sel=round(t_sel_mm,2),
        margin=round(margin,2),
        status="PASS ✅" if margin >= 0 else "FAIL ❌",
    )

# ================================================================
# UNIT CONVERSION — natural gas flow rate
# ================================================================

def to_kg_per_s(value, unit):
    """Convert natural gas flow to kg/s."""
    RHO_STD = 0.679  # kg/Sm³ for methane at 15°C, 1.01325 bar
    if unit == "kg/s":
        return value
    elif unit == "kg/hr":
        return value / 3600
    elif unit == "Sm³/hr":
        return value * RHO_STD / 3600
    elif unit == "MMscfd":
        # 1 MMscfd = 1e6 scf/day × 0.028317 m³/scf / 86400 s/day
        Sm3_per_s = value * 1e6 * 0.028317 / 86400
        return Sm3_per_s * RHO_STD
    return value

# ================================================================
# ITERATIVE h_fo / h_wo  (water bath side, needs skin temp)
# ================================================================

def iterate_h_bath(Q_W, A_o, D_o_m, T_bath, fluid_wb,
                   T_surf_guess, hot_side=True, n_iter=10):
    """
    Iterate Churchill-Chu to converge skin temperature.
    hot_side=True  → tube is hotter than bath (firing tube)
    hot_side=False → tube is cooler than bath (process coil)
    Returns converged h_bath (W/m²·K) and T_surf (°C).
    """
    T_surf = T_surf_guess
    h      = 500.0
    for _ in range(n_iter):
        h      = h_churchill_chu(D_o_m, T_surf, T_bath, fluid_wb)
        q      = Q_W / max(A_o, 0.001)
        if hot_side:
            T_surf = T_bath + q / max(h, 1.0)
        else:
            T_surf = T_bath - q / max(h, 1.0)
    return h, T_surf

# ================================================================
# SKIN TEMPERATURE INCREMENTS
# ================================================================

def skin_temp_increments(Q_W, A_o, D_i_m, D_o_m, h_i, h_o, k_wall):
    """
    Returns temperature rises/drops from bath outward (or inward).
    dT_o  = rise from bath to outer tube skin  = q / h_o
    dT_w  = rise across wall                   = q × ro × ln(ro/ri) / k
    dT_i  = rise from inner skin to fluid film = q × (D_o/D_i) / h_i
    All are positive magnitudes.
    """
    ro  = D_o_m / 2
    ri  = D_i_m / 2
    q   = Q_W / max(A_o, 0.001)
    dT_o = q / max(h_o, 1.0)
    dT_w = q * ro * np.log(ro / ri) / k_wall
    dT_i = q * (D_o_m / D_i_m) / max(h_i, 1.0)
    return dict(q=q, dT_o=dT_o, dT_w=dT_w, dT_i=dT_i)

# ================================================================
# SECTION  —  SIDEBAR INPUTS
# ================================================================

sb = st.sidebar
sb.markdown("## 🔥 WBH Heat Transfer v3")
sb.markdown("---")

# ── SECTION A: VESSEL ──────────────────────────────────────────
sb.markdown("### A · Vessel geometry")
ves_ID   = sb.number_input("Shell inside diameter (mm)", 500, 5000, 1500, 50)
ves_L    = sb.number_input("Length — tangent to tangent (mm)", 1000, 20000, 6000, 500)
T_bath   = sb.number_input("Water bath temperature (°C)", 60.0, 95.0, 82.0, 1.0)
sb.markdown("---")

# ── SECTION B: FIRING TUBE ─────────────────────────────────────
sb.markdown("### B · Firing tube")
ft_config = sb.selectbox("Configuration",
    ["Single U-tube (2 passes)",
     "2 × U-tubes in series (4 passes)",
     "3 × U-tubes in series (6 passes)",
     "4 × U-tubes in series (8 passes)"])
N_utubes  = int(ft_config.split("×")[0].split("Single")[0].strip() or "1")
N_utubes  = {"Single":1,"2":2,"3":3,"4":4}[ft_config.split()[0]]
N_ft_passes = N_utubes * 2

ft_nps    = sb.selectbox("Firing tube NPS",
    [k for k in PIPE_DATA if int(k.split('"')[0].strip().split('.')[0]) >= 4])
ft_sch    = sb.selectbox("Schedule — firing tube",
    list(PIPE_DATA[ft_nps]["sch"].keys()))
ft_wall   = PIPE_DATA[ft_nps]["sch"][ft_sch]
ft_OD     = PIPE_DATA[ft_nps]["OD"]
ft_ID     = ft_OD - 2 * ft_wall
ft_r_bend = sb.number_input("U-bend centreline radius (mm)", 100, 2000,
                              int(ft_OD * 1.5), 50,
                              help="Typically 1.5–3× tube OD")
ft_kmat   = sb.selectbox("Firing tube material", list(K_MAT.keys()))
k_ft      = K_MAT[ft_kmat]
sb.info(f"OD = {ft_OD:.1f} mm  |  Wall = {ft_wall:.2f} mm  |  ID = {ft_ID:.1f} mm")
sb.markdown("---")

# ── SECTION C: BURNER / FLUE GAS ──────────────────────────────
sb.markdown("### C · Burner & flue gas")
Q_burner  = sb.number_input("Burner rated duty (kW)", 100.0, 50000.0, 1000.0, 50.0)
eta_burner= sb.slider("Thermal efficiency (%)", 60, 95, 85)
T_fg_in   = sb.number_input("Flue gas inlet temperature (°C)", 400, 1400, 900, 10)
T_fg_out  = sb.number_input("Flue gas outlet temperature (°C)", 150, 600, 330, 10)
foul_fg   = sb.number_input("Fouling — flue gas side (m²·K/W)",
                              0.0, 0.005, 0.0002, format="%.4f")
sb.markdown("---")

# ── SECTION D: PROCESS COIL ────────────────────────────────────
sb.markdown("### D · Process coil geometry")
pc_nps    = sb.selectbox("Process coil NPS",
    [k for k in PIPE_DATA if int(k.split('"')[0].strip().split('.')[0]) <= 6])
pc_sch    = sb.selectbox("Schedule — process coil",
    list(PIPE_DATA[pc_nps]["sch"].keys()))
pc_wall   = PIPE_DATA[pc_nps]["sch"][pc_sch]
pc_OD     = PIPE_DATA[pc_nps]["OD"]
pc_ID     = pc_OD - 2 * pc_wall
N_paths   = sb.number_input("Number of parallel flow paths", 1, 12, 2, 1,
    help="Gas flow splits equally into this many parallel circuits")
N_rows    = sb.number_input("Number of rows (passes) per path", 1, 30, 8, 1,
    help="Each row = one tube run across vessel length. Rows connected by U-bends.")
pc_r_bend = sb.number_input("U-bend centreline radius — coil (mm)", 50, 500,
                              int(pc_OD * 1.5), 25)
pc_kmat   = sb.selectbox("Process coil material", list(K_MAT.keys()))
k_pc      = K_MAT[pc_kmat]
foul_ng   = sb.number_input("Fouling — gas side (m²·K/W)",
                              0.0, 0.005, 0.0002, format="%.4f")
foul_wb   = sb.number_input("Fouling — water bath side (m²·K/W)",
                              0.0, 0.005, 0.0001, format="%.4f")
sb.info(f"OD = {pc_OD:.1f} mm  |  Wall = {pc_wall:.2f} mm  |  ID = {pc_ID:.1f} mm")
sb.markdown("---")

# ── SECTION E: NATURAL GAS ─────────────────────────────────────
sb.markdown("### E · Natural gas conditions")
ng_flow_unit = sb.selectbox("Flow rate unit",
    ["kg/s", "kg/hr", "Sm³/hr", "MMscfd"])
ng_flow_val  = sb.number_input(f"Gas flow rate ({ng_flow_unit})",
    0.01, 100000.0, 2.5 if ng_flow_unit=="kg/s" else
    9000.0 if ng_flow_unit=="kg/hr" else
    13000.0 if ng_flow_unit=="Sm³/hr" else 0.5, format="%.3f")
ng_mdot      = to_kg_per_s(ng_flow_val, ng_flow_unit)
T_ng_in      = sb.number_input("Gas inlet temperature (°C)", 0.0, 60.0, 15.0, 1.0)
T_ng_out     = sb.number_input("Gas outlet temperature (°C)", 20.0, 90.0, 60.0, 1.0)
P_ng_op      = sb.number_input("Operating pressure (barg)", 1.0, 200.0, 50.0, 1.0)
sb.info(f"ṁ = {ng_mdot:.4f} kg/s  ({ng_mdot*3600:.1f} kg/hr)")
sb.markdown("---")

# ── SECTION F: DESIGN / B31.3 ──────────────────────────────────
sb.markdown("### F · Design & B31.3 check")
b313_mat  = sb.selectbox("B31.3 material — process coil", list(B313.keys()))
P_design  = sb.number_input("Design pressure — coil (barg)", 1.0, 500.0, 100.0, 5.0)
T_design  = sb.number_input("Design temperature — coil (°C)", 0.0, 300.0, 80.0, 5.0)
sb.markdown("---")
sb.caption("v3 · All U values calculated correctly from flow → velocity → Re → h → U")

# ================================================================
# GEOMETRY — DERIVED
# ================================================================

L_ves_m   = ves_L / 1000
ft_OD_m   = ft_OD / 1000
ft_ID_m   = ft_ID / 1000
ft_rb_m   = ft_r_bend / 1000
pc_OD_m   = pc_OD / 1000
pc_ID_m   = pc_ID / 1000
pc_rb_m   = pc_r_bend / 1000

# Firing tube areas
#   Straight:  N_passes × π × D_o × L_vessel
#   U-bends:   N_utubes × π² × D_o × r_bend  (each U-bend = π×r arc × π×D_o circ)
A_ft_straight = N_ft_passes * np.pi * ft_OD_m * L_ves_m
A_ft_ubend    = N_utubes    * np.pi**2 * ft_OD_m * ft_rb_m
A_ft_total    = A_ft_straight + A_ft_ubend
A_ft_cs       = np.pi / 4 * ft_ID_m**2         # cross-section for velocity

# Firing tube straight length (for ΔP)
L_ft_straight = N_ft_passes * L_ves_m          # total straight length

# Process coil areas
#   Straight:  N_paths × N_rows × π × D_o × L_vessel
#   U-bends:   N_paths × (N_rows-1) × π² × D_o × r_bend  (per path: N_rows-1 bends)
A_pc_straight = N_paths * N_rows * np.pi * pc_OD_m * L_ves_m
N_pc_ubends_total = N_paths * (N_rows - 1)
A_pc_ubend    = N_pc_ubends_total * np.pi**2 * pc_OD_m * pc_rb_m
A_pc_total    = A_pc_straight + A_pc_ubend
A_pc_cs       = np.pi / 4 * pc_ID_m**2         # single tube cross-section

# Process coil straight length per path (for ΔP)
L_pc_per_path = N_rows * L_ves_m               # straight only (bends counted separately)

# ================================================================
# CALCULATIONS — FLUE GAS SIDE
# ================================================================

# Heat available to water bath from burner
Q_to_bath_W   = Q_burner * 1000 * eta_burner / 100   # W

# Mean flue gas temperature
T_fg_mean     = (T_fg_in + T_fg_out) / 2

# Fluid properties at mean temperature
fp_fg         = props_flue_gas(T_fg_mean)

# Flue gas mass flow DERIVED from energy balance
#   Q = ṁ × Cp × ΔT
dT_fg         = max(T_fg_in - T_fg_out, 1.0)
mdot_fg       = Q_to_bath_W / (fp_fg["Cp"] * dT_fg)  # kg/s

# Velocity — all passes in series, single tube cross-section
v_fg          = mdot_fg / (fp_fg["rho"] * A_ft_cs)

# Reynolds number — flow-derived
Re_fg         = fp_fg["rho"] * v_fg * ft_ID_m / fp_fg["mu"]

# h inside firing tube
if Re_fg > 3000:
    h_fi = h_gnielinski(Re_fg, fp_fg["Pr"], fp_fg["k"], ft_ID_m, L_ft_straight)
else:
    h_fi = h_dittus_boelter(Re_fg, fp_fg["Pr"], fp_fg["k"], ft_ID_m, heating=False)

# Water bath props
fp_wb = props_water(T_bath)

# h outside firing tube — iterate
h_fo, T_ft_outer = iterate_h_bath(
    Q_to_bath_W, A_ft_total, ft_OD_m, T_bath, fp_wb,
    T_bath + 15, hot_side=True)

# Overall U — firing tube (referenced to outer area)
U_ft = calc_U(h_fi, h_fo, k_ft, ft_ID_m, ft_OD_m, foul_fg, foul_wb)

# Skin temperatures — firing tube
sk_ft = skin_temp_increments(Q_to_bath_W, A_ft_total,
                              ft_ID_m, ft_OD_m, h_fi, h_fo, k_ft)
T_ft_outer_skin = T_bath        + sk_ft["dT_o"]
T_ft_inner_skin = T_ft_outer_skin + sk_ft["dT_w"]
T_ft_film       = T_ft_inner_skin + sk_ft["dT_i"]

# Pressure drop — firing tube
dp_fg = calc_pressure_drop(
    mdot_total=mdot_fg, rho=fp_fg["rho"], mu=fp_fg["mu"],
    D_i_m=ft_ID_m, L_straight_m=L_ft_straight,
    N_ubends=N_utubes, r_ubend_m=ft_rb_m,
    N_parallel=1,
    roughness=4.6e-5)

# ================================================================
# CALCULATIONS — PROCESS COIL SIDE
# ================================================================

T_ng_mean = (T_ng_in + T_ng_out) / 2
fp_ng     = props_natural_gas(T_ng_mean, P_ng_op)

# Heat duty required by natural gas
Q_gas_W   = ng_mdot * fp_ng["Cp"] * (T_ng_out - T_ng_in)   # W

# Velocity per tube — flow splits into N_paths parallel circuits
v_ng  = ng_mdot / (N_paths * fp_ng["rho"] * A_pc_cs)

# Reynolds number
Re_ng = fp_ng["rho"] * v_ng * pc_ID_m / fp_ng["mu"]

# h inside process coil
L_per_pass = L_ves_m  # each straight run = vessel length
if Re_ng > 3000:
    h_gi = h_gnielinski(Re_ng, fp_ng["Pr"], fp_ng["k"],
                         pc_ID_m, L_pc_per_path)
else:
    h_gi = h_dittus_boelter(Re_ng, fp_ng["Pr"], fp_ng["k"],
                              pc_ID_m, heating=True)

# h outside process coil — iterate
h_wo, T_pc_outer = iterate_h_bath(
    Q_gas_W, A_pc_total, pc_OD_m, T_bath, fp_wb,
    T_bath - 10, hot_side=False)

# Overall U — process coil
U_pc = calc_U(h_gi, h_wo, k_pc, pc_ID_m, pc_OD_m, foul_ng, foul_wb)

# LMTD
lmtd_val = calc_lmtd(T_bath, T_ng_in, T_ng_out)

# Required area and coil sizing
A_req     = Q_gas_W / (U_pc["U_o"] * max(lmtd_val, 0.1))
oversurf  = (A_pc_total - A_req) / max(A_req, 0.001) * 100

# Required vessel length (back-calculation)
# A_req = N_paths × N_rows × π × D_o × L_req  +  N_paths×(N_rows-1) × π²×D_o×r_bend
# Solve for L_req:
denom_L = N_paths * N_rows * np.pi * pc_OD_m
A_bend_term = N_paths * (N_rows - 1) * np.pi**2 * pc_OD_m * pc_rb_m
L_req = (A_req - A_bend_term) / max(denom_L, 0.001)

# Skin temperatures — process coil
sk_pc = skin_temp_increments(Q_gas_W, A_pc_total,
                              pc_ID_m, pc_OD_m, h_gi, h_wo, k_pc)
T_pc_outer_skin = T_bath        - sk_pc["dT_o"]
T_pc_inner_skin = T_pc_outer_skin - sk_pc["dT_w"]

# Pressure drop — process coil (per path)
dp_ng = calc_pressure_drop(
    mdot_total=ng_mdot, rho=fp_ng["rho"], mu=fp_ng["mu"],
    D_i_m=pc_ID_m, L_straight_m=L_pc_per_path,
    N_ubends=(N_rows - 1), r_ubend_m=pc_rb_m,
    N_parallel=N_paths,
    roughness=4.6e-5)

# B31.3 check
b313 = b313_check(P_design, pc_OD, T_design, b313_mat, pc_wall)

# Pinch checks
def pinch_check():
    msgs = []
    ok = True
    ap_fg = T_fg_out - T_bath
    ap_ng = T_bath   - T_ng_out
    if ap_fg < 30:
        msgs.append(f"⚠️  Flue gas exit ({T_fg_out:.0f}°C) only {ap_fg:.0f}°C above bath "
                    f"— acid dew point / condensation risk. Recommend ≥30°C approach.")
        ok = False
    if T_ng_out >= T_bath:
        msgs.append(f"❌  Gas outlet ({T_ng_out:.0f}°C) ≥ bath ({T_bath:.0f}°C) — "
                    f"thermodynamically impossible.")
        ok = False
    elif ap_ng < 5:
        msgs.append(f"⚠️  Gas outlet only {ap_ng:.0f}°C below bath — LMTD too small.")
        ok = False
    if ok:
        msgs.append(f"✅  Pinch checks OK. Flue exit approach = {ap_fg:.0f}°C  |  "
                    f"Gas outlet approach = {ap_ng:.0f}°C below bath.")
    return msgs, ok

pinch_msgs, pinch_ok = pinch_check()

# ================================================================
# END ELEVATION SCHEMATIC
# ================================================================

def draw_end_elevation():
    """
    End-view cross-section matching actual tube sheet drawing:
    - Circular vessel
    - Firing tube circles (teal, lower section)
    - Process coil circles (amber/orange, upper section)
    - Burner at bottom, stack at top, gas nozzles on side
    """
    fig, ax = plt.subplots(figsize=(7, 8))
    fig.patch.set_facecolor("#F8F8F6")
    ax.set_facecolor("#F8F8F6")
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.55, 1.45)
    ax.axis("off")

    # Scale: vessel ID → radius = 1.0 in plot units
    scale   = 2.0 / (ves_ID)              # 1 mm = scale plot units
    ft_r_pl = (ft_OD / 2) * scale
    pc_r_pl = (pc_OD / 2) * scale

    # ── Water fill
    ax.add_patch(plt.Circle((0,0), 0.98, fc="#C8DFF5", alpha=0.5, zorder=1))

    # ── Vessel shell
    ax.add_patch(plt.Circle((0,0), 1.0,  fc="none", ec="#4A4A48", lw=3, zorder=6))

    # ── Firing tubes: N_utubes × 2 circles (both legs of each U)
    n_ft_circles = N_utubes * 2
    cols_ft = min(n_ft_circles, 4)
    rows_ft = int(np.ceil(n_ft_circles / cols_ft))
    gap_ft  = ft_r_pl * 2 + max(0.04, 0.02)
    ft_centers = []
    for row in range(rows_ft):
        n_row = min(cols_ft, n_ft_circles - row * cols_ft)
        for col in range(n_row):
            x = (col - (n_row - 1) / 2) * gap_ft
            y = -0.72 + row * gap_ft
            ft_centers.append((x, y))

    for i, (cx, cy) in enumerate(ft_centers):
        ax.add_patch(plt.Circle((cx, cy), ft_r_pl,
                                fc="#5DCAA5", ec="#085041", lw=1.5, zorder=4))
        ax.text(cx, cy, f"FT{i+1}", fontsize=max(6, int(ft_r_pl*40)),
                ha="center", va="center", color="#04342C", fontweight="bold", zorder=5)

    ax.text(0, -0.72 - ft_r_pl - 0.08,
            f"Firing tube  {ft_nps.split('(')[0].strip()}  {ft_sch}",
            ha="center", va="top", fontsize=8.5, color="#085041", fontweight="bold")

    # ── Process coil circles
    n_pc_total  = int(N_paths * N_rows)
    cols_pc     = min(n_pc_total, 12)
    rows_pc     = int(np.ceil(n_pc_total / cols_pc))
    gap_pc      = pc_r_pl * 2 + max(0.02, 0.01)
    y_pc_top    = 0.78

    placed = 0
    for row in range(rows_pc):
        if placed >= n_pc_total: break
        n_row = min(cols_pc, n_pc_total - placed)
        row_y = y_pc_top - row * gap_pc
        for col in range(n_row):
            cx = (col - (n_row - 1) / 2) * gap_pc
            cy = row_y
            if cx**2 + cy**2 > 0.91**2: continue
            ax.add_patch(plt.Circle((cx, cy), pc_r_pl,
                                    fc="#EF9F27", ec="#633806", lw=0.8, zorder=4))
            placed += 1

    ax.text(0, y_pc_top + pc_r_pl + 0.06,
            f"Process coil  {pc_nps.split('(')[0].strip()}  {pc_sch}  "
            f"({N_paths} paths × {N_rows} rows)",
            ha="center", va="bottom", fontsize=8.5, color="#633806", fontweight="bold")

    # ── Burner (bottom)
    bx, by = 0.0, -1.28
    ax.annotate("", xy=(0, -1.02), xytext=(0, -1.22),
                arrowprops=dict(arrowstyle="->", color="#D85A30", lw=2.5))
    ax.text(bx, by - 0.04, "Burner", ha="center", va="top",
            fontsize=9, color="#D85A30", fontweight="bold")

    # ── Flue stack (top centre)
    ax.plot([0, 0], [1.02, 1.28], color="#5F5E5A", lw=6, solid_capstyle="round", zorder=5)
    ax.text(0, 1.31, "Flue stack", ha="center", va="bottom", fontsize=8.5, color="#5F5E5A")

    # ── Gas nozzles (left side — from drawing)
    for y_n, lbl, col, T_val in [
        (-0.25, f"Gas IN  {T_ng_in:.0f}°C",  "#E24B4A", T_ng_in),
        ( 0.25, f"Gas OUT {T_ng_out:.0f}°C", "#BA7517", T_ng_out),
    ]:
        ax.annotate("", xy=(-1.01, y_n), xytext=(-1.22, y_n),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax.text(-1.27, y_n, lbl, ha="right", va="center", fontsize=8, color=col)

    # ── Key metrics box
    metrics_txt = (
        f"U firing = {U_ft['U_o']:.0f} W/m²·K\n"
        f"U coil   = {U_pc['U_o']:.0f} W/m²·K\n"
        f"Q burner = {Q_burner:.0f} kW\n"
        f"Q gas    = {Q_gas_W/1000:.0f} kW"
    )
    ax.text(1.05, 0.85, metrics_txt, ha="left", va="top", fontsize=7.5,
            color="#1a1a18", linespacing=1.7,
            bbox=dict(fc="#EAF3DE", ec="#3B6D11", boxstyle="round,pad=0.4", lw=0.8))

    ax.set_title(
        f"Water bath heater — end elevation (tube sheet view)\n"
        f"Vessel ID {ves_ID:.0f} mm  ×  {ves_L:.0f} mm L   |   Bath {T_bath:.0f}°C",
        fontsize=10, pad=6, color="#1a1a18")

    plt.tight_layout(pad=0.8)
    return fig

# ================================================================
# PDF GENERATOR
# ================================================================

def build_pdf():
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=18*mm, rightMargin=18*mm,
                             topMargin=18*mm, bottomMargin=18*mm)
    BLU = colors.HexColor("#185FA5")
    TEA = colors.HexColor("#0F6E56")
    RED = colors.HexColor("#D85A30")
    GRY = colors.HexColor("#f5f4f0")
    MID = colors.HexColor("#6b6b65")
    BRD = colors.HexColor("#d8d7cf")
    DRK = colors.HexColor("#1a1a18")

    PS = lambda nm, **kw: ParagraphStyle(nm, **kw)
    h1 = PS("h1", fontName="Helvetica-Bold", fontSize=13, textColor=BLU,
             spaceBefore=8, spaceAfter=4)
    h2 = PS("h2", fontName="Helvetica-Bold", fontSize=10, textColor=TEA,
             spaceBefore=6, spaceAfter=3)
    bd = PS("bd", fontName="Helvetica", fontSize=9, textColor=DRK, leading=13)
    sm = PS("sm", fontName="Helvetica", fontSize=8, textColor=MID, leading=12)

    def HR(): return HRFlowable(width="100%", thickness=0.5, color=BRD, spaceAfter=3, spaceBefore=3)
    def SP(n=4): return Spacer(1, n)

    def make_table(rows, cw, hdr_color=BLU):
        t = Table(rows, colWidths=cw)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), hdr_color),
            ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRY]),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("LINEBELOW",     (0,0),(-1,-1), 0.3, BRD),
            ("BOX",           (0,0),(-1,-1), 0.5, BRD),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        return t

    story = []
    CW2 = [100*mm, 70*mm]
    CW3 = [70*mm, 55*mm, 45*mm]

    # Title
    story.append(Paragraph("Direct-Fired Water Bath Heater — Heat Transfer Calculation Sheet v3", h1))
    story.append(Paragraph(
        f"Firing: {ft_nps.split('(')[0].strip()} {ft_sch}, {ft_config}  |  "
        f"Coil: {pc_nps.split('(')[0].strip()} {pc_sch}, {N_paths} paths × {N_rows} rows  |  "
        f"Bath: {T_bath:.0f}°C  |  Burner: {Q_burner:.0f} kW", bd))
    story.append(HR())

    # Heat duties
    story.append(Paragraph("Heat duties", h2))
    story.append(make_table([
        ["Parameter", "Value"],
        ["Burner rated duty",                   f"{Q_burner:.0f} kW"],
        ["Thermal efficiency",                  f"{eta_burner} %"],
        ["Heat delivered to water bath",        f"{Q_to_bath_W/1000:.1f} kW"],
        ["Process gas duty (ṁ × Cp × ΔT)",     f"{Q_gas_W/1000:.1f} kW"],
        ["Heat balance (bath − gas)",           f"{(Q_to_bath_W - Q_gas_W)/1000:.1f} kW"],
    ], CW2, BLU))
    story.append(SP())

    # Firing tube
    story.append(Paragraph("Firing tube analysis", h2))
    story.append(make_table([
        ["Parameter", "Value"],
        ["NPS / Schedule / Configuration",      f"{ft_nps.split('(')[0].strip()}  {ft_sch}  |  {ft_config}"],
        ["OD / Wall / ID",                      f"{ft_OD:.1f} / {ft_wall:.2f} / {ft_ID:.1f}  mm"],
        ["U-bend radius (CL)",                  f"{ft_r_bend:.0f} mm"],
        ["Total straight length",               f"{L_ft_straight:.2f} m"],
        ["Straight tube area",                  f"{A_ft_straight:.2f} m²"],
        ["U-bend area",                         f"{A_ft_ubend:.3f} m²"],
        ["Total heat transfer area",            f"{A_ft_total:.2f} m²"],
        ["Flue gas mass flow (derived)",        f"{mdot_fg:.4f} kg/s  ({mdot_fg*3600:.1f} kg/hr)"],
        ["Flue gas velocity",                   f"{v_fg:.2f} m/s"],
        ["Reynolds number",                     f"{Re_fg:,.0f}  —  {dp_fg['regime']}"],
        ["Prandtl number",                      f"{fp_fg['Pr']:.3f}"],
        ["h_fi — flue gas inside",              f"{h_fi:.1f} W/m²·K"],
        ["h_fo — water bath outside",           f"{h_fo:.1f} W/m²·K"],
        ["Overall U_o — firing tube",           f"{U_ft['U_o']:.1f} W/m²·K"],
        ["Dominant resistance",                 f"Flue gas film = {U_ft['pct_i']:.0f}% of total R"],
        ["Heat flux — outer surface",           f"{sk_ft['q']/1000:.2f} kW/m²"],
        ["Outer skin temperature",              f"{T_ft_outer_skin:.1f} °C"],
        ["Inner skin temperature",              f"{T_ft_inner_skin:.1f} °C"],
        ["ΔP — flue gas",                       f"{dp_fg['dP_kPa']:.2f} kPa  /  {dp_fg['dP_mbar']:.1f} mbar"],
    ], CW2, RED))
    story.append(SP())

    # Process coil
    story.append(Paragraph("Process coil analysis", h2))
    story.append(make_table([
        ["Parameter", "Value"],
        ["NPS / Schedule",                      f"{pc_nps.split('(')[0].strip()}  {pc_sch}"],
        ["OD / Wall / ID",                      f"{pc_OD:.1f} / {pc_wall:.2f} / {pc_ID:.1f}  mm"],
        ["Parallel flow paths",                 str(int(N_paths))],
        ["Rows (passes) per path",              str(int(N_rows))],
        ["Total tubes in cross-section",        str(int(N_paths * N_rows))],
        ["U-bend radius (CL)",                  f"{pc_r_bend:.0f} mm"],
        ["Straight tube area",                  f"{A_pc_straight:.2f} m²"],
        ["U-bend area (all bends)",             f"{A_pc_ubend:.3f} m²"],
        ["Total heat transfer area",            f"{A_pc_total:.2f} m²"],
        ["Gas mass flow",                       f"{ng_mdot:.4f} kg/s  ({ng_mdot*3600:.1f} kg/hr)"],
        ["Velocity per tube",                   f"{v_ng:.2f} m/s"],
        ["Reynolds number",                     f"{Re_ng:,.0f}  —  {dp_ng['regime']}"],
        ["h_gi — gas inside coil",              f"{h_gi:.1f} W/m²·K"],
        ["h_wo — water bath outside",           f"{h_wo:.1f} W/m²·K"],
        ["Overall U_o — process coil",          f"{U_pc['U_o']:.1f} W/m²·K"],
        ["LMTD — bath to gas",                  f"{lmtd_val:.1f} °C"],
        ["Required heat transfer area",         f"{A_req:.2f} m²"],
        ["Installed area",                      f"{A_pc_total:.2f} m²"],
        ["Over-surface margin",                 f"{oversurf:+.1f}%"],
        ["Required vessel length",              f"{L_req:.2f} m  (installed: {ves_L/1000:.2f} m)"],
        ["Outer skin temperature",              f"{T_pc_outer_skin:.1f} °C"],
        ["ΔP — natural gas (per path)",         f"{dp_ng['dP_kPa']:.2f} kPa  /  {dp_ng['dP_mbar']:.1f} mbar"],
    ], CW2, colors.HexColor("#1D9E75")))
    story.append(SP())

    # B31.3
    story.append(Paragraph(f"B31.3 process coil wall check — {b313_mat}", h2))
    story.append(make_table([
        ["Parameter", "Value"],
        ["Design pressure",                     f"{P_design:.0f} barg  ({P_design*0.1:.2f} MPa)"],
        ["Design temperature",                  f"{T_design:.0f} °C"],
        ["Allowable stress S",                  f"{b313['S']:.0f} MPa"],
        ["Min calculated thickness t_min",      f"{b313['t_min']:.2f} mm"],
        ["Required incl. 12.5% mill tolerance", f"{b313['t_req']:.2f} mm"],
        ["Selected wall thickness",             f"{b313['t_sel']:.2f} mm"],
        ["Margin",                              f"{b313['margin']:+.2f} mm"],
        ["Result",                              b313["status"]],
    ], CW2, colors.HexColor("#0F6E56") if "PASS" in b313["status"] else colors.HexColor("#A32D2D")))

    story.append(SP(6))
    story.append(HR())
    for msg in pinch_msgs:
        story.append(Paragraph(msg, bd))
    story.append(HR())
    story.append(Paragraph(
        "Note: Fluid properties are polynomial approximations. "
        "Correlations: Gnielinski (Re>3000), Dittus-Boelter (Re≤3000), "
        "Churchill-Chu (natural convection). "
        "Verify against HTRI/HTFS and applicable codes (API 530, ASME B31.3) for final design.", sm))

    doc.build(story)
    buf.seek(0)
    return buf

# ================================================================
# TABBED LAYOUT
# ================================================================

st.markdown("## 🔥 Direct-Fired Water Bath Heater — Heat Transfer Analyser v3")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏭 Overview",
    "🔥 Firing tube",
    "🌀 Process coil",
    "📊 Temperature profiles",
    "⚙️  Sizing & ΔP",
    "📄 PDF report",
])

# ──────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────
with tab1:
    # Pinch warnings
    for msg in pinch_msgs:
        if "❌" in msg:   st.error(msg)
        elif "⚠️" in msg: st.warning(msg)
        else:              st.success(msg)

    st.markdown("---")

    # Row 1 — duties and overall U
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Burner duty",       f"{Q_burner:.0f} kW")
    c2.metric("Heat to bath",      f"{Q_to_bath_W/1000:.0f} kW")
    c3.metric("Gas duty required", f"{Q_gas_W/1000:.1f} kW")
    c4.metric("U — firing tube",   f"{U_ft['U_o']:.0f} W/m²·K",
              help="Overall U referenced to outer area")
    c5.metric("U — process coil",  f"{U_pc['U_o']:.0f} W/m²·K",
              help="Overall U referenced to outer area")
    c6.metric("LMTD (coil)",       f"{lmtd_val:.1f} °C")

    st.markdown("---")

    # Row 2 — geometry and flow
    c7, c8, c9, c10 = st.columns(4)
    c7.metric("Firing tube area",  f"{A_ft_total:.2f} m²",
              delta=f"Straight: {A_ft_straight:.2f}  Bends: {A_ft_ubend:.3f}")
    c8.metric("Process coil area", f"{A_pc_total:.2f} m²",
              delta=f"Straight: {A_pc_straight:.2f}  Bends: {A_pc_ubend:.3f}")
    c9.metric("Flue gas ṁ (derived)", f"{mdot_fg:.3f} kg/s",
              delta=f"Re = {Re_fg:,.0f}  v = {v_fg:.1f} m/s")
    c10.metric("Gas ṁ (input)", f"{ng_mdot:.3f} kg/s",
               delta=f"Re = {Re_ng:,.0f}  v = {v_ng:.1f} m/s")

    st.markdown("---")
    st.subheader("End elevation — tube sheet view")
    fig_sch = draw_end_elevation()
    st.pyplot(fig_sch, use_container_width=False)
    plt.close(fig_sch)

# ──────────────────────────────────────────────────
# TAB 2 — FIRING TUBE
# ──────────────────────────────────────────────────
with tab2:
    st.subheader(f"Firing tube  ·  {ft_nps.split('(')[0].strip()}  {ft_sch}  ·  {ft_config}")

    # Overall U prominent at top
    ua, ub, uc, ud = st.columns(4)
    ua.metric("Overall U_o",        f"{U_ft['U_o']:.1f} W/m²·K")
    ub.metric("h_fi — flue gas",    f"{h_fi:.1f} W/m²·K")
    uc.metric("h_fo — water bath",  f"{h_fo:.1f} W/m²·K")
    ud.metric("Heat flux",          f"{sk_ft['q']/1000:.2f} kW/m²")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Geometry**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["OD / Wall / ID",
                          "Configuration",
                          "Straight passes × vessel length",
                          "Straight tube area",
                          "U-bends × area each",
                          "U-bend area total",
                          "Total heat transfer area",
                          "Cross-section area (flow)"],
            "Value":     [f"{ft_OD:.1f} / {ft_wall:.2f} / {ft_ID:.1f} mm",
                          ft_config,
                          f"{N_ft_passes} × {L_ves_m:.2f} m = {L_ft_straight:.2f} m",
                          f"{A_ft_straight:.3f} m²",
                          f"{N_utubes} × {np.pi**2*ft_OD_m*ft_rb_m:.4f} m²",
                          f"{A_ft_ubend:.4f} m²",
                          f"{A_ft_total:.3f} m²",
                          f"{A_ft_cs*1e4:.2f} cm²"],
        }), hide_index=True, use_container_width=True)

        st.markdown("**Flue gas flow (all derived from burner duty)**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Burner duty → Q_bath",
                          "Derived ṁ_fg",
                          "Density at mean temp",
                          "Velocity (single tube)",
                          "Reynolds number",
                          "Prandtl number",
                          "Flow regime",
                          "Correlation used"],
            "Value":     [f"{Q_burner:.0f} kW × {eta_burner}% = {Q_to_bath_W/1000:.1f} kW",
                          f"{mdot_fg:.4f} kg/s  =  Q/(Cp×ΔT)",
                          f"{fp_fg['rho']:.3f} kg/m³",
                          f"{v_fg:.3f} m/s",
                          f"{Re_fg:,.0f}",
                          f"{fp_fg['Pr']:.3f}",
                          dp_fg["regime"],
                          "Gnielinski" if Re_fg > 3000 else "Dittus-Boelter"],
        }), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Skin temperatures (bath → flue gas)**")
        st.dataframe(pd.DataFrame({
            "Location":         ["Water bath",
                                 "Outer tube skin",
                                 "Inner tube skin",
                                 "Flue gas film"],
            "Temperature (°C)": [f"{T_bath:.1f}",
                                  f"{T_ft_outer_skin:.1f}  (+{sk_ft['dT_o']:.1f})",
                                  f"{T_ft_inner_skin:.1f}  (+{sk_ft['dT_w']:.2f})",
                                  f"{T_ft_film:.1f}  (+{sk_ft['dT_i']:.1f})"],
        }), hide_index=True, use_container_width=True)

        st.markdown("**Thermal resistance breakdown**")
        fig_r, ax_r = plt.subplots(figsize=(5.5, 1.1))
        fig_r.patch.set_facecolor("#F8F8F6")
        ax_r.set_facecolor("#F8F8F6"); ax_r.axis("off")
        vals = [U_ft["pct_i"], U_ft["pct_w"], U_ft["pct_o"]]
        lbls = [f"Flue gas\nfilm {vals[0]:.0f}%",
                f"Wall\n{vals[1]:.0f}%",
                f"Water bath\n{vals[2]:.0f}%"]
        cols = ["#D85A30","#888780","#185FA5"]
        lft = 0
        for v, c, lb in zip(vals, cols, lbls):
            ax_r.barh(0, v, left=lft, color=c, height=0.6)
            if v > 4:
                ax_r.text(lft+v/2, 0, lb, ha="center", va="center",
                          fontsize=8, color="white", fontweight="bold")
            lft += v
        ax_r.set_xlim(0, 100)
        fig_r.tight_layout(pad=0.1)
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

        st.markdown(
            f"**Dominant resistance: Flue gas film ({U_ft['pct_i']:.0f}% of total)**  \n"
            f"This is typical for gas-fired heaters — flue gas convection limits "
            f"heat transfer. Increasing velocity (larger duty or smaller tube ID) "
            f"would raise U."
        )

# ──────────────────────────────────────────────────
# TAB 3 — PROCESS COIL
# ──────────────────────────────────────────────────
with tab3:
    st.subheader(f"Process coil  ·  {pc_nps.split('(')[0].strip()}  {pc_sch}  ·  "
                 f"{int(N_paths)} paths × {int(N_rows)} rows")

    pa, pb, pc_, pd_ = st.columns(4)
    pa.metric("Overall U_o",       f"{U_pc['U_o']:.1f} W/m²·K")
    pb.metric("h_gi — gas",        f"{h_gi:.1f} W/m²·K")
    pc_.metric("h_wo — water bath",f"{h_wo:.1f} W/m²·K")
    pd_.metric("LMTD",             f"{lmtd_val:.1f} °C")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Geometry**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["OD / Wall / ID",
                          "Parallel flow paths",
                          "Rows per path",
                          "Total tubes in bundle",
                          "U-bends per path",
                          "Total U-bends",
                          "Straight tube area",
                          "U-bend area",
                          "Total heat transfer area",
                          "Cross-section per tube"],
            "Value":     [f"{pc_OD:.1f} / {pc_wall:.2f} / {pc_ID:.1f} mm",
                          str(int(N_paths)),
                          str(int(N_rows)),
                          str(int(N_paths * N_rows)),
                          str(int(N_rows - 1)),
                          str(int(N_pc_ubends_total)),
                          f"{A_pc_straight:.3f} m²",
                          f"{A_pc_ubend:.4f} m²",
                          f"{A_pc_total:.3f} m²",
                          f"{A_pc_cs*1e4:.2f} cm²"],
        }), hide_index=True, use_container_width=True)

        st.markdown("**Natural gas flow**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Input flow rate",
                          "= in kg/s",
                          "Density at mean temp",
                          "Velocity per tube",
                          "Reynolds number",
                          "Prandtl number",
                          "Flow regime",
                          "Correlation"],
            "Value":     [f"{ng_flow_val:.3f} {ng_flow_unit}",
                          f"{ng_mdot:.4f} kg/s",
                          f"{fp_ng['rho']:.3f} kg/m³",
                          f"{v_ng:.3f} m/s",
                          f"{Re_ng:,.0f}",
                          f"{fp_ng['Pr']:.3f}",
                          dp_ng["regime"],
                          "Gnielinski" if Re_ng > 3000 else "Dittus-Boelter"],
        }), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Sizing — required vs installed**")
        status_col = "✅" if oversurf >= 0 else "❌"
        st.dataframe(pd.DataFrame({
            "Parameter": ["Gas duty (ṁ×Cp×ΔT)",
                          "U_o × A_installed",
                          "LMTD",
                          "Required area (Q/U·LMTD)",
                          "Installed area",
                          "Over-surface margin",
                          "Required vessel length",
                          "Installed vessel length"],
            "Value":     [f"{Q_gas_W/1000:.2f} kW",
                          f"{U_pc['U_o']*A_pc_total/1000:.2f} kW/K × LMTD",
                          f"{lmtd_val:.1f} °C",
                          f"{A_req:.3f} m²",
                          f"{A_pc_total:.3f} m²",
                          f"{oversurf:+.1f}%  {status_col}",
                          f"{L_req:.2f} m",
                          f"{ves_L/1000:.2f} m"],
        }), hide_index=True, use_container_width=True)

        if oversurf < 0:
            st.error(f"⚠️ Coil is UNDERSURFACED by {abs(oversurf):.1f}%. "
                     f"Increase N_rows, N_paths, or vessel length.")
        elif oversurf < 10:
            st.warning(f"Coil has only {oversurf:.1f}% margin. Consider adding one more row.")
        else:
            st.success(f"Coil has {oversurf:.1f}% design margin — adequate.")

        st.markdown("**B31.3 wall check**")
        b_color = "✅" if "PASS" in b313["status"] else "❌"
        st.dataframe(pd.DataFrame({
            "Parameter": ["Allowable stress S",
                          "Min t_min (calc.)",
                          "Required t (+ mill tol.)",
                          "Selected wall",
                          "Margin",
                          "Status"],
            "Value":     [f"{b313['S']:.0f} MPa",
                          f"{b313['t_min']:.2f} mm",
                          f"{b313['t_req']:.2f} mm",
                          f"{b313['t_sel']:.2f} mm",
                          f"{b313['margin']:+.2f} mm",
                          b313["status"]],
        }), hide_index=True, use_container_width=True)

        st.markdown("**Thermal resistance breakdown**")
        fig_r2, ax_r2 = plt.subplots(figsize=(5.5, 1.1))
        fig_r2.patch.set_facecolor("#F8F8F6")
        ax_r2.set_facecolor("#F8F8F6"); ax_r2.axis("off")
        vals2 = [U_pc["pct_i"], U_pc["pct_w"], U_pc["pct_o"]]
        lbls2 = [f"Gas film\n{vals2[0]:.0f}%",
                 f"Wall\n{vals2[1]:.0f}%",
                 f"Water bath\n{vals2[2]:.0f}%"]
        lft2 = 0
        for v, c, lb in zip(vals2, ["#993C1D","#888780","#185FA5"], lbls2):
            ax_r2.barh(0, v, left=lft2, color=c, height=0.6)
            if v > 4:
                ax_r2.text(lft2+v/2, 0, lb, ha="center", va="center",
                           fontsize=8, color="white", fontweight="bold")
            lft2 += v
        ax_r2.set_xlim(0, 100)
        fig_r2.tight_layout(pad=0.1)
        st.pyplot(fig_r2, use_container_width=True)
        plt.close(fig_r2)

# ──────────────────────────────────────────────────
# TAB 4 — TEMPERATURE PROFILES
# ──────────────────────────────────────────────────
with tab4:
    st.subheader("Temperature profiles — combined view")
    N  = 80
    xf = np.linspace(0, L_ft_straight, N)
    xp = np.linspace(0, L_pc_per_path * N_paths, N)  # total coil path length

    T_fg_prof  = T_fg_in  - (T_fg_in  - T_fg_out) * xf / max(L_ft_straight, 0.001)
    T_ft_o_pr  = T_bath   + sk_ft["dT_o"] * (1 - xf / max(L_ft_straight, 0.001))
    T_ft_i_pr  = T_ft_o_pr + sk_ft["dT_w"]

    T_ng_prof  = T_ng_in  + (T_ng_out - T_ng_in) * xp / max(xp[-1], 0.001)
    T_pc_o_pr  = T_bath   - sk_pc["dT_o"] * xp / max(xp[-1], 0.001)

    fig5, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig5.patch.set_facecolor("#F8F8F6")

    ax = axes[0]
    ax.set_facecolor("#F8F8F6")
    ax.fill_between(xf, T_ft_o_pr, T_ft_i_pr, alpha=0.2, color="#888780")
    ax.plot(xf, T_fg_prof, color="#D85A30", lw=2.5, label="Flue gas bulk")
    ax.plot(xf, T_ft_i_pr, color="#BA7517", lw=1.8, ls="--", label="Inner skin")
    ax.plot(xf, T_ft_o_pr, color="#378ADD", lw=1.8, ls="--", label="Outer skin")
    ax.axhline(T_bath, color="#1D9E75", lw=1.5, ls=":", label=f"Water bath {T_bath:.0f}°C")
    ax.set_xlabel("Total length along passes (m)", fontsize=10)
    ax.set_ylabel("Temperature (°C)", fontsize=10)
    ax.set_title(f"Firing tube — {ft_config}\nU = {U_ft['U_o']:.0f} W/m²·K  |  "
                 f"h_fi = {h_fi:.0f}  |  h_fo = {h_fo:.0f} W/m²·K", fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.8)
    ax.grid(True, alpha=0.15, ls="--")

    ax2 = axes[1]
    ax2.set_facecolor("#F8F8F6")
    ax2.axhline(T_bath, color="#1D9E75", lw=1.5, ls=":", label=f"Water bath {T_bath:.0f}°C")
    ax2.fill_between(xp, T_pc_o_pr, T_bath, alpha=0.1, color="#1D9E75")
    ax2.plot(xp, T_pc_o_pr, color="#378ADD", lw=1.8, ls="--", label="Coil outer skin")
    ax2.plot(xp, T_ng_prof,  color="#993C1D", lw=2.5, label="Natural gas bulk")
    ax2.set_xlabel("Total coil path length (m)", fontsize=10)
    ax2.set_ylabel("Temperature (°C)", fontsize=10)
    ax2.set_title(f"Process coil — {int(N_paths)} paths × {int(N_rows)} rows\n"
                  f"U = {U_pc['U_o']:.0f} W/m²·K  |  "
                  f"h_gi = {h_gi:.0f}  |  h_wo = {h_wo:.0f} W/m²·K", fontsize=10)
    ax2.legend(fontsize=8.5, framealpha=0.8)
    ax2.grid(True, alpha=0.15, ls="--")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)

    # LMTD diagram
    st.markdown("---")
    st.subheader("LMTD diagram — process coil")
    fig6, ax6 = plt.subplots(figsize=(7, 3.2))
    fig6.patch.set_facecolor("#F8F8F6"); ax6.set_facecolor("#F8F8F6")
    ax6.plot([0,1], [T_bath, T_bath], color="#1D9E75", lw=3,
             label=f"Water bath {T_bath:.0f}°C (isothermal)")
    ax6.plot([0,1], [T_ng_in, T_ng_out], color="#993C1D", lw=2.5, label="Natural gas")
    dT1 = T_bath - T_ng_in; dT2 = T_bath - T_ng_out
    ax6.annotate("", xy=(0, T_bath), xytext=(0, T_ng_in),
                 arrowprops=dict(arrowstyle="<->", color="#E24B4A", lw=1.5))
    ax6.annotate("", xy=(1, T_bath), xytext=(1, T_ng_out),
                 arrowprops=dict(arrowstyle="<->", color="#BA7517", lw=1.5))
    ax6.text(0.05, (T_bath+T_ng_in)/2, f" ΔT₁ = {dT1:.1f}°C", color="#E24B4A", fontsize=10)
    ax6.text(0.75, (T_bath+T_ng_out)/2, f"ΔT₂ = {dT2:.1f}°C ", color="#BA7517",
             fontsize=10, ha="right")
    ax6.text(0.5, T_bath - (dT1+dT2)/4,
             f"LMTD = {lmtd_val:.1f}°C", ha="center", fontsize=12,
             color="#185FA5", fontweight="bold")
    ax6.set_xticks([0,1]); ax6.set_xticklabels(["Gas inlet end","Gas outlet end"])
    ax6.set_ylabel("Temperature (°C)", fontsize=10)
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.15, ls="--")
    plt.tight_layout()
    st.pyplot(fig6, use_container_width=True)
    plt.close(fig6)

# ──────────────────────────────────────────────────
# TAB 5 — SIZING & PRESSURE DROP
# ──────────────────────────────────────────────────
with tab5:
    st.subheader("Coil sizing optimiser")
    sa, sb_, sc, sd = st.columns(4)
    sa.metric("Required area",     f"{A_req:.2f} m²")
    sb_.metric("Installed area",   f"{A_pc_total:.2f} m²")
    sc.metric("Over-surface",      f"{oversurf:+.1f}%",
              delta_color="normal" if oversurf >= 0 else "inverse")
    sd.metric("Required length",   f"{L_req:.2f} m",
              delta=f"Installed: {ves_L/1000:.2f} m")

    st.markdown("---")
    st.subheader("Pressure drops")
    dp1, dp2 = st.columns(2)
    with dp1:
        st.markdown("**Flue gas — firing tube (series)**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Velocity","Reynolds No.","Flow regime",
                          "Friction factor","ΔP straight","ΔP U-bends",
                          "Total ΔP","Total ΔP (mbar)"],
            "Value":     [f"{dp_fg['velocity']:.2f} m/s",
                          f"{dp_fg['Re']:,.0f}",
                          dp_fg["regime"],
                          f"{dp_fg['f']:.4f}",
                          f"{dp_fg['dP_straight']/1000:.3f} kPa",
                          f"{dp_fg['dP_bends']/1000:.3f} kPa",
                          f"{dp_fg['dP_kPa']:.3f} kPa",
                          f"{dp_fg['dP_mbar']:.1f} mbar"],
        }), hide_index=True, use_container_width=True)

    with dp2:
        st.markdown(f"**Natural gas — process coil (per path, {int(N_paths)} parallel)**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Velocity per tube","Reynolds No.","Flow regime",
                          "Friction factor","ΔP straight","ΔP U-bends",
                          "ΔP per path","ΔP per path (mbar)"],
            "Value":     [f"{dp_ng['velocity']:.2f} m/s",
                          f"{dp_ng['Re']:,.0f}",
                          dp_ng["regime"],
                          f"{dp_ng['f']:.4f}",
                          f"{dp_ng['dP_straight']/1000:.3f} kPa",
                          f"{dp_ng['dP_bends']/1000:.3f} kPa",
                          f"{dp_ng['dP_kPa']:.3f} kPa",
                          f"{dp_ng['dP_mbar']:.1f} mbar"],
        }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Sensitivity — effect of N_rows on U and duty")
    rows_range = np.arange(1, 25)
    U_sens = []; duty_sens = []; area_sens = []
    for nr in rows_range:
        A_s = N_paths * nr * np.pi * pc_OD_m * L_ves_m
        A_b = N_paths * (nr-1) * np.pi**2 * pc_OD_m * pc_rb_m
        A_t = A_s + A_b
        v_  = ng_mdot / (N_paths * fp_ng["rho"] * A_pc_cs)
        Re_ = fp_ng["rho"] * v_ * pc_ID_m / fp_ng["mu"]
        h_  = (h_gnielinski(Re_, fp_ng["Pr"], fp_ng["k"], pc_ID_m, nr*L_ves_m)
               if Re_ > 3000 else
               h_dittus_boelter(Re_, fp_ng["Pr"], fp_ng["k"], pc_ID_m, True))
        U_  = calc_U(h_, h_wo, k_pc, pc_ID_m, pc_OD_m, foul_ng, foul_wb)["U_o"]
        U_sens.append(U_)
        duty_sens.append(U_ * A_t * lmtd_val / 1000)
        area_sens.append(A_t)

    fig7, axes7 = plt.subplots(1, 3, figsize=(13, 3.5))
    fig7.patch.set_facecolor("#F8F8F6")
    for ax_ in axes7: ax_.set_facecolor("#F8F8F6"); ax_.grid(True, alpha=0.15, ls="--")

    axes7[0].plot(rows_range, U_sens, "#185FA5", lw=2, marker="o", ms=4)
    axes7[0].axvline(N_rows, color="#D85A30", lw=1.5, ls="--", label=f"Current: {int(N_rows)}")
    axes7[0].set_xlabel("Rows per path", fontsize=10); axes7[0].set_ylabel("U (W/m²·K)", fontsize=10)
    axes7[0].set_title("Overall U vs rows", fontsize=10); axes7[0].legend(fontsize=8)

    axes7[1].plot(rows_range, area_sens, "#0F6E56", lw=2, marker="o", ms=4)
    axes7[1].axvline(N_rows, color="#D85A30", lw=1.5, ls="--", label=f"Current: {int(N_rows)}")
    axes7[1].axhline(A_req, color="#993C1D", lw=1.2, ls=":", label=f"Required: {A_req:.1f} m²")
    axes7[1].set_xlabel("Rows per path", fontsize=10); axes7[1].set_ylabel("Total area (m²)", fontsize=10)
    axes7[1].set_title("Total area vs rows", fontsize=10); axes7[1].legend(fontsize=8)

    axes7[2].plot(rows_range, duty_sens, "#BA7517", lw=2, marker="o", ms=4)
    axes7[2].axhline(Q_gas_W/1000, color="#993C1D", lw=1.5, ls=":", label=f"Required: {Q_gas_W/1000:.0f} kW")
    axes7[2].axvline(N_rows, color="#D85A30", lw=1.5, ls="--", label=f"Current: {int(N_rows)}")
    axes7[2].set_xlabel("Rows per path", fontsize=10); axes7[2].set_ylabel("Achievable duty (kW)", fontsize=10)
    axes7[2].set_title("Duty vs rows", fontsize=10); axes7[2].legend(fontsize=8)

    plt.tight_layout(pad=1.2)
    st.pyplot(fig7, use_container_width=True)
    plt.close(fig7)

# ──────────────────────────────────────────────────
# TAB 6 — PDF
# ──────────────────────────────────────────────────
with tab6:
    st.subheader("Download calculation sheet")
    col_dl, col_inf = st.columns([1, 3])
    with col_dl:
        pdf_buf = build_pdf()
        st.download_button(
            "⬇️  Download PDF",
            data=pdf_buf,
            file_name="WBH_HeatTransfer_v3.pdf",
            mime="application/pdf",
        )
    with col_inf:
        st.info(
            f"**{ft_config}  ·  {pc_nps.split('(')[0].strip()} {pc_sch}  ·  "
            f"{int(N_paths)} paths × {int(N_rows)} rows**\n\n"
            f"U firing = {U_ft['U_o']:.0f} W/m²·K  |  "
            f"U coil = {U_pc['U_o']:.0f} W/m²·K  |  "
            f"LMTD = {lmtd_val:.1f}°C  |  "
            f"Over-surface = {oversurf:+.1f}%  |  "
            f"B31.3: {b313['status']}"
        )
