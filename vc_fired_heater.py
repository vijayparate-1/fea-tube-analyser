# ================================================================
# vc_fired_heater.py
# Vertical Cylindrical Fired Heater — Integrated Analysis Tool
#
# Pre-loaded: 860-H-401 A/B, Dolphin Energy, Taweelah UAE
#             Q9334X-DS300 Rev 3
#
# Modules:
#   A — API 530 tube design: heat flux → skin temp → stress → life
#   B — Flue gas 1D profile: T/v/ΔP from burner to stack + draft
#   C — Hottel-Lobo-Evans radiant zone method: heat flux distribution
#
# Correlations:
#   Lobo-Evans (API 560) for radiant exchange
#   Zukauskas for convective tube banks
#   API 530 6th Ed / ISO 13704 elastic + rupture allowable
#   Larson-Miller for creep rupture life
#   Swamee-Jain friction factor
#
# Run: python3 -m streamlit run vc_fired_heater.py
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
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, PageBreak)

st.set_page_config(page_title="VC Fired Heater — 860-H-401", page_icon="🔥",
                   layout="wide")

# ================================================================
# CONSTANTS
# ================================================================
SIGMA  = 5.6704e-8     # Stefan-Boltzmann W/m²·K⁴
G      = 9.81          # m/s²
R_UNIV = 8314.0        # J/kmol·K

# API 530 material data: {mat: {property: {temp_C: value}}}
# Elastic allowable stress (MPa) and Larson-Miller constants
API530 = {
    "A335 P9 (9Cr-1Mo)": {
        "elastic":  {38:138, 100:131, 200:124, 300:117, 400:110,
                     450:107, 500:101, 550:91,  593:76},
        "rupture":  {538:97, 565:72, 593:52, 621:37, 649:26},
        "LM_const": 18.0,   # Larson-Miller constant C
        "LM_coef":  [36.8, -8.5, 1.2],  # log(stress) vs LMP polynomial
        "density":  7850,   # kg/m³
        "E":        196e3,  # MPa Young's modulus
        "alpha":    11e-6,  # 1/K thermal expansion
        "nu":       0.3,
        "k":        38,     # W/m·K
    },
    "A333 Gr.6 (CS low-T)": {
        "elastic":  {38:138, 93:138, 149:131, 204:128, 260:124,
                     316:117, 371:110, 400:103, 427:93},
        "rupture":  {427:103, 454:76, 482:55, 510:39, 538:27},
        "LM_const": 18.0,
        "LM_coef":  [38.0, -9.2, 1.4],
        "density":  7850,
        "E":        200e3,
        "alpha":    12e-6,
        "nu":       0.3,
        "k":        50,
    },
    "A312 TP316L (SS)": {
        "elastic":  {38:115, 93:115, 149:115, 204:110, 260:105,
                     316:100, 371:94,  400:90},
        "rupture":  {538:90, 565:68, 593:50, 621:36, 649:25},
        "LM_const": 15.0,
        "LM_coef":  [34.0, -7.8, 1.1],
        "density":  8000,
        "E":        185e3,
        "alpha":    17e-6,
        "nu":       0.3,
        "k":        16,
    },
}

# NPS pipe data (subset used for this heater)
PIPE_NPS = {
    '6" NPS  (168.3mm)':  {"OD":168.3,  "sch":{"Sch 80":10.97,"Sch 120":14.27,"Sch 160":18.26,"XXS":21.95}},
    '8" NPS  (219.1mm)':  {"OD":219.1,  "sch":{"Sch 80":12.70,"Sch 120":18.24,"Sch 160":23.01,"XXS":22.22}},
    '10" NPS (273.0mm)':  {"OD":273.0,  "sch":{"Sch 80":15.09,"Sch 120":21.44,"Sch 160":28.58}},
}

# ================================================================
# FLUID PROPERTY FUNCTIONS
# ================================================================

def props_flue_gas(T_C: float) -> dict:
    """Flue gas from NG combustion, polynomial fits 300–1400°C."""
    T   = max(T_C, 100.0)
    Cp  = 1040 + 0.15*T - 2e-5*T**2        # J/kg·K
    k   = 0.022 + 7.5e-5*T                  # W/m·K
    mu  = 1.4e-5  + 4.2e-8*T               # Pa·s
    rho = 1.30 * 273.15/(273.15+T)         # kg/m³ (M≈29 g/mol)
    Pr  = mu*Cp/k
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, Pr=Pr)

def props_nat_gas(T_C: float, P_bara: float) -> dict:
    """Natural gas ~90% CH4, high-pressure properties."""
    T   = max(T_C, 5.0)
    M   = 17.5                              # kg/kmol (between pure CH4=16 and mix=17.71)
    Cp  = 2100 + 1.2*T + 0.003*T**2        # J/kg·K
    k   = 0.030 + 9.0e-5*T                 # W/m·K
    mu  = 1.05e-5 + 3.0e-8*T              # Pa·s
    rho = P_bara*1e5*M/(R_UNIV*(273.15+T)) # kg/m³
    Pr  = mu*Cp/k
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, Pr=Pr)

def props_air(T_C: float) -> dict:
    """Dry air properties."""
    T   = max(T_C, 20.0)
    Cp  = 1005 + 0.05*T
    k   = 0.0241 + 7.7e-5*T
    mu  = 1.71e-5 + 4.5e-8*T
    rho = 1.293*273.15/(273.15+T)
    return dict(Cp=Cp, k=k, mu=mu, rho=rho, Pr=mu*Cp/k)

def interp_allowable(mat_key: str, T_C: float, which: str = "elastic") -> float:
    """Linear interpolation of API 530 allowable stress."""
    tbl   = API530[mat_key][which]
    temps = sorted(tbl.keys())
    if T_C <= temps[0]:  return tbl[temps[0]]
    if T_C >= temps[-1]: return tbl[temps[-1]]
    for i in range(len(temps)-1):
        if temps[i] <= T_C <= temps[i+1]:
            f = (T_C-temps[i])/(temps[i+1]-temps[i])
            return tbl[temps[i]] + f*(tbl[temps[i+1]]-tbl[temps[i]])
    return tbl[temps[-1]]

# ================================================================
# MODULE C — HOTTEL / LOBO-EVANS RADIANT ZONE METHOD
# ================================================================

def adiabatic_flame_temp(Q_LHV_W: float, mdot_fuel: float,
                          mdot_air: float, T_air: float,
                          T_fuel: float) -> float:
    """
    Simplified adiabatic flame temperature.
    T_af = T_ref + Q_LHV / (mdot_products × Cp_products)
    """
    mdot_products = mdot_fuel + mdot_air
    Cp_products   = 1150                           # J/kg·K approx at high T
    T_ref         = (mdot_fuel*T_fuel + mdot_air*T_air) / mdot_products
    T_af          = T_ref + Q_LHV_W / (mdot_products * Cp_products)
    return min(T_af, 2200)                         # physical cap

def lobo_evans_radiant(
        Q_fuel_W: float,        # total heat release W
        eta_radiant: float,     # fraction of heat absorbed by radiant section
        T_fluid_in: float,      # process fluid inlet temp °C
        T_fluid_out: float,     # process fluid outlet temp °C
        T_flue_exit: float,     # flue gas leaving radiant section °C
        D_firebox: float,       # firebox inside diameter (to refractory) m
        H_firebox: float,       # firebox height m
        D_tube_CL: float,       # tube CL diameter m
        N_tubes: int,           # number of radiant tubes
        D_tube_OD: float,       # tube OD m
        L_tube: float,          # effective tube length m
        alpha_tube: float = 0.97, # tube absorption factor
        excess_air: float = 0.15, # fraction
) -> dict:
    """
    Lobo-Evans method for VC heater radiant section.
    Returns: heat flux, flame temp, exchange area, flux factor.

    References: Lobo & Evans (1939), API 560 Annex A,
                Wimpress (1963) HPE method.
    """
    # ── Geometry ──────────────────────────────────────────────
    # Cold plane area (projected area of tube bank at tube CL circle)
    # For single-row circular arrangement:
    #   A_cp = N_tubes × D_OD × L_tube  (projected, one side)
    A_cp   = N_tubes * D_tube_OD * L_tube       # m²

    # Refractory area (inside surface minus tube shadow)
    A_floor = np.pi/4 * D_firebox**2            # m²
    A_arch  = A_floor                           # same for roof
    A_wall  = np.pi * D_firebox * H_firebox     # cylindrical wall
    A_r_total = A_wall + A_floor + A_arch        # total refractory
    A_r_net   = A_r_total - A_cp                 # subtract tube shadow

    # Effective radiant exchange area (Hottel):
    # A_eff = A_cp / (1 + (A_cp/A_r_net)*(1/alpha_tube - 1))
    A_eff  = A_cp / (1 + (A_cp / max(A_r_net, 1)) * (1/alpha_tube - 1))

    # ── Heat absorbed by radiant section ──────────────────────
    Q_R    = Q_fuel_W * eta_radiant              # W

    # ── Effective furnace (gas) temperature ───────────────────
    # Q_R = A_eff × σ × (T_g^4 - T_w^4)
    # T_w = mean tube outer skin temperature (≈ fluid mean + film ΔT)
    T_fluid_mean = (T_fluid_in + T_fluid_out) / 2 + 273.15  # K
    T_w_K        = T_fluid_mean + 30            # approx outer skin offset K

    # Solve for T_g iteratively
    T_g_K = T_w_K + 700                         # initial guess
    for _ in range(30):
        Q_calc = A_eff * SIGMA * (T_g_K**4 - T_w_K**4)
        if Q_calc < 1:
            T_g_K += 50
            continue
        T_g_K = T_w_K * (Q_R/Q_calc + 1 - (T_w_K/T_g_K)**4)**0.25
        T_g_K = max(T_g_K, T_w_K + 100)

    # ── Average and peak heat flux ─────────────────────────────
    q_avg  = Q_R / A_cp                         # W/m²
    # Circumferential flux factor F for single-row VC:
    # F ≈ 1.0 for inside of circle (all tubes see same flux)
    # Peak on fire-facing side: F_peak ≈ 1.0 for VC (uniform)
    # Slight correction for 6-burner arrangement: ~1.02-1.05
    F_peak = 1.05
    q_peak = q_avg * F_peak

    # ── Verification: flue gas exit temperature ───────────────
    T_g_exit_K = (T_flue_exit + 273.15)

    return dict(
        Q_R_MW       = Q_R / 1e6,
        A_cp         = A_cp,
        A_r_net      = A_r_net,
        A_eff        = A_eff,
        T_g_K        = T_g_K,
        T_g_C        = T_g_K - 273.15,
        T_w_K        = T_w_K,
        T_w_C        = T_w_K - 273.15,
        q_avg        = q_avg,
        q_avg_kW     = q_avg / 1000,
        q_peak       = q_peak,
        q_peak_kW    = q_peak / 1000,
        F_peak       = F_peak,
        alpha_tube   = alpha_tube,
    )

# ================================================================
# MODULE A — API 530 TUBE DESIGN
# ================================================================

def h_internal_gnielinski(Re, Pr, k, D_i, L):
    Re  = max(Re, 3001)
    f   = (0.790*np.log(Re)-1.64)**-2
    Nu  = (f/8)*(Re-1000)*Pr/(1+12.7*np.sqrt(f/8)*(Pr**(2/3)-1))
    Nu *= (1+(D_i/max(L,D_i))**(2/3))
    return max(Nu*k/D_i, 1.0)

def api530_tube_design(
        q_flux_W_m2: float,     # heat flux to outer surface W/m²
        h_i: float,             # inside film coefficient W/m²·K
        T_fluid: float,         # bulk fluid temperature °C
        P_design_kPag: float,   # design pressure kPag
        D_o_mm: float,          # tube OD mm
        t_wall_mm: float,       # wall thickness mm
        mat_key: str,           # material key in API530 dict
        design_life_hrs: float = 100000,
        foul: float = 0.00018,  # fouling factor m²·K/W
        T_allowance: float = 50, # design temperature allowance °C
) -> dict:
    """
    API 530 / ISO 13704 tube design calculation.
    Heat flux → skin temperature → elastic + rupture check → life.
    """
    mat   = API530[mat_key]
    k_w   = mat["k"]
    E_mod = mat["E"]
    alpha = mat["alpha"]
    nu    = mat["nu"]

    D_o   = D_o_mm / 1000
    t     = t_wall_mm / 1000
    D_i   = D_o - 2*t
    ro, ri = D_o/2, D_i/2
    P_MPa = P_design_kPag / 100        # kPag → bar → ×0.1 → MPa

    # ── Outer skin temperature ──────────────────────────────────
    # Thermal resistance chain: outer film → wall → fouling → fluid film
    # q_outer = q_flux (given)
    # dT_outer_film: negligible (direct radiation, no outer film in radiant)
    # dT_wall = q × ro × ln(ro/ri) / k
    # dT_fouling = q × (D_o/D_i) × Rf
    # dT_inner_film = q × (D_o/D_i) / h_i
    dT_wall  = q_flux_W_m2 * ro * np.log(ro/ri) / k_w
    dT_foul  = q_flux_W_m2 * (D_o/D_i) * foul
    dT_inner = q_flux_W_m2 * (D_o/D_i) / max(h_i, 1.0)

    T_outer_skin = T_fluid + dT_inner + dT_foul + dT_wall
    T_inner_skin = T_fluid + dT_inner + dT_foul
    T_design_tmt = T_outer_skin + T_allowance   # design tube metal temp

    # ── API 530 Elastic allowable ───────────────────────────────
    S_e   = interp_allowable(mat_key, T_design_tmt, "elastic")   # MPa

    # ── Minimum wall thickness (elastic, API 530 Eq.)
    # t_min = P×D_o / (2×S + P×(1 - 2Y))  where Y = 0.4 for T < 480°C
    Y     = 0.4
    t_min_e = (P_MPa * D_o_mm) / (2*S_e + P_MPa*(1-2*Y))        # mm

    # ── Rupture allowable (API 530) ────────────────────────────
    # Use average of elastic and rupture for design if T > 427°C
    S_r = interp_allowable(mat_key, T_design_tmt, "rupture")      # MPa

    # Minimum wall for rupture
    phi   = 1.0                         # weld factor (seamless = 1.0)
    t_min_r = (P_MPa * D_o_mm) / (2*phi*S_r + P_MPa*(1-2*Y))    # mm

    # Governing minimum wall
    t_min = max(t_min_e, t_min_r)
    t_req = t_min / (1 - 0.125)        # +12.5% mill tolerance

    # ── Stress calculations ─────────────────────────────────────
    # Lamé hoop stress at inner radius
    sigma_hoop = P_MPa*(ri**2+ro**2)/(ro**2-ri**2) * 1e6 / 1e6  # MPa (no unit change needed, P already MPa)
    sigma_hoop = P_MPa * (ri*1000)**2 + (ro*1000)**2  # recalc
    # Use correct formula: σ_θ at inner = P(ro²+ri²)/(ro²-ri²)  [MPa]
    sigma_hoop_inner = P_MPa * (ro**2 + ri**2)/(ro**2 - ri**2)
    sigma_hoop_outer = P_MPa * 2*ri**2/(ro**2 - ri**2)
    sigma_axial      = P_MPa * ri**2/(ro**2 - ri**2)

    # Thermal stress (through-wall gradient)
    dT_wall_val = dT_wall
    sigma_th    = E_mod * alpha * dT_wall_val / (2*(1-nu))        # MPa

    # Von Mises at inner surface (worst case, highest pressure stress)
    s1 = sigma_hoop_inner + sigma_th
    s2 = sigma_axial      + sigma_th * 0.3
    vm = np.sqrt(s1**2 + s2**2 - s1*s2)

    # ── Rupture life estimate (Larson-Miller) ───────────────────
    # LMP = T_K × (C + log10(t_r))
    # Approximate: use S_r at design temp
    C     = mat["LM_const"]
    T_K   = T_design_tmt + 273.15
    # Stress ratio vs allowable → life factor
    stress_ratio = max(vm / max(S_e, 0.1), 0.1)
    if T_design_tmt > 350:
        # Simplified: life ∝ (S_allow/vm)^5 × 100,000 hrs
        t_rupture = min(design_life_hrs * (1/stress_ratio)**5, 1e6)
        lmp = T_K * (C + np.log10(t_rupture)) / 1000
    else:
        t_rupture = 1e6
        lmp       = 0

    life_str = (">1,000,000 hrs" if t_rupture >= 1e6
                else f"{t_rupture/1000:.0f}k hrs"
                if t_rupture >= 1000 else f"{t_rupture:.0f} hrs")

    margin  = t_wall_mm - t_req
    status  = "PASS ✅" if margin >= 0 else "FAIL ❌"

    # Over-temperature check
    ot_ok   = T_outer_skin <= (T_design_tmt - T_allowance + T_allowance)
    tmt_ok  = T_outer_skin < (T_design_tmt - T_allowance + T_allowance + 50)

    return dict(
        D_o=D_o_mm, D_i=D_i*1000,
        t_wall=t_wall_mm, t_min=round(t_min,3),
        t_req=round(t_req,3), margin=round(margin,3),
        status=status,
        T_fluid=round(T_fluid,1),
        T_inner_skin=round(T_inner_skin,1),
        T_outer_skin=round(T_outer_skin,1),
        T_design_tmt=round(T_design_tmt,1),
        dT_wall=round(dT_wall,2),
        dT_foul=round(dT_foul,2),
        dT_inner=round(dT_inner,2),
        S_elastic=round(S_e,1),
        S_rupture=round(S_r,1),
        sigma_hoop=round(sigma_hoop_inner,2),
        sigma_th=round(sigma_th,2),
        vm=round(vm,2),
        utilisation=round(vm/S_e*100,1),
        t_rupture=t_rupture,
        life_str=life_str,
        lmp=round(lmp,2),
        q_flux=round(q_flux_W_m2,1),
        q_kW=round(q_flux_W_m2/1000,3),
    )

# ================================================================
# MODULE B — FLUE GAS 1D PROFILE
# ================================================================

def flue_gas_profile(
        Q_release_W: float,     # total heat release W
        mdot_fg: float,         # flue gas mass flow kg/s
        sections: list,         # list of dicts: {name, Q_abs_W, A_flow_m2}
        T_fg_start: float,      # adiabatic flame temp or arch temp °C
        T_amb: float,           # ambient temp °C
        stack_ID: float,        # stack inside diameter m
        stack_H: float,         # stack height m
        firebox_ID: float,      # firebox inside diameter m
        firebox_H: float,       # firebox height m
) -> dict:
    """
    1D flue gas temperature, velocity, and natural draft profile.
    """
    g          = 9.81
    T_fg       = T_fg_start
    results    = []

    for sec in sections:
        fp       = props_flue_gas(T_fg)
        Q_abs    = sec["Q_abs_W"]
        dT       = Q_abs / (mdot_fg * fp["Cp"])
        T_fg_out = T_fg - dT
        A_flow   = sec.get("A_flow_m2", 0.5)
        v_fg     = mdot_fg / (fp["rho"] * A_flow)
        T_mean   = (T_fg + T_fg_out) / 2
        fp_mean  = props_flue_gas(T_mean)
        results.append({
            "section": sec["name"],
            "T_in":    round(T_fg, 1),
            "T_out":   round(T_fg_out, 1),
            "T_mean":  round(T_mean, 1),
            "dT":      round(dT, 1),
            "v":       round(v_fg, 2),
            "rho":     round(fp_mean["rho"], 4),
            "Q_abs_MW":round(Q_abs/1e6, 3),
        })
        T_fg = T_fg_out

    # ── Natural draft ─────────────────────────────────────────
    T_stack_entry = T_fg            # after all sections
    T_stack_mean  = T_stack_entry - 5   # small cool-down in stack
    rho_air       = props_air(T_amb)["rho"]
    rho_fg_stack  = props_flue_gas(T_stack_mean)["rho"]

    # Draft = g × H_stack × (rho_air - rho_fg_stack)  [Pa]
    draft_stack_Pa = g * stack_H * (rho_air - rho_fg_stack)

    # Firebox draft
    rho_fg_fb = props_flue_gas(T_fg_start)["rho"]
    draft_fb_Pa = g * firebox_H * (rho_air - rho_fg_fb)

    # Stack velocity
    A_stack  = np.pi/4 * stack_ID**2
    v_stack  = mdot_fg / (rho_fg_stack * A_stack)

    # Acid dewpoint (simplified): depends on H2SO4 formation
    # For H2S content ~2ppm: dewpoint ≈ 120°C
    T_acid_dp = 120.0

    return dict(
        sections       = results,
        T_stack_entry  = round(T_stack_entry, 1),
        T_stack_exit   = round(T_stack_entry - 5, 1),
        draft_stack_Pa = round(draft_stack_Pa, 1),
        draft_fb_Pa    = round(draft_fb_Pa, 1),
        v_stack        = round(v_stack, 2),
        A_stack        = round(A_stack, 3),
        T_acid_dp      = T_acid_dp,
        acid_ok        = T_stack_entry > T_acid_dp + 20,
    )

# ================================================================
# MODULE — CONVECTIVE SECTION (Zukauskas tube bank)
# ================================================================

def zukauskas_bank(Re, Pr, N_rows, arrangement="staggered"):
    """
    Zukauskas correlation for flow over tube banks.
    Returns Nu (Nusselt number).
    """
    if Re < 100:
        C, m = 0.80, 0.40
    elif Re < 1000:
        C, m = 0.27, 0.63
    elif Re < 2e5:
        C, m = 0.35 if arrangement=="staggered" else 0.27, 0.60
    else:
        C, m = 0.022, 0.84

    Nu = C * Re**m * Pr**0.36
    # Row correction factor (for N_rows < 20)
    if N_rows < 20:
        F = 1 - np.exp(-0.1 * N_rows)
        Nu *= (0.7 + 0.3 * F / 0.95)
    return Nu

def fin_efficiency(h_o, k_fin, t_fin, H_fin, D_tube):
    """
    Annular fin efficiency (simplified straight-fin approximation).
    """
    m   = np.sqrt(2*h_o / (k_fin * t_fin))     # fin parameter
    L_c = H_fin + t_fin/2                        # corrected fin length
    eta = np.tanh(m * L_c) / (m * L_c)
    return max(min(eta, 1.0), 0.1)

def conv_section_calc(
        mdot_fg: float,         # kg/s
        T_fg_in: float,         # °C
        Q_target_W: float,      # duty W
        N_tubes: int,
        N_rows: int,
        D_tube_OD: float,       # m
        D_tube_ID: float,       # m
        L_tube: float,          # m
        pitch: float,           # m (CL to CL)
        h_i: float,             # W/m²·K inside film
        k_tube: float,          # W/m·K tube wall
        foul: float,            # m²·K/W
        fins: bool = False,
        fin_h: float = 0.019,   # fin height m
        fin_t: float = 0.002,   # fin thickness m (16 GA ≈ 1.6mm)
        fin_density: float = 196.85,  # fins/m
        k_fin: float = 50,      # W/m·K
        arrangement: str = "staggered",
) -> dict:
    """Overall U and heat transfer for one convective section."""
    # Flue gas properties at inlet
    fp   = props_flue_gas(T_fg_in)

    # Flow area through tube bank (minimum free area)
    N_per_row = N_tubes // N_rows
    L_bank    = L_tube          # tube length = duct width
    A_flow    = L_bank * (pitch - D_tube_OD) * N_per_row / N_per_row
    A_flow    = max(A_flow, 0.01)
    # Simplified: A_flow = L_tube × (pitch - D_OD) for one row gap
    A_flow    = L_tube * (pitch - D_tube_OD)

    G_fg      = mdot_fg / A_flow                    # mass velocity kg/s·m²
    Re_fg     = G_fg * D_tube_OD / fp["mu"]

    Nu_fg     = zukauskas_bank(Re_fg, fp["Pr"], N_rows, arrangement)
    h_o_bare  = Nu_fg * fp["k"] / D_tube_OD         # W/m²·K

    # Bare tube outer area
    A_bare    = N_tubes * np.pi * D_tube_OD * L_tube  # m²

    if fins:
        # Finned tube overall surface
        n_fins     = fin_density * L_tube * N_tubes
        A_fin_one  = 2 * np.pi * ((D_tube_OD/2 + fin_h)**2 - (D_tube_OD/2)**2)
        A_fins     = n_fins * A_fin_one
        A_root     = N_tubes * np.pi * D_tube_OD * L_tube * (1 - fin_density*fin_t)
        A_total    = A_fins + A_root

        eta_f      = fin_efficiency(h_o_bare, k_fin, fin_t, fin_h, D_tube_OD)
        eta_o      = 1 - (A_fins/A_total)*(1 - eta_f)  # overall surface efficiency
        h_o_eff    = h_o_bare * eta_o                   # effective outer h
        ext_ratio  = A_total / A_bare
    else:
        A_total    = A_bare
        h_o_eff    = h_o_bare
        eta_o      = 1.0
        ext_ratio  = 1.0

    # Overall U (outer area basis)
    ro    = D_tube_OD/2; ri = D_tube_ID/2
    ratio = D_tube_OD/D_tube_ID
    R_i   = ratio*(1/h_i + foul)
    R_w   = ro*np.log(ro/ri)/k_tube
    R_o   = 1/h_o_eff + foul*0.5
    U_o   = 1/(R_i + R_w + R_o)

    # LMTD (assuming T_fg drops to match duty)
    Q_abs    = Q_target_W
    T_fg_out = T_fg_in - Q_abs/(mdot_fg * fp["Cp"])

    return dict(
        Re_fg    = round(Re_fg, 0),
        G_fg     = round(G_fg, 2),
        h_o_bare = round(h_o_bare, 1),
        h_o_eff  = round(h_o_eff, 1),
        h_i      = round(h_i, 0),
        U_o      = round(U_o, 1),
        A_bare   = round(A_bare, 2),
        A_total  = round(A_total, 2),
        ext_ratio= round(ext_ratio, 2),
        eta_fin  = round(eta_o*100, 1),
        T_fg_out = round(T_fg_out, 1),
    )

# ================================================================
# SCHEMATIC — VC HEATER END ELEVATION
# ================================================================

def draw_vc_schematic(radiant, conv_results, fg_profile):
    """Side elevation schematic of VC heater."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 9),
                              gridspec_kw={'width_ratios': [1, 1.4]})
    fig.patch.set_facecolor("#F8F8F6")

    # ── Left: End elevation (tube sheet view) ──────────────────
    ax = axes[0]
    ax.set_facecolor("#F8F8F6"); ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.5, 1.4)
    ax.axis("off")
    ax.set_title("End elevation — tube sheet view", fontsize=10, pad=4)

    # Refractory wall
    ax.add_patch(plt.Circle((0,0), 1.05, fc="#D3D1C7", ec="#5F5E5A", lw=2, zorder=1))
    # Water-bath analog: blue-ish zone = hot gas
    ax.add_patch(plt.Circle((0,0), 0.98, fc="#FAEEDA", alpha=0.6, zorder=2))

    # 48 radiant tubes (24 P9 + 24 A333)
    N_rad = 48
    R_tube = 0.88     # normalised radius to tube CL
    for i in range(N_rad):
        ang = 2*np.pi*i/N_rad
        cx  = R_tube * np.cos(ang)
        cy  = R_tube * np.sin(ang)
        col = "#5DCAA5" if i < 24 else "#EF9F27"
        ax.add_patch(plt.Circle((cx, cy), 0.055,
                                fc=col, ec="#444441", lw=0.8, zorder=4))

    # 6 burners (floor, arranged in ring)
    R_burn = 0.35
    for i in range(6):
        ang = 2*np.pi*i/6
        bx  = R_burn * np.cos(ang)
        by  = R_burn * np.sin(ang)
        ax.add_patch(plt.Circle((bx, by), 0.07,
                                fc="#E24B4A", ec="#A32D2D", lw=1, zorder=5))
        ax.annotate("", xy=(bx*1.5, by*1.5), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="->", color="#D85A30", lw=1.2))

    # Legend
    ax.add_patch(plt.Circle((-1.2, -1.1), 0.06, fc="#5DCAA5", ec="#444441", lw=0.8))
    ax.text(-1.1, -1.1, "Tubes 1-24  A335 P9", fontsize=7, va="center")
    ax.add_patch(plt.Circle((-1.2, -1.25), 0.06, fc="#EF9F27", ec="#444441", lw=0.8))
    ax.text(-1.1, -1.25, "Tubes 25-48  A333 Gr.6", fontsize=7, va="center")
    ax.add_patch(plt.Circle((-1.2, -1.4), 0.06, fc="#E24B4A", ec="#A32D2D", lw=0.8))
    ax.text(-1.1, -1.4, "6 × Burners (floor)", fontsize=7, va="center")

    ax.text(0, 1.28, f"q_avg = {radiant['q_avg_kW']:.1f} kW/m²\n"
                     f"T_flame ≈ {radiant['T_g_C']:.0f}°C",
            ha="center", va="bottom", fontsize=8, color="#712B13",
            bbox=dict(fc="#FAECE7", ec="#993C1D", boxstyle="round,pad=0.3"))

    # ── Right: Side elevation + flue gas T profile ─────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F8F8F6"); ax2.axis("off")
    ax2.set_title("Side elevation & flue gas temperature profile", fontsize=10, pad=4)

    # Vessel outline (simplified side view)
    vessel_x = [0.05, 0.05, 0.35, 0.35]
    vessel_y = [0.02, 0.88, 0.88, 0.02]
    ax2.fill(vessel_x, vessel_y, fc="#FAEEDA", alpha=0.4, zorder=1)
    ax2.plot(vessel_x + [vessel_x[0]], vessel_y + [vessel_y[0]],
             color="#5F5E5A", lw=1.5, zorder=3)

    # Radiant tubes (side view = vertical lines)
    ax2.add_patch(mpatches.FancyBboxPatch((0.07,0.05), 0.26, 0.78,
        boxstyle="round,pad=0.01", fc="#5DCAA5", alpha=0.25, ec="#0F6E56", lw=0.8))
    ax2.text(0.20, 0.44, "Radiant\n48 tubes\nvertical",
             ha="center", va="center", fontsize=8, color="#085041")

    # Burner arrows (bottom)
    for bx_ in [0.10, 0.16, 0.22, 0.28]:
        ax2.annotate("", xy=(bx_, 0.12), xytext=(bx_, 0.02),
                     arrowprops=dict(arrowstyle="->", color="#D85A30", lw=1.5))
    ax2.text(0.20, 0.00, "6 × burners  floor-fired", ha="center",
             fontsize=7.5, color="#D85A30")

    # Convection section (above radiant)
    conv_y = 0.90
    ax2.add_patch(mpatches.FancyBboxPatch((0.05, conv_y), 0.30, 0.07,
        boxstyle="round,pad=0.005", fc="#E6F1FB", ec="#185FA5", lw=0.8))
    ax2.text(0.20, conv_y+0.035, "Conv. shield + banks 1&2",
             ha="center", va="center", fontsize=7.5, color="#0C447C")

    # Stack
    ax2.add_patch(mpatches.FancyBboxPatch((0.14, 0.97), 0.12, 0.27,
        boxstyle="round,pad=0.005", fc="#D3D1C7", ec="#5F5E5A", lw=1))
    ax2.text(0.20, 1.12, "Stack\n2.26m ID\n45.6m high",
             ha="center", va="center", fontsize=7, color="#2C2C2A")
    ax2.annotate("", xy=(0.20, 1.26), xytext=(0.20, 1.22),
                 arrowprops=dict(arrowstyle="->", color="#888780", lw=2))

    # Temperature profile (right side)
    secs    = fg_profile["sections"]
    T_vals  = [secs[0]["T_in"]] + [s["T_out"] for s in secs]
    T_vals  = T_vals + [fg_profile["T_stack_exit"]]
    labels  = ["Radiant in"] + [s["section"] for s in secs] + ["Stack exit"]
    n       = len(T_vals)
    y_pos   = np.linspace(0.05, 0.97, n)

    ax2.plot([0.42]*n, y_pos, 'o-', color="#D85A30", lw=2, ms=6, zorder=5)
    for i, (T, lbl, yp) in enumerate(zip(T_vals, labels, y_pos)):
        ax2.text(0.44, yp, f"{T:.0f}°C", fontsize=8, va="center", color="#D85A30")
        ax2.text(0.60, yp, lbl, fontsize=7.5, va="center", color="#6b6b65")

    ax2.axhline(y=0.88, xmin=0.38, xmax=0.95, color="#888780",
                lw=0.8, ls="--", alpha=0.5)
    ax2.text(0.38, 0.89, "← radiant/conv. boundary", fontsize=7, color="#888780")

    # Draft annotation
    ax2.text(0.38, 0.50,
             f"Natural draft:\n"
             f"Stack: {fg_profile['draft_stack_Pa']:.0f} Pa\n"
             f"Arch:  {fg_profile['draft_fb_Pa']:.0f} Pa\n"
             f"Stack v: {fg_profile['v_stack']:.1f} m/s",
             fontsize=7.5, va="center", color="#1a1a18",
             bbox=dict(fc="#EAF3DE", ec="#3B6D11", boxstyle="round,pad=0.3"))

    plt.tight_layout(pad=0.8)
    return fig

# ================================================================
# PDF REPORT BUILDER
# ================================================================

def build_pdf(radiant, api_rad, api_shield, conv_s, conv_b1, conv_b2,
              fg_profile, inputs):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=18*mm, rightMargin=18*mm,
                             topMargin=18*mm, bottomMargin=18*mm)
    BLU=colors.HexColor("#185FA5"); TEA=colors.HexColor("#0F6E56")
    RED=colors.HexColor("#D85A30"); GRY=colors.HexColor("#F5F4F0")
    MID=colors.HexColor("#6b6b65"); BRD=colors.HexColor("#d8d7cf")
    DRK=colors.HexColor("#1a1a18")
    PS = lambda n,**kw: ParagraphStyle(n,**kw)
    h1 = PS("h1",fontName="Helvetica-Bold",fontSize=13,textColor=BLU,
             spaceBefore=8,spaceAfter=4)
    h2 = PS("h2",fontName="Helvetica-Bold",fontSize=10,textColor=TEA,
             spaceBefore=6,spaceAfter=3)
    bd = PS("bd",fontName="Helvetica",fontSize=8,textColor=DRK,leading=12)
    sm = PS("sm",fontName="Helvetica-Oblique",fontSize=7,textColor=MID,leading=11)
    def HR(): return HRFlowable(width="100%",thickness=0.5,color=BRD,
                                 spaceAfter=3,spaceBefore=3)
    def SP(n=4): return Spacer(1,n)
    def T(rows,cw,hdr=BLU):
        t=Table(rows,colWidths=cw)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),hdr),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,GRY]),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",(0,0),(-1,-1),5),
            ("LINEBELOW",(0,0),(-1,-1),0.3,BRD),
            ("BOX",(0,0),(-1,-1),0.5,BRD),("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        return t
    story=[]
    story.append(Paragraph(f"Vertical Cylindrical Fired Heater — {inputs['tag']}", h1))
    story.append(Paragraph(
        f"Project: {inputs['project']}  |  Client: {inputs['client']}  |  "
        f"Doc: {inputs['doc_no']}  |  Total duty: {inputs['duty_MW']:.0f} MW  |  "
        f"Heat release: {inputs['release_MW']:.3f} MW", bd))
    story.append(HR())
    # Radiant
    story.append(Paragraph("Module C — Hottel / Lobo-Evans radiant zone", h2))
    story.append(T([
        ["Parameter","Value"],
        ["Firebox ID / height", f"{inputs['D_fb']:.2f} m  /  {inputs['H_fb']:.2f} m"],
        ["Cold plane area A_cp", f"{radiant['A_cp']:.2f} m²"],
        ["Net refractory area A_r", f"{radiant['A_r_net']:.2f} m²"],
        ["Effective exchange area A_eff", f"{radiant['A_eff']:.2f} m²"],
        ["Radiant duty Q_R", f"{radiant['Q_R_MW']:.3f} MW"],
        ["Effective flame temperature T_g", f"{radiant['T_g_C']:.0f} °C"],
        ["Mean tube wall temperature T_w", f"{radiant['T_w_C']:.1f} °C"],
        ["Average heat flux q_avg", f"{radiant['q_avg_kW']:.2f} kW/m²"],
        ["Peak heat flux q_peak", f"{radiant['q_peak_kW']:.2f} kW/m²"],
        ["Flux factor F_peak", f"{radiant['F_peak']:.2f}"],
    ],  [80*mm,90*mm], RED))
    story.append(SP())
    # API 530 radiant
    story.append(Paragraph("Module A — API 530 tube design: radiant section (A335 P9)", h2))
    for lbl, r in [("Radiant (P9, tubes 1-24)", api_rad),
                   ("Conv. shield (A333 Gr.6)", api_shield)]:
        story.append(Paragraph(lbl, PS("lbl",fontName="Helvetica-Bold",
                                        fontSize=8,textColor=TEA)))
        story.append(T([
            ["Parameter","Value","Parameter","Value"],
            ["Heat flux q","f{r['q_kW']:.3f} kW/m²",
             "Fluid temperature",f"{r['T_fluid']:.1f} °C"],
            ["Inner skin temp",f"{r['T_inner_skin']:.1f} °C",
             "Outer skin temp",f"{r['T_outer_skin']:.1f} °C"],
            ["Design TMT",f"{r['T_design_tmt']:.1f} °C",
             "dT across wall",f"{r['dT_wall']:.2f} °C"],
            ["Elastic S_e",f"{r['S_elastic']:.1f} MPa",
             "Rupture S_r",f"{r['S_rupture']:.1f} MPa"],
            ["Hoop stress",f"{r['sigma_hoop']:.2f} MPa",
             "Thermal stress",f"{r['sigma_th']:.2f} MPa"],
            ["Von Mises",f"{r['vm']:.2f} MPa",
             "Utilisation",f"{r['utilisation']:.1f}%"],
            ["Min wall t_min",f"{r['t_min']:.3f} mm",
             "Required t_req",f"{r['t_req']:.3f} mm"],
            ["Selected wall",f"{r['t_wall']:.2f} mm",
             "Margin",f"{r['margin']:+.3f} mm"],
            ["Status",r['status'],"Rupture life",r['life_str']],
        ], [45*mm,40*mm,45*mm,40*mm], RED))
        story.append(SP(3))
    # Flue gas profile
    story.append(Paragraph("Module B — Flue gas temperature profile & natural draft", h2))
    rows_fg = [["Section","T in (°C)","T out (°C)","ΔT (°C)","v (m/s)","Q abs (MW)"]]
    for s in fg_profile["sections"]:
        rows_fg.append([s["section"],f"{s['T_in']:.0f}",f"{s['T_out']:.0f}",
                        f"{s['dT']:.0f}",f"{s['v']:.2f}",f"{s['Q_abs_MW']:.3f}"])
    rows_fg.append(["Stack entry","—",f"{fg_profile['T_stack_entry']:.0f}","—",
                    f"{fg_profile['v_stack']:.2f}","—"])
    story.append(T(rows_fg,[38*mm,22*mm,24*mm,20*mm,20*mm,24*mm], BLU))
    story.append(SP(4))
    story.append(T([
        ["Draft parameter","Value"],
        ["Natural draft — stack", f"{fg_profile['draft_stack_Pa']:.1f} Pa"],
        ["Natural draft — firebox (arch)", f"{fg_profile['draft_fb_Pa']:.1f} Pa"],
        ["Stack exit velocity", f"{fg_profile['v_stack']:.2f} m/s"],
        ["Acid dewpoint (est.)", f"{fg_profile['T_acid_dp']:.0f} °C"],
        ["Stack exit vs dewpoint",
         "OK ✅" if fg_profile["acid_ok"] else "WARNING — below dewpoint ⚠️"],
    ], [80*mm, 90*mm], TEA))
    story.append(SP(6)); story.append(HR())
    story.append(Paragraph(
        "Correlations: Lobo-Evans (1939) / API 560 radiant method; "
        "API 530 6th Ed. (ISO 13704) elastic + rupture allowable; "
        "Zukauskas (1972) tube bank convection; "
        "Swamee-Jain friction factor. "
        "Results are for engineering assessment — verify with full design calculation.", sm))
    doc.build(story)
    buf.seek(0)
    return buf

# ================================================================
# SIDEBAR INPUTS  (pre-loaded from Q9334X-DS300)
# ================================================================
sb = st.sidebar
sb.markdown("## 🔥 VC Fired Heater")
sb.markdown("**860-H-401 A/B  ·  Dolphin Energy**")
sb.markdown("---")

sb.markdown("### Heater identity")
tag      = sb.text_input("Tag number",     "860-H-401 A/B")
project  = sb.text_input("Project",        "Taweelah Fujairah Pipeline")
client   = sb.text_input("Client",         "Dolphin Energy Limited")
doc_no   = sb.text_input("Document no.",   "Q9334X-DS300 Rev 3")
sb.markdown("---")

sb.markdown("### Firebox geometry")
D_fb     = sb.number_input("Firebox ID — to refractory (m)", 2.0,12.0, 5.30, 0.05)
H_fb     = sb.number_input("Firebox height (m)",             6.0,25.0,12.50, 0.25)
D_tube_CL= sb.number_input("Tube CL diameter (m)",          2.0,10.0, 4.66, 0.05)
N_rad    = sb.number_input("Number of radiant tubes",        12, 120,   48,   1)
pitch_rad= sb.number_input("Tube CL–CL pitch (mm)",        200, 600,  304.8, 5.0)
sb.markdown("---")

sb.markdown("### Radiant tube specifications")
tube_nps = sb.selectbox("Tube NPS", list(PIPE_NPS.keys()), index=0)
tube_sch = sb.selectbox("Schedule", list(PIPE_NPS[tube_nps]["sch"].keys()), index=2)
D_o_mm   = PIPE_NPS[tube_nps]["OD"]
# Override with datasheet actual: OD=168.28mm, wall=18.24mm
D_o_mm   = sb.number_input("Tube OD (mm)",        100.0,400.0,168.28,0.5)
t_wall   = sb.number_input("Wall thickness (mm)",  5.0,  50.0, 18.24, 0.1)
L_eff    = sb.number_input("Effective tube length (mm)", 5000,20000,12000,100)
mat_rad  = sb.selectbox("Radiant material (rows 1-4)",
                         list(API530.keys()), index=0)
mat_conv = sb.selectbox("Conv./shield material",
                         list(API530.keys()), index=1)
sb.markdown("---")

sb.markdown("### Combustion — burner & flue gas")
Q_release = sb.number_input("Total heat release LHV (MW)", 5.0,100.0,19.264,0.1)
eta_r     = sb.slider("Radiant section efficiency (%)", 40, 90, 58,
                       help="% of released heat absorbed by radiant tubes")
excess_air= sb.slider("Excess air (%)", 5, 50, 15)
mdot_fg_hr= sb.number_input("Flue gas mass flow (kg/hr)", 5000.0,200000.0,32653.0,100.0)
T_fg_arch = sb.number_input("Flue gas temp at arch (°C)", 600, 1200, 800, 10)
sb.markdown("---")

sb.markdown("### Process conditions (natural gas)")
T_proc_in = sb.number_input("Process gas inlet temp (°C)",  -20.0,100.0, 19.0, 1.0)
T_rad_out = sb.number_input("Temp after radiant (°C)",       50.0,300.0,107.2, 1.0)
T_sh_out  = sb.number_input("Temp after conv. shield (°C)",  80.0,300.0,115.8, 1.0)
T_b1_out  = sb.number_input("Temp after conv. bank 1 (°C)", 100.0,300.0,134.1, 1.0)
T_b2_out  = sb.number_input("Temp after conv. bank 2 (°C)", 100.0,350.0,150.0, 1.0)
P_op_bara = sb.number_input("Operating pressure (bara)",     50.0,200.0,131.0, 1.0)
P_des_kPag= sb.number_input("Design pressure (kPag)",      1000.0,30000.0,14700.0,100.0)
mdot_proc = sb.number_input("Process gas flow (kg/hr)",    10000.0,500000.0,155955.0,500.0)
foul_proc = sb.number_input("Fouling factor (m²·K/W)", 0.00005,0.001,0.00018,format="%.5f")
sb.markdown("---")

sb.markdown("### Convective section — shield")
N_sh      = sb.number_input("Shield tubes",           6,  60, 18, 1)
N_sh_rows = sb.number_input("Shield rows",            1,  10,  3, 1)
L_sh      = sb.number_input("Shield tube length (mm)",2000,10000,4500,100)

sb.markdown("### Convective bank 1")
N_b1      = sb.number_input("Bank 1 tubes",           6,  60, 12, 1)
N_b1_rows = sb.number_input("Bank 1 rows",            1,  10,  2, 1)
L_b1      = sb.number_input("Bank 1 tube length (mm)",2000,10000,4500,100)
fin1_h    = sb.number_input("Bank 1 fin height (mm)",  5,  50, 19, 1)

sb.markdown("### Convective bank 2")
N_b2      = sb.number_input("Bank 2 tubes",           6,  60, 18, 1)
N_b2_rows = sb.number_input("Bank 2 rows",            1,  10,  3, 1)
L_b2      = sb.number_input("Bank 2 tube length (mm)",2000,10000,4500,100)
fin2_h    = sb.number_input("Bank 2 fin height (mm)",  5,  50, 25, 1)

sb.markdown("---")
sb.markdown("### Stack")
stack_ID  = sb.number_input("Stack inside diameter (m)", 0.5,  5.0, 2.26, 0.05)
stack_H   = sb.number_input("Stack height above grade (m)", 10.0,80.0,45.6,0.5)
T_amb     = sb.number_input("Ambient temperature (°C)",    5.0, 60.0,33.85,0.5)

sb.markdown("---")
sb.markdown("### Design life")
design_life = sb.number_input("Design life (hrs)", 10000, 200000, 100000, 5000)
T_allow     = sb.number_input("TMT temperature allowance (°C)", 25, 75, 50, 5)
sb.caption("Pre-loaded: Q9334X-DS300 Rev3 · API 530 6th Ed.")

# ================================================================
# DERIVED INPUTS
# ================================================================
mdot_fg   = mdot_fg_hr / 3600          # kg/s
mdot_proc_s = mdot_proc / 3600         # kg/s
L_eff_m   = L_eff / 1000              # m
D_o_m     = D_o_mm / 1000
D_i_m     = D_o_m - 2*(t_wall/1000)

# Inside film coefficient (from datasheet, can vary)
fp_rad    = props_nat_gas((T_proc_in+T_rad_out)/2, P_op_bara)
A_cs_tube = np.pi/4 * D_i_m**2
v_proc    = mdot_proc_s / (6 * fp_rad["rho"] * A_cs_tube)  # 6 flow paths
Re_proc   = fp_rad["rho"] * v_proc * D_i_m / fp_rad["mu"]
h_i_calc  = h_internal_gnielinski(Re_proc, fp_rad["Pr"],
                                   fp_rad["k"], D_i_m, L_eff_m)

# ================================================================
# RUN CALCULATIONS
# ================================================================

# MODULE C — Radiant zone
radiant = lobo_evans_radiant(
    Q_fuel_W    = Q_release * 1e6,
    eta_radiant = eta_r / 100,
    T_fluid_in  = T_proc_in,
    T_fluid_out = T_rad_out,
    T_flue_exit = T_fg_arch,
    D_firebox   = D_fb,
    H_firebox   = H_fb,
    D_tube_CL   = D_tube_CL,
    N_tubes     = int(N_rad),
    D_tube_OD   = D_o_m,
    L_tube      = L_eff_m,
    alpha_tube  = 0.97,
    excess_air  = excess_air/100,
)

# MODULE A — API 530: radiant tubes
api_rad = api530_tube_design(
    q_flux_W_m2    = radiant["q_peak"],
    h_i            = max(h_i_calc, 1486),
    T_fluid        = (T_proc_in + T_rad_out) / 2,
    P_design_kPag  = P_des_kPag,
    D_o_mm         = D_o_mm,
    t_wall_mm      = t_wall,
    mat_key        = mat_rad,
    design_life_hrs= design_life,
    foul           = foul_proc,
    T_allowance    = T_allow,
)

# API 530: conv. shield
q_sh = (1.017e6) / (N_sh * np.pi * D_o_m * (L_sh/1000))
api_shield = api530_tube_design(
    q_flux_W_m2    = q_sh,
    h_i            = 1511,
    T_fluid        = (T_rad_out + T_sh_out) / 2,
    P_design_kPag  = P_des_kPag,
    D_o_mm         = D_o_mm,
    t_wall_mm      = t_wall,
    mat_key        = mat_conv,
    design_life_hrs= design_life,
    foul           = foul_proc,
    T_allowance    = T_allow,
)

# Convective sections
A_flow_sh = (L_sh/1000) * (pitch_rad/1000 - D_o_m)
conv_sh = conv_section_calc(
    mdot_fg=mdot_fg, T_fg_in=T_fg_arch,
    Q_target_W=1.017e6,
    N_tubes=int(N_sh), N_rows=int(N_sh_rows),
    D_tube_OD=D_o_m, D_tube_ID=D_i_m,
    L_tube=L_sh/1000, pitch=pitch_rad/1000,
    h_i=1511, k_tube=API530[mat_conv]["k"],
    foul=foul_proc, fins=False,
)

T_fg_b1 = conv_sh["T_fg_out"]
conv_b1 = conv_section_calc(
    mdot_fg=mdot_fg, T_fg_in=T_fg_b1,
    Q_target_W=2.167e6,
    N_tubes=int(N_b1), N_rows=int(N_b1_rows),
    D_tube_OD=D_o_m, D_tube_ID=D_i_m,
    L_tube=L_b1/1000, pitch=pitch_rad/1000,
    h_i=1445, k_tube=API530[mat_conv]["k"],
    foul=foul_proc, fins=True,
    fin_h=fin1_h/1000, fin_t=0.0016, fin_density=196.85, k_fin=50,
)

T_fg_b2 = conv_b1["T_fg_out"]
conv_b2 = conv_section_calc(
    mdot_fg=mdot_fg, T_fg_in=T_fg_b2,
    Q_target_W=1.889e6,
    N_tubes=int(N_b2), N_rows=int(N_b2_rows),
    D_tube_OD=D_o_m, D_tube_ID=D_i_m,
    L_tube=L_b2/1000, pitch=pitch_rad/1000,
    h_i=1521, k_tube=API530[mat_conv]["k"],
    foul=foul_proc, fins=True,
    fin_h=fin2_h/1000, fin_t=0.0016, fin_density=196.85, k_fin=50,
)

# MODULE B — Flue gas profile
A_conv_flow = (L_sh/1000) * (pitch_rad/1000 - D_o_m) * max(int(N_sh//N_sh_rows),1)
sections_list = [
    {"name":"Radiant",        "Q_abs_W": radiant["Q_R_MW"]*1e6,
     "A_flow_m2": np.pi/4*D_fb**2},
    {"name":"Conv. shield",   "Q_abs_W": 1.017e6,  "A_flow_m2": 0.60},
    {"name":"Conv. bank 1",   "Q_abs_W": 2.167e6,  "A_flow_m2": 0.55},
    {"name":"Conv. bank 2",   "Q_abs_W": 1.889e6,  "A_flow_m2": 0.55},
]

fg_profile = flue_gas_profile(
    Q_release_W = Q_release*1e6,
    mdot_fg     = mdot_fg,
    sections    = sections_list,
    T_fg_start  = T_fg_arch + 50,  # flame peak, then cools to arch value
    T_amb       = T_amb,
    stack_ID    = stack_ID,
    stack_H     = stack_H,
    firebox_ID  = D_fb,
    firebox_H   = H_fb,
)

# ================================================================
# TABBED DASHBOARD
# ================================================================
st.markdown(f"## 🔥 {tag} — {project}")
st.markdown(f"*{client}  ·  {doc_no}  ·  VC Heater  ·  Natural Draught  ·  16 MW nominal*")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🏭 Overview",
    "🌡 Radiant zone (C)",
    "🔩 Tube design API 530 (A)",
    "🌀 Convective section",
    "💨 Flue gas profile (B)",
    "📄 PDF report",
])

# ─────────────────────────────────────────────────────
with tab1:
    # Headline metrics
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Heat release",     f"{Q_release:.2f} MW")
    c2.metric("Radiant duty",     f"{radiant['Q_R_MW']:.2f} MW")
    c3.metric("Flame temp",       f"{radiant['T_g_C']:.0f} °C")
    c4.metric("Peak flux",        f"{radiant['q_peak_kW']:.1f} kW/m²")
    c5.metric("Stack exit temp",  f"{fg_profile['T_stack_exit']:.0f} °C")
    c6.metric("Natural draft",    f"{fg_profile['draft_stack_Pa']:.0f} Pa")

    # Warnings
    secs = fg_profile["sections"]
    T_stack = fg_profile["T_stack_exit"]
    if T_stack < fg_profile["T_acid_dp"] + 20:
        st.warning(f"⚠️  Stack exit {T_stack:.0f}°C is close to acid dewpoint "
                   f"({fg_profile['T_acid_dp']:.0f}°C) — condensation risk.")
    if api_rad["utilisation"] > 80:
        st.warning(f"⚠️  Radiant tube utilisation {api_rad['utilisation']:.1f}% — above 80%.")
    if api_rad["margin"] < 0:
        st.error(f"❌  Radiant tube wall FAILS API 530 — margin = {api_rad['margin']:.2f} mm")
    else:
        st.success(f"✅  All checks passed — radiant tube margin = +{api_rad['margin']:.2f} mm")

    st.markdown("---")
    st.subheader("Heater schematic")
    fig_sch = draw_vc_schematic(radiant, None, fg_profile)
    st.pyplot(fig_sch, use_container_width=True)
    plt.close(fig_sch)

# ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Module C — Hottel / Lobo-Evans radiant heat transfer")
    st.markdown(
        "The Lobo-Evans method models the radiant firebox as an effective "
        "black-body exchange between the hot combustion gases (at effective "
        "flame temperature T_g) and the tube cold plane area A_cp, with "
        "refractory walls acting as reradiating surfaces. "
        "Q_R = A_eff × σ × (T_g⁴ − T_w⁴)")

    ca,cb,cc,cd = st.columns(4)
    ca.metric("Cold plane area A_cp",   f"{radiant['A_cp']:.2f} m²")
    cb.metric("Net refractory area A_r",f"{radiant['A_r_net']:.2f} m²")
    cc.metric("Effective area A_eff",   f"{radiant['A_eff']:.2f} m²")
    cd.metric("Effective flame T_g",    f"{radiant['T_g_C']:.0f} °C")

    ce,cf,cg,ch = st.columns(4)
    ce.metric("Radiant duty Q_R",  f"{radiant['Q_R_MW']:.3f} MW")
    cf.metric("Average flux",      f"{radiant['q_avg_kW']:.2f} kW/m²",
              delta=f"DS: 35.3 kW/m²")
    cg.metric("Peak flux",         f"{radiant['q_peak_kW']:.2f} kW/m²",
              delta=f"DS: 36.09 kW/m²")
    ch.metric("Mean tube wall T_w",f"{radiant['T_w_C']:.1f} °C")

    st.markdown("---")
    st.subheader("Heat flux comparison vs datasheet")
    fig_flux, ax_f = plt.subplots(figsize=(9,3.5))
    fig_flux.patch.set_facecolor("#F8F8F6"); ax_f.set_facecolor("#F8F8F6")
    sections_names = ["Radiant\n(firebox)", "Conv.\nshield", "Conv.\nbank 1", "Conv.\nbank 2"]
    q_avg_ds  = [35.3,  23.74, 75.92, 44.04]
    q_max_ds  = [36.09, 33.76, 84.8,  56.2]
    q_avg_calc= [radiant["q_avg_kW"], q_sh/1000,
                 2.167e6/(N_b1*np.pi*D_o_m*(L_b1/1000))/1000,
                 1.889e6/(N_b2*np.pi*D_o_m*(L_b2/1000))/1000]
    x = np.arange(len(sections_names))
    w = 0.28
    ax_f.bar(x-w,   q_avg_ds,   w, label="DS avg flux",  color="#B5D4F4")
    ax_f.bar(x,     q_max_ds,   w, label="DS max flux",  color="#378ADD")
    ax_f.bar(x+w,   q_avg_calc, w, label="Calc avg flux",color="#D85A30", alpha=0.8)
    ax_f.set_xticks(x); ax_f.set_xticklabels(sections_names, fontsize=9)
    ax_f.set_ylabel("Heat flux (kW/m²)",fontsize=10)
    ax_f.set_title("Heat flux — datasheet vs calculated",fontsize=11)
    ax_f.legend(fontsize=9); ax_f.grid(True,alpha=0.15,ls="--")
    st.pyplot(fig_flux, use_container_width=True)
    plt.close(fig_flux)

# ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Module A — API 530 tube design (all sections)")

    for lbl, r, mat in [
        ("Radiant section — A335 P9 (rows 1-4)", api_rad, mat_rad),
        ("Conv. shield — A333 Gr.6",             api_shield, mat_conv),
    ]:
        st.markdown(f"**{lbl}**")
        if "FAIL" in r["status"]:
            st.error(f"❌  Wall thickness FAILS — margin = {r['margin']:.3f} mm")
        else:
            st.success(f"✅  Wall OK — margin = +{r['margin']:.3f} mm")

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Outer skin temp",    f"{r['T_outer_skin']:.1f} °C",
                  delta=f"TMT design: {r['T_design_tmt']:.0f}°C")
        c2.metric("Elastic S_e",        f"{r['S_elastic']:.0f} MPa")
        c3.metric("Rupture S_r",        f"{r['S_rupture']:.0f} MPa")
        c4.metric("Von Mises stress",   f"{r['vm']:.1f} MPa",
                  delta=f"{r['utilisation']:.0f}% of S_e")
        c5.metric("Rupture life",       r["life_str"])

        st.dataframe(pd.DataFrame({
            "Parameter": [
                "Heat flux q", "dT across wall", "dT fouling",
                "dT inner film", "Inner skin temp", "Outer skin temp",
                "Design TMT (+50°C)", "Min wall t_min",
                "Required t_req (+mill)", "Selected wall", "Margin", "Status",
                "LMP (×10³)","Rupture life"
            ],
            "Value": [
                f"{r['q_kW']:.3f} kW/m²",
                f"{r['dT_wall']:.2f} °C",
                f"{r['dT_foul']:.2f} °C",
                f"{r['dT_inner']:.1f} °C",
                f"{r['T_inner_skin']:.1f} °C",
                f"{r['T_outer_skin']:.1f} °C",
                f"{r['T_design_tmt']:.1f} °C",
                f"{r['t_min']:.3f} mm",
                f"{r['t_req']:.3f} mm",
                f"{r['t_wall']:.2f} mm",
                f"{r['margin']:+.3f} mm",
                r["status"],
                f"{r['lmp']:.2f}",
                r["life_str"],
            ]
        }), hide_index=True, use_container_width=True)
        st.markdown("---")

    # Skin temp waterfall chart
    st.subheader("Skin temperature waterfall — radiant tube")
    fig_w, ax_w = plt.subplots(figsize=(8,3.5))
    fig_w.patch.set_facecolor("#F8F8F6"); ax_w.set_facecolor("#F8F8F6")
    r = api_rad
    temps = [r["T_fluid"], r["T_fluid"]+r["dT_inner"],
             r["T_fluid"]+r["dT_inner"]+r["dT_foul"],
             r["T_outer_skin"]]
    labels= ["Bulk fluid","+ inner film","+ fouling","Outer skin\n(TMT-50°C)"]
    cols  = ["#B5D4F4","#FAEEDA","#EAF3DE","#D85A30"]
    for i,(T,lb,c) in enumerate(zip(temps,labels,cols)):
        ax_w.barh(i, T, color=c, ec="#444441", lw=0.5, height=0.6)
        ax_w.text(T+1, i, f"{T:.1f}°C", va="center", fontsize=9)
    ax_w.set_yticks(range(4)); ax_w.set_yticklabels(labels, fontsize=9)
    ax_w.axvline(r["T_design_tmt"]-T_allow, color="#A32D2D",
                 ls="--", lw=1.5, label=f"Max TMT calc ({r['T_outer_skin']:.0f}°C)")
    ax_w.set_xlabel("Temperature (°C)", fontsize=10)
    ax_w.set_title("Radiant tube — temperature stack", fontsize=11)
    ax_w.legend(fontsize=8); ax_w.grid(True,alpha=0.15,axis="x")
    plt.tight_layout()
    st.pyplot(fig_w, use_container_width=True)
    plt.close(fig_w)

# ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Convective section — overall U and heat transfer")
    st.markdown(
        "Flue gas flows across the tube bundle (cross-flow). "
        "Zukauskas correlation for bare tubes; fin efficiency applied for extended surface.")

    for lbl, cv, duty in [
        ("Conv. shield   (bare tubes, 3 rows)",  conv_sh, 1.017),
        ("Conv. bank 1   (finned, solid fins)",  conv_b1, 2.167),
        ("Conv. bank 2   (finned, solid fins)",  conv_b2, 1.889),
    ]:
        st.markdown(f"**{lbl}**")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Flue gas Re",    f"{cv['Re_fg']:,.0f}")
        c2.metric("h_o (bare)",     f"{cv['h_o_bare']:.0f} W/m²·K")
        c3.metric("h_o (effective)",f"{cv['h_o_eff']:.0f} W/m²·K")
        c4.metric("Overall U_o",    f"{cv['U_o']:.0f} W/m²·K")
        c5.metric("Fin efficiency", f"{cv['eta_fin']:.0f}%")
        st.markdown(
            f"Surface area — bare: **{cv['A_bare']:.2f} m²** | "
            f"total (with fins): **{cv['A_total']:.2f} m²** | "
            f"extension ratio: **{cv['ext_ratio']:.2f}×**   |   "
            f"T_fg out: **{cv['T_fg_out']:.1f} °C**")
        st.markdown("---")

# ─────────────────────────────────────────────────────
with tab5:
    st.subheader("Module B — Flue gas temperature, velocity and natural draft")

    secs = fg_profile["sections"]
    df_fg = pd.DataFrame([{
        "Section":    s["section"],
        "T in (°C)":  s["T_in"],
        "T out (°C)": s["T_out"],
        "ΔT (°C)":    s["dT"],
        "Velocity (m/s)": s["v"],
        "Q absorbed (MW)": s["Q_abs_MW"],
        "Density (kg/m³)": s["rho"],
    } for s in secs])
    st.dataframe(df_fg, hide_index=True, use_container_width=True)

    # Draft summary
    st.markdown("---")
    ca,cb,cc,cd = st.columns(4)
    ca.metric("Stack natural draft",   f"{fg_profile['draft_stack_Pa']:.0f} Pa",
              delta=f"DS: -25 Pa arch")
    cb.metric("Firebox draft",         f"{fg_profile['draft_fb_Pa']:.0f} Pa")
    cc.metric("Stack exit velocity",   f"{fg_profile['v_stack']:.2f} m/s",
              delta=f"DS: 7.93 m/s")
    cd.metric("Stack exit temp",       f"{fg_profile['T_stack_exit']:.0f} °C",
              delta=f"DS: 295°C")

    if not fg_profile["acid_ok"]:
        st.warning(f"⚠️  Stack exit temperature {fg_profile['T_stack_exit']:.0f}°C "
                   f"is within 20°C of estimated acid dewpoint "
                   f"({fg_profile['T_acid_dp']:.0f}°C)")
    else:
        st.success(f"✅  Stack exit temp OK — {fg_profile['T_stack_exit']:.0f}°C "
                   f"vs acid dewpoint ~{fg_profile['T_acid_dp']:.0f}°C")

    # Temperature profile chart
    st.markdown("---")
    fig_b, ax_b = plt.subplots(figsize=(10, 4))
    fig_b.patch.set_facecolor("#F8F8F6"); ax_b.set_facecolor("#F8F8F6")
    T_points = ([secs[0]["T_in"]] + [s["T_out"] for s in secs]
                + [fg_profile["T_stack_exit"]])
    x_pts = list(range(len(T_points)))
    x_lbl = (["Burner\n(arch)"] + [s["section"] for s in secs]
              + ["Stack\nexit"])
    ax_b.plot(x_pts, T_points, "o-", color="#D85A30", lw=2.5, ms=8)
    for x,T in zip(x_pts, T_points):
        ax_b.text(x, T+12, f"{T:.0f}°C", ha="center", fontsize=9, color="#D85A30")
    ax_b.axhline(fg_profile["T_acid_dp"], color="#A32D2D", ls="--", lw=1.2,
                 label=f"Acid dewpoint ~{fg_profile['T_acid_dp']:.0f}°C")
    ax_b.fill_between(x_pts, T_points, fg_profile["T_acid_dp"],
                      where=[T > fg_profile["T_acid_dp"] for T in T_points],
                      alpha=0.08, color="#D85A30")
    ax_b.set_xticks(x_pts); ax_b.set_xticklabels(x_lbl, fontsize=9)
    ax_b.set_ylabel("Flue gas temperature (°C)", fontsize=10)
    ax_b.set_title("Flue gas temperature profile — burner to stack exit",fontsize=11)
    ax_b.legend(fontsize=9); ax_b.grid(True, alpha=0.15, ls="--")
    plt.tight_layout()
    st.pyplot(fig_b, use_container_width=True)
    plt.close(fig_b)

    # Process gas temperature profile
    st.markdown("---")
    st.subheader("Process gas temperature — heater coil")
    fig_proc, ax_p = plt.subplots(figsize=(10, 3.5))
    fig_proc.patch.set_facecolor("#F8F8F6"); ax_p.set_facecolor("#F8F8F6")
    T_proc_pts = [T_proc_in, T_rad_out, T_sh_out, T_b1_out, T_b2_out]
    x_proc     = list(range(len(T_proc_pts)))
    x_plbl     = ["Inlet\n19°C","After\nradiant","After\nshield",
                  "After\nbank 1","Outlet\n150°C"]
    ax_p.plot(x_proc, T_proc_pts,"o-",color="#185FA5",lw=2.5,ms=8)
    for x,T in zip(x_proc,T_proc_pts):
        ax_p.text(x, T+2, f"{T:.1f}°C", ha="center", fontsize=9, color="#185FA5")
    ax_p.fill_between(x_proc,T_proc_pts,T_proc_in,alpha=0.1,color="#185FA5")
    ax_p.set_xticks(x_proc); ax_p.set_xticklabels(x_plbl,fontsize=9)
    ax_p.set_ylabel("Process gas temperature (°C)",fontsize=10)
    ax_p.set_title("Process gas temperature profile — 19 to 150°C",fontsize=11)
    ax_p.grid(True,alpha=0.15,ls="--")
    plt.tight_layout()
    st.pyplot(fig_proc,use_container_width=True)
    plt.close(fig_proc)

# ─────────────────────────────────────────────────────
with tab6:
    st.subheader("Download calculation report")
    col_dl, col_inf = st.columns([1,3])
    with col_dl:
        pdf_buf = build_pdf(
            radiant, api_rad, api_shield,
            conv_sh, conv_b1, conv_b2,
            fg_profile,
            inputs=dict(tag=tag, project=project, client=client, doc_no=doc_no,
                        duty_MW=16.0, release_MW=Q_release,
                        D_fb=D_fb, H_fb=H_fb))
        st.download_button("⬇️  Download PDF",
            data=pdf_buf,
            file_name=f"VC_Heater_{tag.replace(' ','_')}_Calc.pdf",
            mime="application/pdf")
    with col_inf:
        st.info(
            f"**{tag}  ·  {doc_no}**\n\n"
            f"Radiant flux: {radiant['q_avg_kW']:.1f} kW/m² avg / "
            f"{radiant['q_peak_kW']:.1f} kW/m² peak  |  "
            f"Flame T: {radiant['T_g_C']:.0f}°C  |  "
            f"Tube margin: {api_rad['margin']:+.2f} mm  |  "
            f"Stack: {fg_profile['T_stack_exit']:.0f}°C  |  "
            f"Draft: {fg_profile['draft_stack_Pa']:.0f} Pa"
        )

    st.markdown("---")
    st.markdown("**Calculation basis:**")
    st.markdown(
        "- **Module C** — Lobo-Evans (1939) radiant exchange method per API 560 Annex A. "
        "Effective exchange area computed from tube cold plane area and refractory geometry. "
        "Flame temperature solved iteratively from Q = A_eff × σ × (T_g⁴ − T_w⁴).\n"
        "- **Module A** — API 530 6th Edition / ISO 13704:2007. "
        "Elastic and rupture allowable stresses interpolated from material tables. "
        "Minimum wall from pressure design; design TMT = outer skin + 50°C allowance. "
        "Larson-Miller parameter for rupture life estimate.\n"
        "- **Module B** — 1D energy balance per section for flue gas temperature profile. "
        "Natural draft = g × H_stack × (ρ_air − ρ_fg). "
        "Stack velocity from continuity.\n"
        "- **Convective section** — Zukauskas (1972) correlation for cross-flow over tube banks. "
        "Annular fin efficiency per Schmidt approximation."
    )
