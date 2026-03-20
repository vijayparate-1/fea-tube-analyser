# =============================================================
# main.py  —  Flask backend for Combustion Tube FEA Analyser
# Paste this into Replit (main.py) then click Run.
# =============================================================

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# ─── Material library ──────────────────────────────────────────────────────────
# Each material has:
#   E     — Young's modulus (MPa)
#   alpha — thermal expansion coefficient (1/°C)
#   nu    — Poisson's ratio
#   k     — thermal conductivity (W/m·°C)
#   allow — function: allowable stress (MPa) at a given temperature (°C)
#            simplified step-wise approximation of published design curves

MATERIALS = {
    "cs": {
        "name": "Carbon steel A106",
        "E": 200_000, "alpha": 12e-6, "nu": 0.3, "k": 48,
        "allow": lambda T: (
            138 if T < 450 else 124 if T < 480 else 110 if T < 510 else
            90  if T < 540 else 70  if T < 570 else 45
        ),
    },
    "p91": {
        "name": "9Cr-1Mo P91",
        "E": 196_000, "alpha": 11e-6, "nu": 0.3, "k": 28,
        "allow": lambda T: (
            138 if T < 550 else 117 if T < 580 else 94 if T < 610 else
            72  if T < 640 else 52  if T < 670 else 35 if T < 700 else 20
        ),
    },
    "hp": {
        "name": "HP alloy 25Cr-35Ni",
        "E": 170_000, "alpha": 16e-6, "nu": 0.3, "k": 18,
        "allow": lambda T: (
            80 if T < 750 else 55 if T < 800 else 35 if T < 850 else
            18 if T < 900 else 10 if T < 950 else 6
        ),
    },
    "347h": {
        "name": "347H stainless",
        "E": 185_000, "alpha": 17e-6, "nu": 0.3, "k": 16,
        "allow": lambda T: (
            130 if T < 600 else 100 if T < 650 else 72 if T < 700 else
            50  if T < 750 else 32  if T < 800 else 18 if T < 850 else 10
        ),
    },
}


# ─── Core engineering calculations ─────────────────────────────────────────────

def tube_analysis(OD_mm, t_mm, P_barg, T_fluid_C, Q_kW_m2, mat_key):
    """
    Full stress analysis of a thick-wall tube under combined
    internal pressure and thermal loading.

    Returns a dict of results suitable for JSON serialisation.
    """
    mat = MATERIALS[mat_key]

    # Convert units
    P   = P_barg * 0.1          # barg → MPa
    Q   = Q_kW_m2 * 1_000       # kW/m² → W/m²
    t   = t_mm / 1_000          # mm → m
    OD  = OD_mm / 1_000         # mm → m
    ro  = OD / 2
    ri  = ro - t

    E     = mat["E"]
    alpha = mat["alpha"]
    nu    = mat["nu"]
    k     = mat["k"]

    # 1. Temperature profile through wall (steady-state, cylindrical)
    #    For a thin-ish wall we use the planar (slab) approximation:
    #    dT = Q * t / k
    dT      = Q * t / k                    # °C, inner-to-outer temperature rise
    T_inner = T_fluid_C                    # inner wall ≈ fluid temperature
    T_outer = T_inner + dT                 # outer wall (fire side) is hotter
    T_mean  = (T_inner + T_outer) / 2

    # Temperature profile at N points through wall thickness
    N      = 40
    x_frac = np.linspace(0, 1, N)          # 0 = inner, 1 = outer
    T_profile = T_inner + dT * x_frac

    # 2. Lamé thick-wall hoop stress (pressure contribution)
    #    Maximum is at inner radius
    sigma_hoop_inner = P * (ri**2 + ro**2) / (ro**2 - ri**2)
    sigma_hoop_outer = P * (2 * ri**2)     / (ro**2 - ri**2)
    sigma_axial      = P * ri**2           / (ro**2 - ri**2)   # closed-end axial

    # 3. Thermal stress (through-wall ΔT drives bending-like stress)
    #    Classical result for a tube with free ends:
    #    σ_thermal = ±E·α·ΔT / (2·(1−ν))
    #    Tension on cold (inner) side, compression on hot (outer) side
    sigma_thermal_magnitude = E * alpha * dT / (2 * (1 - nu))

    # 4. Combined stress — Von Mises at outer (fire-side) surface
    #    Outer surface: hoop (reduced), axial, thermal (compressive in hoop dir)
    s_hoop_outer_combined = sigma_hoop_outer - sigma_thermal_magnitude
    s_axial_outer         = sigma_axial      - sigma_thermal_magnitude * nu
    tau                   = 0.0              # no shear in this simplified model

    vm_outer = np.sqrt(
        s_hoop_outer_combined**2 + s_axial_outer**2
        - s_hoop_outer_combined * s_axial_outer
        + 3 * tau**2
    )

    # Combined Von Mises at inner (process-side) surface (usually lower)
    s_hoop_inner_combined = sigma_hoop_inner + sigma_thermal_magnitude
    vm_inner = np.sqrt(
        s_hoop_inner_combined**2 + sigma_axial**2
        - s_hoop_inner_combined * sigma_axial
    )

    vm_max = max(vm_outer, vm_inner)

    # 5. Allowable stress at the outer (hottest) wall temperature
    allowable = mat["allow"](T_outer)
    utilisation = vm_max / allowable

    if utilisation < 0.8:
        status = "ok"
        status_text = "Within allowable limits"
    elif utilisation < 1.0:
        status = "warning"
        status_text = "Caution — above 80% utilisation"
    else:
        status = "fail"
        status_text = "Overstress — exceeds allowable"

    # 6. Indicative creep life via Larson-Miller parameter
    #    LMP = T(K) × (C + log10(t_r))   where C ≈ 20 for most alloys
    #    We back-calculate rupture time from vm_max vs allowable ratio
    C = 20
    if vm_max > 0 and T_outer > 400:
        T_K     = T_outer + 273.15
        stress_ratio = max(vm_max / allowable, 0.05)
        # Simplified: life scales inversely with stress ratio raised to power ~5
        life_hrs = 100_000 * (1 / stress_ratio) ** 5
        life_hrs = min(life_hrs, 500_000)
        lmp = T_K * (C + np.log10(life_hrs)) / 1_000
    else:
        life_hrs = 500_000
        lmp = 0

    # Format life for display
    if life_hrs >= 200_000:
        life_str = ">200,000 hrs"
    elif life_hrs >= 1_000:
        life_str = f"{life_hrs/1000:.0f}k hrs"
    else:
        life_str = f"{life_hrs:.0f} hrs"

    # 7. Safety factors
    sf_yield  = allowable / vm_max if vm_max > 0 else 99

    return {
        # Inputs echoed back
        "material_name": mat["name"],
        "OD_mm": OD_mm,
        "t_mm": t_mm,
        "P_barg": P_barg,
        "T_fluid": T_fluid_C,
        "Q_kW": Q_kW_m2,

        # Temperatures
        "T_inner": round(T_inner, 1),
        "T_outer": round(T_outer, 1),
        "T_mean":  round(T_mean,  1),
        "delta_T": round(dT, 1),

        # Stresses (MPa)
        "sigma_hoop_inner":     round(sigma_hoop_inner, 2),
        "sigma_hoop_outer":     round(sigma_hoop_outer, 2),
        "sigma_axial":          round(sigma_axial, 2),
        "sigma_thermal":        round(sigma_thermal_magnitude, 2),
        "vm_outer":             round(vm_outer, 2),
        "vm_inner":             round(vm_inner, 2),
        "vm_max":               round(vm_max, 2),

        # Code check
        "allowable":            round(allowable, 1),
        "utilisation_pct":      round(utilisation * 100, 1),
        "safety_factor":        round(sf_yield, 2),
        "status":               status,
        "status_text":          status_text,

        # Creep life
        "life_str":             life_str,
        "lmp":                  round(lmp, 2),

        # Wall temperature profile for chart
        "wall_profile": {
            "x":  [round(float(v), 3) for v in x_frac],    # 0→1 normalised
            "T":  [round(float(v), 1) for v in T_profile],
        },
    }


# ─── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the HTML frontend."""
    return render_template("index.html")


@app.route("/api/analyse", methods=["POST"])
def analyse():
    """
    Receive JSON from the browser, run the analysis, return JSON results.
    POST body: { OD, t, P, T_fluid, Q, material }
    """
    data = request.get_json()

    try:
        result = tube_analysis(
            OD_mm     = float(data["OD"]),
            t_mm      = float(data["t"]),
            P_barg    = float(data["P"]),
            T_fluid_C = float(data["T_fluid"]),
            Q_kW_m2   = float(data["Q"]),
            mat_key   = data["material"],
        )
        return jsonify({"ok": True, "result": result})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    # Replit requires host="0.0.0.0" to be accessible publicly
    app.run(host="0.0.0.0", port=8080, debug=True)
