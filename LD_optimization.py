# file: fixed_airfoil_optimizer.py
import csv, json, math, random, time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
OUTPUT_CSV = "results_naca4412.csv"
N_ITERATIONS = 300
N_INIT_SAMPLES = 20

BASELINE_REAL_L_D = 14.5

RANGES = {
    "wing_span":        (1.40, 2.20),   # m
    "root_chord":       (0.18, 0.35),   # m
    "taper_ratio":      (0.40, 0.80),   # -
    "sweep_deg":        (0.0,  15.0),   # deg
    "twist_deg":        (0.0,  4.0),    # deg
    "dihedral_deg":     (2.0,  6.0),    # deg
    "tail_arm_m":       (0.50, 0.90),   # m
    "htail_area":       (0.04, 0.09),   # m^2
    "vtail_area":       (0.02, 0.05),   # m^2
}
KEYS = list(RANGES.keys())

RHO = 1.225
V_CRUISE = 18.0
WEIGHT_N = 55.0
CAL_FACTOR = 1.0

# -----------------------------------------------------------
# Physics Engine (Fixed Airfoil)
# -----------------------------------------------------------
def get_physics_score(p: dict) -> float:
    # 1. Geometry
    b = p["wing_span"]
    c_root = p["root_chord"]
    lam = p["taper_ratio"]
    c_tip = c_root * lam
    S = b * (c_root + c_tip) / 2.0
    
    AR = (b**2) / S
    MAC = S / b
    
    # 2. Fixed Airfoil Constants (NACA 4412 / 0012)
    THICK_MAIN = 0.12  # 12%
    THICK_TAIL = 0.12  # 12%
    
    # 3. Drag Calculation
    # Form Factor equation for 12% thickness
    def get_form_factor(t_ratio):
        return 1.0 + 2.0*t_ratio + 60.0*(t_ratio**4)

    ff_main = get_form_factor(THICK_MAIN)
    ff_tail = get_form_factor(THICK_TAIL)

    l_t, S_h, S_v = p["tail_arm_m"], p["htail_area"], p["vtail_area"]
    
    # Wetted Area Drag
    swet_wing = 2.02 * S
    swet_tail = 2.02 * (S_h + S_v)
    swet_fuse = 0.15 + 0.1 * l_t
    
    # Calculate weighted average CD0
    drag_area_wing = swet_wing * ff_main
    drag_area_tail = swet_tail * ff_tail
    # Fuselage assumes similar FF or just simpler addition
    drag_area_total = drag_area_wing + drag_area_tail + (swet_fuse * 1.1)
    
    cd0 = 0.0035 * (drag_area_total / S)

    # 4. Lift & Induced Drag
    q = 0.5 * RHO * V_CRUISE**2
    CL_req = WEIGHT_N / (q * S)

    swp, twt = p["sweep_deg"], p["twist_deg"]
    e = 0.85 - 0.0005*(swp**1.5)
    if lam < 0.6 and twt > 1.0: e += 0.02 
    e = max(0.6, min(0.98, e))
    
    k = 1.0 / (math.pi * e * AR)
    cdi = k * (CL_req**2)

    # 5. Penalties
    penalty = 1.0
    Vh = (S_h * l_t) / (S * MAC)
    
    if Vh < 0.35: penalty *= 0.5
    if S_v < 0.05 * S: penalty *= 0.8
    if AR > 14.0: penalty *= 0.85
    if CL_req > 1.0: penalty *= 0.5

    total_drag = cd0 + cdi
    if total_drag <= 0: return 0.0
    
    return (CL_req / total_drag) * penalty

def evaluate_calibrated(params):
    return get_physics_score(params) * CAL_FACTOR

# -----------------------------------------------------------
# Utilities & Main
# -----------------------------------------------------------
def dict_to_vector(d): return [d[k] for k in KEYS]

def generate_random_sample():
    d = {}
    for k in KEYS:
        lo, hi = RANGES[k]
        d[k] = round(random.uniform(lo, hi), 4)
    return d

def calibrate_system():
    global CAL_FACTOR
    base_params = {
        "wing_span": 1.60, "root_chord": 0.27, "taper_ratio": 0.50,
        "sweep_deg": 0.0, "twist_deg": 0.0, "dihedral_deg": 4.0,
        "tail_arm_m": 0.60, "htail_area": 0.05, "vtail_area": 0.03
    }
    raw = get_physics_score(base_params)
    if raw == 0: raise ValueError("Baseline Error")
    CAL_FACTOR = BASELINE_REAL_L_D / raw
    print(f"[System] Calibrated. Target L/D: {BASELINE_REAL_L_D}")

def main():
    calibrate_system()
    
    X_train = []
    y_train = []
    all_results = []

    kernel = Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-5, 1e6)) + WhiteKernel(noise_level=0.1)
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

    print(f"--- Initializing ({N_INIT_SAMPLES}) ---")
    for i in range(N_INIT_SAMPLES):
        p = generate_random_sample()
        s = evaluate_calibrated(p)
        X_train.append(dict_to_vector(p))
        y_train.append(s)
        all_results.append({**p, "score": s})

    print(f"--- AI Optimization ({N_ITERATIONS}) ---")
    for i in range(N_ITERATIONS):
        gp_model.fit(np.array(X_train), np.array(y_train))

        candidates = [generate_random_sample() for _ in range(500)]
        X_cand = np.array([dict_to_vector(c) for c in candidates])
        y_pred, sigma = gp_model.predict(X_cand, return_std=True)
        
        next_params = candidates[np.argmax(y_pred + 1.96 * sigma)]
        real_score = evaluate_calibrated(next_params)
        
        X_train.append(dict_to_vector(next_params))
        y_train.append(real_score)
        
        # Fabrication Calc
        c_root = next_params["root_chord"]
        lam = next_params["taper_ratio"]
        c_tip = c_root * lam
        
        # Calculate specific Thickness (mm) for NACA 4412 (12%)
        t_root_mm = c_root * 0.12 * 1000
        t_tip_mm  = c_tip  * 0.12 * 1000
        
        record = {**next_params}
        record["calc_tip_chord"] = round(c_tip, 4)
        record["calc_root_thick_mm"] = round(t_root_mm, 1)
        record["calc_tip_thick_mm"] = round(t_tip_mm, 1)
        record["score"] = real_score
        all_results.append(record)

        if (i+1) % 50 == 0:
            print(f"Iter {i+1:3d} | Best L/D: {max(y_train):.4f}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    save_keys = KEYS + ["calc_tip_chord", "calc_root_thick_mm", "calc_tip_thick_mm", "score"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=save_keys)
        writer.writeheader()
        writer.writerows(all_results)

    best = all_results[0]
    print("\n" + "="*50)
    print(" [NACA 4412/0012] BEST CONFIGURATION ")
    print("="*50)
    print(f"  * Score (L/D) : {best['score']:.4f}")
    print("-" * 30)
    print(f"  1. Span       : {best['wing_span']*1000:.1f} mm")
    print(f"  2. Root Chord : {best['root_chord']*1000:.1f} mm")
    print(f"  3. Tip Chord  : {best['calc_tip_chord']*1000:.1f} mm")
    print("-" * 30)
    print(f"  4. Root Thick : {best['calc_root_thick_mm']} mm (NACA4412)")
    print(f"  5. Tip Thick  : {best['calc_tip_thick_mm']} mm (NACA4412)")
    print("-" * 30)
    print(f"  6. Twist      : {best['twist_deg']:.2f} deg")
    print(f"  7. Sweep      : {best['sweep_deg']:.2f} deg")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()