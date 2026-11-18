"""Shared constants, guideline limits and data generation."""
import numpy as np, pandas as pd

FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

# name: (min, max, unit, safe_low, safe_high)  -> safe range used for guideline checks (WHO/BIS style)
LIMITS = {
    "ph": (0, 14, "", 6.5, 8.5),
    "Hardness": (47, 323, "mg/L", 0, 300),
    "Solids": (320, 61000, "ppm", 0, 30000),
    "Chloramines": (0.3, 13, "ppm", 0, 8.5),
    "Sulfate": (129, 481, "mg/L", 0, 400),
    "Conductivity": (181, 753, "uS/cm", 0, 800),
    "Organic_carbon": (2, 28, "ppm", 0, 18),
    "Trihalomethanes": (0, 124, "ug/L", 0, 80),
    "Turbidity": (1.5, 6.7, "NTU", 0, 4.8),
}

def make_synthetic(n=3000, seed=42):
    """Synthetic samples labelled with the limits above (+5% label noise)."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ph": r.normal(7.1, 1.3, n).clip(3, 12),
        "Hardness": r.normal(196, 32, n),
        "Solids": r.normal(21900, 8800, n).clip(300, None),
        "Chloramines": r.normal(7.1, 1.6, n).clip(0.3, None),
        "Sulfate": r.normal(333, 40, n),
        "Conductivity": r.normal(426, 80, n),
        "Organic_carbon": r.normal(14.3, 3.3, n).clip(1, None),
        "Trihalomethanes": r.normal(66, 16, n).clip(1, None),
        "Turbidity": r.normal(3.97, 0.78, n).clip(0.2, None),
    })
    bad = ((df.ph < 6.5) | (df.ph > 8.5)).astype(int) + (df.Solids > 30000) \
        + (df.Chloramines > 8.5) + (df.Trihalomethanes > 80) \
        + (df.Turbidity > 4.8) + (df.Organic_carbon > 18)
    y = (bad == 0).astype(int)
    flip = r.random(n) < 0.05
    df["Potability"] = np.where(flip, 1 - y, y)
    return df

def check_limits(row):
    """Return a DataFrame comparing one sample to guideline limits."""
    out = []
    for f in FEATURES:
        lo, hi = LIMITS[f][3], LIMITS[f][4]
        v = float(row[f])
        out.append({"Parameter": f, "Value": round(v, 2), "Safe range": f"{lo} - {hi} {LIMITS[f][2]}".strip(),
                    "Status": "OK" if lo <= v <= hi else "OUT OF RANGE"})
    return pd.DataFrame(out)
