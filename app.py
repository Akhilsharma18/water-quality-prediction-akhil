"""Streamlit web app:  streamlit run app.py"""
import json, os, joblib, pandas as pd, streamlit as st
from water_utils import FEATURES, LIMITS, check_limits

st.set_page_config(page_title="Water Quality Predictor", page_icon="💧", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists("models/water_model.pkl"):
        import train; train.main()
    return joblib.load("models/water_model.pkl")

model = load_model()
st.title("💧 Smart Water Quality & Potability Predictor")
st.caption("SDG 6 - Clean Water and Sanitation | ML screening tool, not a substitute for lab testing")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Sample", "📁 Batch CSV", "📊 Model Performance", "🌍 About SDG 6"])

with tab1:
    st.subheader("Enter water test readings")
    cols = st.columns(3); vals = {}
    defaults = {"ph": 7.0, "Hardness": 196.0, "Solids": 20000.0, "Chloramines": 7.0, "Sulfate": 333.0,
                "Conductivity": 420.0, "Organic_carbon": 14.0, "Trihalomethanes": 66.0, "Turbidity": 4.0}
    for i, f in enumerate(FEATURES):
        lo, hi, unit = LIMITS[f][:3]
        vals[f] = cols[i % 3].number_input(f"{f} {('(' + unit + ')') if unit else ''}", float(lo), float(hi * 1.5 if f != "ph" else 14),
                                           float(defaults[f]), step=0.1)
    if st.button("Predict potability", type="primary"):
        row = pd.DataFrame([vals])
        prob = float(model.predict_proba(row)[0][1])
        c1, c2 = st.columns([1, 2])
        with c1:
            if prob >= 0.5: st.success(f"✅ Likely POTABLE\n\nConfidence: {prob:.1%}")
            else: st.error(f"⚠️ Likely UNSAFE\n\nPotable probability: {prob:.1%}")
            st.progress(prob)
        with c2:
            chk = check_limits(row.iloc[0])
            st.dataframe(chk.style.map(lambda s: "color:#c0392b;font-weight:bold" if s == "OUT OF RANGE" else "color:#27ae60",
                                       subset=["Status"]), width='stretch', hide_index=True)
            bad = chk[chk.Status != "OK"].Parameter.tolist()
            if bad: st.warning("Treat/test further: " + ", ".join(bad))

with tab2:
    st.write("Upload a CSV with columns: `" + ", ".join(FEATURES) + "`")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        d = pd.read_csv(up)
        miss = [c for c in FEATURES if c not in d.columns]
        if miss: st.error(f"Missing columns: {miss}")
        else:
            d["Prediction"] = ["Potable" if p else "Unsafe" for p in model.predict(d[FEATURES])]
            d["Potable_prob"] = model.predict_proba(d[FEATURES])[:, 1].round(3)
            st.dataframe(d, width='stretch')
            st.bar_chart(d["Prediction"].value_counts())
            st.download_button("Download results", d.to_csv(index=False), "predictions.csv", "text/csv")

with tab3:
    if os.path.exists("models/metrics.json"):
        m = json.load(open("models/metrics.json"))
        st.dataframe(pd.DataFrame(m["metrics"]).T.round(3), width='stretch')
        a, b = st.columns(2)
        a.image("images/confusion.png", caption="Confusion matrix (Random Forest)")
        b.bar_chart(pd.Series(m["importance"]).sort_values(), horizontal=True)
    else: st.info("Run `python train.py` first.")

with tab4:
    st.markdown("""**SDG 6** aims to ensure availability and sustainable management of water and sanitation for all.
This tool supports **Target 6.1** (safe, affordable drinking water) by enabling fast, low-cost first-level screening
of water samples so unsafe sources can be flagged early for proper testing and treatment.""")
