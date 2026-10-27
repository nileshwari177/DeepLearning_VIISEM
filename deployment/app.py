import os, io, numpy as np, pandas as pd, streamlit as st
from dq_checker import DataQualityChecker

st.set_page_config(page_title="RBM Data-Quality Checker", layout="wide")
st.title("RBM-based Data Quality Checker")
st.caption("Upload a CSV with the 30 Wisconsin breast-cancer feature columns. Educational demo, not a medical tool.")

checker = DataQualityChecker()
file = st.file_uploader("Upload CSV", type="csv")
if st.button("Use demo data (clean rows + corrupted rows)"):
    from sklearn.datasets import load_breast_cancer
    fr = load_breast_cancer(as_frame=True)
    d = fr.data[fr.target == 1]                          # benign (normal) rows only
    demo = d.iloc[:40].copy()
    demo.iloc[:10, :3] = demo.iloc[:10, :3] * 5          # inject outliers
    demo.iloc[10:15, 5:8] = 0.0                          # inject zero-filled values
    file = io.StringIO(demo.to_csv(index=False))

if file is not None:
    df = pd.read_csv(file)
    rep = checker.assess(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows checked", len(rep)); c2.metric("Rows flagged", int(rep.flagged.sum()))
    c3.metric("Rows with missing cells", int((rep.missing_cells > 0).sum()))
    st.subheader("Anomaly probability per row"); st.bar_chart(rep["anomaly_probability"])
    st.subheader("Flagged rows and the columns that look wrong")
    st.dataframe(rep[rep.flagged].sort_values("anomaly_probability", ascending=False))
    st.download_button("Download full report (CSV)", rep.to_csv().encode(), "data_quality_report.csv")
