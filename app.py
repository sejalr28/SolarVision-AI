


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import os, json, datetime, io
import plotly.express as px
import plotly.graph_objects as go


IMG_SIZE    = (128, 128)
MODEL_PATH  = "solar_model.h5"
LABELS_PATH = "class_labels.json"

SEVERITY_MAP = {
    "Clean":              (" None",   "green",  0),
    "Dusty":              (" Low",    "gold",   15),
    "Bird-drop":          (" Medium", "orange", 35),
    "Electrical-damage":  (" High",   "red",    80),
    "Physical-damage":    (" High",   "red",    90),
}

MAINTENANCE_MAP = {
    "Clean":             "No action needed",
    "Dusty":             "Schedule routine cleaning within 2 weeks",
    "Bird-drop":         "Clean within 3 days – moderate efficiency loss",
    "Electrical-damage": "Immediate electrical inspection required",
    "Physical-damage":   "Panel replacement required – critical fault",
}


@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Please run `python train_model.py` first.")
        st.stop()
    if not os.path.exists(LABELS_PATH):
        st.error("❌ class_labels.json not found. Please run `python train_model.py` first.")
        st.stop()
    model  = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}
    return model, labels

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(model, labels, img: Image.Image):
    x     = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = labels[idx]
    conf  = float(probs[idx]) * 100
    all_probs = {labels[i]: float(probs[i]) * 100 for i in range(len(labels))}
    return label, conf, all_probs


st.set_page_config(
    page_title="SolarVision Analytics",
    
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
        color: white;
    }
    .kpi-card {
        background: #f8f9fa; border-radius: 10px; padding: 15px;
        text-align: center; border-left: 4px solid #1F6FEB;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1> SolarVision AI</h1>
    <p style='margin:0; opacity:0.8'>AI-Powered Solar Infrastructure Inspection Platform</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Deep learning-based defect detection and analytics for solar infrastructure inspection."
)


model, labels = load_model_and_labels()
num_classes   = len(labels)

# ── Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/solar-panel.png", width=80)
    st.title("Inspection Console")
    st.markdown("---")
    st.markdown(f"**Model:** `{MODEL_PATH}`")
    st.markdown(f"**Classes:** {num_classes}")
    st.markdown(f"**Input Size:** {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    st.markdown("---")
    conf_threshold = st.slider("Confidence Threshold (%)", 30, 95, 60)
    mission_id = st.text_input("Mission ID", value=f"MISS-{datetime.date.today().strftime('%Y%m%d')}")
    site_name  = st.text_input("Site Name", value="Solar Park - Site A")

tab1, tab2 = st.tabs([" Inspect Images", " Analytics Dashboard"])


with tab1:
    st.subheader("Upload Inspection Images")
    uploaded_files = st.file_uploader(
        "Upload one or more solar panel images (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        st.markdown(f"### Inspecting **{len(uploaded_files)}** image(s)...")

        # Show images in a grid with predictions
        cols_per_row = 2
        for i in range(0, len(uploaded_files), cols_per_row):
            row_files = uploaded_files[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, file in enumerate(row_files):
                img   = Image.open(file)
                label, conf, all_probs = predict_image(model, labels, img)
                sev_text, sev_color, energy_loss = SEVERITY_MAP.get(label, ("Unknown", "gray", 0))
                action = MAINTENANCE_MAP.get(label, "Inspect manually")
                low_conf = conf < conf_threshold

                with cols[j]:
                    st.image(img, caption=file.name, use_container_width=True)
                    if low_conf:
                        st.warning(f"Low confidence prediction — manual review recommended")
                    else:
                        st.markdown(f"**Condition:** `{label}`")
                        st.markdown(f"**Confidence:** `{conf:.1f}%`")
                        st.markdown(f"**Severity:** {sev_text}")
                        st.markdown(f"**Action:** _{action}_")

                    # Mini probability bar chart
                    prob_df = pd.DataFrame(list(all_probs.items()), columns=["Class", "Probability (%)"])
                    fig = px.bar(
                        prob_df,
                        x="Class",
                        y="Probability (%)",
                        color="Probability (%)",
                        color_continuous_scale="Blues",
                        height=420
                    )
                    fig.update_layout(
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False,
                        xaxis_tickangle=-25,
                        font=dict(size=16),
                        xaxis_title="Condition",
                        yaxis_title="Confidence (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                results.append({
                    "Mission_ID":       mission_id,
                    "Site":             site_name,
                    "File":             file.name,
                    "Condition":        label,
                    "Confidence_%":     round(conf, 2),
                    "Low_Confidence":   low_conf,
                    "Severity":         sev_text,
                    "Est_Energy_Loss_%": energy_loss,
                    "Action_Required":  action,
                    "Timestamp":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        # ── Summary KPIs
        st.markdown("---")
        st.subheader(" Inspection Summary")
        df = pd.DataFrame(results)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Panels Inspected",  len(df))
        k2.metric("Defects Detected",         len(df[df["Condition"] != "Clean"]))
        k3.metric("Defect Rate",       f"{len(df[df['Condition'] != 'Clean'])/len(df)*100:.1f}%")
        k4.metric("Critical Cases",     len(df[df["Severity"].str.contains("High")]))
        k5.metric("Avg Energy Loss",   f"{df['Est_Energy_Loss_%'].mean():.1f}%")

        st.dataframe(df, use_container_width=True)

        # ── Download 
        csv = df.to_csv(index=False)
        st.download_button(
            " Download Inspection Report (CSV)",
            data=csv,
            file_name=f"solar_inspection_{mission_id}_{datetime.date.today()}.csv",
            mime="text/csv"
        )


with tab2:
    st.subheader("Inspection Analytics")
    st.caption("Upload historical inspection reports for trend analysis.")
    csv_upload = st.file_uploader("Upload a previously generated Inspection Report CSV", type=["csv"])

    if csv_upload:
        df_hist = pd.read_csv(csv_upload)
    else:
        # Demo data
        st.caption("Displaying sample analytics data.")
        np.random.seed(42)
        conditions = np.random.choice(
            ["Clean", "Dusty", "Bird-drop", "Electrical-damage", "Physical-damage"],
            size=120, p=[0.40, 0.25, 0.15, 0.10, 0.10]
        )
        sites = np.random.choice(["Site A", "Site B", "Site C"], size=120)
        dates = pd.date_range("2024-01-01", periods=120, freq="3D")
        df_hist = pd.DataFrame({
            "Date":             dates,
            "Site":             sites,
            "Condition":        conditions,
            "Confidence_%":     np.random.uniform(70, 99, 120).round(1),
            "Est_Energy_Loss_%": [SEVERITY_MAP.get(c, ("", "", 0))[2] for c in conditions],
        })

    df_hist["Date"] = pd.to_datetime(df_hist["Date"])

    # KPIs
    total    = len(df_hist)
    defects  = len(df_hist[df_hist["Condition"] != "Clean"])
    avg_loss = df_hist["Est_Energy_Loss_%"].mean() if "Est_Energy_Loss_%" in df_hist.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Inspections", total)
    c2.metric("Defects Detected",  defects)
    c3.metric("Defect Rate",       f"{defects/total*100:.1f}%")
    c4.metric("Avg Energy Loss",   f"{avg_loss:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Defect Type Breakdown")
        vc = df_hist["Condition"].value_counts().reset_index()
        vc.columns = ["Condition", "Count"]
        fig = px.pie(vc, names="Condition", values="Count",
                     color="Condition")
                     
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Defect Trend Over Time")
        df_hist["Month"] = df_hist["Date"].dt.to_period("M").astype(str)
        trend = df_hist.groupby(["Month", "Condition"]).size().reset_index(name="Count")
        fig2  = px.line(trend, x="Month", y="Count", color="Condition", markers=True)
        fig2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    if "Site" in df_hist.columns:
        st.markdown("#### Defect Rate by Site")
        site_df = df_hist.groupby("Site").apply(
            lambda x: pd.Series({
                "Total": len(x),
                "Defective": (x["Condition"] != "Clean").sum(),
                "Defect_Rate_%": round((x["Condition"] != "Clean").mean() * 100, 1)
            })
        ).reset_index()
        fig3 = px.bar(site_df, x="Site", y="Defect_Rate_%", color="Site",
                      text="Defect_Rate_%", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig3, use_container_width=True)

