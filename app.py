import streamlit as st
import pandas as pd
import pickle
import time

# ===========================
# LOAD MACHINE LEARNING MODEL
# ===========================
MODEL_PATH = "knn_classifier.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===========================
# DASHBOARD CONFIG
# ===========================
st.set_page_config(
    page_title="River Monitoring Dashboard",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 **River Monitoring Dashboard**")

DATA_PATH = "river_data_log.csv"


# ===========================
# HELPER: LOAD CSV
# ===========================
def load_data():
    return pd.read_csv(DATA_PATH)


# ===========================
# HELPER: PREDICT USING .PKL
# ===========================
def get_prediction(water, rain):
    """
    Model input format:
    X = [water_level, rain_level]
    """
    X = [[water, rain]]
    pred = model.predict(X)[0]
    return pred


# ===========================
# HELPER: BIG STATUS BOX
# ===========================
def status_box(title, level, mode="danger"):
    if mode == "danger":
        if level == 0:
            color = "#1b9e77"
            emoji = "🟢"
            text = "SAFE"
        elif level == 1:
            color = "#e6ab02"
            emoji = "🟡"
            text = "WARNING"
        else:
            color = "#d95f02"
            emoji = "🔴"
            text = "DANGEROUS"

    elif mode == "rain":
        if level == 0:
            color = "#1b9e77"
            emoji = "🌤️"
            text = "NO RAIN"
        elif level == 1:
            color = "#e6ab02"
            emoji = "🌦️"
            text = "LIGHT RAIN"
        else:
            color = "#2b46d9"
            emoji = "🌧️"
            text = "HEAVY RAIN"

    st.markdown(
        f"""
        <div style="padding:20px; border-radius:15px; 
             background:{color}; text-align:center;">
            <h2 style="color:white;">{title}</h2>
            <h1 style="color:white; font-size:60px;">{emoji}</h1>
            <h1 style="color:white;">{text}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


# ===========================
# MAIN LOOP (AUTO REFRESH)
# ===========================
placeholder = st.empty()

while True:
    with placeholder.container():
        df = load_data()

        # ambil data terbaru
        latest = df.iloc[-1]
        water = latest["water_level_cm"]
        danger = latest["danger_level"]
        rain = latest["rain_level"]


        # ===========================
        # --- COLUMN STATUS ---
        # ===========================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Latest Water Level")
            st.metric("Water Level (cm)", water)

        with col2:
            status_box("Danger Level", danger, mode="danger")

        with col3:
            status_box("Rain Level", rain, mode="rain")

        # ===========================
        # --- GRAPHIC WATER LEVEL ---
        # ===========================
        st.subheader("📈 Water Level Chart")
        st.line_chart(df["water_level_cm"])

        # ===========================
        # --- MODEL PREDICTION ---
        # ===========================
        st.subheader("🤖 AI Prediction (KNN Model)")
        prediction = get_prediction(water, rain)

        pred_text = ["SAFE", "WARNING", "DANGER"][prediction]
        pred_emoji = ["🟢", "🟡", "🔴"][prediction]

        st.markdown(
            f"""
            <div style="padding:20px; border-radius:15px; background:#4b4b4b; text-align:center;">
                <h2 style="color:white;">Predicted Condition</h2>
                <h1 style="color:white; font-size:60px;">{pred_emoji}</h1>
                <h1 style="color:white;">{pred_text}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    time.sleep(3)
