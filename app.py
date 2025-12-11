import streamlit as st
import pandas as pd
import pickle
import time
import threading
import json
import os
from paho.mqtt import client as mqtt

# ===========================
# CONFIG
# ===========================
MODEL_PATH = "knn_classifier.pkl"
CSV_PATH = "river_data_log.csv"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "river/monitoring/data"

# ===========================
# LOAD MODEL
# ===========================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===========================
# INIT CSV IF NOT EXISTS
# ===========================
if not os.path.exists(CSV_PATH):
    df0 = pd.DataFrame(columns=[
        "water_level_cm",
        "rain_level",
        "danger_level",
        "humidity_pct",
        "datetime"
    ])
    df0.to_csv(CSV_PATH, index=False)

# ===========================
# GLOBAL VARIABLE FOR MQTT DATA
# ===========================
latest_mqtt = None
mqtt_lock = threading.Lock()

# ===========================
# MQTT CALLBACK
# ===========================
def on_message(client, userdata, msg):
    global latest_mqtt
    try:
        payload = msg.payload.decode()

        # expect JSON
        try:
            data = json.loads(payload)
        except:
            # fallback CSV-like "water,rain,danger,hum"
            parts = payload.split(",")
            data = {
                "water_level_cm": float(parts[0]),
                "rain_level": int(parts[1]),
                "danger_level": int(parts[2]),
                "humidity_pct": float(parts[3])
            }

        with mqtt_lock:
            latest_mqtt = data

    except Exception as e:
        print("MQTT error:", e)


def mqtt_thread():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.subscribe(MQTT_TOPIC)
    client.loop_forever()


mqtt_bg = threading.Thread(target=mqtt_thread, daemon=True)
mqtt_bg.start()

# ===========================
# STREAMLIT SETTINGS
# ===========================
st.set_page_config(page_title="River Monitor + MQTT + ML", layout="wide")
st.title("🌊 River Monitoring Dashboard — Real-Time + Prediction")

# ===========================
# HELPERS
# ===========================
def load_data():
    return pd.read_csv(CSV_PATH)

def append_data(data):
    df = load_data()
    data["datetime"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

def normalize_pred(pred):
    try:
        pi = int(pred)
        mapping = ["SAFE","WARNING","DANGER","CRITICAL"]
        if pi < len(mapping):
            return mapping[pi]
    except:
        return str(pred).upper()
    return str(pred)

def normalize_emoji(label):
    l = label.upper()
    return {
        "SAFE": "🟢",
        "WARNING": "🟡",
        "DANGER": "🔴",
        "CRITICAL": "⚠️"
    }.get(l, "❓")

def status_box(title, level, mode="danger"):
    if mode == "danger":
        if level == 0: color="#1b9e77"; emoji="🟢"; text="SAFE"
        elif level == 1: color="#e6ab02"; emoji="🟡"; text="WARNING"
        else: color="#d95f02"; emoji="🔴"; text="DANGEROUS"

    elif mode == "rain":
        if level == 0: color="#1b9e77"; emoji="🌤️"; text="NO RAIN"
        elif level == 1: color="#e6ab02"; emoji="🌦️"; text="LIGHT RAIN"
        else: color="#2b46d9"; emoji="🌧️"; text="HEAVY RAIN"

    st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background:{color}; text-align:center;">
            <h2 style="color:white;">{title}</h2>
            <h1 style="color:white; font-size:50px;">{emoji}</h1>
            <h3 style="color:white;">{text}</h3>
        </div>
    """, unsafe_allow_html=True)


# ===========================
# MAIN LOOP
# ===========================
while True:

    # MQTT incoming → append to CSV
    with mqtt_lock:
        md = latest_mqtt
        latest_mqtt = None

    if md is not None:
        append_data(md)

    df = load_data()

    if df.empty:
        st.info("Waiting for data...")
        time.sleep(2)
        st.experimental_rerun()

    last = df.iloc[-1]

    water = last["water_level_cm"]
    rain = last["rain_level"]
    danger = last["danger_level"]
    hum = last["humidity_pct"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Water Level (cm)", water)
    with col2:
        status_box("Danger Level", int(danger), mode="danger")
    with col3:
        status_box("Rain Level", int(rain), mode="rain")

    st.subheader("📈 Water Level Over Time")
    st.line_chart(df["water_level_cm"])

    # ML prediction
    try:
        pred = model.predict([[float(water), float(rain), float(danger), float(hum)]])[0]
    except:
        pred = None

    st.subheader("🤖 Predicted Condition (KNN)")

    if pred is not None:
        label = normalize_pred(pred)
        emoji = normalize_emoji(label)

        st.markdown(f"""
            <div style="padding:25px; border-radius:15px; background:#333; color:white; text-align:center;">
                <h2>Prediction:</h2>
                <h1 style="font-size:60px;">{emoji}</h1>
                <h1>{label}</h1>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Prediction unavailable (model error)")

    time.sleep(3)
    st.rerun()
