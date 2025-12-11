# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import queue
import threading
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
import os

# ---------------------------
# CONFIG
# ---------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "river/monitoring/data"
MODEL_PATH = "knn_classifier.pkl"       # put your .pkl here
CSV_PATH = "river_data_log.csv"      # persistent CSV

# timezone GMT+7
TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# GLOBAL QUEUE (for MQTT thread)
# ---------------------------
GLOBAL_MQ = queue.Queue()

# ---------------------------
# STREAMLIT PAGE SETUP
# ---------------------------
st.set_page_config(page_title="River Monitoring Dashboard", layout="wide")
st.title("🌊 River Monitoring Dashboard — Realtime")

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    # logs will contain dicts with canonical columns
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

if "ml_model" not in st.session_state:
    st.session_state.ml_model = None

if "connected" not in st.session_state:
    st.session_state.connected = False

# ---------------------------
# CSV BOILERPLATE — ensure CSV exists and load recent
# ---------------------------
CSV_COLUMNS = [
    "timestamp", "datetime",
    "water_level_cm", "temperature_c", "humidity_pct",
    "danger_level", "rain_level",
    "model_pred", "model_conf"
]

if not os.path.exists(CSV_PATH):
    df_init = pd.DataFrame(columns=CSV_COLUMNS)
    df_init.to_csv(CSV_PATH, index=False)

# pre-load existing CSV into session logs (limit to last 5000)
try:
    df_existing = pd.read_csv(CSV_PATH)
    # normalize columns and append to logs
    for _, r in df_existing.tail(1000).iterrows():
        row = {
            "timestamp": int(r.get("timestamp", int(time.time()*1000))),
            "datetime": str(r.get("datetime", "")),
            "water_level_cm": float(r.get("water_level_cm", np.nan)) if not pd.isna(r.get("water_level_cm")) else None,
            "temperature_c": float(r.get("temperature_c", np.nan)) if not pd.isna(r.get("temperature_c")) else None,
            "humidity_pct": float(r.get("humidity_pct", np.nan)) if not pd.isna(r.get("humidity_pct")) else None,
            "danger_level": int(r.get("danger_level", 0)) if not pd.isna(r.get("danger_level")) else 0,
            "rain_level": int(r.get("rain_level", 0)) if not pd.isna(r.get("rain_level")) else 0,
            "model_pred": r.get("model_pred", ""),
            "model_conf": r.get("model_conf", None)
        }
        st.session_state.logs.append(row)
    # trim
    if len(st.session_state.logs) > 5000:
        st.session_state.logs = st.session_state.logs[-5000:]
except Exception:
    # file may be empty or new - ignore
    pass

# ---------------------------
# Load ML MODEL (classification -> Safe/Warning/Danger)
# ---------------------------
@st.cache_resource
def load_model(path):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.warning(f"Could not load model '{path}': {e}")
        return None

if st.session_state.ml_model is None:
    st.session_state.ml_model = load_model(MODEL_PATH)

if st.session_state.ml_model:
    st.success(f"Model loaded: {MODEL_PATH}")
else:
    st.info("No model loaded. Place river_model.pkl in repo to enable predictions.")

# ---------------------------
# MQTT CALLBACKS
# ---------------------------
def _on_connect(client, userdata, flags, rc):
    st.session_state.connected = (rc == 0)
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})
    try:
        client.subscribe(TOPIC_SENSOR)
    except Exception:
        pass

def _on_message(client, userdata, msg):
    # safe decode
    payload = msg.payload.decode(errors="ignore")
    try:
        data = json.loads(payload)
    except Exception:
        GLOBAL_MQ.put({"_type": "raw", "payload": payload, "ts": time.time(), "topic": msg.topic})
        return

    GLOBAL_MQ.put({"_type": "sensor", "data": data, "ts": time.time(), "topic": msg.topic})

# ---------------------------
# START MQTT THREAD (background)
# ---------------------------
def start_mqtt_thread_once():
    def worker():
        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                GLOBAL_MQ.put({"_type": "error", "msg": f"MQTT worker error: {e}", "ts": time.time()})
                time.sleep(5)

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True, name="mqtt_worker")
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05)

start_mqtt_thread_once()

# ---------------------------
# HELPER: MODEL PREDICTION (safe)
# ---------------------------
def model_predict(model, features_dict):
    """
    features_dict may contain: water_level_cm, temperature_c, humidity_pct, danger_level, rain_level
    We'll attempt to form a feature vector in a sensible order. If model fails, return (None, None).
    """
    if model is None:
        return (None, None)
    # candidate feature orders to try
    attempts = [
        ["water_level_cm", "temperature_c", "humidity_pct", "danger_level", "rain_level"],
        ["water_level_cm", "temperature_c", "humidity_pct"],
        ["temperature_c", "humidity_pct", "water_level_cm"],
        ["water_level_cm"]
    ]
    for feat_order in attempts:
        try:
            X = []
            valid = True
            for f in feat_order:
                v = features_dict.get(f)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    valid = False
                    break
                X.append(float(v))
            if not valid:
                continue
            X = [X]
            pred = model.predict(X)
            label = pred[0]
            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    prob = model.predict_proba(X)
                    conf = float(np.max(prob))
                except Exception:
                    conf = None
            return (str(label), conf)
        except Exception:
            continue
    return (None, None)

# ---------------------------
# PROCESS QUEUE (drain items from GLOBAL_MQ)
# ---------------------------
def process_queue():
    updated = False
    q = st.session_state.msg_queue
    while not q.empty():
        item = q.get()
        ttype = item.get("_type")
        if ttype == "status":
            st.session_state.connected = item.get("connected", False)
            updated = True
        elif ttype == "error":
            st.error(item.get("msg"))
            updated = True
        elif ttype == "raw":
            row = {"ts": now_str(), "raw": item.get("payload")}
            st.session_state.logs.append(row)
            st.session_state.last = row
            updated = True
        elif ttype == "sensor":
            d = item.get("data", {})
            # read expected fields (use robust defaults)
            try:
                water = float(d.get("water_level_cm")) if d.get("water_level_cm") is not None else None
            except Exception:
                water = None
            try:
                temp = float(d.get("temperature_c")) if d.get("temperature_c") is not None else None
            except Exception:
                temp = None
            try:
                hum = float(d.get("humidity_pct")) if d.get("humidity_pct") is not None else None
            except Exception:
                hum = None
            try:
                danger = int(d.get("danger_level")) if d.get("danger_level") is not None else 0
            except Exception:
                danger = 0
            try:
                rain = int(d.get("rain_level")) if d.get("rain_level") is not None else 0
            except Exception:
                rain = 0
            timestamp = int(d.get("timestamp", time.time()*1000))

            features = {
                "water_level_cm": water,
                "temperature_c": temp,
                "humidity_pct": hum,
                "danger_level": danger,
                "rain_level": rain
            }

            pred_label, pred_conf = model_predict(st.session_state.ml_model, features)
            if pred_label is None:
                pred_label = ""
            # create canonical row
            row = {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "water_level_cm": water,
                "temperature_c": temp,
                "humidity_pct": hum,
                "danger_level": danger,
                "rain_level": rain,
                "model_pred": pred_label,
                "model_conf": pred_conf
            }
            # append to session logs and CSV
            st.session_state.logs.append(row)
            # keep bounded logs
            if len(st.session_state.logs) > 5000:
                st.session_state.logs = st.session_state.logs[-5000:]

            # append to CSV file
            try:
                pd.DataFrame([row])[CSV_COLUMNS].to_csv(CSV_PATH, mode='a', header=False, index=False)
            except Exception:
                # best-effort: try without model fields
                try:
                    safe_row = {k: row.get(k, "") for k in CSV_COLUMNS}
                    pd.DataFrame([safe_row]).to_csv(CSV_PATH, mode='a', header=False, index=False)
                except Exception:
                    pass

            st.session_state.last = row
            updated = True
    return updated

# run once to pick up any queued messages
_ = process_queue()

# ---------------------------
# UI LAYOUT
# ---------------------------
if st.button("Force refresh (drain queue)"):
    _ = process_queue()

# Auto-refresh small (optional, non-blocking)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2000, limit=None, key="autorefresh")
except Exception:
    pass

left, right = st.columns([1, 2])

with left:
    st.header("Connection & Metrics")
    st.write(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    st.metric("MQTT Connected", "Yes" if st.session_state.connected else "No")
    st.write("Topic:", TOPIC_SENSOR)
    st.markdown("---")
    st.header("Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"Time: {last.get('datetime')}")
        st.write(f"Water level (cm): {last.get('water_level_cm')}")
        st.write(f"Temperature (°C): {last.get('temperature_c')}")
        st.write(f"Humidity (%): {last.get('humidity_pct')}")
        st.write(f"Danger level: {last.get('danger_level')}")
        st.write(f"Rain level: {last.get('rain_level')}")
        st.write(f"Model Prediction: {last.get('model_pred')} (conf: {last.get('model_conf')})")
        if str(last.get("model_pred")).lower() == "danger":
            st.error("🚨 MODEL PREDICTION: DANGER")
        elif str(last.get("model_pred")).lower() == "warning":
            st.warning("⚠ MODEL PREDICTION: WARNING")
        else:
            st.success("✅ MODEL PREDICTION: SAFE / OK")
    else:
        st.info("Waiting for data...")

    st.markdown("---")
    st.header("Download & Controls")
    if st.button("Download full CSV"):
        try:
            df_dl = pd.read_csv(CSV_PATH)
            csv_bytes = df_dl.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV file", data=csv_bytes, file_name=f"river_data_{int(time.time())}.csv")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    st.markdown("---")
    st.header("Smoothing & Plot Options")
    smoothing = st.checkbox("Enable moving average smoothing", value=False)
    window = st.number_input("Smoothing window (points)", min_value=1, max_value=200, value=5, step=1)
    show_points = st.checkbox("Show markers on lines", value=False)
    st.markdown("---")
    st.write("Rows in memory:", len(st.session_state.logs))

with right:
    st.header("Charts — Latest data (live)")
    # prepare DataFrame from session logs (last N)
    MAX_POINTS = st.slider("Points to show (most recent)", min_value=50, max_value=2000, value=400, step=50)
    df_plot = pd.DataFrame(st.session_state.logs[-MAX_POINTS:])

    if df_plot.empty:
        st.info("No data available yet. Waiting for MQTT messages...")
    else:
        # Ensure types
        df_plot["ts_order"] = df_plot["timestamp"].fillna(0).astype(int)
        df_plot = df_plot.sort_values("ts_order")
        # x axis label: use datetime if available, else generated from timestamp
        if "datetime" in df_plot.columns and df_plot["datetime"].notnull().any():
            xvals = df_plot["datetime"]
        else:
            xvals = pd.to_datetime(df_plot["timestamp"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")

        # helper for smoothing
        def maybe_smooth(series):
            if smoothing and len(series) >= window and window > 1:
                return series.rolling(window=window, min_periods=1).mean()
            return series

        # 1) Water level chart
        st.subheader("Water Level (cm)")
        if "water_level_cm" in df_plot.columns and df_plot["water_level_cm"].notnull().any():
            y = pd.to_numeric(df_plot["water_level_cm"], errors="coerce")
            ys = maybe_smooth(y)
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=xvals, y=ys, mode="lines+markers" if show_points else "lines", name="Water level (cm)", line=dict(width=2)))
            fig_w.update_layout(title="Water level over time", xaxis_title="Time", yaxis_title="Distance (cm, smaller = higher water)", height=320)
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("No water level data yet.")

        # 2) Danger level chart
        st.subheader("Danger Level")
        if "danger_level" in df_plot.columns and df_plot["danger_level"].notnull().any():
            y = pd.to_numeric(df_plot["danger_level"], errors="coerce").fillna(0).astype(int)
            ys = maybe_smooth(y)
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=xvals, y=ys, mode="lines+markers" if show_points else "lines", name="Danger level", line=dict(width=2), fill='tozeroy'))
            fig_d.update_layout(title="Danger level (0=no, 1,2,3)", xaxis_title="Time", yaxis_title="Danger Level", height=240)
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No danger level data yet.")

        # 3) Rain sensor chart
        st.subheader("Rain Level")
        if "rain_level" in df_plot.columns and df_plot["rain_level"].notnull().any():
            y = pd.to_numeric(df_plot["rain_level"], errors="coerce").fillna(0).astype(int)
            ys = maybe_smooth(y)
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(x=xvals, y=ys, name="Rain level (0-3)"))
            fig_r.update_layout(title="Rain intensity (0=none, 1=drizzle, 2=moderate, 3=heavy)", xaxis_title="Time", yaxis_title="Rain Level", height=260)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No rain level data yet.")

        # 4) Combined multi-axis chart
        st.subheader("Combined Chart — Water (left) + Rain (bar) + Danger (right)")
        fig = go.Figure()
        # Water level on primary y
        if "water_level_cm" in df_plot.columns and df_plot["water_level_cm"].notnull().any():
            y_w = pd.to_numeric(df_plot["water_level_cm"], errors="coerce")
            ys_w = maybe_smooth(y_w)
            fig.add_trace(go.Scatter(x=xvals, y=ys_w, mode="lines+markers" if show_points else "lines", name="Water level (cm)", yaxis="y1"))
        # Rain level as bar on same axis (secondary)
        if "rain_level" in df_plot.columns and df_plot["rain_level"].notnull().any():
            y_r = pd.to_numeric(df_plot["rain_level"], errors="coerce").fillna(0)
            ys_r = maybe_smooth(y_r)
            fig.add_trace(go.Bar(x=xvals, y=ys_r, name="Rain level", yaxis="y2", opacity=0.6))
        # Danger level as line on tertiary axis (we'll overlay on right)
        if "danger_level" in df_plot.columns and df_plot["danger_level"].notnull().any():
            y_d = pd.to_numeric(df_plot["danger_level"], errors="coerce").fillna(0)
            ys_d = maybe_smooth(y_d)
            fig.add_trace(go.Scatter(x=xvals, y=ys_d, mode="lines+markers" if show_points else "lines", name="Danger level", yaxis="y3", line=dict(dash="dot", width=3)))

        # layout - three axes
        fig.update_layout(
            title="Combined: water level (left) • rain (bar) • danger (right)",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Water level (cm)", side="left"),
            yaxis2=dict(title="Rain level", overlaying="y", side="right", position=0.98, showgrid=False),
            yaxis3=dict(title="Danger level", overlaying="y", side="right", position=1.0, anchor="free"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Recent Logs (latest first)")
    if st.session_state.logs:
        df_recent = pd.DataFrame(st.session_state.logs)[::-1].head(200)
        st.dataframe(df_recent)
    else:
        st.write("—")

# After UI render, drain queue so next rerun shows fresh data
_ = process_queue()
