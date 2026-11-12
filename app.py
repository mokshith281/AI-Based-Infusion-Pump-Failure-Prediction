"""
app.py
======
Streamlit dashboard for the AI Smart Infusion Pump monitoring prototype.

Run:
    streamlit run dashboard/app.py

Views:
  * Live monitoring  - reads the `data/latest.json` snapshot written by
                       python_bridge/serial_bridge.py
  * AI simulation    - uses generated sample values with the trained
                       Random Forest model (ml_model/rf_model.joblib)
  * Saved live data  - charts the CSV log written by the serial bridge

NOTE: the live AI prediction is paused when the model expects inputs the
ESP32 hardware does not provide (pressure, operating hours). The displayed
NORMAL/WARNING/CRITICAL label in Live mode is the *firmware* status, not a
verified machine-learning prediction.
"""

import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
LATEST_JSON = os.path.join(DATA_DIR, "latest.json")
CSV_LOG     = os.path.join(DATA_DIR, "telemetry.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "ml_model", "rf_model.joblib")

MODEL_FEATURES = ["temperature_c", "pressure", "flow_lpm", "vibration",
                  "motor_current", "battery_voltage", "operating_hours"]
STALE_SECONDS = 30

st.set_page_config(page_title="Infusion Pump AI Monitor", layout="wide")
st.title("🧠 INFUSION PUMP AI MONITOR")
st.caption("Real-Time Medical Equipment Failure Prediction Using Machine Learning")

# ------------------------------ Sidebar ------------------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")
    interval = st.slider("Monitoring interval (seconds)", 1, 30, 3)

    st.divider()
    st.header("🧠 AI Model")
    st.write("Algorithm: **Random Forest**")
    st.write("Features: **7**")
    model_loaded = os.path.exists(MODEL_PATH)
    st.write("Model Status:", "Loaded ✅" if model_loaded else "Not found ⚠️")
    if model_loaded:
        st.caption(f"Trained on: {MODEL_FEATURES}")

    st.divider()
    st.header("📊 Dashboard view")
    view = st.radio("Choose a view",
                    ["Live monitoring", "AI simulation", "Saved live data"])


# --------------------------- Helper functions ------------------------------
def read_latest():
    """Return (record, age_seconds) from latest.json, or (None, None)."""
    if not os.path.exists(LATEST_JSON):
        return None, None
    try:
        with open(LATEST_JSON) as f:
            payload = json.load(f)
        return payload["record"], time.time() - payload["received_at"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None, None


def load_model():
    if not model_loaded:
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def metric_card(col, title, value, unit=""):
    col.metric(title, f"{value} {unit}".strip())


# =========================== LIVE MONITORING ===============================
if view == "Live monitoring":
    st.subheader("📡 Live ESP32 Telemetry")
    st.info("The Python bridge reads COM3, saves each complete DATA row for this "
            "page, and forwards it to the UNO on COM6. Keep the PlatformIO "
            "Monitor closed while the bridge is running.")

    record, age = read_latest()

    if record is None:
        st.warning("No telemetry received yet. Start `python python_bridge/serial_bridge.py`.")
        st.stop()

    if age is not None and age > STALE_SECONDS:
        st.warning(f"Bridge data is stale ({int(age)} seconds old). "
                   "Check the bridge terminal.")
    else:
        st.success(f"Dashboard ready  ·  last update {int(age)} s ago")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Temperature", f"{record['temperature_c']:.2f}", "°C")
    metric_card(c2, "Voltage",     f"{record['voltage_v']:.2f}", "V")
    metric_card(c3, "Current",     f"{record['current_ma']:.2f}", "mA")
    metric_card(c4, "Flow",        f"{record['flow_lpm']:.3f}", "L/min")

    c5, c6, c7, c8 = st.columns(4)
    metric_card(c5, "Load cell (uncalibrated)", f"{record['weight_kg']:.3f}", "kg")
    metric_card(c6, "Pump", "ON" if record["pump_on"] else "OFF")
    metric_card(c7, "Firmware status", record["firmware_status"])
    metric_card(c8, "Acceleration X/Y/Z",
                f"{record['accel_x_g']:.3f} / {record['accel_y_g']:.3f} / "
                f"{record['accel_z_g']:.3f}", "g")

    st.caption(f"Total pulses: {record['total_pulses']}   ·   "
               f"Total flow: {record['total_flow_l']:.3f} L   ·   "
               f"Timestamp: {record['timestamp']}")

    st.divider()
    st.subheader("🧠 AI Prediction (offline model)")
    model = load_model()
    if model is None:
        st.warning("Random Forest model not found. Train it first with "
                   "`python ml_model/train_model.py`.")
    else:
        st.warning("Live AI prediction is paused: the trained model expects "
                   "pressure and operating-hours inputs that the ESP32 does not "
                   "provide. Retrain the model on hardware-matched features to "
                   "enable live prediction.")
        st.caption("The status shown above is the firmware rule-based label, "
                   "not a verified machine-learning prediction.")

    time.sleep(interval)
    st.rerun()

# =========================== AI SIMULATION =================================
elif view == "AI simulation":
    st.subheader("🧪 AI Simulation")
    st.caption("Uses generated sample values for the model. This does not read "
               "the physical prototype.")

    model = load_model()
    if model is None:
        st.error("Model not found. Run `python ml_model/train_model.py` first.")
        st.stop()

    rng = np.random.default_rng()
    sim = pd.DataFrame({
        "temperature_c":   rng.normal(36.5, 2.0, 50).round(2),
        "pressure":        rng.normal(102.0, 5.0, 50).round(2),
        "flow_lpm":        rng.normal(100.0, 5.0, 50).round(2),
        "vibration":       rng.uniform(0.2, 1.0, 50).round(2),
        "motor_current":   rng.normal(1.1, 0.15, 50).round(2),
        "battery_voltage": rng.normal(12.0, 0.5, 50).round(2),
        "operating_hours": rng.uniform(50, 500, 50).round(2),
    })
    preds = model.predict(sim[MODEL_FEATURES])
    proba = model.predict_proba(sim[MODEL_FEATURES]).max(axis=1)

    sim["status"] = preds
    sim["confidence"] = (proba * 100).round(1).astype(str) + "%"
    sim.insert(0, "time",
               [(datetime.now() - pd.Timedelta(seconds=4 * i)).strftime("%H:%M:%S")
                for i in range(len(sim))])

    st.subheader("📋 Recent Predictions")
    st.dataframe(sim[["time"] + MODEL_FEATURES + ["status", "confidence"]],
                 use_container_width=True)

    st.subheader("📈 Individual Sensor Trends")
    st.line_chart(sim.set_index("time")[["temperature_c", "pressure",
                                         "flow_lpm", "vibration"]])
    st.line_chart(sim.set_index("time")[["motor_current", "battery_voltage"]])

    st.subheader("🎲 AI Prediction Probabilities (latest sample)")
    classes = list(model.classes_)
    st.bar_chart(pd.DataFrame({"probability": model.predict_proba(sim[MODEL_FEATURES])[-1]},
                              index=classes))

    st.success(f"Simulation log active — {len(sim)} records generated.")

# =========================== SAVED LIVE DATA ===============================
else:
    st.subheader("💾 Saved Live Data")
    if not os.path.exists(CSV_LOG):
        st.warning("No telemetry log found yet. Run the serial bridge first.")
        st.stop()

    df = pd.read_csv(CSV_LOG, parse_dates=["timestamp"]).set_index("timestamp")
    st.caption(f"{len(df)} records logged.")

    st.subheader("📈 Real-Time Sensor Trends")
    st.line_chart(df[["temperature_c", "voltage_v", "current_ma", "flow_lpm"]])
    st.line_chart(df[["accel_x_g", "accel_y_g", "accel_z_g"]])

    st.subheader("📋 Logged Records")
    st.dataframe(df.tail(50), use_container_width=True)
