"""
serial_bridge.py
================
Reads ESP32 telemetry from COM3 (USB serial), appends it to a CSV log,
writes a `latest.json` snapshot for the Streamlit dashboard, and forwards
each DATA row to the Arduino UNO on COM6 for the 2.4" TFT display.

Close the PlatformIO Serial Monitor before running this script, because only
one program can own COM3 at a time.

Usage:
    python serial_bridge.py
"""

import csv
import json
import os
import time
from datetime import datetime

import serial

# ----------------------------- Configuration -------------------------------
ESP32_PORT   = "COM3"        # ESP32 USB serial
ESP32_BAUD   = 115200
UNO_PORT     = "COM6"        # Arduino UNO USB serial (TFT display)
UNO_BAUD     = 115200
FORWARD_TO_UNO = True

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "..", "data")
CSV_LOG      = os.path.join(DATA_DIR, "telemetry.csv")
LATEST_JSON  = os.path.join(DATA_DIR, "latest.json")

FIELDS = [
    "temperature_c", "voltage_v", "current_ma", "flow_lpm", "weight_kg",
    "accel_x_g", "accel_y_g", "accel_z_g",
    "pump_on", "firmware_status", "total_pulses", "total_flow_l",
]
EXPECTED_TOKENS = 1 + len(FIELDS)   # "DATA" + fields


def parse_data_row(line: str):
    """Parse one `DATA,...` row into a dict. Returns None if invalid."""
    line = line.strip()
    if not line.startswith("DATA,"):
        return None
    tokens = line.split(",")
    if len(tokens) != EXPECTED_TOKENS:
        return None
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "temperature_c":   float(tokens[1]),
            "voltage_v":       float(tokens[2]),
            "current_ma":      float(tokens[3]),
            "flow_lpm":        float(tokens[4]),
            "weight_kg":       float(tokens[5]),
            "accel_x_g":       float(tokens[6]),
            "accel_y_g":       float(tokens[7]),
            "accel_z_g":       float(tokens[8]),
            "pump_on":         int(tokens[9]) == 1,
            "firmware_status": tokens[10],
            "total_pulses":    int(tokens[11]),
            "total_flow_l":    float(tokens[12]),
        }
        return record
    except (ValueError, IndexError):
        return None


def append_csv(record: dict):
    write_header = not os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp"] + FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def write_latest(record: dict):
    payload = {"received_at": time.time(), "record": record}
    tmp = LATEST_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, LATEST_JSON)   # atomic-ish update for the dashboard


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"[bridge] Opening ESP32 on {ESP32_PORT} @ {ESP32_BAUD}...")
    esp = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=2)
    time.sleep(2)   # let the board reset after the port opens

    uno = None
    if FORWARD_TO_UNO:
        try:
            print(f"[bridge] Opening UNO on {UNO_PORT} @ {UNO_BAUD}...")
            uno = serial.Serial(UNO_PORT, UNO_BAUD, timeout=1)
            time.sleep(2)
        except serial.SerialException as exc:
            print(f"[bridge] WARNING: UNO not available ({exc}). "
                  f"Dashboard logging continues without TFT forwarding.")
            uno = None

    print("[bridge] Running. Press Ctrl+C to stop.")
    try:
        while True:
            raw = esp.readline().decode("utf-8", errors="replace")
            record = parse_data_row(raw)
            if record is None:
                continue

            append_csv(record)
            write_latest(record)

            if uno is not None:
                try:
                    uno.write((raw.strip() + "\n").encode("utf-8"))
                except serial.SerialException:
                    uno = None
                    print("[bridge] WARNING: UNO disconnected.")

            print(f"[bridge] {record['timestamp']}  "
                  f"T={record['temperature_c']:.1f}C  "
                  f"V={record['voltage_v']:.2f}  "
                  f"status={record['firmware_status']}")
    except KeyboardInterrupt:
        print("\n[bridge] Stopped by user.")
    finally:
        esp.close()
        if uno is not None:
            uno.close()


if __name__ == "__main__":
    main()
