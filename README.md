# Real-Time Medical Equipment Failure Prediction Using ML Techniques

AI-based smart infusion-pump monitoring prototype: an ESP32 collects
pump-related sensor telemetry, a Python serial bridge forwards it to a
Streamlit dashboard and an Arduino UNO + 2.4" TFT local display, and a
Random Forest model classifies pump operating condition (NORMAL / WARNING /
FAILURE) from dataset features.

> **⚠️ Academic prototype only.** This system is **not** a clinically validated
> medical device. The pump stays **OFF** by default and may only run in
> supervised bench demonstrations. Dashboard/model output supports human
> review; it does not automatically control any real clinical equipment.

---

## Repository structure

```
infusion-pump-ai-monitor/
├── esp32_firmware/        PlatformIO project (ESP32 DOIT DEVKIT V1)
│   ├── platformio.ini
│   └── src/main.cpp       Sensor reading, pump control, DATA serial output
├── python_bridge/
│   └── serial_bridge.py   COM3 (ESP32) -> CSV log + latest.json -> COM6 (UNO)
├── dashboard/
│   └── app.py             Streamlit dashboard (live / simulation / saved data)
├── ml_model/
│   └── train_model.py     Random Forest + Logistic Regression training script
├── arduino_uno_tft/
│   └── tft_display.ino    2.4" TFT local display sketch (MCUFRIEND_kbv)
├── data/                  Runtime CSV logs, latest.json snapshot (git-ignored)
├── requirements.txt
└── README.md
```

## Hardware

| Component | Function | Wiring |
|---|---|---|
| ESP32 DOIT DEVKIT V1 | Main controller | reads sensors, controls pump, USB serial to laptop |
| R385 12 V DC pump | Pump mechanism (demo) | driven via BTS7960, external 12 V supply |
| BTS7960 / IBT-2 motor driver | Pump control | R_EN=25, L_EN=26, RPWM=27, LPWM=14 |
| DHT22 | Temperature & humidity | data GPIO 4 |
| INA219 | Voltage / current / power | I2C SDA=21, SCL=22 |
| MPU6050 | Vibration (accel + gyro) | I2C SDA=21, SCL=22 |
| ZJ-S201 | Flow pulses | pulse input GPIO 33 |
| Load cell + HX711 | Weight / fluid-level proxy | DT=34, SCK=15 |
| Arduino UNO + 2.4" TFT | Local display | receives forwarded rows on COM6 |
| LEDs + buzzer | Local indication | green=32, yellow=19, red=18, white=23, buzzer=5 |

## Serial protocol

The ESP32 prints a human-readable block plus one machine-readable row per
second at 115200 baud:

```
DATA,temp,voltage,current,flow_rate,weight,accel_x,accel_y,accel_z,pump,status,pulses,total_flow
DATA,27.00,4.31,0.10,0.000,8.000,0.156,0.004,1.055,0,NORMAL,0,0.000
```

`status` is the **firmware rule-based label, not an AI prediction**.

## Setup & run

1. **ESP32 firmware** (PlatformIO / VS Code):
   ```bash
   cd esp32_firmware
   pio run --target upload
   ```

2. **Python environment**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (Random Forest + Logistic Regression):
   ```bash
   python ml_model/train_model.py
   ```
   If no real dataset is found, a clearly-labelled synthetic dataset of
   10,000 records is generated for pipeline demonstration — replace it with
   your measured data before citing any performance numbers.

4. **Serial bridge** (close PlatformIO Monitor first — only one program can
   own COM3):
   ```bash
   python python_bridge/serial_bridge.py
   ```

5. **Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

6. **TFT display**: open `arduino_uno_tft/tft_display.ino` in Arduino IDE,
   install the **MCUFRIEND_kbv** library, select Arduino UNO, upload.

## Current prototype status & known limitations (from project testing)

- Live AI prediction is **paused**: the trained model expects pressure and
  operating-hours inputs that the ESP32 does not provide, and the hardware has
  no pressure sensor or battery monitor — those fault classes cannot be
  verified from live readings.
- Flow-sensor readings are **uncalibrated**; the load cell requires calibration
  against known weights; INA219 readings should be checked against the actual
  pump supply.
- The TFT has been tested separately; continuous live ESP32→UNO forwarding
  still needs verification on your exact shield revision.
- Offline dataset metrics (e.g., ~99%+ Random Forest accuracy) do **not**
  establish live prototype performance; independent validation on unseen data
  is required.

## Future work

Calibrate/validate sensors; retrain the model on hardware-matched features;
verify the full COM3→Python→dashboard/COM6 pipeline; mobile app with verified
phone-number accounts and SMS alerts; multi-device dashboards; data export.

## Team

Chandru S (1VJ23AI012) · Harshal M (1VJ23AI020) · Hemanth H A (1VJ23AI022) ·
Mokshith R (1VJ23AI031)

Guide: Prof. Sangeetha C, Dept. of AI & ML, Vijaya Vittala Institute of
Technology, Bengaluru (VTU, 2026–2027)
