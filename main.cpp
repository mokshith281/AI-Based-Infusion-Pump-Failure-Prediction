/**
 * ============================================================================
 * Real-Time Medical Equipment Failure Prediction Using ML Techniques
 * ESP32 Firmware  -  ESP32 DOIT DEVKIT V1 (PlatformIO / Arduino framework)
 *
 * Academic prototype ONLY - NOT a clinically validated medical device.
 * The pump stays OFF by default and may only run in supervised bench tests.
 *
 * Wiring (see README.md for the full table):
 *   DHT22  (temp/humidity)     -> GPIO 4
 *   INA219 + MPU6050 (I2C)     -> SDA GPIO 21, SCL GPIO 22
 *   ZJ-S201 flow sensor pulse  -> GPIO 33
 *   HX711 load cell            -> DT GPIO 34, SCK GPIO 15
 *   BTS7960 motor driver       -> R_EN 25, L_EN 26, RPWM 27, LPWM 14
 *   LEDs                       -> GREEN 32, YELLOW 19, RED 18, WHITE 23
 *   Buzzer                     -> GPIO 5
 *
 * Serial telemetry format (comma separated, fixed field order):
 * DATA,temp,voltage,current,flow_rate,weight,accel_x,accel_y,accel_z,
 *      pump,status,pulses,total_flow
 * Example:
 * DATA,27.00,4.31,0.10,0.000,8.000,0.156,0.004,1.055,0,NORMAL,0,0.000
 * ============================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <Adafruit_INA219.h>
#include <Adafruit_MPU6050.h>
#include <HX711.h>

// ----------------------------- Pin map -------------------------------------
#define PIN_DHT        4
#define PIN_FLOW       33
#define PIN_HX711_DT   34
#define PIN_HX711_SCK  15
#define PIN_R_EN       25
#define PIN_L_EN       26
#define PIN_RPWM       27
#define PIN_LPWM       14
#define LED_GREEN      32
#define LED_YELLOW     19
#define LED_RED        18
#define LED_WHITE      23
#define PIN_BUZZER     5

// I2C
#define I2C_SDA 21
#define I2C_SCL 22

// --------------------------- Configuration ---------------------------------
#define DHT_TYPE           DHT22
#define SAMPLE_INTERVAL_MS 1000

// ZJ-S201 calibration: pulses per litre. UNCALIBRATED placeholder value -
// must be validated against a known volume before trusting flow readings.
#define FLOW_PULSES_PER_LITRE 1500.0

// HX711 calibration factor. UNCALIBRATED placeholder - verify with known
// weights before trusting load-cell readings.
#define LOADCELL_CALIBRATION_FACTOR 986.0

// Proposed demonstration cycle: 6 s ON / 114 s OFF.
// The pump is OFF by default. Set to true ONLY for a supervised bench test.
static const bool PUMP_DEMO_ENABLED = false;
static const unsigned long PUMP_ON_MS  = 6000UL;
static const unsigned long PUMP_OFF_MS = 114000UL;

// Simple firmware rule limits (firmware status label, NOT an AI prediction)
#define TEMP_WARNING_C   38.0
#define TEMP_CRITICAL_C  45.0
#define FLOW_MIN_LPM     0.05   // expected minimum flow while pump is ON

// ----------------------------- Objects --------------------------------------
DHT            dht(PIN_DHT, DHT_TYPE);
Adafruit_INA219 ina219(0x40);
Adafruit_MPU6050 mpu;
HX711            scale;

// ----------------------------- Globals --------------------------------------
volatile unsigned long flowPulseCount = 0;
unsigned long          lastSampleMs   = 0;
unsigned long          pumpCycleStart = 0;
bool                   pumpState      = false;   // OFF by default
unsigned long          totalPulses    = 0;
float                  totalFlowLitres = 0.0;

bool dhtOK = false, inaOK = false, mpuOK = false, flowOK = false, hxOK = false;

// ----------------------------- ISR ------------------------------------------
void IRAM_ATTR flowPulseISR() { flowPulseCount++; }

// ----------------------------- Setup ----------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println(F("=== AI Smart Infusion Pump Monitor - ESP32 firmware ==="));

  // Indicator outputs
  pinMode(LED_GREEN, OUTPUT);  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED,   OUTPUT);  pinMode(LED_WHITE,  OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(LED_GREEN, HIGH);   // power-on self test
  delay(200);
  digitalWrite(LED_GREEN, LOW);

  // Motor driver pins
  pinMode(PIN_R_EN, OUTPUT); pinMode(PIN_L_EN, OUTPUT);
  pinMode(PIN_RPWM, OUTPUT); pinMode(PIN_LPWM, OUTPUT);
  pumpStop();

  // Flow sensor pulse input
  pinMode(PIN_FLOW, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_FLOW), flowPulseISR, RISING);

  // I2C bus shared by INA219 and MPU6050
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);   // 100 kHz - more robust with long breadboard wiring

  dhtOK  = dht.begin();
  inaOK  = ina219.begin();
  mpuOK  = mpu.begin(0x68, &Wire);
  if (mpuOK) {
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  flowOK = true;   // pulse input armed

  hxOK   = scale.begin(PIN_HX711_DT, PIN_HX711_SCK);
  if (hxOK) {
    scale.set_scale(LOADCELL_CALIBRATION_FACTOR);
    scale.tare(20);   // zero reference with no load applied
  }

  printSystemStatus();

  pumpCycleStart = millis();
  lastSampleMs   = millis();
}

// ----------------------------- Pump control ---------------------------------
void pumpStart() {
  digitalWrite(PIN_R_EN, HIGH);
  digitalWrite(PIN_L_EN, HIGH);
  analogWrite(PIN_RPWM, 200);   // modest duty for bench demonstration
  digitalWrite(PIN_LPWM, LOW);
  pumpState = true;
}

void pumpStop() {
  digitalWrite(PIN_RPWM, LOW);
  digitalWrite(PIN_LPWM, LOW);
  digitalWrite(PIN_R_EN, LOW);
  digitalWrite(PIN_L_EN, LOW);
  pumpState = false;
}

// ----------------------------- Status logic ---------------------------------
// Firmware rule-based status ONLY. It is NOT the machine-learning prediction.
String computeFirmwareStatus(float tempC, float flowLpm) {
  if (tempC >= TEMP_CRITICAL_C) return "CRITICAL";
  if (tempC >= TEMP_WARNING_C)  return "WARNING";
  if (pumpState && flowLpm < FLOW_MIN_LPM) return "WARNING";
  return "NORMAL";
}

void updateIndicators(const String& status) {
  digitalWrite(LED_GREEN,  status == "NORMAL"   ? HIGH : LOW);
  digitalWrite(LED_YELLOW, status == "WARNING"  ? HIGH : LOW);
  digitalWrite(LED_RED,    status == "CRITICAL" ? HIGH : LOW);
  digitalWrite(LED_WHITE,  status == "CRITICAL" ? HIGH : LOW); // all LEDs on critical
  if (status == "CRITICAL") {
    tone(PIN_BUZZER, 2000, 200);
  }
}

// ----------------------------- Reporting ------------------------------------
void printSystemStatus() {
  Serial.println(F("--------------- SYSTEM STATUS ---------------"));
  Serial.print(F("DHT22       : ")); Serial.println(dhtOK  ? F("WORKING") : F("ERROR"));
  Serial.print(F("INA219      : ")); Serial.println(inaOK  ? F("WORKING") : F("ERROR"));
  Serial.print(F("MPU6050     : ")); Serial.println(mpuOK  ? F("WORKING") : F("ERROR"));
  Serial.print(F("Flow Sensor : ")); Serial.println(flowOK ? F("WORKING") : F("ERROR"));
  Serial.print(F("HX711       : ")); Serial.println(hxOK   ? F("WORKING") : F("ERROR"));
}

// ----------------------------- Main loop ------------------------------------
void loop() {
  unsigned long now = millis();

  // Optional supervised bench demonstration cycle (disabled by default)
  if (PUMP_DEMO_ENABLED) {
    unsigned long elapsed = now - pumpCycleStart;
    if (!pumpState && elapsed >= PUMP_OFF_MS) { pumpStart(); pumpCycleStart = now; }
    else if (pumpState && elapsed >= PUMP_ON_MS) { pumpStop(); pumpCycleStart = now; }
  } else if (pumpState) {
    pumpStop();
  }

  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = now;

  // ---- Read sensors ----
  float tempC = NAN, hum = NAN;
  if (dhtOK) {
    tempC = dht.readTemperature();
    hum   = dht.readHumidity();
    if (isnan(tempC)) tempC = 0.0;
  }

  float busV = 0.0, currentMa = 0.0, powerMw = 0.0;
  if (inaOK) {
    busV     = ina219.getBusVoltage_V();
    currentMa = ina219.getCurrent_mA();
    powerMw   = ina219.getPower_mW();
  }

  float ax = 0.0, ay = 0.0, az = 1.0;   // az ~ 1 g at rest
  if (mpuOK) {
    sensors_event_t a, g, tEvent;
    mpu.getEvent(&a, &g, &tEvent);
    ax = a.acceleration.x / 9.80665;
    ay = a.acceleration.y / 9.80665;
    az = a.acceleration.z / 9.80665;
  }

  // Flow: convert pulses counted in the last interval
  noInterrupts();
  unsigned long pulses = flowPulseCount;
  flowPulseCount = 0;
  interrupts();
  totalPulses += pulses;
  float flowLpm = (pulses / FLOW_PULSES_PER_LITRE) * 60.0;  // UNCALIBRATED
  totalFlowLitres += pulses / FLOW_PULSES_PER_LITRE;

  float weightKg = 0.0;
  if (hxOK) {
    weightKg = scale.get_units(10);      // UNCALIBRATED placeholder factor
    if (weightKg < 0) weightKg = 0.0;
  }

  String status = computeFirmwareStatus(tempC, flowLpm);
  updateIndicators(status);

  // ---- Human-readable serial block ----
  Serial.println(F("=============================================="));
  Serial.println(F("                 SENSOR DATA"));
  Serial.println(F("=============================================="));
  Serial.print(F("Temperature   : ")); Serial.print(tempC, 2); Serial.println(F(" C"));
  Serial.print(F("Humidity      : ")); Serial.print(hum, 2);   Serial.println(F(" %"));
  Serial.print(F("Pump Voltage  : ")); Serial.print(busV, 2);  Serial.println(F(" V"));
  Serial.print(F("Pump Current  : ")); Serial.print(currentMa, 2); Serial.println(F(" mA"));
  Serial.print(F("Pump Power    : ")); Serial.print(powerMw, 2);  Serial.println(F(" mW"));
  Serial.print(F("Acceleration X: ")); Serial.print(ax, 3); Serial.print(F(" g | Y: "));
  Serial.print(ay, 3); Serial.print(F(" g | Z: ")); Serial.print(az, 3); Serial.println(F(" g"));
  Serial.print(F("Flow Pulses   : ")); Serial.println(pulses);
  Serial.print(F("Flow Rate     : ")); Serial.print(flowLpm, 3); Serial.println(F(" L/min"));
  Serial.print(F("Total Flow    : ")); Serial.print(totalFlowLitres, 3); Serial.println(F(" L"));
  Serial.print(F("Weight        : ")); Serial.print(weightKg, 3); Serial.println(F(" kg"));
  Serial.print(F("Pump Status   : ")); Serial.println(pumpState ? F("ON") : F("OFF"));

  // ---- Machine-readable DATA row (fixed order, for the Python bridge) ----
  Serial.print(F("DATA,"));
  Serial.print(tempC, 2);      Serial.print(F(","));
  Serial.print(busV, 2);       Serial.print(F(","));
  Serial.print(currentMa, 2);  Serial.print(F(","));
  Serial.print(flowLpm, 3);    Serial.print(F(","));
  Serial.print(weightKg, 3);   Serial.print(F(","));
  Serial.print(ax, 3);         Serial.print(F(","));
  Serial.print(ay, 3);         Serial.print(F(","));
  Serial.print(az, 3);         Serial.print(F(","));
  Serial.print(pumpState ? 1 : 0); Serial.print(F(","));
  Serial.print(status);        Serial.print(F(","));
  Serial.print(totalPulses);   Serial.print(F(","));
  Serial.println(totalFlowLitres, 3);
}
