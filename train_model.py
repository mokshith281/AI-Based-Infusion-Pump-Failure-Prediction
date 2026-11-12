"""
train_model.py
==============
Trains the Machine Learning classifiers for the infusion-pump monitoring
dashboard: Random Forest (primary) and Logistic Regression (benchmark).

Features (7): temperature_c, pressure, flow_lpm, vibration, motor_current,
battery_voltage, operating_hours
Classes:      NORMAL / WARNING / FAILURE

IMPORTANT (honesty note):
-------------------------
The report's dataset details were not supplied with the repository, so this
script generates a clearly-labelled SYNTHETIC dataset of 10,000 records for
demonstration and pipeline testing. Replace `make_synthetic_dataset()` with
loading of your real collected dataset before citing any performance number.
Offline accuracy on a synthetic or single dataset does NOT establish live
prototype performance - validate on independent, unseen data.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "pump_dataset.csv")
RF_PATH      = os.path.join(BASE_DIR, "rf_model.joblib")
LR_PATH      = os.path.join(BASE_DIR, "lr_model.joblib")

FEATURES = ["temperature_c", "pressure", "flow_lpm", "vibration",
            "motor_current", "battery_voltage", "operating_hours"]


def make_synthetic_dataset(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic labelled dataset for pipeline demonstration.

    NORMAL:  typical operating ranges.
    WARNING: drifting toward limits (slightly high temp/vibration, etc.).
    FAILURE: clearly out-of-range measurements.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for label, n_cls in [("NORMAL", int(n * 0.70)),
                         (WARNING_PLACEHOLDER := "WARNING", int(n * 0.20)),
                         ("FAILURE", n - int(n * 0.70) - int(n * 0.20))]:
        if label == "NORMAL":
            t   = rng.normal(32, 2, n_cls)
            p   = rng.normal(102, 4, n_cls)
            f   = rng.normal(100, 3, n_cls)
            vib = rng.uniform(0.1, 0.4, n_cls)
            cur = rng.normal(1.1, 0.1, n_cls)
            bat = rng.normal(12.0, 0.3, n_cls)
            hrs = rng.uniform(0, 2000, n_cls)
        elif label == "WARNING":
            t   = rng.normal(38, 1.5, n_cls)
            p   = rng.normal(98, 6, n_cls)
            f   = rng.normal(92, 6, n_cls)
            vib = rng.uniform(0.4, 0.8, n_cls)
            cur = rng.normal(1.3, 0.15, n_cls)
            bat = rng.normal(11.2, 0.5, n_cls)
            hrs = rng.uniform(1000, 4000, n_cls)
        else:  # FAILURE
            t   = rng.normal(45, 2, n_cls)
            p   = rng.normal(85, 10, n_cls)
            f   = rng.normal(75, 10, n_cls)
            vib = rng.uniform(0.8, 1.5, n_cls)
            cur = rng.normal(1.8, 0.3, n_cls)
            bat = rng.normal(9.5, 1.0, n_cls)
            hrs = rng.uniform(3000, 8000, n_cls)

        rows.append(pd.DataFrame({
            "temperature_c": t, "pressure": p, "flow_lpm": f,
            "vibration": vib, "motor_current": cur,
            "battery_voltage": bat, "operating_hours": hrs,
            "status": label,
        }))

    df = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed)
    return df.round(2)


def evaluate(name: str, y_true, y_pred) -> dict:
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro') * 100:.2f}%")
    print(f"Recall   : {recall_score(y_true, y_pred, average='macro') * 100:.2f}%")
    print(f"F1-score : {f1_score(y_true, y_pred, average='macro') * 100:.2f}%")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred,
                           labels=["NORMAL", "WARNING", "FAILURE"]))
    print(classification_report(y_true, y_pred,
                                labels=["NORMAL", "WARNING", "FAILURE"]))
    return {"model": name,
            "accuracy": accuracy_score(y_true, y_pred)}


def main():
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
    else:
        print("No real dataset found - generating a SYNTHETIC demo dataset "
              "(10,000 records). Replace it with your measured data.")
        df = make_synthetic_dataset(10_000)
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print(f"Saved synthetic dataset to {DATASET_PATH}")

    X = df[FEATURES]
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ----- Random Forest (primary model) -----
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results = [evaluate("Random Forest", y_test, rf.predict(X_test))]

    # ----- Logistic Regression (benchmark) -----
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    results.append(evaluate("Logistic Regression", y_test, lr.predict(X_test)))

    joblib.dump(rf, RF_PATH)
    joblib.dump(lr, LR_PATH)
    print(f"\nSaved models:\n  {RF_PATH}\n  {LR_PATH}")

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nBest model: {best['model']} "
          f"(test accuracy {best['accuracy'] * 100:.2f}%)")
    print("\nNOTE: these are OFFLINE dataset results. They do not establish "
          "live prototype performance; validate on independent unseen data.")


if __name__ == "__main__":
    main()
