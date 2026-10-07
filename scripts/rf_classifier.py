"""
PROJECT ECHO Real-Time RF Classifier (v2 — Environment-Independent)
Reads live CSI from serial, classifies gestures using movement-only features,
and outputs labels. Uses the shared feature_engine for zero train/inference mismatch.

Usage (standalone test):
    python rf_classifier.py --port COM8
"""

import serial
import os
import sys
import time
import pickle
import numpy as np
import threading
from collections import deque

from feature_engine import extract_features, extract_doppler_for_display, FEATURE_COUNT, DOPPLER_BINS

BAUD_RATE = 115200
WINDOW_SIZE = 50
NUM_SUBCARRIERS = 64
CONFIDENCE_THRESHOLD = 0.30
COOLDOWN_SECONDS = 1.0

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODEL_FILE = os.path.join(MODELS_DIR, "echo_model.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "echo_scaler.pkl")


class RealtimeClassifier:
    """Real-time CSI gesture classifier using environment-independent features."""

    def __init__(self, port, baud=BAUD_RATE):
        self.port = port
        self.baud = baud
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.current_label = "idle"
        self.current_confidence = 0.0
        self.current_doppler = np.zeros(DOPPLER_BINS)
        self.last_gesture_time = 0
        self.running = False
        self.lock = threading.Lock()

        print("[MODEL] Loading trained model...")
        if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
            print("[ERROR] Trained model not found!")
            print("        Expected files:")
            print(f"          - {MODEL_FILE}")
            print(f"          - {SCALER_FILE}")
            print("        Run 'python field_calibrate.py --port COM8' first to train the model.")
            sys.exit(1)
        with open(MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_FILE, "rb") as f:
            self.scaler = pickle.load(f)
        print(f"[MODEL] ✓ Model loaded ({len(self.model.classes_)} classes, {FEATURE_COUNT} features).")

    def parse_csi_line(self, line):
        try:
            if not line.startswith("CSI_DATA"):
                return None

            bracket_start = line.index("[")
            bracket_end = line.index("]")
            csi_part = line[bracket_start + 1:bracket_end].strip()
            csi_raw = [int(x) for x in csi_part.split() if x.strip()]

            amplitudes = []
            for i in range(0, len(csi_raw) - 1, 2):
                amp = np.sqrt(csi_raw[i]**2 + csi_raw[i + 1]**2)
                amplitudes.append(amp)

            if len(amplitudes) < NUM_SUBCARRIERS:
                amplitudes.extend([0.0] * (NUM_SUBCARRIERS - len(amplitudes)))
            return amplitudes[:NUM_SUBCARRIERS]

        except (ValueError, IndexError):
            return None

    def classify_window(self):
        if len(self.buffer) < WINDOW_SIZE:
            return "idle", 0.0

        window = np.array(list(self.buffer))  # (50, 64)

        # Use the EXACT SAME feature extraction as training
        features = extract_features(window)

        # Also compute raw Doppler spectrum for HUD visualization
        doppler = extract_doppler_for_display(window)

        features = features.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)

        proba = self.model.predict_proba(features_scaled)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        label = self.model.classes_[pred_idx]

        now = time.time()
        if label not in ["idle", "empty_room"] and confidence < CONFIDENCE_THRESHOLD:
            label = "idle"
            confidence = 0.0
        elif label not in ["idle", "empty_room"] and (now - self.last_gesture_time) < COOLDOWN_SECONDS:
            pass
        else:
            if label not in ["idle", "empty_room"]:
                self.last_gesture_time = now

        return label, confidence, doppler

    def get_state(self):
        with self.lock:
            return self.current_label, self.current_confidence, self.current_doppler.copy()

    def serial_loop(self):
        print(f"\n[SERIAL] Connecting to {self.port} at {self.baud}...")
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
        except serial.SerialException as e:
            print(f"[ERROR] Cannot open {self.port}: {e}")
            self.running = False
            return

        print(f"[SERIAL] ✓ Connected! Classifying gestures live...\n")
        self.running = True

        count = 0
        while self.running:
            try:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                amplitudes = self.parse_csi_line(line)
                if amplitudes is None:
                    continue

                self.buffer.append(amplitudes)
                count += 1

                if count % 10 == 0 and len(self.buffer) >= WINDOW_SIZE:
                    label, confidence, doppler = self.classify_window()
                    with self.lock:
                        self.current_label = label
                        self.current_confidence = confidence
                        self.current_doppler = doppler

                    icon = {"empty_room": "👻", "idle": "⏸", "gesture_halt": "✊", "gesture_advance": "🫸"}.get(label, "?")
                    print(
                        f"\r  {icon}  {label.upper():<20} "
                        f"confidence: {confidence:.0%}  "
                        f"buffer: {len(self.buffer)}/{WINDOW_SIZE}",
                        end="", flush=True
                    )

            except Exception as e:
                print(f"\n[ERROR] {e}")
                continue

        ser.close()
        print("\n[SERIAL] Disconnected.")

    def start(self):
        thread = threading.Thread(target=self.serial_loop, daemon=True)
        thread.start()
        return thread

    def stop(self):
        self.running = False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PROJECT ECHO Real-Time Classifier v2")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help="Baud rate")
    args = parser.parse_args()

    classifier = RealtimeClassifier(args.port, args.baud)
    thread = classifier.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[EXIT] Stopping classifier...")
        classifier.stop()


if __name__ == "__main__":
    main()
