"""
PROJECT ECHO Field Calibrate — Rapid Environmental Tuning
Run this script at the demo location to automatically record 25 seconds
of each gesture and train the model for the new environment.

Usage:
    python field_calibrate.py --port COM8
"""

import serial
import csv
import os
import sys
import time
import argparse
import numpy as np
import subprocess

# ─── Config ───────────────────────────────────────────────────────────
BAUD_RATE = 115200
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DURATION_PER_GESTURE = 45  # seconds per gesture (longer = more accurate model)

LABELS = [
    ("empty_room", "EMPTY ROOM (Step completely away from the boards)"),
    ("idle", "IDLE (Stand perfectly still between the boards)"),
    ("gesture_halt", "HALT (Raise right fist up and down)"),
    ("gesture_advance", "ADVANCE (Sweep arms out wide and back)"),
]

def parse_csi_line(line):
    try:
        if not line.startswith("CSI_DATA"): return None, None
        bracket_start = line.index("[")
        bracket_end = line.index("]")
        metadata_part = line[:bracket_start].rstrip(",")
        csi_part = line[bracket_start + 1:bracket_end].strip()
        
        meta_fields = metadata_part.split(",")
        metadata = {"rssi": meta_fields[3].strip()} if len(meta_fields) > 3 else {"rssi": "0"}
        csi_raw = [int(x) for x in csi_part.split() if x.strip()]
        return metadata, csi_raw
    except (ValueError, IndexError):
        return None, None

def compute_amplitudes(csi_raw):
    amplitudes = []
    for i in range(0, len(csi_raw) - 1, 2):
        I = csi_raw[i]
        Q = csi_raw[i + 1]
        amp = np.sqrt(I**2 + Q**2)
        amplitudes.append(round(amp, 4))
    return amplitudes

def main():
    parser = argparse.ArgumentParser(description="PROJECT ECHO Field Calibrate")
    parser.add_argument("--port", required=True, help="Serial port (e.g. COM8)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"╔═════════════════════════════════════════════════════════════╗")
    print(f"║     PROJECT ECHO — Tactical Rapid Field Calibration         ║")
    print(f"╚═════════════════════════════════════════════════════════════╝\n")

    print(f"[SERIAL] Connecting to {args.port}...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {args.port}: {e}")
        sys.exit(1)

    print("\n[READY] Preparing to record 3 gestures.")
    print(f"Each gesture takes {DURATION_PER_GESTURE} seconds.")
    print("Clear the space between the boards.")
    
    for label, desc in LABELS:
        input(f"\n>> Press ENTER when ready to record: {desc}...")
        
        filepath = os.path.join(DATA_DIR, f"{label}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "rssi"] + [f"amp_{i}" for i in range(64)])
            
            print(f"\n[RECORDING] {label.upper()} for {DURATION_PER_GESTURE} seconds. Go!")
            start_time = time.time()
            count = 0
            
            while time.time() - start_time < DURATION_PER_GESTURE:
                raw = ser.readline()
                if not raw: continue
                try: line = raw.decode("utf-8", errors="ignore").strip()
                except: continue
                
                meta, csi_raw = parse_csi_line(line)
                if meta is None: continue
                
                amps = compute_amplitudes(csi_raw)
                if len(amps) < 64: amps.extend([0.0]*(64-len(amps)))
                amps = amps[:64]
                
                writer.writerow([time.time(), meta["rssi"]] + amps)
                count += 1
                
                elapsed = time.time() - start_time
                remain = max(0, DURATION_PER_GESTURE - elapsed)
                print(f"\r  Progress: {elapsed/DURATION_PER_GESTURE:6.1%} | Time left: {remain:4.1f}s | Samples: {count:4d}", end="", flush=True)
                
        print(f"\n[DONE] Saved {count} samples for {label.upper()}.")

    ser.close()

    # Validate minimum sample counts before training
    MIN_SAMPLES = 50
    all_ok = True
    for label, desc in LABELS:
        filepath = os.path.join(DATA_DIR, f"{label}.csv")
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            if len(df) < MIN_SAMPLES:
                print(f"\n⚠ WARNING: {label.upper()} has only {len(df)} samples (need ≥{MIN_SAMPLES}). Consider re-recording!")
                all_ok = False
            else:
                print(f"  ✓ {label.upper()}: {len(df)} samples recorded.")
        except Exception:
            print(f"\n⚠ WARNING: Could not read {filepath}. Re-record this gesture!")
            all_ok = False

    if not all_ok:
        resp = input("\n⚠ Some gestures may have insufficient data. Continue training anyway? (y/n): ").strip().lower()
        if resp != "y":
            print("[ABORT] Fix the data and re-run field_calibrate.py.")
            return
    
    print("\n[TRAINING] Calibration complete. Training model now...")
    try:
        print("\n--- Running Signal Processing ---")
        subprocess.run([sys.executable, "signal_process.py"], check=True)
        print("\n--- Running Training ---")
        subprocess.run([sys.executable, "train_model.py"], check=True)
        print("\n✅ TACTICAL CALIBRATION SUCCESSFUL! PROJECT ECHO is ready for CQB deployment.")
        print("Run Tactical HUD: python echo_main.py --port " + args.port)
    except subprocess.CalledProcessError:
        print("\n❌ Error during training. Please check the logs above.")

if __name__ == "__main__":
    main()
