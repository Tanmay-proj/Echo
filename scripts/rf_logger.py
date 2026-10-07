"""
PROJECT ECHO RF Logger
Reads CSI data from ESP32 serial port and logs to labeled CSV files.

Usage:
    python rf_logger.py --port COM8

Controls (press key + Enter):
    1 = Record IDLE (Squad Standby)
    2 = Record GESTURE_HALT (raise right fist)
    3 = Record GESTURE_ADVANCE (sweep arms out)
    s = Stop recording (just monitor)
    q = Quit
"""

import serial
import csv
import os
import sys
import time
import threading
import argparse
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────
BAUD_RATE = 115200
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

LABELS = {
    "1": "idle",
    "2": "gesture_halt",
    "3": "gesture_advance",
}

# ─── Globals ──────────────────────────────────────────────────────────
current_label = None
recording = False
sample_count = 0
lock = threading.Lock()


def parse_csi_line(line):
    """
    Parse a CSI_DATA line from ESP32-CSI-Tool.
    Format: CSI_DATA,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,
            not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,
            channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,
            real_time_set,real_timestamp,len,[i0 q0 i1 q1 ...]
    Returns: (metadata_dict, csi_raw_list) or (None, None) on failure.
    """
    try:
        if not line.startswith("CSI_DATA"):
            return None, None

        bracket_start = line.index("[")
        bracket_end = line.index("]")

        metadata_part = line[:bracket_start].rstrip(",")
        csi_part = line[bracket_start + 1:bracket_end].strip()

        meta_fields = metadata_part.split(",")
        meta_keys = [
            "type", "role", "mac", "rssi", "rate", "sig_mode", "mcs",
            "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc",
            "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel",
            "secondary_channel", "local_timestamp", "ant", "sig_len",
            "rx_state", "real_time_set", "real_timestamp", "len"
        ]

        metadata = {}
        for i, key in enumerate(meta_keys):
            if i < len(meta_fields):
                metadata[key] = meta_fields[i].strip()

        csi_raw = [int(x) for x in csi_part.split() if x.strip()]

        return metadata, csi_raw

    except (ValueError, IndexError):
        return None, None


def compute_amplitudes(csi_raw):
    """
    Convert raw I/Q pairs to amplitude per subcarrier.
    csi_raw = [I0, Q0, I1, Q1, ...] → amplitudes = [amp0, amp1, ...]
    """
    amplitudes = []
    for i in range(0, len(csi_raw) - 1, 2):
        I = csi_raw[i]
        Q = csi_raw[i + 1]
        amp = np.sqrt(I**2 + Q**2)
        amplitudes.append(round(amp, 4))
    return amplitudes


def input_listener():
    """Background thread: listen for keyboard commands."""
    global current_label, recording, sample_count

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     PROJECT ECHO TSCC RF Logger — Controls               ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  1 + Enter = Record IDLE (Standby)                       ║")
    print("║  2 + Enter = Record GESTURE_HALT (Hold Position)         ║")
    print("║  3 + Enter = Record GESTURE_ADVANCE (Assault Forward)    ║")
    print("║  s + Enter = Stop recording (monitor only)               ║")
    print("║  q + Enter = Quit                                        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            break

        with lock:
            if cmd == "q":
                print("\n[EXIT] Shutting down...")
                os._exit(0)
            elif cmd == "s":
                recording = False
                current_label = None
                print("[STOPPED] Monitoring only — not recording.")
            elif cmd in LABELS:
                current_label = LABELS[cmd]
                recording = True
                sample_count = 0
                print(f"[RECORDING] Label: {current_label.upper()} → saving to {current_label}.csv")
            else:
                print(f"[?] Unknown command: '{cmd}'. Use 1/2/3/s/q.")


def main():
    global sample_count

    parser = argparse.ArgumentParser(description="PROJECT ECHO RF Logger")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help="Baud rate")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    csv_files = {}
    csv_writers = {}
    for label in LABELS.values():
        filepath = os.path.join(DATA_DIR, f"{label}.csv")
        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        f = open(filepath, "a", newline="")
        writer = csv.writer(f)
        if not file_exists:
            header = ["timestamp", "rssi"] + [f"amp_{i}" for i in range(64)]
            writer.writerow(header)
        csv_files[label] = f
        csv_writers[label] = writer

    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()

    print(f"\n[SERIAL] Opening {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"[SERIAL] Connected! Waiting for CSI data...\n")
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {args.port}: {e}")
        print("[HELP]  Check Device Manager → Ports (COM & LPT) for the correct port.")
        print("[HELP]  Make sure no other program (Arduino IDE Serial Monitor) is using the port.")
        sys.exit(1)

    lines_received = 0
    csi_lines_parsed = 0

    try:
        while True:
            raw_line = ser.readline()
            if not raw_line:
                continue

            try:
                line = raw_line.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            lines_received += 1

            metadata, csi_raw = parse_csi_line(line)
            if metadata is None:
                if lines_received <= 30 or "CSI" in line.upper():
                    print(f"[DEBUG] {line[:120]}")
                continue

            csi_lines_parsed += 1
            amplitudes = compute_amplitudes(csi_raw)

            if len(amplitudes) < 64:
                amplitudes.extend([0.0] * (64 - len(amplitudes)))
            amplitudes = amplitudes[:64]

            rssi = metadata.get("rssi", "0")
            timestamp = time.time()

            avg_amp = np.mean(amplitudes)
            bar_len = int(min(avg_amp, 40))
            bar = "█" * bar_len + "░" * (40 - bar_len)

            with lock:
                status = f"REC:{current_label.upper()}" if recording else "MONITOR"
                print(
                    f"\r[{status}] CSI#{csi_lines_parsed:>6} | RSSI:{rssi:>4} | "
                    f"AvgAmp:{avg_amp:>6.1f} |{bar}| "
                    f"Samples:{sample_count:>5}",
                    end="", flush=True
                )

                if recording and current_label:
                    row = [timestamp, rssi] + amplitudes
                    csv_writers[current_label].writerow(row)
                    csv_files[current_label].flush()
                    sample_count += 1

    except KeyboardInterrupt:
        print("\n\n[EXIT] Ctrl+C received. Closing...")
    finally:
        ser.close()
        for f in csv_files.values():
            f.close()
        print("[DONE] All CSV files saved in:", DATA_DIR)


if __name__ == "__main__":
    main()
