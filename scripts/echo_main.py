"""
PROJECT ECHO Main — Full Tactical Integration
Connects the real-time RF classifier to the Tactical Squad Leader HUD.

Usage:
    python echo_main.py --port COM8
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rf_classifier import RealtimeClassifier
from tactical_hud import TacticalHUD


def main():
    parser = argparse.ArgumentParser(description="PROJECT ECHO TSCC — Full Tactical Integration")
    parser.add_argument("--port", required=True, help="Serial port for ESP32 AP (e.g., COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    args = parser.parse_args()

    print("=" * 66)
    print("  PROJECT ECHO — Tactical Silent Command Communication (TSCC)")
    print("  Full Pipeline: Silent Gesture → WiFi CSI → AI → Squad HUD")
    print("=" * 66)

    print("\n[1/2] Starting real-time CSI tactical classifier...")
    classifier = RealtimeClassifier(args.port, args.baud)
    classifier_thread = classifier.start()

    print("[WAIT] Waiting for CSI data stream...")
    timeout = 10
    start = time.time()
    while not classifier.running and (time.time() - start) < timeout:
        time.sleep(0.3)

    if not classifier.running:
        print("[ERROR] Could not connect to ESP32. Check:")
        print("        - Correct COM port?")
        print("        - ESP32 AP board is plugged in and flashed?")
        print("        - No other program using the port?")
        sys.exit(1)

    print("[2/2] Launching Tactical Squad Leader HUD...")
    print("\n" + "─" * 66)
    print("  LIVE! Perform tactical gestures between the ESP32 boards.")
    print("  The squad HUD will display your silent orders in real time.")
    print("  Press ESC in the HUD window to quit.")
    print("─" * 66 + "\n")

    demo = TacticalHUD()
    try:
        demo.run(get_external_state=classifier.get_state)
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        classifier.stop()
        print("\n[EXIT] PROJECT ECHO shut down cleanly.")


if __name__ == "__main__":
    main()
