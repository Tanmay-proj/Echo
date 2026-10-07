import pickle
import os

MODEL_FILE = r"E:\WIFI_INTERFEROMETRIC_NEURAL_SENSING\echo\models\echo_model.pkl"
SCALER_FILE = r"E:\WIFI_INTERFEROMETRIC_NEURAL_SENSING\echo\models\echo_scaler.pkl"
FEATURES_FILE = r"E:\WIFI_INTERFEROMETRIC_NEURAL_SENSING\echo\data\features.pkl"

print("=" * 60)
print("PROJECT ECHO — Full Diagnostic Report")
print("=" * 60)

# 1. Check data files
print("\n--- DATA FILES ---")
data_dir = r"E:\WIFI_INTERFEROMETRIC_NEURAL_SENSING\echo\data"
for f in ["empty_room.csv", "idle.csv", "gesture_halt.csv", "gesture_advance.csv"]:
    fp = os.path.join(data_dir, f)
    if os.path.exists(fp):
        size = os.path.getsize(fp)
        # Count lines
        with open(fp, "r") as fh:
            lines = sum(1 for _ in fh) - 1  # minus header
        print(f"  ✓ {f}: {size} bytes, {lines} data rows")
    else:
        print(f"  ✗ {f}: NOT FOUND")

# 2. Check features.pkl
print("\n--- FEATURES ---")
if os.path.exists(FEATURES_FILE):
    with open(FEATURES_FILE, "rb") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]
    import numpy as np
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels: {dict(zip(unique, counts))}")
else:
    print("  ✗ features.pkl NOT FOUND")

# 3. Check model
print("\n--- MODEL ---")
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"  Model type: {type(model).__name__}")
    print(f"  Classes: {model.classes_}")
    print(f"  Num classes: {len(model.classes_)}")
    print(f"  Num estimators: {model.n_estimators}")
else:
    print("  ✗ echo_model.pkl NOT FOUND")

if os.path.exists(SCALER_FILE):
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Scaler type: {type(scaler).__name__}")
    print(f"  Scaler n_features: {scaler.n_features_in_}")
else:
    print("  ✗ echo_scaler.pkl NOT FOUND")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
