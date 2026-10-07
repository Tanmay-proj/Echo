"""
PROJECT ECHO ML Training Pipeline
Trains a Random Forest classifier on preprocessed CSI features.

Usage:
    python train_model.py
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
FEATURES_FILE = os.path.join(DATA_DIR, "features.pkl")
MODEL_FILE = os.path.join(MODELS_DIR, "echo_model.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "echo_scaler.pkl")


def main():
    print("=" * 60)
    print("PROJECT ECHO — ML Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading preprocessed features...")
    if not os.path.exists(FEATURES_FILE):
        print(f"[ERROR] {FEATURES_FILE} not found!")
        print("        Run signal_process.py first.")
        return

    with open(FEATURES_FILE, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    print(f"  ✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  ✓ Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("\n[2/5] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Train: {len(y_train)} samples")
    print(f"  ✓ Test:  {len(y_test)} samples")

    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[4/5] Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train_scaled, y_train)

    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"  ✓ Cross-validation accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    print("\n[5/5] Evaluating on test set...")
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'─' * 50}")
    print(f"  TEST ACCURACY: {accuracy:.1%}")
    print(f"{'─' * 50}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Confusion Matrix:")
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"{'':>20}", end="")
    for l in labels:
        print(f"{l:>18}", end="")
    print()
    for i, row_label in enumerate(labels):
        print(f"  Actual {row_label:>10}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>18}", end="")
        print()

    print(f"\nTop 10 Most Important Features:")
    feature_names = data.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {rank:>2}. {name:<30} {importances[idx]:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n✓ Model saved to: {MODEL_FILE}")

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {SCALER_FILE}")

    print(f"\n{'=' * 60}")
    if accuracy >= 0.80:
        print(f"  ✅ PASS — {accuracy:.0%} accuracy. Ready for real-time classification!")
    elif accuracy >= 0.60:
        print(f"  ⚠️  OKAY — {accuracy:.0%} accuracy. Workable but record more data for improvement.")
    else:
        print(f"  ❌ LOW — {accuracy:.0%} accuracy. Record more data or review gesture technique.")
    print(f"{'=' * 60}")
    print("\nNext step: Run rf_classifier.py or echo_main.py for live classification!")


if __name__ == "__main__":
    main()
