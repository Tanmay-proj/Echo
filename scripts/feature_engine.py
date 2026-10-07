"""
PROJECT ECHO — Shared Feature Extraction Engine v2 (Environment-Independent)

This module is the SINGLE SOURCE OF TRUTH for feature computation.
Both signal_process.py (offline training) and rf_classifier.py (real-time)
import and use these exact same functions.

DESIGN PRINCIPLES:
  1. All features measure MOVEMENT and CHANGE — never absolute amplitude values.
  2. Doppler FFT decomposes motion into velocity components.
     Static environment = DC (frequency 0) = automatically removed.
  3. This makes the system environment-independent by physics, not heuristics.

Feature Vector (226 features total):
  ── Movement Statistics (198) ──
  - Per-subcarrier temporal std (64)
  - Per-subcarrier temporal range (64)
  - Per-subcarrier mean |diff| (64)
  - 6 global motion descriptors

  ── Doppler FFT Features (28) ──
  - Doppler power spectrum bins 1-24 (24): Motion speed distribution (DC removed)
  - Peak Doppler bin (1): Dominant speed of motion
  - Total Doppler energy (1): Total motion intensity
  - Doppler centroid (1): Center-of-mass speed
  - Doppler spread (1): Bandwidth of motion speeds
"""

import numpy as np

NUM_SUBCARRIERS = 64
SMOOTHING_WINDOW = 5
DOPPLER_BINS = 24  # FFT bins to keep (excluding DC and mirror)


def smooth_window(arr):
    """Apply causal moving average smoothing (window=5) to a 2D numpy array.
    arr shape: (num_frames, num_subcarriers)
    Returns smoothed array of same shape.
    """
    import pandas as pd
    return pd.DataFrame(arr).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values


def compute_doppler_spectrum(arr):
    """Compute the Doppler power spectrum from a CSI time-series window.

    For each subcarrier, compute FFT along the time axis.
    Sum magnitudes across all subcarriers to get a single Doppler spectrum.
    Remove DC component (static environment) — this is the key to
    environment independence.

    Args:
        arr: numpy array of shape (window_size, 64)

    Returns:
        doppler_spectrum: 1D array of length DOPPLER_BINS (DC removed)
    """
    n_frames = arr.shape[0]

    # Apply Hanning window to reduce spectral leakage
    hanning = np.hanning(n_frames).reshape(-1, 1)
    windowed = arr * hanning

    # FFT along time axis for each subcarrier
    fft_result = np.fft.fft(windowed, axis=0)
    fft_magnitude = np.abs(fft_result)

    # Sum across all subcarriers to get a single Doppler spectrum
    doppler_full = fft_magnitude.sum(axis=1)  # (window_size,)

    # Remove DC component (index 0) — this eliminates static environment
    # Keep only positive frequencies (bins 1 to N//2)
    n_keep = min(DOPPLER_BINS, n_frames // 2)
    doppler_spectrum = doppler_full[1:1 + n_keep]

    # Pad if window is too small
    if len(doppler_spectrum) < DOPPLER_BINS:
        doppler_spectrum = np.pad(doppler_spectrum, (0, DOPPLER_BINS - len(doppler_spectrum)))

    return doppler_spectrum


def extract_features(window_arr):
    """Extract environment-independent features from a single window.

    Args:
        window_arr: numpy array of shape (window_size, 64) — raw amplitudes.

    Returns:
        1D numpy array of features (226 elements).
    """
    # Apply smoothing
    arr = smooth_window(window_arr)

    # ══════════════════════════════════════════════════════════════════
    # PART 1: Movement Statistics (198 features)
    # ══════════════════════════════════════════════════════════════════

    # Per-subcarrier temporal statistics
    std_per_sc = np.std(arr, axis=0, ddof=1)       # (64,)
    range_per_sc = np.ptp(arr, axis=0)              # (64,)

    # Frame-to-frame absolute differences
    frame_diffs = np.abs(np.diff(arr, axis=0))      # (window-1, 64)
    mean_diff_per_sc = frame_diffs.mean(axis=0)     # (64,)

    # Global motion descriptors
    var_per_sc = np.var(arr, axis=0, ddof=1)        # (64,)

    total_motion_energy = np.array([np.sum(var_per_sc)])

    max_std = np.max(std_per_sc) + 1e-9
    active_count = np.sum(std_per_sc > 0.1 * max_std)
    motion_spread = np.array([active_count / NUM_SUBCARRIERS])

    peak_sc = np.array([np.argmax(var_per_sc) / NUM_SUBCARRIERS])

    total_diff_per_frame = frame_diffs.sum(axis=1)
    median_diff = np.median(total_diff_per_frame) + 1e-9
    burst_ratio = np.array([np.mean(total_diff_per_frame > 2.0 * median_diff)])

    diff_energy_std = np.array([np.std(total_diff_per_frame)])

    var_clipped = np.clip(var_per_sc, 1e-9, None)
    log_mean = np.mean(np.log(var_clipped))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(var_clipped)
    spectral_flatness = np.array([geo_mean / (arith_mean + 1e-9)])

    # ══════════════════════════════════════════════════════════════════
    # PART 2: Doppler FFT Features (28 features)
    # Environment independence by physics: DC removal = room removal
    # ══════════════════════════════════════════════════════════════════

    doppler_spectrum = compute_doppler_spectrum(arr)  # (24,)

    # Normalize Doppler spectrum (relative power, not absolute)
    doppler_total = np.sum(doppler_spectrum) + 1e-9
    doppler_normalized = doppler_spectrum / doppler_total

    # Doppler statistics
    freq_bins = np.arange(1, DOPPLER_BINS + 1)

    # Peak Doppler bin (which speed is dominant, normalized)
    peak_doppler = np.array([np.argmax(doppler_spectrum) / DOPPLER_BINS])

    # Total Doppler energy (log scale for better dynamic range)
    total_doppler_energy = np.array([np.log1p(doppler_total)])

    # Doppler centroid (center-of-mass of the spectrum)
    centroid = np.sum(freq_bins * doppler_normalized) / DOPPLER_BINS
    doppler_centroid = np.array([centroid])

    # Doppler spread (std of the spectrum)
    spread = np.sqrt(np.sum(doppler_normalized * (freq_bins - centroid * DOPPLER_BINS)**2))
    doppler_spread = np.array([spread / DOPPLER_BINS])

    # ══════════════════════════════════════════════════════════════════
    # ASSEMBLE FINAL FEATURE VECTOR
    # ══════════════════════════════════════════════════════════════════

    features = np.concatenate([
        # Part 1: Movement (198)
        std_per_sc,           # 64
        range_per_sc,         # 64
        mean_diff_per_sc,     # 64
        total_motion_energy,  # 1
        motion_spread,        # 1
        peak_sc,              # 1
        burst_ratio,          # 1
        diff_energy_std,      # 1
        spectral_flatness,    # 1
        # Part 2: Doppler (28)
        doppler_normalized,   # 24
        peak_doppler,         # 1
        total_doppler_energy, # 1
        doppler_centroid,     # 1
        doppler_spread,       # 1
    ])

    return features


def extract_doppler_for_display(window_arr):
    """Extract just the Doppler spectrum for HUD visualization.
    Returns the raw (un-normalized) Doppler spectrum for display.
    """
    arr = smooth_window(window_arr)
    return compute_doppler_spectrum(arr)


def get_feature_names():
    """Return human-readable names for each feature."""
    names = []
    for prefix in ["std", "range", "diff"]:
        for i in range(NUM_SUBCARRIERS):
            names.append(f"{prefix}_sc{i}")
    names += [
        "total_motion_energy",
        "motion_spread",
        "peak_motion_sc",
        "burst_ratio",
        "diff_energy_std",
        "spectral_flatness",
    ]
    for i in range(DOPPLER_BINS):
        names.append(f"doppler_bin{i+1}")
    names += [
        "peak_doppler",
        "total_doppler_energy",
        "doppler_centroid",
        "doppler_spread",
    ]
    return names


# Total feature count for validation
FEATURE_COUNT = NUM_SUBCARRIERS * 3 + 6 + DOPPLER_BINS + 4  # 226
