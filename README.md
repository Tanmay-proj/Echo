# PROJECT ECHO — Tactical Silent Command Communication (TSCC) System

> **Electromagnetic Command & Human Operation via WiFi Interferometric Neural Sensing**

![Status](https://img.shields.io/badge/Status-Operational_PoC-brightgreen)
![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20Python-blue)
![Domain](https://img.shields.io/badge/Domain-Tactical_RF_Sensing%20%7C%20CQB-orange)
![Security](https://img.shields.io/badge/EW_Resilience-Project_SHADOW_Ready-purple)

---

## 1. The Tactical Problem in CQB & Zero-Visibility Operations

In Close Quarters Battle (CQB), underground bunker clearing, or night operations involving heavy smoke, tear gas, or blackout conditions, traditional squad communication breaks down:
1. **Optical Failure:** Smoke, dust, and darkness completely blind optical cameras, night-vision devices (NVDs), and visual line-of-sight hand signals.
2. **Acoustic Vulnerability:** Speaking into tactical radios breaks audio discipline, revealing squad leader positions to enemy sensors or nearby combatants.
3. **Through-Wall Obstacles:** Squad members are often separated by interior drywall, wooden doors, or foliage where line-of-sight is impossible.

## 2. The RF Physics Solution: WiFi CSI Interferometry

**PROJECT ECHO** solves this by using **Radio Frequency (RF) Interferometry** on standard 2.4 GHz WiFi frequencies (Channel 6).
* Radio waves at 2.4 GHz naturally penetrate smoke, dust, complete darkness, and non-metallic interior walls.
* As a tactical commander performs specific hand signals within the RF field between two portable transceiver nodes, the physical movement distorts the multipath propagation of the electromagnetic waves.
* By extracting the **Channel State Information (CSI)** across 64 OFDM subcarriers, PROJECT ECHO applies a trained Machine Learning model to translate micro-Doppler RF disturbances into instant, silent squad orders displayed on tactical HUDs or wrist-receivers.

---

## 3. System Architecture

```mermaid
graph LR
    subgraph Tactical Field Nodes (2-3 Meters Apart)
        AP["Node 1 (TX/AP)<br/>ESP-WROOM-32"]
        STA["Node 2 (RX/STA)<br/>ESP-WROOM-32"]
    end
    
    subgraph Commander
        CMD["🪖 Squad Leader<br/>Silent Hand Signal"] -.->|"Distorts<br/>RF Multipath"| AP
        AP -->|"CSI Probe Packets<br/>(2.4 GHz WiFi)"| STA
    end
    
    subgraph Tactical Processing Unit (Laptop / Edge PC)
        SER["Serial Stream<br/>(COM Port)"]
        PRE["CSI Preprocessor<br/>(324 Windowed Features)"]
        ML["Random Forest<br/>Tactical Classifier"]
        HUD["Squad Leader HUD<br/>Real-Time Command Display"]
    end
    
    STA -->|USB Serial| SER
    SER --> PRE --> ML --> HUD
```

---

## 4. Tactical Command Dictionary

| Command | Gesture Action | Tactical Meaning | Squad HUD Reaction |
|---|---|---|---|
| **IDLE** | Sit / Stand perfectly still | **SQUAD STANDBY** | Squad maintains standard defensive perimeter |
| **HALT ✊** | Raise right arm with clenched fist | **HOLD POSITION / FREEZE** | Squad units halt immediately, locking defensive angles |
| **ADVANCE 🫸** | Sweep both arms outward from chest | **ASSAULT FORWARD** | Squad units fan out in tactical V-formation to advance |

---

## 5. Hardware & Software Setup

### Hardware Requirements
* **2x ESP32 Development Boards** (ESP-WROOM-32 or WROVER)
* **2x USB Data Cables** (One connected to laptop/PC, one powered via wall adapter or battery bank)
* **Laptop / Edge Computer** (Windows 10/11, Python 3.8+)

### Software Structure
```
ECHO/
├── README.md                     # Tactical System Documentation
├── data/                         # Recorded CSI gesture datasets (.csv)
├── models/                       # Trained Random Forest models (.pkl)
└── scripts/
    ├── field_calibrate.py        # 90-second Tactical Field Calibration
    ├── rf_logger.py              # Manual CSI RF logger
    ├── signal_process.py         # Feature extraction (Window size: 50)
    ├── train_model.py            # Model training & serialization
    ├── rf_classifier.py          # Thread-safe real-time inference engine
    ├── tactical_hud.py           # Tactical CQB Squad HUD visualization
    └── echo_main.py              # Primary TSCC operational execution script
```

---

## 6. Rapid Field Deployment Guide

Because RF multipath reflections vary between rooms (bunkers, hallways, outdoor terrain), PROJECT ECHO features a **Rapid Environmental Calibration (REC)** mode. When deploying in a new tactical environment, the system calibrates in under 90 seconds.

### Step 1: Position Nodes
1. Plug **Node 1 (AP)** into your laptop USB port (e.g., `COM8`).
2. Power **Node 2 (STA)** using a USB battery bank or wall adapter, placed **2 to 3 meters apart** from Node 1.
3. Position yourself (or the squad leader) between the two nodes.

### Step 2: Run 90-Second Rapid Calibration
Open PowerShell in the `scripts` directory and run:
```powershell
python field_calibrate.py --port COM8
```
* Stand still for **25 seconds** when prompted for `IDLE`.
* Perform the **HALT ✊** gesture repeatedly for **25 seconds**.
* Perform the **ADVANCE 🫸** gesture repeatedly for **25 seconds**.

The script will automatically extract 324 spatial-temporal features per window and train the Random Forest model, outputting the environmental verification accuracy (target: ≥80%).

### Step 3: Launch Tactical Squad HUD
To initiate real-time silent command recognition:
```powershell
python echo_main.py --port COM8
```
A live tactical CQB display will appear. As the commander performs gestures in zero visibility, the squad HUD updates instantly with the verified command and ML confidence metrics.

---

## 7. Electronic Warfare (EW) Resilience & Project SHADOW

In contested environments, enemy forces may attempt to blind tactical CSI sensors using RF jamming or spoofing. PROJECT ECHO is engineered to integrate with **Project SHADOW** (an offensive EW deception weapon):
* **The Attack:** An adversary deploys a rogue ESP32 (`SHADOW`) flooding Channel 6 with rhythmic, mechanical packets to simulate false movement ("digital ghosts").
* **The Defense (Synergy):** Because mechanical RF jamming lacks the complex, non-linear micro-Doppler signature of human biomechanics, PROJECT ECHO can be trained with an adversarial `EW_ATTACK` classification class. When under attack, the UI triggers a `🔴 JAMMING DETECTED 🔴` failsafe, switching squad units to autonomous defensive holding patterns.

---

## 8. Future Scope: Hardware & AI Scaling

1. **USRP SDR Integration:** Upgrading from ESP32 nodes to Full-Duplex Software Defined Radios (e.g., USRP B210) running custom **FMCW Radar chirps** on 5.8 GHz.
2. **Moving Target Indication (MTI):** Implementing phase-coherent MTI filtering to mathematically erase static walls and background room reflections, achieving 100% zero-calibration environment independence.
3. **Domain-Adversarial Neural Networks (DANN):** Replacing Random Forest with PyTorch DANN architectures that use Gradient Reversal Layers (GRL) to make gesture classification completely invariant to room geometry.
