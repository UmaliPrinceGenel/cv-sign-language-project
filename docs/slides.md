# Real-Time Sign Language Word Gesture Recognition
## Milestone 4 (80% Progress Check) Presentation Deck

**Course:** CSST 106 — Computer Vision Final Project  
**Repository Tag:** `cp4-80`  
**Pipeline:** MediaPipe Tasks API + 63-D MLP + Geometric Landmark Refinement  
**Interactive Interface:** Gradio Web Application (`src/app.py`) & Desktop OpenCV Pipeline (`src/run.py`)  

---

<!-- slide -->
# Slide 1: Title & Team Overview

### Real-Time Sign Language Word Gesture Recognition on Commodity Laptop CPU
**CSST 106 Computer Vision • Milestone 4 (80% Progress Check)**

- **Target Deployment:** Standard consumer laptop without GPU acceleration
- **Current Milestone Achievements:**
  - 8-Class Vocabulary recognition at **53.0 FPS** (18.87 ms latency)
  - **86.25% Test Accuracy** and **0.897 Macro-F1** on 240 unseen test images
  - Full-featured interactive web application with confidence controls, live webcam stream, and cumulative counters
  - 10 documented test cases (`TC01`–`TC10`) including 3 hard failure modes

---

<!-- slide -->
# Slide 2: Problem Statement & Motivation

### The Challenge of Accessible Sign Language Interfaces
- **The Human Problem:** Deaf and hard-of-hearing individuals face persistent communication barriers when interacting with non-signing individuals in daily educational, healthcare, and retail environments.
- **Hardware Limitations:** Academic systems often rely on multi-GPU server clusters or expensive depth cameras (LiDAR, RealSense), making them impractical for standard user devices.
- **Technical Hurdles:**
  1. *CPU Real-Time Constraint:* System must sustain $\ge 15$ FPS (target $\ge 25$ FPS) on basic x86 laptop cores.
  2. *Illumination & Clutter Brittleness:* Classical skin-color filters fail when faces, hands, or wood furniture share similar chromaticity.
  3. *Look-Alike Gestures:* Gestures with identical digit counts (e.g. `PALM` vs `STOP` vs `OK`) trigger severe classification overlap.

---

<!-- slide -->
# Slide 3: Dataset & Target Vocabulary

### 8 Core Static Word Gestures (1,120 Total Labeled Images)
- **Controlled Splits:**
  - **Train:** 720 images (90 per class)
  - **Validation:** 160 images (20 per class)
  - **Test (Unseen):** 240 images (30 per class, frozen partition)
- **Vocabulary Categories:**
  1. `CALL`: Thumb and pinky extended, intermediate fingers curled.
  2. `DISLIKE`: Closed fist with thumb pointed vertically downward.
  3. `LIKE`: Closed fist with thumb pointed vertically upward.
  4. `MUTE`: Closed fist gesture.
  5. `OK`: Thumb and index fingertip pinched into an 'O' ring, 3 fingers extended.
  6. `PALM`: Open hand facing forward with fingers spread apart ($\ge 6.0^\circ$).
  7. `PEACE`: V-sign with index and middle fingers extended.
  8. `STOP`: Open hand facing forward with fingers held tightly together ($< 6.0^\circ$).

---

<!-- slide -->
# Slide 4: Methodology & Pipeline Architecture

### End-to-End CPU Pipeline Workflow
```
[Raw Webcam/Image] ➔ [MediaPipe Tasks API] ➔ [Active Hand Selector] ➔ [StandardScaler]
                            │ (21 3D Knuckles)        │ (min-y elevation)
                            ▼                         ▼
                   [2-Layer MLP 128x64] ➔ [Geometric Refiner] ➔ [Majority Vote] ➔ [Interactive UI]
```
- **Key Architectural Innovations:**
  1. **Active Hand Elevation Filter:** Instantiates MediaPipe with `num_hands=2` and selects the hand with minimum $y$ (highest elevation), completely eliminating resting lap hand distraction.
  2. **Geometric Landmark Refinement:** Scale- and position-invariant anatomical ratios verify thumb-index pinch (OK), inter-finger spread angle (PALM vs STOP), and pinky abduction (CALL).
  3. **Temporal Majority Voting:** 5-frame sliding window stabilizes output and eliminates single-frame flickers.

---

<!-- slide -->
# Slide 5: Quantitative Experimental Results

### Comparative Benchmark: Baseline vs Core vs Refined Milestone 4
All metrics evaluated on the frozen 240-image unseen test split:

| Metric / Dimension | Milestone 2 Baseline (HOG + SVM) | Milestone 3 Core (Raw MediaPipe + MLP) | Milestone 4 Current (`cp4-80` Refined) |
| :--- | :---: | :---: | :---: |
| **Pipeline** | YCrCb Mask + HOG + SVM | MediaPipe Tasks + MLP | MediaPipe + MLP + Geometric Refinement |
| **Unseen Test Accuracy** | < 30.0% (fails on scenes) | 73.75% (177/240) | **86.25% (207/240)** *(+12.50%)* |
| **Macro-F1 Score** | 0.284 | 0.777 | **0.897** *(+0.120)* |
| **Detected Hand Accuracy** | ~42.0% | 82.71% | **93.67%** *(+10.96%)* |
| **Inference Latency** | 6.09 ms | 21.18 ms | **18.87 ms** |
| **Throughput Speed** | 164.3 FPS (cropped ROI) | 47.2 FPS | **53.0 FPS** (3.5x real-time target) |
| **Model Storage Footprint** | 23.3 MB | 613.8 KB | **613.8 KB (`models/best.pt`)** |

*Exact reproduction: `python src/evaluate.py --split test --weights models/best.pt`*

---

<!-- slide -->
# Slide 6: Milestone 4 Per-Class Benchmark Breakdown

### Precision, Recall, and F1-Score on Frozen 240-Sample Test Set
- **`PEACE` (0.967 F1):** Exceptional discrimination of the V-shape (29/30 correct).
- **`OK` (0.949 F1):** Geometric pinch ratio boosted recall from 0.857 to 0.933, resolving historical confusion with STOP.
- **`PALM` (0.931 F1):** Inter-finger angular spread measurement elevated F1 from 0.745 to 0.931.
- **`DISLIKE` (0.915 F1):** Clear vertical orientation vectors ensure 90.0% recall and 93.1% precision.
- **`STOP` (0.897 F1):** Correctly separated from PALM using finger spread threshold ($< 6.0^\circ$).
- **`CALL` (0.852 F1) & `LIKE` (0.847 F1):** Improved via pinky-separation heuristics; minor foreshortening errors remain.
- **`MUTE` (0.815 F1):** Closed fist posture correctly recognized in majority of samples.

---

<!-- slide -->
# Slide 7: Interactive Application Features

### Dual Application Interfaces (`src/app.py` & `src/run.py`)
1. **Gradio Browser Web Application (`src/app.py`):**
   - **One-Time Model Pre-loading:** Kept resident in RAM; zero reload lag across interactions.
   - **Multi-Modal Input:** Single image file upload with quick-test gallery, live browser webcam streaming, and `.mp4` video batch processing.
   - **Interactive HUD Controls:** Real-time Confidence Threshold slider ($0.0 - 1.0$), toggleable skeleton keypoints, toggleable prediction HUD banner.
   - **Cumulative Tally Table:** Live counters tracking occurrences of each gesture class, with a one-click Reset button.
2. **Optimized Desktop HUD (`src/run.py`):**
   - Threaded background camera capture with Windows Media Foundation (`MSMF`).
   - Idle palm detector subsampling (keeps camera at 30 FPS even when no hand is present).
   - Bottom overlay bar with live detection counts.

---

<!-- slide -->
# Slide 8: Operating Limitations & Error Analysis

### Real-World Operating Envelope & Concrete Failure Cases
- **Target Operating Envelope:** Ambient room lighting $\ge 150\text{ lux}$, subject distance $0.4\text{m} - 1.8\text{m}$, unoccluded hand.
- **Failure Mode 1: Severe Foreshortening (`TC07`):**
  - *Symptom:* When the wrist is tilted directly toward the camera, the extended pinky collapses in 2D projection, causing `CALL` to be classified as `MUTE` or `LIKE`.
  - *Fix for MS5:* Transition to 3D world landmarks (`world_landmarks`) measured in metric millimeters.
- **Failure Mode 2: Finger Spread Boundary Zone (`TC08`):**
  - *Symptom:* When a subject holds their fingers slightly loose ($~6.2^\circ$), `STOP` can cross into `PALM`.
- **Failure Mode 3: Out-of-Vocabulary Gestures (`TC09`):**
  - *Symptom:* Untrained signs (e.g. closed fist) are assigned to the nearest class with high softmax score.
  - *Fix for MS5:* Add open-set rejection threshold to label untrained poses as "Unknown Gesture".

---

<!-- slide -->
# Slide 9: Live Demo Plan & Milestone 5 Roadmap

### Demo Execution Strategy for In-Class Presentation
1. **Step 1: Test Cases Validation:** Walk through `TC01`–`TC10` using the quick-test gallery in `src/app.py`.
2. **Step 2: Live Webcam Demonstration:** Show real-time streaming, latency readout (14 ms), and dynamic slider thresholding.
3. **Step 3: Teacher's Test Input:** Ingest teacher-supplied challenge images live through the web interface.
4. **Step 4: Reproducible CLI Verification:** Execute `python src/evaluate.py --split test --weights models/best.pt` to prove that output metrics match the report exactly.

### Path to 100% Completion (Milestone 5):
- Implement 3D world landmark coordinates for foreshortening invariance.
- Train final model on wrist-centered normalized coordinates.
- Final comprehensive technical report and presentation delivery.

