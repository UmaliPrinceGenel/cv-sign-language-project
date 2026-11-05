# Real-Time Sign Language Recognition & Captioning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe Tasks API](https://img.shields.io/badge/MediaPipe-Tasks%20API-007FFF?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Milestone](https://img.shields.io/badge/Milestone-3%20(cp3--core)-success)](#3-current-state-milestone-3-main-method-cp3-core)
[![Course](https://img.shields.io/badge/Course-CSST%20106-blue)](#7-team--course-attribution)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **CSST 106 - Computer Vision Project**  
> An edge-capable, low-latency computer vision system that detects static hand gestures from live webcam video in real time and automatically renders on-screen word captions to bridge communication gaps for the deaf and hard-of-hearing.

---

## 📑 Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Project History & Evolution (Milestones 1 & 2)](#2-project-history--evolution-milestones-1--2)
  - [Milestone 1: Inception, Scoping & Dataset Curation](#milestone-1-inception-scoping--dataset-curation-cp1-proposal)
  - [Milestone 2: Classical Computer Vision Baseline (HOG + SVM)](#milestone-2-classical-computer-vision-baseline-hog--svm-cp2-50)
  - [The Classical Failure Mode: Why Hand ROIs Failed in the Wild](#the-classical-failure-mode-why-hand-rois-failed-in-the-wild)
- [3. Current State: Milestone 3 Main Method (`cp3-core`)](#3-current-state-milestone-3-main-method-cp3-core)
  - [Deep Learning Architecture: MediaPipe Tasks + MLP](#deep-learning-architecture-mediapipe-tasks--mlp)
  - [Milestone 3 Benchmark Performance](#milestone-3-benchmark-performance)
  - [Baseline vs. Main Method Comparative Evaluation](#baseline-vs-main-method-comparative-evaluation)
  - [Error Analysis & Results Report](#error-analysis--results-report)
- [4. Repository Structure](#4-repository-structure)
- [5. Installation & Setup](#5-installation--setup)
- [6. Usage & Deliverables](#6-usage--deliverables)
- [7. Team & Course Attribution](#7-team--course-attribution)

---

## 1. Project Overview

Deaf and hard-of-hearing individuals frequently encounter communication barriers during everyday face-to-face interactions (e.g., at service counters, classrooms, clinics, and customer support desks) when speaking with non-signers. Relying on smartphone typing or slow finger-spelling charts interrupts natural engagement.

This project delivers a **real-time sign language recognition and captioning system** running entirely on a standard laptop CPU using an off-the-shelf webcam. The system detects, classifies, and overlays instant subtitle banners for **8 static whole-word sign gestures**:

| Gesture | Sign Language Word / Intent | Visual Pose Description |
| :---: | :---: | :--- |
| **`CALL`** | *"Call Me"* / *"Phone"* | Thumb and pinky extended outward; index, middle, and ring fingers folded into palm. |
| **`DISLIKE`** | *"No"* / *"Dislike"* | Closed fist with thumb pointed vertically downward. |
| **`LIKE`** | *"Yes"* / *"Like"* / *"Good"* | Closed fist with thumb pointed vertically upward. |
| **`MUTE`** | *"Quiet"* / *"Mute"* | Closed fist held centrally (or index finger held vertically over lips). |
| **`OK`** | *"OK"* / *"Understood"* | Thumb and index fingertips touching in a ring; middle, ring, and pinky extended upward. |
| **`PALM`** | *"Hello"* / *"Open Hand"* | Open palm facing the camera with all five fingers extended upright. |
| **`PEACE`** | *"Peace"* / *"Two"* | Index and middle fingers extended upward in a 'V' shape; ring, pinky, and thumb folded. |
| **`STOP`** | *"Stop"* / *"Halt"* | Flat, upright open hand held forward at shoulder level with fingers together. |

---

## 2. Project History & Evolution (Milestones 1 & 2)

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│        Milestone 1        │      │        Milestone 2        │      │        Milestone 3        │
│      (cp1-proposal)       │ ───► │         (cp2-50)          │ ───► │        (cp3-core)         │
│  Dataset Curation & Spike │      │    HOG + SVM Baseline     │      │   MediaPipe Tasks + MLP   │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

### Milestone 1: Inception, Scoping & Dataset Curation (`cp1-proposal`)
- **Dataset Assembly:** Curated an 8-class subset sampled from the **HaGRID (HAnd Gesture Recognition Image Dataset)** distribution, combined with locally captured webcam validation samples (see [`docs/datacard.md`](docs/datacard.md)).
- **Stratified Partitioning:** Structured a strict 70% / 15% / 15% stratified split across all 8 classes ensuring zero subject or room overlap between splits:
  - **Training Split:** 1,120 images (140 per class) + horizontal flip augmentation.
  - **Validation Split:** 240 images (30 per class).
  - **Unseen Test Split:** 241 images (~30 per class).
- **Feasibility Spike:** Implemented early 20-sample contact sheet generation and fundamental image operations (grayscale, 5×5 Gaussian blur denoising, Otsu thresholding, Sobel edge detection) in `src/spike.py`.

### Milestone 2: Classical Computer Vision Baseline (HOG + SVM) (`cp2-50`)
In Milestone 2, we developed a classical, hand-crafted feature engineering pipeline (`src/baseline.py`):
1. **Color Segmentation:** Converted BGR frames to **YCrCb color space** and thresholded skin pixels using calibrated chrominance ranges ($Cr \in [133, 173]$, $Cb \in [77, 127]$).
2. **Morphological Filtering:** Applied elliptical $5 \times 5$ morphological closing to fill interior hand voids.
3. **Contour Geometry:** Extracted the largest skin contour to compute **7 Scale- and Rotation-Invariant Hu Moments**, convexity defects, contour aspect ratio, and solidity.
4. **Texture & Orientation:** Applied CLAHE contrast enhancement followed by **Histogram of Oriented Gradients (HOG)** with $16 \times 16$ stride and 9 angular bins, concatenating into a 3,790-dimensional feature vector.
5. **Classification:** Trained an RBF-kernel Support Vector Machine (SVM, $C=10.0$, $\gamma=\text{scale}$).

### The Classical Failure Mode: Why Hand ROIs Failed in the Wild
- **Validation Metric:** On pre-cropped hand Regions-of-Interest (ROIs) utilizing HaGRID's ground-truth JSON bounding boxes, the SVM achieved **96.25% validation accuracy** and a **Macro-F1 of 0.9627**.
- **Real-World Breakdown:** When tested live on raw, uncropped webcam frames without artificial bounding boxes (`src/predict.py`), performance collapsed (<30% accuracy). The classical skin mask repeatedly latched onto faces, bare arms, and warm wooden desks as the "largest contour," blinding the classifier to the actual hand shape.
- **Strategic Decision:** This validated our planned architectural pivot away from color thresholding to deep skeletal landmark tracking in Milestone 3.

---

## 3. Current State: Milestone 3 Main Method (`cp3-core`)

### Deep Learning Architecture: MediaPipe Tasks + MLP

```
Raw Webcam Frame (640x480 RGB)
              │
              ▼
┌───────────────────────────────────────────────┐
│  MediaPipe Tasks API (HandLandmarker)         │
│  - Deep ML Palm Detector & Knuckle Regressor  │
│  - 21 3D Landmarks (x, y, z) = 63 Features    │
│  - Invariant to skin tone, lighting & clutter │
└───────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────┐
│  StandardScaler Normalization (scaler_mp.pkl) │
└───────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────┐
│  Multi-Layer Perceptron (MLP) Classifier      │
│  - Softmax Probability Distribution (8 Class) │
└───────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────┐
│  Post-Processing & HUD Rendering              │
│  - Threshold Gating (τ = 0.75)                │
│  - 5-Frame Rolling Majority Smoothing Buffer  │
│  - High-Contrast Live OpenCV Caption Banner   │
└───────────────────────────────────────────────┘
```

Rather than segmenting fragile skin pixels, our Milestone 3 system tracks the structural geometry of the human hand:
1. **Knuckle Localization:** The MediaPipe Tasks `HandLandmarker` model regresses **21 3D hand-knuckle landmarks** ($x, y, z$ in normalized space) directly from full-resolution RGB video.
2. **Feature Vector:** The 63 coordinates capture finger joint angles, inter-digit spreads, and thumb opposition independently of skin pigmentation, illumination shifts, or background clutter.
3. **MLP Classifier:** A Multi-Layer Perceptron trained on normalized landmark coordinates outputs class probabilities with ultra-low computational overhead.

### Milestone 3 Benchmark Performance

Evaluated strictly on our **241-image unseen test split** using `src/evaluate.py`:

| Performance Metric | Local Benchmark Result | Practical Implication |
| :--- | :---: | :--- |
| **Overall Accuracy** | **`73.86%`** | **Honest system metric** (178 / 241 test images), strictly penalizing all 27 detector dropouts. *(82.71% on detected hands)* |
| **Macro-F1 Score** | **`0.778`** | High balanced performance across all 8 classes (Weighted-F1: **0.781**). |
| **Inference Latency** | **`21.18 ms`** | Pure CPU inference per frame (preprocessing + landmarking + MLP classification). |
| **System Throughput** | **`47.2 FPS`** | **3.1× faster** than the project's real-time threshold of 15 FPS. |

### Baseline vs. Main Method Comparative Evaluation

| Evaluation Dimension | Milestone 2 Baseline (HOG + SVM) | Milestone 3 Main Method (MediaPipe + MLP) | Advantage / Trade-off |
| :--- | :--- | :--- | :--- |
| **Underlying Technique** | YCrCb Skin Mask + Contours + HOG + SVM | MediaPipe Tasks HandLandmarker + 63-D MLP | Landmark model is invariant to background noise |
| **Feature Dimensionality** | 3,790 features (Hu moments, ratios, HOG) | **63 features** (21 landmarks $\times [x,y,z]$) | **60× smaller** feature space; faster training |
| **Input Framing** | Requires pre-cropped hand bounding boxes | **Full uncropped frames** ($640 \times 480$ RGB) | Ready for unconstrained webcam use |
| **Inference Latency** | **6.09 ms per image** | **21.18 ms per image** | Baseline is faster, but functionally unusable live |
| **Throughput (Speed)** | 164.3 FPS | **47.2 FPS** | Main method easily clears the 15 FPS threshold |
| **Controlled Val Accuracy** | **96.25%** (on cropped ROI validation) | 83.2% (on detected hand landmarks) | Baseline excelled only under artificial cropping |
| **Unseen Full-Scene Accuracy** | **< 30.0%** (fails on real-world backgrounds) | **73.86%** (Honest metric on 241 test frames) | **+43.8% practical accuracy gain** in realistic scenes |
| **Macro-F1 Score** | 0.9627 (cropped ROI only) | **0.778** (unseen full-scene split) | Reliable multiclass discrimination |
| **Clutter / Skin Invariance** | **Very Fragile** (faces, wood, arms corrupt mask) | **High** (deep skeletal features ignore clutter) | Immune to room furniture and lighting changes |

### Error Analysis & Results Report

For the complete technical evaluation, refer to our official results documentation:
- **Publication Results PDF (3 Pages):** [`docs/ms3_report.pdf`](docs/ms3_report.pdf)
- **Markdown Error Analysis Report:** [`docs/ms3_report.md`](docs/ms3_report.md)
- **Key Findings:**
  1. **Confusion Matrix Analysis:** [`outputs/ms3_confusion_matrix.png`](outputs/ms3_confusion_matrix.png) demonstrates near-perfect recall on `PEACE` (1.000) and `STOP` (0.929), while identifying `PALM` vs. `STOP` and `CALL` vs. `LIKE` as primary confusions.
  2. **10-Image Failure Gallery:** Generated in [`outputs/failures/`](outputs/failures/) showing the primary mechanical root causes (e.g., wrist tilt hiding the pinky finger in `CALL`, and two-hand interference where `num_hands=1` locks onto a resting lap hand).
  3. **Phase 4 Improvement Plan:** Concrete engineering roadmap addressing each mechanical cause via 3D rotation augmentation, `num_hands=2` active hand elevation heuristics, and 5-frame rolling majority voting.

---

## 4. Repository Structure

```text
signlang/
├── README.md                      # Project overview, evolution, benchmarks & setup guide
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules (weights, cache, venv)
├── data/
│   ├── samples/                   # Representative sample images per class
│   └── test/                      # 241-image unseen test split partitioned by class
│       ├── call/
│       ├── dislike/
│       ├── like/
│       ├── mute/
│       ├── ok/
│       ├── palm/
│       ├── peace/
│       └── stop/
├── docs/
│   ├── datacard.md                # HaGRID subset provenance, licensing & stratified splits
│   ├── progress_report.md         # Milestone 2 (cp2-50) progress check documentation
│   ├── ms3_report.md              # Milestone 3 error analysis, metrics & improvement plan
│   ├── ms3_report.pdf             # Formatted 3-page submission PDF report
│   └── Project_Outline_CV.pdf     # Official CSST 106 project proposal and timeline
├── models/                        # Serialized models and feature scalers
│   ├── hand_landmarker.task       # MediaPipe HandLandmarker TFLite model asset
│   ├── mediapipe_mlp_model.pkl    # Trained MLP Classifier for 8-class gestures
│   ├── scaler_mp.pkl              # StandardScaler for 63-D MediaPipe landmark coordinates
│   ├── svm_baseline_model.pkl     # Classical Milestone 2 SVM baseline model
│   └── scaler.pkl                 # StandardScaler for classical HOG baseline
├── notebooks/
│   └── CV_project_sign.ipynb      # Exploration, feature extraction & training notebook
├── outputs/
│   ├── ms3_confusion_matrix.png   # 300 DPI Confusion Matrix display
│   ├── contact_sheet.png          # 20-sample labeled dataset contact sheet
│   ├── failures/                  # 10 annotated failure gallery images from evaluate.py
│   └── predictions/               # Visual inference outputs with HUD banners
└── src/
    ├── baseline.py                # Classical YCrCb, Hu Moments & HOG feature pipeline
    ├── evaluate.py                # Benchmark evaluation script (speed, F1, confusion matrix)
    ├── generate_failures.py       # Visual failure hunting and skeleton annotation tool
    ├── generate_pdf_report.py     # Results PDF compiler (builds docs/ms3_report.pdf)
    ├── predict.py                 # Single-image classical SVM prototype inference
    ├── run.py                     # Real-time live webcam recognition application with HUD
    └── spike.py                   # Milestone 1 feasibility spike script
```

---

## 5. Installation & Setup

### Step 1: Clone the Repository & Set Up Virtual Environment

```bash
# Clone the project repository
git clone https://github.com/Prince-Genel/signlang.git
cd signlang

# Create and activate a Python 3.12 virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate
```

### Step 2: Install Required Dependencies

```bash
pip install -r requirements.txt
```

> **Requirements:** Python $\ge 3.10$, `mediapipe>=0.10.14`, `opencv-python>=4.9.0`, `scikit-learn>=1.6.0`, `matplotlib>=3.8.0`, `joblib>=1.3.0`, `pillow>=10.0.0`.

### Step 3: Download Pre-Trained Model Weights

> **Important:** Model weights are excluded from Git tracking due to file size limits and must be placed manually inside the `models/` folder.

1. Download the trained Milestone 3 weights (`mediapipe_mlp_model.pkl` and `scaler_mp.pkl`) from Google Drive:  
   🔗 **[Download Model Weights (Google Drive)](https://drive.google.com/drive/folders/1y-HSH_zvyWyW8_Z57QIqf8QoLx51tEPz?usp=sharing)**
2. Place both `.pkl` files inside the `models/` directory:
   ```text
   models/
   ├── mediapipe_mlp_model.pkl
   └── scaler_mp.pkl
   ```
3. *(Optional)* The MediaPipe asset `models/hand_landmarker.task` will automatically download from Google Cloud Storage on first run if not present.

---

## 6. Usage & Deliverables

### A. Run Live Webcam Inference (Main Application)
Launch the real-time sign language recognition application with live OpenCV HUD caption overlay:

```bash
python src/run.py -webcam 0
```
- **Controls:** Press `q` to safely exit the webcam stream.
- **Features:** Real-time skeletal landmark rendering, classification confidence score, and on-screen caption subtitle bar.

### B. Evaluate Test Split & Generate Confusion Matrix
Run the comprehensive benchmark evaluation across the 241-image unseen test set:

```bash
python src/evaluate.py -d data/test
```
- Measures individual and average inference latency (ms per frame and FPS throughput).
- Prints the scikit-learn `classification_report` formatted with `digits=3`.
- Generates and saves the high-resolution confusion matrix to `outputs/ms3_confusion_matrix.png`.

### C. Generate the 10-Image Failure Gallery
Extract and annotate 10 misclassified frames with skeletal overlays and prediction error tags:

```bash
python src/generate_failures.py
```
- Saves annotated images (`fail_1_...jpg` through `fail_10_...jpg`) into `outputs/failures/`.

### D. Compile Results PDF Report
Rebuild the 3-page results report PDF:

```bash
python src/generate_pdf_report.py
```
- Compiles [`docs/ms3_report.pdf`](docs/ms3_report.pdf).

---

## 7. Team & Course Attribution

- **Course:** CSST 106 - Computer Vision
- **Institution:** Department of Computer Science & Information Technology
- **Project Repository:** [`github.com/Prince-Genel/signlang`](https://github.com/Prince-Genel/signlang)

### Team Members & Contributions
- **Prince Genel Umali** — Feature extraction pipelines, SVM baseline training, quantitative evaluation framework (`src/evaluate.py`), confusion matrix diagnostics, results PDF compilation (`src/generate_pdf_report.py`).
- **Kean Gabriel Salvahan** — HaGRID dataset curation, stratified splitting (70/15/15), data normalization, Gaussian denoising, and exploratory dataset contact sheets.
- **Justin Angelo Luzande** — MediaPipe Tasks API integration, deep MLP classifier training, live webcam OpenCV HUD captioning application (`src/run.py`), temporal smoothing filter.
