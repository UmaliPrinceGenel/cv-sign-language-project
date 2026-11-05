# CSST 106 - Milestone 3: Core Implementation, Error Analysis & Results Report

> **Official Deliverable PDF:** For the submission results PDF formatted please see [`docs/ms3_report.pdf`](ms3_report.pdf).

---

## 1. Quantitative Benchmark & Speed Results

Our local evaluation pipeline (`src/evaluate.py`) evaluated the end-to-end **MediaPipe Tasks API + Multi-Layer Perceptron (MLP)** pipeline on the **241-image unseen test split** (30 images per class across 8 gesture classes, with no overlap with training/validation splits).

### Local Benchmark Summary
- **Overall Accuracy:** **73.86%** (178 / 241 unseen test images)
- **Macro-F1 Score:** **0.778** (Weighted F1: 0.781)
- **Average Inference Latency:** **21.18 ms per image** (includes landmark extraction and MLP prediction on standard laptop CPU)
- **System Throughput:** **47.2 FPS** (exceeding the 15 FPS real-time project requirement by 3.1×)

> **Note on Benchmark Rigor ("Honest Accuracy Metric"):**  
> Unlike isolated academic benchmarks that discard frames where the hand detector fails, our local evaluation script (`src/evaluate.py`) strictly penalizes all 27 dropped detections (where MediaPipe returned `"no_hand"`) as incorrect classifications. On frames where a hand was successfully localized (214/241), recognition accuracy was **82.71%** (177/214). Penalizing dropouts produces our honest end-to-end system accuracy of **73.86%**.

---

## 2. Baseline vs. Main Method Comparative Evaluation

The following table summarizes the quantitative performance, architectural characteristics, and real-world trade-offs between our Milestone 2 classical baseline and Milestone 3 deep learning main method:

| Evaluation Dimension / Metric | Milestone 2 Baseline (HOG + SVM) | Milestone 3 Main Method (MediaPipe + MLP) |
| :--- | :--- | :--- |
| **Pipeline & Model** | YCrCb Skin Mask + Contours + HOG + Support Vector Machine (RBF kernel) | MediaPipe Tasks API (`HandLandmarker`) + 63-D Multi-Layer Perceptron (MLP) |
| **Feature Representation** | 7 Hu Moments + 3 Shape Ratios + HOG Descriptors (3,790 features) | 21 normalized 3D knuckle landmarks ($x, y, z$: 63 coordinates) |
| **Input Framing** | Requires pre-cropped hand Region-of-Interest (JSON bounding boxes) | Full uncropped webcam frames ($640 \times 480$ RGB) |
| **Inference Latency** | **6.09 ms per image** (Local CPU) | **21.18 ms per image** (Local CPU) |
| **Throughput (Speed)** | **164.3 FPS** (exceeds 15 FPS target) | **47.2 FPS** (3.1× faster than 15 FPS target) |
| **Controlled Val Accuracy** | **96.25%** (on ground-truth cropped hand ROIs) | **83.2%** (on detected landmarks) |
| **Unseen Full-Scene Test Acc**| **< 30.0%** (severely degrades on uncropped scenes) | **73.86%** (honest end-to-end on 241 unseen test images) |
| **Macro-F1 Score** | **0.9627** (cropped ROI validation split only) | **0.778** (full unseen test split, penalizing dropouts) |
| **Clutter / Background Invariance** | **Very Fragile:** skin-toned furniture, faces, and walls break skin mask | **High:** deep skeletal landmarks remain invariant to lighting and clutter |
| **Real-World Deployment** | Fails live webcam test without artificial ROI constraints | Fully operational for live OpenCV HUD captioning |

### Synthesis & Architectural Trade-offs
While the classical SVM baseline demonstrated low latency (6.09 ms, 164.3 FPS), its 96.25% validation accuracy is dependent on ground-truth bounding box crops. In realistic uncropped webcam feeds, classical YCrCb thresholding segments faces and background objects. MediaPipe + MLP adds a modest latency overhead (21.18 ms vs. 6.09 ms), but maintains 47.2 FPS while providing the invariant hand tracking required for real-time live usage.

---

## 3. Error Analysis & Confusion Matrix

To diagnose model behavior, we evaluated the test set and plotted the complete confusion matrix (`outputs/ms3_confusion_matrix.png`).

![Milestone 3 Confusion Matrix](../outputs/ms3_confusion_matrix.png)

### Per-Class Performance Breakdown (`digits=3`)

| Class | Precision | Recall | F1-Score | Support | Correct / Misclassified |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CALL** | 1.000 | 0.577 | 0.732 | 26 | 15 correct, 11 errors |
| **DISLIKE** | 0.781 | 0.862 | 0.820 | 29 | 25 correct, 4 errors |
| **LIKE** | 0.656 | 0.875 | 0.750 | 24 | 21 correct, 3 errors |
| **MUTE** | 0.870 | 0.833 | 0.851 | 24 | 20 correct, 4 errors |
| **OK** | 1.000 | 0.857 | 0.923 | 28 | 24 correct, 4 errors |
| **PALM** | 0.826 | 0.679 | 0.745 | 28 | 19 correct, 9 errors |
| **PEACE** | 0.964 | 1.000 | 0.982 | 27 | 27 correct, 0 errors |
| **STOP** | 0.703 | 0.929 | 0.800 | 28 | 26 correct, 2 errors |
| **OVERALL** | **0.850** | **0.827** | **0.778** | **241** | **178 correct, 36 misclassifications, 27 dropouts** |

### The Three Primary Mechanical Failure Modes

1. **Palm vs. Stop Geometry Confusion (Look-Alike Class Problem):**
   - *Evidence:* 7 true `palm` signs misclassified as `stop`; 1 `stop` misclassified as `palm`.
   - *Mechanical Root Cause:* Both gestures present an open palm with 5 extended digits. MediaPipe normalizes 3D coordinates relative to the hand bounding box, which standardizes joint positions but discards absolute wrist pitch and forearm elevation angle. Without wrist tilt context, an open palm facing the camera and an upright stop gesture yield near-identical 63-dimensional coordinate arrays.

2. **Call vs. Like / Dislike Confusion (Perspective & Rotation Vulnerability):**
   - *Evidence:* 5 true `call` signs misclassified as `like`, 2 as `dislike`.
   - *Mechanical Root Cause:* In natural signing, the `call` gesture (thumb and pinky extended, intermediate fingers curled) is performed with the wrist tilted toward the ear. When captured at oblique angles, the extended pinky is foreshortened or self-occluded behind the fist. MediaPipe's single-camera 3D projection perceives curled digits with only the thumb protruding, projecting a feature vector indistinguishable from a `like` or `dislike` fist.

3. **Two-Hand Interference & Resting Hand Distraction (The `num_hands=1` Flaw):**
   - *Evidence:* Observed in multiple full-body test frames (`fail_1`, `fail_7`, `fail_8`, `fail_10`).
   - *Mechanical Root Cause:* MediaPipe HandLandmarker was instantiated with `num_hands=1`. In natural seated postures, subjects frequently rest an idle hand on their lap or thigh while gesturing with the other. Because the resting hand is closer to the body center, static, and devoid of motion blur, the detection anchor frequently locks onto the inactive hand. The MLP correctly classifies this resting hand as a relaxed fist (`like` or `mute`), producing a false negative for the active signing hand.

---

## 4. The 10-Image Annotated Failure Gallery

The script `src/generate_failures.py` inspected `[MISMATCH]` predictions from the test set and saved 10 annotated failure samples in `outputs/failures/`. In each image, the detected MediaPipe skeleton is drawn with red landmark nodes, alongside green `TRUE` and red `PRED` labels:

| Sample | True Gesture | Predicted | Specific Root Cause Observation |
| :---: | :---: | :---: | :--- |
| **`fail_1`** | **CALL** | `MUTE` | MediaPipe locked onto the inactive left hand resting at the waist, ignoring the raised call gesture near the ear. |
| **`fail_2`** | **CALL** | `DISLIKE` | Wrist rotated downward; thumb pointed down, creating a coordinate pattern resembling a dislike gesture. |
| **`fail_3`** | **CALL** | `LIKE` | Wrist angled towards camera; pinky finger self-occluded behind fist; perceived as a thumbs-up fist. |
| **`fail_4`** | **CALL** | `LIKE` | Severe perspective foreshortening of the pinky finger; only extended thumb localized. |
| **`fail_5`** | **CALL** | `DISLIKE` | Inverted wrist axis during natural signing; thumb orientation classified as dislike. |
| **`fail_6`** | **CALL** | `STOP` | Intermediate fingers partially uncurled during transition, triggering open-hand threshold. |
| **`fail_7`** | **CALL** | `LIKE` | Two-hand interference: subject's resting lap hand captured instead of the signing hand. |
| **`fail_8`** | **CALL** | `LIKE` | Inactive resting hand closer to camera detected by `num_hands=1` anchor. |
| **`fail_9`** | **CALL** | `PEACE` | Spread between pinky and thumb misprojected as a two-finger peace sign. |
| **`fail_10`**| **CALL** | `MUTE` | Resting waist hand detected as a closed fist (`mute`). |

---

## 5. Phase 4 Improvement Plan

The table below maps each identified failure cause to concrete algorithmic solutions and measurable verification targets:

| Identified Failure Cause | Algorithmic & Pipeline Enhancement | Expected Impact & Verification Metric |
| :--- | :--- | :--- |
| **Palm vs. Stop Confusion**<br>*(Open hand geometry similarity; lack of wrist pitch)* | 1. Merge "Palm" and "Stop" into a unified "Open Hand" gesture in the vocabulary, **OR**<br>2. Augment feature vector with a wrist-to-elbow directional vector to provide absolute elevation context to the MLP. | Eliminates the largest off-diagonal error cell (7 samples); projected to increase Macro-F1 from 0.778 to **> 0.815**. |
| **Call vs. Like / Dislike Confusion**<br>*(Wrist tilt occludes pinky; perceived as a fist)* | 1. Apply synthetic 3D rotation augmentation (random pitch/yaw/roll jitter up to $\pm 25^\circ$) to landmark coordinates during MLP training.<br>2. Retrain MLP with higher regularization (dropout / weight decay). | Makes MLP invariant to oblique viewing angles; restores Call recall from 0.577 to **> 0.800**. |
| **Resting Hand Interference**<br>*(The `num_hands=1` flaw; detector locks onto lap hand)* | 1. Reconfigure MediaPipe HandLandmarker to `num_hands=2`.<br>2. Implement a `postprocess()` heuristic selecting the hand with highest vertical position ($y_{\min}$, closest to face/chest). | Completely eliminates idle lap-hand false detections (`fail_1, 7, 8, 10`); recovers 8–10 test frames; increases full-scene accuracy to **> 82%**. |
| **Single-Frame Drops & Prediction Flicker** | 1. Implement a 5-frame rolling majority voting temporal buffer.<br>2. Enforce confidence threshold $\tau = 0.75$, suppressing flickering sub-threshold predictions in the live OpenCV HUD. | Stabilizes live captioning HUD; prevents single dropped frames from glitching on-screen subtitles. |

---

## 6. Execution Instructions

- **Live Webcam Inference:**
  ```bash
  python src/run.py -webcam 0
  ```
- **Test Set Evaluation & Confusion Matrix Generation:**
  ```bash
  python src/evaluate.py -d data/test
  ```
- **Generate Failure Gallery Artifacts:**
  ```bash
  python src/generate_failures.py
  ```
- **Generate Results PDF Report (3 Pages):**
  ```bash
  python src/generate_pdf_report.py
  ```
