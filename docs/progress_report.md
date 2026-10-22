# Milestone 2 Progress Report

**Overall Completion:** ~50% (Milestone 2 core requirements fully met).
* **How we got this:** We planned 4 major tasks. The first three tasks (Dataset Preparation, HOG+SVM Baseline, and the Command-Line Prototype) are completely finished and account for approximately 50% of the project's total technical effort.

**Target for Phase 3:** Implement Deep Learning/MediaPipe fallback for cluttered backgrounds.

## 1. Task Status

| Task (from proposal) | Owner | Status | Evidence (file, commit, screenshot) |
| :--- | :--- | :--- | :--- |
| **Collect and split dataset** | Kean | Done | `docs/datacard.md`, 70/15/15 split complete |
| **HOG + SVM Baseline** | Prince | Done | `src/baseline.py`, Val Accuracy: **96.25%**, Macro-F1: **0.9627** |
| **Prediction Prototype** | Justin | Done | `src/predict.py`, CLI arguments operational |

## 2. Current Blockers / Risks

As noted in the original project proposal, the classical YCrCb skin mask heavily relies on perfect bounding boxes. On raw, uncropped images with skin-colored backgrounds (e.g., wooden desks, faces), the largest contour often misses the hand and segments background noise instead. This validates our planned transition to MediaPipe landmark detection in Phase 3 to handle real-world webcam clutter.

## 3. Analysis of Results & Current Risks

**Why Validation Accuracy is High (96.25%):**  
Our SVM baseline performs exceptionally well during validation because those images are perfectly cropped to the hand Region of Interest (ROI) using the provided HaGRID `.json` bounding boxes. When the background is removed, our HOG features and Hu Moments can classify the gesture shapes with high precision.

**Why the Live Prototype Struggles:**  
As noted in our original risk assessment, the live prototype (`predict.py`) must process raw, uncropped images without the help of JSON bounding boxes. In these real-world scenarios, our classical YCrCb skin mask often mistakenly targets the largest skin-colored contour in the room (e.g., a face, a neck, or a warm-colored wooden desk) instead of the hand, leading to misclassifications like predicting "MUTE" on a "PEACE" sign.

**Next Steps:**  
This discrepancy between perfectly cropped data and uncropped real-world data perfectly validates our planned transition to MediaPipe landmark detection in Phase 3. This will allow us to build the Deep Learning model that inherently tracks hand joints and ignores background clutter, bypassing the limitations of classical skin-masking.