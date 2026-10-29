# Real-Time Sign Language Recognition

Milestone 3 (`cp3-core`) completion for CSST 106.

---

## 1. Project Overview

This project is a real-time sign language recognition system built using the **MediaPipe Tasks API** for hand landmark extraction and a **Multi-Layer Perceptron (MLP)** classifier for gesture prediction. The system is designed for high-throughput, low-latency live inference as well as automated batch evaluation.

---

## 2. Milestone 3 Performance

Our local benchmark results evaluated on the unseen test dataset:

- **Accuracy:** 73.86% on the 241-image unseen test split
- **Speed:** 21.18 ms per image (47.2 FPS)

---

## 3. Installation & Model Weights

> **Note:** The model weights are too large for GitHub and must be downloaded manually.

1. Clone the repository and run pip install -r requirements.txt
2. Download the trained model weights (mediapipe_mlp_model.pkl and scaler_mp.pkl) from this Google Drive link: [INSERT GOOGLE DRIVE LINK HERE]
3. Place both .pkl files inside the models/ directory.

---

## 4. Usage

Run the Milestone 3 deliverables using the following terminal commands:

- **Live Inference:**
  ```bash
  python src/run.py -webcam 0
  ```

- **Test Set Evaluation:**
  ```bash
  python src/evaluate.py -d data/test
  ```

- **Failure Generator:**
  ```bash
  python src/generate_failures.py
  ```

---

## 5. Documentation

For the full error analysis (including the 10-failure gallery, mechanical root causes, and evaluation breakdown) and the Phase 4 improvement plan, please see [`docs/ms3_report.md`](docs/ms3_report.md).
