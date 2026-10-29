"""
Usage:
  python src/evaluate.py
  python src/evaluate.py --test-dir data/test
"""

import sys
import time
import argparse
import urllib.request
from pathlib import Path

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------------------------------------------------
# Path Resolution and Configuration
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe_mlp_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler_mp.pkl"
TASK_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "test"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CM_OUTPUT_PATH = OUTPUTS_DIR / "ms3_confusion_matrix.png"

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# The 8 target gesture classes recognized by mediapipe_mlp_model
TARGET_CLASSES = ["call", "dislike", "like", "mute", "ok", "palm", "peace", "stop"]


# -------------------------------------------------------------------------
# Asset Initialization
# -------------------------------------------------------------------------
def initialize_assets(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    task_path: Path = TASK_PATH
):
    """
    Checks for the MediaPipe task asset, downloads if missing, loads the
    scaler and MLP model via joblib, and initializes HandLandmarker.
    """
    # 1. Download hand_landmarker.task if not present
    if not task_path.exists():
        print(f"[+] Downloading hand_landmarker.task from Google Storage...")
        task_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, str(task_path))
            print(f"[OK] Download complete: {task_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to download hand_landmarker.task: {e}")

    # 2. Verify model and scaler
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

    print(f"[+] Loading MLP Model: {model_path.name}")
    mlp_model = joblib.load(str(model_path))

    print(f"[+] Loading Scaler: {scaler_path.name}")
    scaler = joblib.load(str(scaler_path))

    print("[+] Initializing MediaPipe Tasks HandLandmarker...")
    base_options = python.BaseOptions(model_asset_path=str(task_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("[OK] Assets and HandLandmarker initialized successfully.\n")

    return detector, mlp_model, scaler


# -------------------------------------------------------------------------
# Feature Extraction and Inference
# -------------------------------------------------------------------------
def predict_image(image_bgr: np.ndarray, detector, mlp_model, scaler):
    """
    Extracts 63 landmark coordinates (21 landmarks x [x, y, z]) using MediaPipe Tasks API,
    scales them with scaler_mp.pkl, and returns the predicted class label.
    """
    if image_bgr is None:
        return "no_hand"

    rgb_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return "no_hand"

    landmarks = result.hand_landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])

    features_arr = np.array(features, dtype=np.float32).reshape(1, -1)
    scaled_features = scaler.transform(features_arr)
    pred_label = mlp_model.predict(scaled_features)[0].lower()

    return pred_label


def extract_ground_truth(img_path: Path, test_dir: Path) -> str:
    """
    Determines the ground truth label from either:
    1. Parent directory name (e.g., data/test/peace/001.jpg -> 'peace')
    2. File prefix (e.g., data/test/peace_01.jpg -> 'peace')
    """
    if img_path.parent != test_dir and img_path.parent.name.lower() in TARGET_CLASSES:
        return img_path.parent.name.lower()

    stem_prefix = img_path.stem.split("_")[0].lower()
    if stem_prefix in TARGET_CLASSES:
        return stem_prefix

    for c in TARGET_CLASSES:
        if c in img_path.stem.lower():
            return c

    return stem_prefix


# -------------------------------------------------------------------------
# Main Evaluation Routine
# -------------------------------------------------------------------------
def run_evaluation(test_dir_path: Path):
    print("=" * 65)
    print(" Computer Vision Project - Milestone 3 MediaPipe Evaluation")
    print("=" * 65)
    print(f"Test Directory : {test_dir_path}")
    print(f"Target Classes : {', '.join(TARGET_CLASSES)}")
    print("-" * 65)

    if not test_dir_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir_path}")

    # Discover all supported test images
    image_paths = sorted([
        p for p in test_dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not image_paths:
        raise FileNotFoundError(f"No valid test images found in: {test_dir_path}")

    print(f"[+] Discovered {len(image_paths)} test image(s).\n")

    # 1. Initialize models and MediaPipe landmarker
    detector, mlp_model, scaler = initialize_assets()

    # 2. Rubric Requirement: Warm-up run on first image
    warmup_path = image_paths[0]
    print(f"[+] Executing warm-up run on: {warmup_path.name}")
    warmup_img = cv2.imread(str(warmup_path))
    if warmup_img is not None:
        _ = predict_image(warmup_img, detector, mlp_model, scaler)
    print("[OK] Warm-up run complete. Starting benchmark timer...\n")

    # 3. Timed Evaluation Loop
    y_true = []
    y_pred = []
    individual_latencies = []

    print("[*] Evaluating test set...")
    total_eval_start = time.perf_counter()

    for idx, img_path in enumerate(image_paths, 1):
        true_label = extract_ground_truth(img_path, test_dir_path)
        img_bgr = cv2.imread(str(img_path))

        # Measure pure inference + preprocessing latency
        t0 = time.perf_counter()
        pred_label = predict_image(img_bgr, detector, mlp_model, scaler)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        individual_latencies.append(latency_ms)

        y_true.append(true_label)
        y_pred.append(pred_label)

        status_flag = "[OK]" if true_label == pred_label else "[MISMATCH]"
        print(f"  [{idx:03d}/{len(image_paths):03d}] {img_path.name:20s} | "
              f"True: {true_label:8s} | Pred: {pred_label:8s} | "
              f"Latency: {latency_ms:5.1f} ms {status_flag}")

    total_eval_elapsed = time.perf_counter() - total_eval_start

    # 4. Latency Statistics
    total_inference_time_ms = sum(individual_latencies)
    avg_speed_ms = total_inference_time_ms / len(image_paths)
    throughput_fps = 1000.0 / avg_speed_ms if avg_speed_ms > 0 else 0.0

    print("\n" + "=" * 65)
    print("BENCHMARK TIMING SUMMARY")
    print("=" * 65)
    print(f"Total Test Samples Evaluated : {len(image_paths)}")
    print(f"Total Inference Time         : {total_inference_time_ms / 1000.0:.3f} s")
    print(f"Total Elapsed Time (with I/O): {total_eval_elapsed:.3f} s")
    print(f"Average Inference Speed      : {avg_speed_ms:.2f} ms per image")
    print(f"Throughput                   : {throughput_fps:.1f} FPS")
    print("=" * 65)

    # 5. Rubric Requirement: Classification Report with digits=3
    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT (sklearn.metrics, digits=3)")
    print("=" * 65)
    report = classification_report(
        y_true,
        y_pred,
        labels=TARGET_CLASSES,
        digits=3,
        zero_division=0
    )
    print(report)

    # Calculate overall Accuracy and Macro-F1
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=TARGET_CLASSES, average="macro", zero_division=0)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Macro-F1 Score  : {macro_f1:.3f}")
    print("=" * 65)

    # 6. Rubric Requirement: ConfusionMatrixDisplay to outputs/ms3_confusion_matrix.png
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[+] Generating Confusion Matrix Display...")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=TARGET_CLASSES,
        display_labels=[c.upper() for c in TARGET_CLASSES],
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=True
    )
    ax.set_title("Milestone 3 MediaPipe + MLP Confusion Matrix", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()

    plt.savefig(CM_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Confusion matrix saved to: {CM_OUTPUT_PATH}")


# -------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Milestone 3 MediaPipe + MLP model on test images."
    )
    parser.add_argument(
        "--test-dir", "-d",
        type=str,
        default=str(DEFAULT_TEST_DIR),
        help=f"Path to test images directory (default: {DEFAULT_TEST_DIR})"
    )
    args = parser.parse_args()

    test_path = Path(args.test_dir)
    run_evaluation(test_path)


if __name__ == "__main__":
    main()
