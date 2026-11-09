"""
Usage:
  python src/evaluate.py
  python src/evaluate.py --split test --weights models/mediapipe_mlp_model.pkl
  python src/evaluate.py --test-dir data/test
"""

import sys
import time
import json
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
EVAL_DIR = OUTPUTS_DIR / "eval"
CM_OUTPUT_PATH = OUTPUTS_DIR / "ms3_confusion_matrix.png"
EVAL_CM_OUTPUT_PATH = EVAL_DIR / "confusion_matrix.png"
METRICS_JSON_PATH = EVAL_DIR / "metrics.json"

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# The 8 target gesture classes recognized by mediapipe_mlp_model
TARGET_CLASSES = ["call", "dislike", "like", "mute", "ok", "palm", "peace", "stop"]


# -------------------------------------------------------------------------
# Geometric Post-Processing Refinement
# -------------------------------------------------------------------------
def refine_prediction(lm: np.ndarray, raw_pred: str, confidence: float) -> tuple[str, float]:
    """
    Position-, scale-, and handedness-invariant geometric refinement using 21 landmarks.
    Resolves look-alike confusions:
    - OK vs STOP/PALM: Pinch distance between thumb & index tips vs palm length
    - PALM vs STOP: Inter-finger angular spread across middle 4 fingers
    - CALL vs LIKE/DISLIKE/MUTE: Pinky-to-ring separation when index/middle/ring curled
    """
    palm_len = float(np.linalg.norm(lm[9, :2] - lm[0, :2])) + 1e-6
    pinch = float(np.linalg.norm(lm[4, :2] - lm[8, :2])) / palm_len
    pinky_sep = float(np.linalg.norm(lm[20, :2] - lm[16, :2])) / palm_len

    ext_index = np.linalg.norm(lm[8, :2] - lm[0, :2]) > np.linalg.norm(lm[6, :2] - lm[0, :2]) * 1.12
    ext_middle = np.linalg.norm(lm[12, :2] - lm[0, :2]) > np.linalg.norm(lm[10, :2] - lm[0, :2]) * 1.12
    ext_ring = np.linalg.norm(lm[16, :2] - lm[0, :2]) > np.linalg.norm(lm[14, :2] - lm[0, :2]) * 1.10
    ext_pinky = np.linalg.norm(lm[20, :2] - lm[0, :2]) > np.linalg.norm(lm[18, :2] - lm[0, :2]) * 1.08

    # 1. OK gesture: thumb tip (4) and index tip (8) pinched together while middle, ring, pinky are extended
    if pinch < 0.36 and ext_middle and ext_ring and ext_pinky:
        return "ok", max(confidence, 0.92)

    # 2. CALL gesture: pinky extended away from ring while index, middle, ring are curled
    if raw_pred in ("like", "dislike", "mute") and not ext_index and not ext_middle and not ext_ring and ext_pinky and pinky_sep > 0.45:
        return "call", max(confidence, 0.88)

    # 3. PALM vs STOP disambiguation when all 4 fingers are extended:
    #    PALM has spread fingers (>= 6 deg average inter-finger angle); STOP has fingers held together (< 6 deg)
    if raw_pred in ("palm", "stop") and ext_index and ext_middle and ext_ring and ext_pinky and pinch >= 0.45:
        vecs = [lm[t, :2] - lm[m, :2] for m, t in ((5, 8), (9, 12), (13, 16), (17, 20))]
        angs = [
            np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1.0, 1.0)))
            for a, b in zip(vecs, vecs[1:])
        ]
        spread = float(np.mean(angs))
        if spread >= 6.0:
            return "palm", max(confidence, 0.90)
        return "stop", max(confidence, 0.90)

    return raw_pred, confidence


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
    if not task_path.exists():
        print(f"[+] Downloading hand_landmarker.task from Google Storage...")
        task_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, str(task_path))
            print(f"[OK] Download complete: {task_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to download hand_landmarker.task: {e}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

    print(f"[+] Loading Model: {model_path.name}")
    mlp_model = joblib.load(str(model_path))

    print(f"[+] Loading Scaler: {scaler_path.name}")
    scaler = joblib.load(str(scaler_path))

    print("[+] Initializing MediaPipe Tasks HandLandmarker (num_hands=2 active hand tracking)...")
    base_options = python.BaseOptions(model_asset_path=str(task_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("[OK] Assets and HandLandmarker initialized successfully.\n")

    return detector, mlp_model, scaler


# -------------------------------------------------------------------------
# Feature Extraction and Inference
# -------------------------------------------------------------------------
def predict_image(image_bgr: np.ndarray, detector, mlp_model, scaler, use_refinement: bool = True):
    """
    Extracts 63 landmark coordinates (21 landmarks x [x, y, z]) using MediaPipe Tasks API,
    scales them with scaler_mp.pkl, and returns the predicted class label and confidence.
    """
    if image_bgr is None:
        return "no_hand", 0.0

    rgb_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return "no_hand", 0.0

    # Active hand selection (highest vertical hand / minimum y coordinate)
    hand = min(result.hand_landmarks, key=lambda h: min(pt.y for pt in h))
    lm_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

    features_arr = lm_arr.reshape(1, -1)
    scaled_features = scaler.transform(features_arr)

    if hasattr(mlp_model, "predict_proba"):
        probs = mlp_model.predict_proba(scaled_features)[0]
        best = int(np.argmax(probs))
        pred_label = str(mlp_model.classes_[best]).lower()
        confidence = float(probs[best])
    else:
        pred_label = str(mlp_model.predict(scaled_features)[0]).lower()
        confidence = 1.0

    if use_refinement:
        pred_label, confidence = refine_prediction(lm_arr, pred_label, confidence)

    return pred_label, confidence


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
def run_evaluation(test_dir_path: Path, weights_path: Path = MODEL_PATH, use_refinement: bool = True):
    print("=" * 65)
    print(" Computer Vision Project - Milestone 4 Evaluation Benchmark")
    print("=" * 65)
    print(f"Test Directory    : {test_dir_path}")
    print(f"Model Weights     : {weights_path}")
    print(f"Geometric Refined : {'ENABLED (M4 Final)' if use_refinement else 'DISABLED (MS3 Raw)'}")
    print(f"Target Classes    : {', '.join(TARGET_CLASSES)}")
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
    detector, mlp_model, scaler = initialize_assets(model_path=weights_path)

    # 2. Rubric Requirement: Warm-up run on first image
    warmup_path = image_paths[0]
    print(f"[+] Executing warm-up run on: {warmup_path.name}")
    warmup_img = cv2.imread(str(warmup_path))
    if warmup_img is not None:
        _ = predict_image(warmup_img, detector, mlp_model, scaler, use_refinement=use_refinement)
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
        pred_label, conf = predict_image(img_bgr, detector, mlp_model, scaler, use_refinement=use_refinement)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        individual_latencies.append(latency_ms)

        y_true.append(true_label)
        y_pred.append(pred_label)

        status_flag = "[OK]" if true_label == pred_label else "[MISMATCH]"
        if idx % 20 == 0 or idx == len(image_paths) or status_flag == "[MISMATCH]":
            print(f"  [{idx:03d}/{len(image_paths):03d}] {img_path.name:22s} | "
                  f"True: {true_label:8s} | Pred: {pred_label:8s} ({conf*100:2.0f}%) | "
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
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=TARGET_CLASSES,
        digits=3,
        zero_division=0,
        output_dict=True
    )
    report_str = classification_report(
        y_true,
        y_pred,
        labels=TARGET_CLASSES,
        digits=3,
        zero_division=0
    )
    print(report_str)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=TARGET_CLASSES, average="macro", zero_division=0)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Macro-F1 Score  : {macro_f1:.3f}")
    print("=" * 65)

    # 6. Milestone 4 Requirement: Output metrics.json and confusion_matrix.png
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Model size in bytes
    weights_size_bytes = weights_path.stat().st_size if weights_path.exists() else 0

    metrics_payload = {
        "milestone": "Milestone 4 (cp4-80)",
        "weights_file": str(weights_path.name),
        "weights_size_bytes": weights_size_bytes,
        "weights_size_kb": round(weights_size_bytes / 1024, 2),
        "geometric_refinement": use_refinement,
        "total_test_samples": len(image_paths),
        "overall_accuracy": round(float(acc), 4),
        "overall_accuracy_pct": round(float(acc) * 100, 2),
        "macro_f1": round(float(macro_f1), 4),
        "average_latency_ms": round(float(avg_speed_ms), 2),
        "throughput_fps": round(float(throughput_fps), 1),
        "per_class": {
            cls: {
                "precision": round(float(report_dict[cls]["precision"]), 3),
                "recall": round(float(report_dict[cls]["recall"]), 3),
                "f1_score": round(float(report_dict[cls]["f1-score"]), 3),
                "support": int(report_dict[cls]["support"])
            }
            for cls in TARGET_CLASSES
        }
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[OK] Evaluation metrics saved to: {METRICS_JSON_PATH}")

    # 7. Generate and save Confusion Matrix Display
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
    title_suffix = " (M4 Refined Pipeline)" if use_refinement else " (MS3 Raw MLP)"
    ax.set_title(f"Sign Language Confusion Matrix{title_suffix}", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()

    # Save to both outputs/eval/confusion_matrix.png and outputs/ms3_confusion_matrix.png
    plt.savefig(EVAL_CM_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.savefig(CM_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Confusion matrix saved to: {EVAL_CM_OUTPUT_PATH}")
    print(f"[OK] Confusion matrix archived to: {CM_OUTPUT_PATH}")


# -------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Sign Language Recognition Pipeline."
    )
    parser.add_argument(
        "--test-dir", "-d",
        type=str,
        default=str(DEFAULT_TEST_DIR),
        help=f"Path to test images directory (default: {DEFAULT_TEST_DIR})"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name to evaluate: 'test' (default: 'test')"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(MODEL_PATH),
        help=f"Path to classifier weights file (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable geometric post-processing refinement (run raw MS3 MLP)"
    )

    args = parser.parse_args()

    # If --split is passed, resolve test directory
    if args.split:
        candidate_dir = PROJECT_ROOT / "data" / args.split
        if candidate_dir.exists():
            test_path = candidate_dir
        else:
            test_path = Path(args.test_dir)
    else:
        test_path = Path(args.test_dir)

    weights_path = Path(args.weights)
    use_refinement = not args.no_refine

    run_evaluation(test_path, weights_path=weights_path, use_refinement=use_refinement)


if __name__ == "__main__":
    main()
