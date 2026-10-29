

import os
import sys
import time
import argparse
import urllib.request
from pathlib import Path
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------------------------------------------------
# Configuration and Path Resolution
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe_mlp_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler_mp.pkl"
TASK_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Hand skeleton connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                 # Palm base
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Global model references
detector = None
mlp_model = None
scaler = None


# -------------------------------------------------------------------------
# Asset Initialization
# -------------------------------------------------------------------------
def initialize_pipeline(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    task_path: Path = TASK_PATH
):
    """
    Ensures model assets and the MediaPipe hand landmarker task file exist,
    loads the MLP classifier and StandardScaler, and initializes the HandLandmarker.
    """
    global detector, mlp_model, scaler

    # 1. Download hand_landmarker.task if not present
    if not task_path.exists():
        print(f"[+] 'hand_landmarker.task' not found. Downloading from Google Storage...")
        task_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, str(task_path))
            print(f"[OK] Download completed successfully: {task_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to download hand_landmarker.task: {e}")

    # 2. Verify model and scaler files exist
    if not model_path.exists():
        raise FileNotFoundError(f"MLP model file not found at: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

    # 3. Load MLP model and Scaler via joblib
    print(f"[+] Loading MLP model: {model_path.name}")
    mlp_model = joblib.load(str(model_path))

    print(f"[+] Loading Scaler: {scaler_path.name}")
    scaler = joblib.load(str(scaler_path))

    # 4. Initialize MediaPipe Tasks API HandLandmarker
    print("[+] Initializing MediaPipe Tasks API HandLandmarker...")
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
    print("[OK] Pipeline successfully initialized.\n")


# -------------------------------------------------------------------------
# Isolated Stage Functions
# -------------------------------------------------------------------------
def preprocess(frame: np.ndarray) -> mp.Image:
    """
    Stage 1: Preprocess.
    Converts a raw OpenCV BGR frame into a MediaPipe Image object in sRGB format.
    """
    if frame is None:
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def infer(mp_image: mp.Image) -> dict:
    """
    Stage 2: Inference.
    Detects hand landmarks using MediaPipe Tasks API, extracts 63 normalized coordinates
    (21 landmarks x 3 coordinates [x, y, z]), scales features, and runs MLP prediction.

    Returns:
        dict: {
            'prediction': str,      # Raw predicted gesture label
            'raw_prediction': str,  # Raw label before temporal smoothing
            'landmarks': list,      # Detected hand landmark list (or None)
            'confidence': float     # Softmax confidence score
        }
    """
    if mp_image is None or detector is None or mlp_model is None or scaler is None:
        return {
            "prediction": "No Hand Detected",
            "raw_prediction": "No Hand Detected",
            "landmarks": None,
            "confidence": 0.0
        }

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        landmarks = detection_result.hand_landmarks[0]
        # Extract 63 features (21 landmarks x 3 coordinates)
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        features_arr = np.array(features, dtype=np.float32).reshape(1, -1)
        scaled_features = scaler.transform(features_arr)
        pred_label = mlp_model.predict(scaled_features)[0]

        confidence = 1.0
        if hasattr(mlp_model, "predict_proba"):
            probs = mlp_model.predict_proba(scaled_features)[0]
            confidence = float(np.max(probs))

        return {
            "prediction": pred_label,
            "raw_prediction": pred_label,
            "landmarks": landmarks,
            "confidence": confidence
        }

    return {
        "prediction": "No Hand Detected",
        "raw_prediction": "No Hand Detected",
        "landmarks": None,
        "confidence": 0.0
    }


def postprocess(prediction_queue: deque) -> str:
    """
    Stage 3: Postprocess.
    Applies majority voting across the sliding window (prediction_queue)
    to stabilize the output and eliminate flickering between consecutive frames.
    """
    if not prediction_queue:
        return "No Hand Detected"

    # Count occurrences in the queue and determine the majority vote
    counter = Counter(prediction_queue)
    majority_vote, _ = counter.most_common(1)[0]
    return majority_vote


def render(frame: np.ndarray, result, fps: float = None) -> np.ndarray:
    """
    Stage 4: Render.
    Draws the high-contrast prediction HUD banner, hand landmark skeleton,
    and real-time FPS on the frame using OpenCV.

    Parameters:
        frame (np.ndarray): The BGR image frame to draw upon.
        result (dict or str): Prediction result containing 'prediction', 'fps',
                              and optional 'landmarks'.
        fps (float, optional): Optional override for FPS counter.
    """
    output = frame.copy()
    h, w = output.shape[:2]

    # Parse result dictionary or string
    if isinstance(result, dict):
        pred_label = result.get("prediction", "No Hand Detected")
        fps_val = result.get("fps", 0.0) if fps is None else fps
        landmarks = result.get("landmarks", None)
    else:
        pred_label = str(result)
        fps_val = 0.0 if fps is None else fps
        landmarks = None

    # 1. Draw hand skeleton connections and landmark points
    if landmarks is not None:
        # Landmark connections (green lines)
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(output, pt1, pt2, (0, 220, 0), 2, cv2.LINE_AA)

        # Landmark points (red with white border)
        for lm in landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(output, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(output, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

    # 2. Draw high-contrast top banner HUD
    banner_height = 55
    cv2.rectangle(output, (0, 0), (w, banner_height), (20, 20, 20), -1)

    is_hand_detected = pred_label.lower() != "no hand detected"
    accent_color = (0, 255, 0) if is_hand_detected else (80, 80, 255)

    # Accent bottom divider line
    cv2.line(output, (0, banner_height), (w, banner_height), accent_color, 2)

    # Prediction text
    pred_text = f"PREDICTION: {pred_label.upper()}"
    cv2.putText(
        output,
        pred_text,
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        accent_color,
        2,
        cv2.LINE_AA
    )

    # FPS counter on top-right
    if fps_val > 0.0:
        fps_text = f"FPS: {fps_val:.1f}"
    else:
        fps_text = "FPS: --"

    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    fps_x = max(16, w - text_size[0] - 20)
    cv2.putText(
        output,
        fps_text,
        (fps_x, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


# -------------------------------------------------------------------------
# Execution Handlers
# -------------------------------------------------------------------------
def process_single_image(image_path: Path):
    """Processes a single image file through the 4-stage pipeline."""
    print(f"[+] Loading image: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    prediction_queue = deque(maxlen=5)

    # Measure inference latency
    start_t = time.time()
    mp_image = preprocess(frame)
    result = infer(mp_image)
    prediction_queue.append(result["raw_prediction"])
    smoothed_pred = postprocess(prediction_queue)
    elapsed = time.time() - start_t

    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    result["prediction"] = smoothed_pred
    result["fps"] = fps

    rendered = render(frame, result)

    print("=" * 60)
    print(f"File       : {image_path.name}")
    print(f"Prediction : {smoothed_pred.upper()}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print(f"Latency    : {elapsed * 1000:.1f} ms (approx {fps:.1f} FPS)")
    print("=" * 60)

    # Save to outputs directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"pred_{image_path.name}"
    cv2.imwrite(str(out_path), rendered)
    print(f"[OK] Output saved to: {out_path}")

    # Display image window
    try:
        cv2.imshow("Sign Language Pipeline (Milestone 3)", rendered)
        print("\nPress any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        # Headless environment
        pass


def process_video_or_webcam(source):
    """
    Streams a video file or live webcam feed through the 4-stage pipeline
    with temporal voting smoothing (deque maxlen=5) and FPS calculation.
    """
    print(f"[+] Opening video/webcam source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/webcam source: {source}")

    prediction_queue = deque(maxlen=5)
    prev_time = time.time()
    fps = 0.0

    print("[*] Stream started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate instantaneous and smoothed FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = 0.9 * fps + 0.1 * inst_fps if fps > 0.0 else inst_fps

        # Pipeline execution
        mp_image = preprocess(frame)
        result = infer(mp_image)
        prediction_queue.append(result["raw_prediction"])
        smoothed_pred = postprocess(prediction_queue)

        result["prediction"] = smoothed_pred
        result["fps"] = fps

        rendered = render(frame, result)

        try:
            cv2.imshow("Sign Language Pipeline (Milestone 3)", rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[*] Quit signal received.")
                break
        except cv2.error:
            # Fallback if headless environment
            print(f"[Stream] Pred: {smoothed_pred} | FPS: {fps:.1f}")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Stream closed.")



def main():
    parser = argparse.ArgumentParser(
        description="Universal Sign Language Pipeline for Milestone 3."
    )
    parser.add_argument(
        "-input", "--input",
        dest="input",
        type=str,
        default=None,
        help="Path to input image or video file."
    )
    parser.add_argument(
        "-webcam", "--webcam",
        dest="webcam",
        type=int,
        default=None,
        help="Integer index of the webcam device (e.g., 0)."
    )

    args = parser.parse_args()

    # Validate CLI arguments
    if args.input is None and args.webcam is None:
        print("[!] No input specified. Defaulting to webcam 0...")
        args.webcam = 0

    # Initialize models and MediaPipe Tasks detector
    initialize_pipeline()

    if args.input is not None:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            process_single_image(input_path)
        else:
            process_video_or_webcam(str(input_path))
    elif args.webcam is not None:
        process_video_or_webcam(args.webcam)


if __name__ == "__main__":
    main()
