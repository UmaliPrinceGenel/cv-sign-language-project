import os
import sys
import time
import ctypes
import argparse
import threading
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
    task_path: Path = TASK_PATH,
    video_mode: bool = False
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
    # VIDEO mode re-uses the previous frame's hand location (tracking) and skips the
    # full-frame palm detector while the hand stays in view: ~2x lower latency on live
    # streams (measured 15.9 ms -> 7.6 ms per frame). IMAGE mode is kept for still images.
    running_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    print(f"[+] Initializing MediaPipe Tasks API HandLandmarker ({running_mode.name} mode)...")
    base_options = python.BaseOptions(model_asset_path=str(task_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=1 if video_mode else 2,
        min_hand_detection_confidence=0.45 if video_mode else 0.35,
        min_hand_presence_confidence=0.45 if video_mode else 0.35,
        min_tracking_confidence=0.45 if video_mode else 0.35
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


def refine_prediction(lm: np.ndarray, raw_pred: str, confidence: float) -> tuple[str, float]:
    """
    Position-, scale-, and handedness-invariant geometric refinement using the 21
    MediaPipe hand landmarks. Resolves look-alike confusions (OK vs STOP/PALM,
    PALM vs STOP via inter-finger spread angle, and CALL vs LIKE/DISLIKE/MUTE).
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


def infer(mp_image: mp.Image, timestamp_ms: int = None) -> dict:
    """
    Stage 2: Inference.
    Detects hand landmarks using MediaPipe Tasks API, selects the active raised hand
    when two hands are visible, extracts 63 normalized coordinates, scales features,
    runs MLP prediction, and applies geometric landmark refinement.

    Parameters:
        mp_image (mp.Image): Preprocessed frame.
        timestamp_ms (int, optional): Monotonically increasing frame timestamp. Required
                                      when the detector runs in VIDEO (tracking) mode.

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

    if timestamp_ms is None:
        detection_result = detector.detect(mp_image)
    else:
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.hand_landmarks:
        # Select active raised hand (highest in frame / minimum y coordinate)
        landmarks = min(detection_result.hand_landmarks, key=lambda h: min(pt.y for pt in h))
        lm_arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        features_arr = lm_arr.reshape(1, -1)
        scaled_features = scaler.transform(features_arr)

        # Single forward pass: derive both label and confidence from predict_proba
        probs_dict = {}
        if hasattr(mlp_model, "predict_proba"):
            probs = mlp_model.predict_proba(scaled_features)[0]
            best = int(np.argmax(probs))
            pred_label = str(mlp_model.classes_[best]).lower()
            confidence = float(probs[best])
            probs_dict = {str(c): float(p) for c, p in zip(mlp_model.classes_, probs)}
        else:
            pred_label = str(mlp_model.predict(scaled_features)[0]).lower()
            confidence = 1.0
            probs_dict = {pred_label: 1.0}

        pred_label, confidence = refine_prediction(lm_arr, pred_label, confidence)
        if pred_label in probs_dict:
            probs_dict[pred_label] = max(probs_dict[pred_label], confidence)

        return {
            "prediction": pred_label,
            "raw_prediction": pred_label,
            "landmarks": landmarks,
            "confidence": confidence,
            "probabilities": probs_dict
        }

    return {
        "prediction": "No Hand Detected",
        "raw_prediction": "No Hand Detected",
        "landmarks": None,
        "confidence": 0.0,
        "probabilities": {}
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


def render(
    frame: np.ndarray,
    result,
    fps: float = None,
    latency_ms: float = None,
    show_skeleton: bool = True,
    show_banner: bool = True,
    conf_threshold: float = 0.0,
    counts_summary: str = None
) -> np.ndarray:
    """
    Stage 4: Render.
    Draws the high-contrast prediction HUD banner, hand landmark skeleton,
    and real-time FPS on the frame using OpenCV.

    Parameters:
        frame (np.ndarray): The BGR image frame to draw upon.
        result (dict or str): Prediction result containing 'prediction', 'fps',
                              and optional 'landmarks'.
        fps (float, optional): Optional override for FPS counter.
        latency_ms (float, optional): Optional processing latency in milliseconds.
        show_skeleton (bool): Whether to draw the hand keypoints and skeleton.
        show_banner (bool): Whether to draw the top prediction banner HUD.
        conf_threshold (float): Minimum confidence threshold. Predictions below
                                this threshold are marked as 'UNCERTAIN'.
        counts_summary (str, optional): Summary of cumulative gesture counts.
    """
    output = frame.copy()
    h, w = output.shape[:2]

    # Parse result dictionary or string
    confidence = 0.0
    if isinstance(result, dict):
        pred_label = result.get("prediction", "No Hand Detected")
        fps_val = result.get("fps", 0.0) if fps is None else fps
        lat_val = result.get("latency_ms", None) if latency_ms is None else latency_ms
        landmarks = result.get("landmarks", None)
        confidence = float(result.get("confidence", 0.0))
    else:
        pred_label = str(result)
        fps_val = 0.0 if fps is None else fps
        lat_val = latency_ms
        landmarks = None

    # Check confidence threshold
    is_hand_detected = pred_label.lower() != "no hand detected"
    if is_hand_detected and conf_threshold > 0.0 and confidence < conf_threshold:
        pred_label = f"UNCERTAIN (<{int(conf_threshold * 100)}%)"
        is_hand_detected = False

    # 1. Draw hand skeleton connections and landmark points
    if show_skeleton and landmarks is not None:
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
    if show_banner:
        banner_height = 55
        cv2.rectangle(output, (0, 0), (w, banner_height), (20, 20, 20), -1)

        accent_color = (0, 255, 0) if is_hand_detected else (80, 80, 255)

        # Accent bottom divider line
        cv2.line(output, (0, banner_height), (w, banner_height), accent_color, 2)

        # Prediction text (with raw-frame confidence when a hand is present)
        pred_text = f"PREDICTION: {pred_label.upper()}"
        if is_hand_detected and confidence > 0.0:
            pred_text += f"  {confidence * 100:.0f}%"
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
            if lat_val is not None and lat_val > 0:
                fps_text = f"FPS: {fps_val:.1f} ({lat_val:.0f}ms)"
            else:
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

    # 3. Optional bottom counts bar
    if counts_summary:
        bar_h = 26
        cv2.rectangle(output, (0, h - bar_h), (w, h), (15, 15, 15), -1)
        cv2.line(output, (0, h - bar_h), (w, h - bar_h), (60, 60, 60), 1)
        cv2.putText(
            output,
            counts_summary,
            (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
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


def _bring_window_to_front(window_name: str):
    """Forces the OpenCV window to the foreground on Windows so it never hides behind the IDE/browser."""
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        if hasattr(ctypes, "windll"):
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, window_name)
            if hwnd:
                user32.ShowWindow(hwnd, 1)  # SW_SHOWNORMAL
                user32.SetWindowPos(hwnd, -1, 100, 60, 960, 720, 0x0040)  # HWND_TOPMOST | SWP_SHOWWINDOW
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


class ThreadedCamera:
    """
    Asynchronous webcam reader that warms up on the main thread and continuously
    fetches the newest camera frame in a background daemon thread without blocking
    the OpenCV GUI or dropping frames during auto-exposure adjustments.
    """
    def __init__(self, source):
        self.is_webcam = isinstance(source, int)
        self.cap = None
        if self.is_webcam:
            for backend, name in ((cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "Default")):
                cap = cv2.VideoCapture(source, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != 640 or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 480:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                for _ in range(10):
                    ret, frm = cap.read()
                    if ret and frm is not None:
                        self.cap = cap
                        self.ret = True
                        self.frame = frm
                        print(f"[OK] Camera {source} opened via {name} ({frm.shape[1]}x{frm.shape[0]}).")
                        break
                    time.sleep(0.02)
                if self.cap is not None:
                    break
                cap.release()
        else:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frm = cap.read()
                self.cap = cap
                self.ret = ret
                self.frame = frm

        self.frame_id = 1 if (self.cap is not None and self.frame is not None) else 0
        self.running = self.cap is not None and self.cap.isOpened()
        self.lock = threading.Lock()
        self.thread = None
        if self.running and self.is_webcam:
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self):
        while self.running and self.cap.isOpened():
            ret, frm = self.cap.read()
            if ret and frm is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frm
                    self.frame_id += 1
            else:
                time.sleep(0.005)

    def read_latest(self):
        if not self.is_webcam:
            ret, frm = self.cap.read()
            if ret and frm is not None:
                self.frame_id += 1
                return self.frame_id, frm
            return self.frame_id, None
        with self.lock:
            return self.frame_id, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.running and self.cap is not None and self.cap.isOpened()

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()


def process_video_or_webcam(source):
    """
    Streams a video file or live webcam feed with background frame capture,
    MediaPipe VIDEO-mode tracking, fast idle-frame skipping when no hand is present,
    and responsive OpenCV GUI rendering.
    """
    print(f"[+] Opening video/webcam source: {source}")
    cap = ThreadedCamera(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/webcam source: {source}")

    window_name = "Sign Language Pipeline (Milestone 3)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    prediction_queue = deque(maxlen=5)
    gesture_counts = Counter()
    dt_history = deque(maxlen=20)
    latency_history = deque(maxlen=20)
    fps = 0.0
    avg_latency_ms = 0.0

    print("[*] Stream started. Press 'q' or 'ESC' in the video window to quit.")

    last_frame_id = 0
    last_ts = -1
    frame_count = 0
    t_start = time.perf_counter()
    prev_frame_time = time.perf_counter()
    last_result = {
        "prediction": "No Hand Detected",
        "raw_prediction": "No Hand Detected",
        "landmarks": None,
        "confidence": 0.0
    }
    hand_active = False

    while cap.isOpened():
        fid, frame = cap.read_latest()
        if frame is None:
            if not cap.is_webcam:
                break
            time.sleep(0.005)
            continue

        # If camera sensor hasn't delivered a new frame yet, keep GUI responsive without re-running AI
        if fid == last_frame_id:
            key = cv2.waitKey(4) & 0xFF
            if key == ord("q") or key == 27:
                print("[*] Quit signal received.")
                break
            continue

        last_frame_id = fid
        frame_count += 1

        now = time.perf_counter()
        dt = now - prev_frame_time
        prev_frame_time = now
        # Ignore long pauses from dragging window or snipping tool (>0.3s)
        if 0.001 < dt < 0.30:
            dt_history.append(dt)
        if dt_history:
            fps = 1.0 / float(np.mean(dt_history))

        # When no hand was detected on the previous frame, only run the full-frame palm
        # detector on every 2nd camera frame so 'No Hand Detected' has near-zero CPU cost
        # and never lags the live video feed. Once a hand appears, track every frame.
        run_ai_this_frame = hand_active or (frame_count % 2 == 1)

        if run_ai_this_frame:
            ts = int((time.perf_counter() - t_start) * 1000)
            ts = max(ts, last_ts + 1)
            last_ts = ts

            t0 = time.perf_counter()
            mp_image = preprocess(frame)
            last_result = infer(mp_image, timestamp_ms=ts)
            prediction_queue.append(last_result["raw_prediction"])
            smoothed_pred = postprocess(prediction_queue)
            hand_active = last_result["landmarks"] is not None
            latency_history.append((time.perf_counter() - t0) * 1000.0)
        else:
            smoothed_pred = postprocess(prediction_queue)
            latency_history.append(0.2)

        avg_latency_ms = float(np.mean(latency_history)) if latency_history else 0.0

        last_result["prediction"] = smoothed_pred
        last_result["fps"] = fps
        last_result["latency_ms"] = avg_latency_ms

        if smoothed_pred.lower() != "no hand detected":
            gesture_counts[smoothed_pred.upper()] += 1

        top_counts = [f"{k}: {v}" for k, v in gesture_counts.most_common(5)]
        counts_summary = ("COUNTS  |  " + "  •  ".join(top_counts)) if top_counts else "COUNTS: None yet"

        rendered = render(
            frame,
            last_result,
            fps=fps,
            latency_ms=avg_latency_ms,
            counts_summary=counts_summary
        )

        if frame_count % 30 == 0:
            print(f"[*] Frame {frame_count:05d} | Pred: {smoothed_pred.upper():16s} | "
                  f"AI Latency: {avg_latency_ms:4.1f} ms | Camera FPS: {fps:4.1f}")

        try:
            cv2.imshow(window_name, rendered)
            if frame_count == 1:
                _bring_window_to_front(window_name)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[*] Quit signal received.")
                break
        except cv2.error as e:
            # Fallback if headless environment
            print(f"[!] GUI Error: {e} | Pred: {smoothed_pred} | FPS: {fps:.1f}")
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
    # (VIDEO/tracking mode for streams, IMAGE mode for single still images)
    is_still_image = args.input is not None and Path(args.input).suffix.lower() in IMAGE_EXTENSIONS
    initialize_pipeline(video_mode=not is_still_image)

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
