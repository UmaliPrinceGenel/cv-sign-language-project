"""
Interactive Web Application for Real-time Sign Language Word Gesture Recognition.
Milestone 4 (80% Progress Check) Interactive Application.

Wraps the unified end-to-end pipeline in src/run.py:
- Models loaded once in memory (MediaPipe HandLandmarker + Scaler + MLP classifier).
- Interactive controls: Confidence threshold slider, Skeleton toggle.
- Clean visual output: Skeleton keypoints only; no redundant text banners on the image.
- Native HTTP MJPEG Live Webcam Stream (Zero browser freeze, locked 30 FPS hardware capture).
- Dedicated cards for Prediction, Confidence, Latency & FPS.
- Cumulative gesture counter and tally statistics.
- Direct desktop launcher for native OpenCV window.
"""

import sys
import time
import subprocess
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import gradio as gr
from fastapi.responses import StreamingResponse, JSONResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run import (
    initialize_pipeline,
    preprocess,
    infer,
    postprocess,
    render,
    ThreadedCamera,
    PROJECT_ROOT
)

# -------------------------------------------------------------------------
# Global State & Model Pre-loading (Loaded ONCE in memory)
# -------------------------------------------------------------------------
print("[*] Initializing Sign Language AI Pipeline for Web Application...")
initialize_pipeline(video_mode=False)

CLASSES = ["CALL", "DISLIKE", "LIKE", "MUTE", "OK", "PALM", "PEACE", "STOP"]
gesture_counts = Counter({cls_name: 0 for cls_name in CLASSES})
gesture_counts["NO HAND"] = 0
total_frames_processed = 0

# Dynamic stream controls
current_conf_threshold = 0.50
current_show_skeleton = True
camera_active = True

latest_telemetry = {
    "prediction": "No Hand Detected",
    "confidence": "--",
    "latency": "--",
    "fps": "--"
}

desktop_process = None


def get_tally_df() -> pd.DataFrame:
    """Returns current cumulative gesture counts as a Pandas DataFrame."""
    data = []
    for cls_name in CLASSES:
        data.append({"Gesture": cls_name, "Detections": gesture_counts[cls_name]})
    data.append({"Gesture": "NO HAND DETECTED", "Detections": gesture_counts["NO HAND"]})
    return pd.DataFrame(data)


def reset_tally():
    """Resets cumulative gesture counters."""
    global total_frames_processed, gesture_counts
    total_frames_processed = 0
    gesture_counts = Counter({cls_name: 0 for cls_name in CLASSES})
    gesture_counts["NO HAND"] = 0
    return get_tally_df(), "Counters reset to 0."


def update_controls(conf: float, skel: bool):
    """Updates dynamic stream settings."""
    global current_conf_threshold, current_show_skeleton
    current_conf_threshold = float(conf)
    current_show_skeleton = bool(skel)


# -------------------------------------------------------------------------
# Single Image Inference Handler (Clean: Skeleton Only, No Top/Bottom Banners)
# -------------------------------------------------------------------------
def process_image(
    image: np.ndarray,
    conf_threshold: float,
    show_skeleton: bool
) -> Tuple[np.ndarray, str, str, str, Dict[str, float], pd.DataFrame]:
    """
    Processes an uploaded still image through the pipeline.
    Renders ONLY the hand skeleton keypoints without any text banners on the image.
    """
    global total_frames_processed, gesture_counts

    if image is None:
        return None, "No Image", "0.0%", "0.0 ms", {}, get_tally_df()

    # Convert RGB to BGR
    bgr_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    mp_image = preprocess(bgr_frame)
    result = infer(mp_image)
    dt = time.perf_counter() - t0

    latency_ms = dt * 1000.0
    fps = (1.0 / dt) if dt > 0 else 0.0

    raw_pred = result["prediction"]
    conf = float(result["confidence"])
    probs = result.get("probabilities", {})

    # Check confidence threshold
    if raw_pred.lower() != "no hand detected":
        if conf < conf_threshold:
            display_pred = f"Uncertain (< {int(conf_threshold * 100)}%)"
            detected_tag = "NO HAND"
        else:
            display_pred = raw_pred.upper()
            detected_tag = raw_pred.upper()
    else:
        display_pred = "No Hand Detected"
        detected_tag = "NO HAND"

    # Update cumulative statistics
    total_frames_processed += 1
    gesture_counts[detected_tag] += 1

    # Render clean image: show_banner=False and counts_summary=None
    rendered_bgr = render(
        bgr_frame,
        result,
        fps=fps,
        latency_ms=latency_ms,
        show_skeleton=show_skeleton,
        show_banner=False,
        conf_threshold=conf_threshold,
        counts_summary=None
    )

    rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
    conf_display = f"{conf * 100:.1f}%" if raw_pred.lower() != "no hand detected" else "N/A"
    lat_display = f"{latency_ms:.1f} ms ({fps:.1f} FPS)"
    formatted_probs = {k.upper(): float(v) for k, v in probs.items()}

    return rendered_rgb, display_pred, conf_display, lat_display, formatted_probs, get_tally_df()


# -------------------------------------------------------------------------
# Native MJPEG Camera Generator (Zero Browser Lag, 30 FPS Hardware Capture)
# -------------------------------------------------------------------------
def generate_mjpeg_stream():
    """
    Captures live webcam frames via Windows Media Foundation at locked 30 FPS,
    runs MediaPipe + MLP inference, and yields JPEG frames over HTTP multipart stream.
    Immune to browser WebSocket freezing and memory leaks.
    """
    global camera_active, latest_telemetry, total_frames_processed, gesture_counts

    print("[*] Starting backend MJPEG webcam stream via ThreadedCamera...")
    cap = ThreadedCamera(0)
    if not cap.isOpened():
        print("[!] Could not open camera 0 for MJPEG stream.")
        # Yield error frame
        err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err_frame, "Camera 0 Busy or Not Available", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode(".jpg", err_frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        return

    prediction_queue = deque(maxlen=5)
    last_frame_id = 0
    t_prev = time.perf_counter()
    fps = 30.0

    try:
        while camera_active and cap.isOpened():
            fid, frame = cap.read_latest()
            if frame is None or fid == last_frame_id:
                time.sleep(0.01)
                continue

            last_frame_id = fid
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if 0.001 < dt < 0.30:
                fps = 1.0 / dt

            # Run inference
            t0 = time.perf_counter()
            mp_image = preprocess(frame)
            result = infer(mp_image)
            prediction_queue.append(result["raw_prediction"])
            smoothed_pred = postprocess(prediction_queue)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            raw_pred = smoothed_pred
            conf = float(result["confidence"])
            if raw_pred.lower() != "no hand detected" and conf >= current_conf_threshold:
                disp_pred = raw_pred.upper()
                gesture_counts[disp_pred] += 1
            else:
                disp_pred = "No Hand Detected"
                gesture_counts["NO HAND"] += 1

            total_frames_processed += 1

            latest_telemetry["prediction"] = disp_pred
            latest_telemetry["confidence"] = f"{conf * 100:.1f}%" if disp_pred != "No Hand Detected" else "N/A"
            latest_telemetry["latency"] = f"{lat_ms:.1f} ms"
            latest_telemetry["fps"] = f"{fps:.1f} FPS"

            # Render clean skeleton overlay (NO top banner, NO bottom bar)
            rendered = render(
                frame,
                result,
                fps=fps,
                latency_ms=lat_ms,
                show_skeleton=current_show_skeleton,
                show_banner=False,
                conf_threshold=current_conf_threshold,
                counts_summary=None
            )

            ret, jpeg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.015)
    finally:
        print("[*] Releasing backend camera for MJPEG stream.")
        cap.release()


def read_latest_telemetry():
    """Polls latest live stream metrics for the UI cards."""
    return (
        latest_telemetry["prediction"],
        latest_telemetry["confidence"],
        f"{latest_telemetry['latency']} ({latest_telemetry['fps']})",
        get_tally_df()
    )


# -------------------------------------------------------------------------
# Launch Native Desktop Camera Window
# -------------------------------------------------------------------------
def launch_native_camera():
    """Launches the dedicated OpenCV MSMF 30 FPS camera window."""
    global desktop_process, camera_active
    # Temporarily pause MJPEG backend so OpenCV desktop window has exclusive camera access
    camera_active = False
    time.sleep(0.3)

    if desktop_process is not None and desktop_process.poll() is None:
        try:
            desktop_process.terminate()
        except Exception:
            pass

    cmd = f'start cmd /c python "{PROJECT_ROOT}\\src\\run.py" --webcam 0'
    desktop_process = subprocess.Popen(cmd, shell=True)
    return "✅ **Native OpenCV Desktop Window Launched!** (Running locked 30 FPS. Look at your taskbar / screen)."


def resume_web_stream():
    """Resumes the browser MJPEG camera stream."""
    global camera_active
    camera_active = True
    return None


# -------------------------------------------------------------------------
# Pre-recorded Video Handler
# -------------------------------------------------------------------------
def process_video_file(
    video_path: str,
    conf_threshold: float,
    show_skeleton: bool,
    progress=gr.Progress()
) -> Tuple[str, str, pd.DataFrame]:
    """Processes an uploaded video file frame-by-frame and exports an annotated video."""
    global total_frames_processed, gesture_counts

    if not video_path:
        return None, "No video provided.", get_tally_df()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error opening video file.", get_tally_df()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = PROJECT_ROOT / "outputs" / "processed_demo.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (w, h))

    frames_processed = 0
    t_start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frames_processed += 1
        progress(frames_processed / max(total_frames, 1), desc=f"Processing frame {frames_processed}/{total_frames}")

        t0 = time.perf_counter()
        mp_image = preprocess(frame)
        result = infer(mp_image)
        lat = (time.perf_counter() - t0) * 1000.0

        raw_pred = result["prediction"]
        conf = float(result["confidence"])
        tag = raw_pred.upper() if (raw_pred.lower() != "no hand detected" and conf >= conf_threshold) else "NO HAND"
        gesture_counts[tag] += 1
        total_frames_processed += 1

        rendered = render(
            frame,
            result,
            fps=fps_in,
            latency_ms=lat,
            show_skeleton=show_skeleton,
            show_banner=False,
            conf_threshold=conf_threshold,
            counts_summary=None
        )
        out_writer.write(rendered)

    cap.release()
    out_writer.release()

    total_time = time.perf_counter() - t_start
    avg_speed = (frames_processed / total_time) if total_time > 0 else 0.0
    summary = f"Processed {frames_processed} frames in {total_time:.2f}s ({avg_speed:.1f} FPS)."

    return str(output_path), summary, get_tally_df()


# -------------------------------------------------------------------------
# Build Sample Image Gallery
# -------------------------------------------------------------------------
sample_images = []
sample_dir = PROJECT_ROOT / "data"
for name in ["call_01.jpg", "dislike_01.jpg", "like_01.jpg", "mute_01.jpg", "ok_01.jpg", "palm_01.jpg", "peace_01.jpg", "stop_01.jpg"]:
    p = sample_dir / name
    if p.exists():
        sample_images.append([str(p)])


# -------------------------------------------------------------------------
# Gradio User Interface Layout
# -------------------------------------------------------------------------
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.hud-title {
    text-align: center;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 12px;
}
.stream-box {
    display: flex;
    justify-content: center;
    align-items: center;
    background: #111;
    border-radius: 8px;
    overflow: hidden;
    padding: 6px;
    border: 2px solid #2a5298;
}
"""

with gr.Blocks(title="Sign Language Gesture Recognition - Milestone 4") as demo:
    gr.HTML(
        """
        <div class="hud-title">
            <h1 style="margin: 0; font-size: 1.8rem;">🤟 Sign Language Gesture Recognition System</h1>
            <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 0.95rem;">
                CSST 106 Computer Vision • Milestone 4 Interactive Application • CPU-Optimized Pipeline
            </p>
        </div>
        """
    )

    with gr.Row():
        # Left Column: Global Controls & Cumulative Stats
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Interactive Controls")
            conf_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.50,
                step=0.05,
                label="Confidence Threshold",
                info="Predictions below this score are filtered out as Uncertain."
            )
            toggle_skeleton = gr.Checkbox(
                value=True,
                label="Show Hand Skeleton & Keypoints",
                info="Draw 21 MediaPipe hand landmarks and bone connections."
            )

            # Sync dynamic slider and checkbox with backend stream
            conf_slider.change(fn=update_controls, inputs=[conf_slider, toggle_skeleton], outputs=[])
            toggle_skeleton.change(fn=update_controls, inputs=[conf_slider, toggle_skeleton], outputs=[])

            gr.Markdown("### 📊 Cumulative Gesture Tally")
            tally_table = gr.DataFrame(
                value=get_tally_df(),
                label="Cumulative Detection Counts",
                interactive=False
            )
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset Counts", variant="secondary")
                sync_btn = gr.Button("📊 Sync Latest", variant="secondary")

            reset_status = gr.Markdown("")
            reset_btn.click(fn=reset_tally, inputs=[], outputs=[tally_table, reset_status])
            sync_btn.click(fn=get_tally_df, inputs=[], outputs=[tally_table])

        # Right Column: Main Application Modes (Tabs)
        with gr.Column(scale=2):
            with gr.Tabs():
                # Tab 1: Image File Upload
                with gr.Tab("🖼️ Image File Upload"):
                    with gr.Row():
                        with gr.Column():
                            img_input = gr.Image(type="numpy", label="Upload Hand Gesture Image")
                            run_img_btn = gr.Button("🔍 Run Prediction", variant="primary")
                        with gr.Column():
                            img_output = gr.Image(type="numpy", label="Annotated Result (Skeleton Only)")

                    with gr.Row():
                        out_pred = gr.Textbox(label="Detected Gesture", value="--")
                        out_conf = gr.Textbox(label="Confidence Score", value="--")
                        out_lat = gr.Textbox(label="Latency & FPS", value="--")

                    out_probs = gr.Label(num_top_classes=4, label="Class Probabilities")

                    run_img_btn.click(
                        fn=process_image,
                        inputs=[img_input, conf_slider, toggle_skeleton],
                        outputs=[img_output, out_pred, out_conf, out_lat, out_probs, tally_table]
                    )

                    if sample_images:
                        gr.Markdown("#### 📂 Quick Test Gallery (Click any sample image below to load and test)")
                        gr.Examples(
                            examples=sample_images,
                            inputs=img_input,
                            fn=process_image,
                            outputs=[img_output, out_pred, out_conf, out_lat, out_probs, tally_table],
                            cache_examples=False
                        )

                # Tab 2: Live Webcam Stream (MJPEG Stream - Zero Freeze)
                with gr.Tab("📹 Live Webcam Stream"):
                    gr.Markdown("### 🎥 Native 30 FPS Video Feed (Zero WebSocket Overhead)")
                    gr.Markdown("Streams directly from your webcam via Windows Media Foundation without browser lag or freeze:")

                    # Native MJPEG Stream Component (Standard HTML <img>)
                    gr.HTML(
                        """
                        <div class="stream-box">
                            <img id="live_mjpeg" src="/video_feed" style="max-width: 640px; width: 100%; height: auto; border-radius: 4px;" alt="Live Webcam Feed" />
                        </div>
                        """
                    )

                    with gr.Row():
                        cam_pred = gr.Textbox(label="Active Gesture", value="--")
                        cam_conf = gr.Textbox(label="Confidence", value="--")
                        cam_lat = gr.Textbox(label="AI Latency & Speed", value="--")

                    # Fast telemetry timer (updates text boxes every 0.5s without touching the video feed)
                    telemetry_timer = gr.Timer(value=0.5, active=True)
                    telemetry_timer.tick(
                        fn=read_latest_telemetry,
                        inputs=[],
                        outputs=[cam_pred, cam_conf, cam_lat, tally_table]
                    )

                    with gr.Row():
                        refresh_stream_btn = gr.Button("🔄 Reconnect Stream", variant="secondary")
                        refresh_stream_btn.click(fn=resume_web_stream, inputs=[], outputs=[])

                    gr.Markdown("---")
                    gr.Markdown("#### 🖥️ Dedicated Hardware Window Alternative")
                    gr.Markdown("Prefer the standalone OpenCV desktop window? Click below:")
                    launch_desktop_btn = gr.Button("🚀 Launch Native Desktop Camera Window (30 FPS OpenCV)", variant="primary")
                    desktop_status = gr.Markdown("")
                    launch_desktop_btn.click(fn=launch_native_camera, inputs=[], outputs=[desktop_status])

                # Tab 3: Video File Input
                with gr.Tab("🎞️ Pre-recorded Video File"):
                    gr.Markdown("Upload a pre-recorded `.mp4` or `.avi` video clip to run end-to-end batch inference.")
                    with gr.Row():
                        vid_input = gr.Video(label="Upload Video File")
                        vid_output = gr.Video(label="Annotated Output Video")

                    vid_status = gr.Textbox(label="Batch Processing Summary", value="Ready.")
                    run_vid_btn = gr.Button("🚀 Process Video", variant="primary")

                    run_vid_btn.click(
                        fn=process_video_file,
                        inputs=[vid_input, conf_slider, toggle_skeleton],
                        outputs=[vid_output, vid_status, tally_table]
                    )

    gr.Markdown(
        """
        ---
        **Pipeline Architecture:** MediaPipe Tasks API (21 3D Landmarks) ➔ StandardScaler ➔ 2-Layer MLP (128x64 ReLU) ➔ Geometric Landmark Refinement ➔ Majority-Vote Smoothing Window.  
        *Model kept persistently resident in RAM without reload.*
        """
    )


# -------------------------------------------------------------------------
# Mount FastAPI App with Dedicated MJPEG Video Feed
# -------------------------------------------------------------------------
from fastapi import FastAPI
import uvicorn

fastapi_app = FastAPI(title="Sign Language Recognition API")


@fastapi_app.get("/video_feed")
def video_feed():
    """Serves continuous MJPEG multipart stream directly to browser <img>."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@fastapi_app.get("/telemetry")
def get_telemetry():
    """JSON telemetry endpoint."""
    return JSONResponse(latest_telemetry)


# Mount Gradio on root path /
app = gr.mount_gradio_app(fastapi_app, demo, path="/")


def find_available_port(start_port: int = 7860, max_attempts: int = 20) -> int:
    import socket
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('127.0.0.1', p))
                return p
            except OSError:
                continue
    return start_port


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sign Language Recognition Interactive App")
    parser.add_argument("--port", type=int, default=7860, help="Web server port (default: 7860)")
    args = parser.parse_args()

    chosen_port = find_available_port(start_port=args.port)
    if chosen_port != args.port:
        print(f"[!] Notice: Port {args.port} is already in use by another process.")
        print(f"[+] Automatically assigned available port {chosen_port}.")

    print(f"\n[+] Launching Zero-Lag Sign Language Web App on http://127.0.0.1:{chosen_port} ...")
    uvicorn.run(app, host="127.0.0.1", port=chosen_port, log_level="warning")


if __name__ == "__main__":
    main()
