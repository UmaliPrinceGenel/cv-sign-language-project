import cv2
import numpy as np
import joblib
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths
TEST_DIR = Path("data/test")
MODEL_PATH = Path("models/mediapipe_mlp_model.pkl")
SCALER_PATH = Path("models/scaler_mp.pkl")
TASK_PATH = Path("models/hand_landmarker.task")
FAILURES_DIR = Path("outputs/failures")
FAILURES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading models for Failure Generator...")
mlp_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

base_options = python.BaseOptions(model_asset_path=str(TASK_PATH))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)

failures_saved = 0
TARGET_CLASSES = ["call", "dislike", "like", "mute", "ok", "palm", "peace", "stop"]

print(f"Hunting for 10 mismatches in {TEST_DIR}...\n")

for class_dir in TEST_DIR.iterdir():
    if not class_dir.is_dir(): continue
    true_label = class_dir.name.lower()
    
    for img_path in class_dir.glob("*.*"):
        if failures_saved >= 10:
            break
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Preprocess and Infer
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        
        if not result.hand_landmarks:
            continue # Skip no-hand drops for the visual gallery
            
        # Extract features
        features = []
        for lm in result.hand_landmarks[0]:
            features.extend([lm.x, lm.y, lm.z])
            
        scaled = scaler.transform([np.array(features)])
        pred_label = mlp_model.predict(scaled)[0].lower()
        
        # If it's a mismatch, draw and save!
        if pred_label != true_label:
            h, w, _ = img.shape
            
            # Draw the MediaPipe skeleton in red
            for lm in result.hand_landmarks[0]:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                
            # Draw True vs Pred text
            cv2.putText(img, f"TRUE: {true_label.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"PRED: {pred_label.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            save_path = FAILURES_DIR / f"fail_{failures_saved+1}_{true_label}_as_{pred_label}.jpg"
            cv2.imwrite(str(save_path), img)
            print(f"Saved {save_path.name}")
            
            failures_saved += 1

    if failures_saved >= 10:
        break
print("done")