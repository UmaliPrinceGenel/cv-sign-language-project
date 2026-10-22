import cv2
import joblib
import argparse
import numpy as np
from pathlib import Path
# Assuming you save your feature extraction function in a file named baseline.py inside src/
from baseline import extract_hand_features 

def run_prototype(image_path, output_dir, model_path="models/svm_baseline_model.pkl", scaler_path="models/scaler.pkl"):
    """
    End-to-end prototype: One input image -> Preprocess -> Predict -> Output image with label.
    """
    print(f"Loading image: {image_path}")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError("Could not read image. Check the path.")

    # 1. Load the trained model and scaler
    print("Loading SVM model and Scaler...")
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Preprocess (We pass img_stem=None because in real life we don't have the JSON boxes)
    print("Extracting features...")
    features = extract_hand_features(img_bgr, img_stem=None)
    
    if features is None or np.all(features == 0):
        predicted_label = "UNKNOWN (No Hand Detected)"
    else:
        # 3. Model Algorithm
        features_scaled = scaler.transform([features])
        predicted_label = svm_model.predict(features_scaled)[0].upper()

    # 4. Draw Result
    print(f"Prediction: {predicted_label}")
    output_img = img_bgr.copy()
    
    # Draw a high-contrast banner at the top
    cv2.rectangle(output_img, (0, 0), (output_img.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(
        output_img, 
        f"PREDICTION: {predicted_label}", 
        (15, 35), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (0, 255, 0), 
        2, 
        cv2.LINE_AA
    )

    # Save output
    out_path = Path(output_dir) / f"pred_{Path(image_path).name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), output_img)
    print(f"Saved result to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the SVM Baseline Prototype")
    parser.add_argument("-input", required=True, help="Path to input test image")
    parser.add_argument("-output", required=True, help="Directory to save the result")
    args = parser.parse_args()
    
    run_prototype(args.input, args.output)