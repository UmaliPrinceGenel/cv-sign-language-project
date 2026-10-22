import cv2
import numpy as np

# Empty lookup for local predictions (Bypassed when img_stem=None)
BBOX_LOOKUP = {}

def crop_to_hand_roi(image_bgr, img_stem):
    if img_stem not in BBOX_LOOKUP or img_stem is None:
        return image_bgr

    img_h, img_w = image_bgr.shape[:2]
    nx, ny, nw, nh = BBOX_LOOKUP[img_stem]

    # Convert normalized [x, y, w, h] to pixel coordinates with 10% context padding
    pad_x = nw * 0.10
    pad_y = nh * 0.10

    x1 = max(0, int((nx - pad_x) * img_w))
    y1 = max(0, int((ny - pad_y) * img_h))
    x2 = min(img_w, int((nx + nw + pad_x) * img_w))
    y2 = min(img_h, int((ny + nh + pad_y) * img_h))

    crop = image_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else image_bgr

def extract_hand_features(image_bgr, img_stem=None):
    """
    Project Outline Classical Baseline Pipeline on Hand ROI:
    1. Crop to Hand ROI (if bbox available) & Resize to 128x128
    2. 5x5 Gaussian Blur Denoising
    3. BGR -> YCrCb Skin Mask + Morphological Closing
    4. Largest Hand Contour -> 7 Hu Moments + Convexity Defects + Shape Ratios
    5. CLAHE + HOG Orientation Features
    """
    if image_bgr is None:
        return None

    # 1. Isolate the hand ROI so faces/walls don't corrupt the skin mask
    if img_stem is not None:
        image_bgr = crop_to_hand_roi(image_bgr, img_stem)

    img = cv2.resize(image_bgr, (128, 128))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. Convert to YCrCb and threshold skin pixels
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. Extract Largest Hand Contour + 7 Hu Moments + Convexity Defects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu_features = [0.0] * 7
    defect_count = 0.0
    aspect_ratio = 1.0
    solidity = 0.0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > 100:
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_features = [
                -1.0 * np.sign(h) * np.log10(abs(h) + 1e-10) if h != 0 else 0.0
                for h in hu_moments
            ]
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / float(h) if h > 0 else 1.0

            hull_pts = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull_pts)
            solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0

            hull_idx = cv2.convexHull(largest_contour, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3 and len(largest_contour) > 3:
                try:
                    defects = cv2.convexityDefects(largest_contour, hull_idx)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            if defects[i, 0][3] > 1000:
                                defect_count += 1.0
                except Exception:
                    defect_count = 0.0

    # 4. CLAHE Contrast Enhancement + HOG Descriptor on the cropped hand
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
    hog_features = hog.compute(gray_enhanced).flatten()

    return np.hstack([hu_features, [defect_count, aspect_ratio, solidity], hog_features])