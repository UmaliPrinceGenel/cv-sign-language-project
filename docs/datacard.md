# Dataset Data Card: HaGRID Subset for Sign Language Recognition

## 1. Sources and Licenses
* **Source:** Subset of the HaGRID (HAnd Gesture Recognition Image Dataset) sampled from the 30k 384p Kaggle distribution[cite: 7].
* **Direct Link:** [HaGRID Sample 30k 384p on Kaggle](https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p)
* **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International).

## 2. Counts per Class and per Split
The dataset consists of **1,600 total images** distributed equally across 8 target gesture classes (`call`, `dislike`, `like`, `mute`, `ok`, `palm`, `peace`, `stop`)
The data was strictly split by source using a **70% / 15% / 15%** stratified distribution:
* **Total per class:** 200 images
* **Training Set:** 140 images per class (1,120 images total)
* **Validation Set:** 30 images per class (240 images total)
* **Test Set:** 30 images per class (240 images total)-

## 3. How Labels Were Made and Checked
* **Annotations:** Labels and bounding boxes were sourced from HaGRID's original JSON annotation files (`ann_train_val`).
* **Validation:** Bounding boxes were programmatically mapped to the target gesture label for each image ID. If an image contained multiple hands, the script specifically isolated the bounding box that matched the assigned class label.

## 4. Image Sizes and Preprocessing Applied
* **Original Size:** Various raw resolutions from the HaGRID dataset.
* **Preprocessing Pipeline:** 
  1. Cropped to the hand Region of Interest (ROI) using the JSON bounding box with an additional 10% context padding.
  2. Resized to a fixed `128x128` resolution.
  3. Denoised using a 5x5 Gaussian Blur.
  4. Converted to YCrCb color space for skin pixel thresholding and morphological closing.
  5. Contrast enhanced using CLAHE (Clip Limit 2.0, 8x8 Grid) on a grayscale copy.
* **Augmentation:** Horizontal flip augmentation was applied strictly to the Training Split.

## 5. Known Gaps and Limitations
* **Skin-Colored Backgrounds:** The classical YCrCb skin mask heavily relies on perfect bounding box cropping. Without these bounding boxes (e.g., in live webcam predictions), the mask often fails by picking up faces, necks, or wooden furniture (like warm-colored desks) as the "largest contour."
* **Lighting:** Extreme shadows or very dark lighting conditions can cause the fixed skin threshold values (`Lower: 0, 133, 77` | `Upper: 255, 173, 127`) to under-segment the hand.