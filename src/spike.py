"""
Spike Script for Computer Vision Pipeline
-----------------------------------------
This script reads sample images from `data/samples/`, generates a 20-sample 
labeled contact sheet, performs standard computer vision processing 
(grayscale, blurring, edge detection, Otsu thresholding), and writes 
visualizations and processed outputs into `outputs/`.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define project directories relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def generate_contact_sheet(image_files, output_dir):
    """Generates a 20-sample labeled contact sheet grid (Deliverable 2)."""
    if len(image_files) < 20:
        print(f"[!] Warning: Found only {len(image_files)} images. Aim for at least 20 for the contact sheet.")
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle("Dataset Contact Sheet (20 Labeled Samples)", fontsize=16, weight="bold")
    
    for ax, img_path in zip(axes.flatten(), image_files[:20]):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # Extract label from filename (e.g., "hello_01.jpg" -> "HELLO")
            label = img_path.stem.split("_")[0].upper()
            ax.imshow(img_rgb)
            ax.set_title(f"Label: {label}", fontsize=11, fontweight="bold")
        ax.axis("off")
        
    plt.tight_layout()
    contact_sheet_path = output_dir / "contact_sheet.png"
    plt.savefig(contact_sheet_path, dpi=150)
    plt.close(fig)
    print(f"[+] Generated contact sheet: {contact_sheet_path.name}")


def process_image(image_path: Path, output_dir: Path) -> None:
    """Processes a single image and saves the 2x2 CV operation results (Deliverable 3)."""
    print(f"[+] Processing spike on: {image_path.name}")

    # Read original image (BGR)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"    [!] Failed to read image: {image_path.name}")
        return

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to Grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(img_blurred, threshold1=50, threshold2=150)

    # Otsu's Thresholding
    _, thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Generate a comparative 2x2 visualization plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Feasibility Spike: {image_path.name}", fontsize=14, weight="bold")

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original (RGB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_gray, cmap="gray")
    axes[0, 1].set_title("Grayscale")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title("Canny Edges")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(thresh, cmap="gray")
    axes[1, 1].set_title("Otsu Threshold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    summary_out_path = output_dir / f"{image_path.stem}_analysis.png"
    plt.savefig(summary_out_path, dpi=150)
    plt.close(fig)

    print(f"    -> Saved spike analysis to: {summary_out_path.name}")


def main():
    print("=" * 60)
    print(" Computer Vision Project - Spike & Contact Sheet Runner")
    print("=" * 60)
    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Samples Dir  : {SAMPLES_DIR}")
    print(f"Outputs Dir  : {OUTPUTS_DIR}")
    print("-" * 60)

    # Ensure output directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find sample images
    image_files = sorted([
        f for f in SAMPLES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print("[!] No sample images found in `data/samples/`.")
        print(f"    Please place your sample images into: {SAMPLES_DIR}")
        print("\nSupported extensions: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        return

    print(f"Found {len(image_files)} image(s) total.\n")

    # 1. Generate Contact Sheet
    generate_contact_sheet(image_files, OUTPUTS_DIR)

    # 2. Run Feasibility Spike on the first 5 images (or all of them)
    print("\nRunning feasibility spike operations...")
    for img_path in image_files[:5]:
        process_image(img_path, OUTPUTS_DIR)

    print("\n[✓] Processing complete! Check your `outputs/` folder for the deliverables.")


if __name__ == "__main__":
    main()