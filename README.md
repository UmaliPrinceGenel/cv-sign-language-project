# Computer Vision Project: Sign Language Recognition

Milestone 2 (50% Progress Check) submission for CSST 106, featuring an end-to-end Support Vector Machine (SVM) baseline prototype for sign language recognition.

> **Note:** Please see [`docs/progress_report.md`](docs/progress_report.md) for the completion status and milestone details.

---

## 📁 Project Structure

```text
cv-sign-language-project/
├── README.md                      # Project documentation and setup instructions
├── requirements.txt               # Python dependencies
├── data/
│   └── test/                      # Sample test images (e.g., peace_01.jpg)
├── docs/
│   ├── datacard.md                # Dataset documentation and summary
│   └── progress_report.md         # Milestone 2 progress report and completion status
├── models/                        # Pre-trained models (download .pkl weights from Drive)
│   ├── scaler.pkl                 # Feature scaler
│   └── svm_baseline_model.pkl     # Trained SVM baseline model
├── notebooks/
│   └── CV_project_sign.ipynb      # Exploration, feature extraction & training notebook
├── outputs/
│   └── predictions/               # Inference outputs with visual prediction banners
└── src/
    ├── baseline.py                # Feature extraction & pipeline utility functions
    └── predict.py                 # Prototype inference script
```

---

## 🚀 Getting Started

### 1. Set Up Python Environment

It is recommended to use a virtual environment:

```bash
# Navigate to the project folder
cd cv-sign-language-project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (cmd):
.\venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

> **Note:** This project requires `scikit-learn==1.6.1` and `opencv-python`.

### 3. Download Assets

Download the dataset and pre-trained model weights (`svm_baseline_model.pkl` and `scaler.pkl`) from Google Drive:

- **Dataset**: [https://drive.google.com/file/d/1ysFX8YRMNX6_WK4RgmRdRlYx4nTRBE5P/view?usp=sharing]
  - Place your test images inside the `data/test/` directory.
- **Model Weights**: [https://drive.google.com/drive/folders/1y-HSH_zvyWyW8_Z57QIqf8QoLx51tEPz?usp=sharing]
  - Download `svm_baseline_model.pkl` and `scaler.pkl` and place both files into the `models/` directory.

### 4. Run the Prototype

Run the inference prototype using `src/predict.py`:

```bash
python src/predict.py -input data/test/peace_01.jpg -output outputs/predictions/
```

This runs the end-to-end recognition pipeline on the specified test image and saves the predicted image with a high-contrast prediction banner to the `outputs/predictions/` folder.
