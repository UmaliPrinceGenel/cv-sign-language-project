# Computer Vision Project (cv-project)

A structured Computer Vision project setup for image processing, experiments, and proposal delivery.

## 📁 Project Structure

```text
cv-project/
├── README.md                # Project documentation and setup instructions
├── requirements.txt         # Python dependencies (OpenCV, Matplotlib, NumPy)
├── data/
│   └── samples/             # Place your 20 sample images here (.jpg, .png, etc.)
├── src/
│   └── spike.py             # Main Python processing script
├── outputs/                 # Processed images and visualizations will be saved here
└── docs/
    └── proposal.pdf         # Project proposal document
```

---

## 🚀 Getting Started

### 1. Set Up Python Environment

It is recommended to use a virtual environment:

```bash
# Navigate to the project folder
cd cv-project

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

```bash
pip install -r requirements.txt
```

### 3. Add Sample Images

Place your 20 test/sample images inside the `data/samples/` directory:
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### 4. Run the Script

Execute the spike pipeline:

```bash
python src/spike.py
```

Processed results and comparative plots will be saved automatically into the `outputs/` directory.
