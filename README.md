# 🏥 SecureScript: Intelligent Medical OCR & PII Redaction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Handwriting%20Recognition-yellow?style=for-the-badge)
![Privacy](https://img.shields.io/badge/Privacy-Microsoft%20Presidio-red?style=for-the-badge)

**SecureScript** is a robust, privacy-focused pipeline designed to digitize and anonymize handwritten medical documents. It traverses the full journey from raw, noisy images to clean, redacted digital records, ensuring patient privacy is protected.

---

## 🚀 Key Features

- **📷 Intelligent Pre-processing**: Automatically handles tilted images, uneven lighting, and noise using adaptive thresholding and denoising.
- **✍️ Handwriting OCR**: Powered by **EasyOCR**, capable of deciphering complex doctor handwriting and form data.
- **🛡️ PII Detection**: Utilizes **Microsoft Presidio** (backed by SpaCy NLP) to intelligently identify sensitive entities like:
  - Patient Names
  - Dates
  - Phone Numbers
  - Email Addresses
- **👁️ Visual Redaction**: Not just text! Uses bounding box mapping to partially obscure the original image where PII is detected, generating a "blacked-out" document.

---

## 🛠️ Installation

### 1. Clone & Setup
Ensure you have Python 3.8+ installed.

```bash
# Install required python packages
pip install -r requirements.txt
```

### 2. Download NLP Models
This project requires a large English language model for accurate entity recognition.
```bash
python -m spacy download en_core_web_lg
```

---

## 💻 Usage

### ⚡ Automatic Pipeline
Place your target images (`.jpg`) in the project root directory and run:

```bash
python pipeline.py
```
**What happens?**
- The script finds all `*.jpg` images.
- Processes them one by one.
- Saves 3 output files in the `output/` folder:
  1. `redacted_[name].jpg`: The image with visual black boxes.
  2. `comparison_[name].png`: A side-by-side view of Before vs. After.
  3. Console logs showing extracted text and detected PII.

### 📓 Interactive Notebook
For research, debugging, or step-by-step visualization:

```bash
jupyter notebook notebook.ipynb
```

---

## 📂 Project Structure

| File/Folder | Description |
|:---|:---|
| `pipeline.py` | 🐍 Main execution script for batch processing. |
| `notebook.ipynb` | 📓 Interactive Jupyter experiment notebook. |
| `requirements.txt` | 📦 List of python dependencies. |
| `dependencies.md` | 📄 Detailed explanation of libraries used. |
| `output/` | 📂 Generated redacted images and comparisons. |

---

## 📊 Sample Result

> *The system takes a raw form, identifies "Santosh Pradhan" as a PERSON, and automatically places a black box over that specific region in the image.*

---

## 🔧 Troubleshooting

- **"Module not found"**: Ensure you ran `pip install -r requirements.txt`.
- **"Can't find model"**: Ensure you ran the `spacy download` command.
- **OCR Accuracy**: Use high-resolution images for best handwriting recognition results.
