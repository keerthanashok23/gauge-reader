# Gauge Reader AI

This project is a complete pipeline to **automatically read analog gauge values** using YOLOv8 object detection and EasyOCR, with both a **command-line interface** and a **web application**.

---

## 🧠 What It Does

It detects four key elements on a gauge dial:
- `minimum`: where the gauge starts
- `maximum`: where the gauge ends
- `base`: the fixed center of the dial
- `tip`: the needle pointer tip

From these detections, it computes the **needle angle** and maps it to a **reading value** between the min and max.

---

## 🚀 Project Structure

```bash
gauge_reader_app/
│
├── weights/                # Trained YOLOv8 model (.pt file)
├── templates/              # HTML template for the web app
│   └── index.html
├── static/outputs/         # Web app output images
├── uploads/                # Temporary upload folder
│
├── app.py                  # Flask web server
├── testGaugeRead.py        # CLI interface for image reading
├── requirements.txt        # Required libraries
└── README.md               # This file
```

---

## 🔬 Model Training Summary

- **Model used**: YOLOv8
- **Framework**: Ultralytics YOLO
- **Data**: Custom labeled dataset with classes: base, minimum, maximum, tip
- **Trained on**: 17,000+ images with augmentation
- **OCR**: EasyOCR to extract min/max values from gauge image

---

## 💻 Web Application

Start the app using:

```bash
python app.py
```

### Features
- Upload gauge image
- OCR automatically detects `min` and `max` values
- If OCR fails, user can input values manually
- Displays bounding boxes (green) for min/max and dots (red) for base/tip
- Result image and value shown on the page
- Deletes files after 1 hour

### Fallback Support
If OCR fails to detect min/max, the app shows a form to enter them manually and recalculate the reading.

---

## ⚙️ Command Line Interface (CLI)

For script-only usage (no browser), run:

```bash
python testGaugeRead.py <path_to_image>
```

If OCR fails, it will **ask you to enter min/max** via terminal input.

### Example:

```bash
python testGaugeRead.py gauge4.jpg
```

### Output:

- Prints:
  - Min & Max (from OCR or user)
  - Calculated reading
- Shows image with drawn output (bounding boxes and dots)

---

## 📦 Requirements

Install requirements:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `ultralytics`
- `easyocr`
- `opencv-python`
- `flask`
- `matplotlib`

---

## 📸 Sample Outputs

- Green box = min/max
- Red dot = base/tip
- OCR box = blue
- Output image shown in browser or plotted in CLI

---

## 🧾 Author & Info

- Date: April 10, 2025
- Author: AI Assistant + User Collaboration
- Status: Production-ready with both GUI and CLI

---
