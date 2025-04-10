import os
import uuid
import math
import re
import cv2
import easyocr
import threading
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

model = YOLO("weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# Helper functions
def calculate_midpoint(bbox):
    return (bbox['x'], bbox['y']) if bbox else None

def calculate_angle(base_point, point1, point2):
    dx1, dy1 = point1[0] - base_point[0], point1[1] - base_point[1]
    dx2, dy2 = point2[0] - base_point[0], point2[1] - base_point[1]
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    angle_diff = angle2 - angle1
    if angle_diff < 0:
        angle_diff += 2 * math.pi
    return math.degrees(angle_diff) % 360

def extract_points_and_boxes(results):
    base = minimum = maximum = tip = None
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        x, y, w, h = box.xywh[0].tolist()
        prediction = {"x": x, "y": y, "width": w, "height": h}
        if label == "base": base = prediction
        elif label == "minimum": minimum = prediction
        elif label == "maximum": maximum = prediction
        elif label == "tip": tip = prediction
    return base, minimum, maximum, tip

def draw_output_image(results, base_pt, min_pt, max_pt, tip_pt, reading, input_path, output_path):
    result_img = cv2.imread(input_path)
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        x, y, w, h = box.xywh[0].tolist()
        center = (int(x), int(y))
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))

        if label in ['minimum', 'maximum']:
            cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
        elif label in ['tip', 'base']:
            cv2.circle(result_img, center, 6, (0, 0, 255), -1)

    cv2.imwrite(output_path, result_img)

def get_ocr_values(image_path, minimum, maximum):
    result_img = cv2.imread(image_path)
    ocr_values = {}
    for label, bbox in [("minimum", minimum), ("maximum", maximum)]:
        if bbox:
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            padding = 15
            x1 = max(0, int(x - w / 2) - padding)
            y1 = max(0, int(y - h / 2) - padding)
            x2 = min(result_img.shape[1], int(x + w / 2) + padding)
            y2 = min(result_img.shape[0], int(y + h / 2) + padding)
            crop = result_img[y1:y2, x1:x2]
            result = reader.readtext(crop, detail=0)
            cleaned = []
            for r in result:
                r_clean = r.replace(",", ".").strip()
                try:
                    val = float(r_clean)
                    cleaned.append(val)
                except ValueError:
                    continue
            if cleaned:
                ocr_values[label] = cleaned[0]
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return ocr_values

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    image_file.save(input_path)

    results = model.predict(source=input_path, save=False, conf=0.5)[0]
    base, minimum, maximum, tip = extract_points_and_boxes(results)

    base_pt = calculate_midpoint(base)
    min_pt = calculate_midpoint(minimum)
    max_pt = calculate_midpoint(maximum)
    tip_pt = calculate_midpoint(tip)

    ocr_values = get_ocr_values(input_path, minimum, maximum)
    min_val = ocr_values.get("minimum")
    max_val = ocr_values.get("maximum")

    reading = "N/A"
    error = None

    if all([base_pt, min_pt, max_pt, tip_pt, min_val is not None, max_val is not None]):
        scale_angle = calculate_angle(base_pt, min_pt, max_pt)
        needle_angle = calculate_angle(base_pt, min_pt, tip_pt)
        gauge_value = min_val + (needle_angle / scale_angle) * (max_val - min_val)
        reading = f"{gauge_value:.2f}"
    else:
        error = "OCR could not detect the minimum or maximum value. Please check the image."

    draw_output_image(results, base_pt, min_pt, max_pt, tip_pt, reading, input_path, output_path)

    result_data = {
        'reading': reading,
        'filename': f"outputs/{filename}",
        'min_val': min_val,
        'max_val': max_val,
        'error': error
    }

    def delete_files():
        try:
            os.remove(input_path)
            os.remove(output_path)
            print(f"Deleted: {input_path}, {output_path}")
        except Exception as e:
            print(f"File cleanup error: {e}")

    threading.Timer(3600.0, delete_files).start()
    return render_template('index.html', result=result_data)

@app.route('/recalculate', methods=['POST'])
def recalculate():
    filename = request.form.get("filename").replace("outputs/", "")    
    min_val = float(request.form.get("min_value"))
    max_val = float(request.form.get("max_value"))

    # Use the original input image from uploads folder for reprocessing
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)

    results = model.predict(source=input_path, save=False, conf=0.5)[0]
    base, minimum, maximum, tip = extract_points_and_boxes(results)

    base_pt = calculate_midpoint(base)
    min_pt = calculate_midpoint(minimum)
    max_pt = calculate_midpoint(maximum)
    tip_pt = calculate_midpoint(tip)

    reading = "N/A"
    if all([base_pt, min_pt, max_pt, tip_pt]):
        scale_angle = calculate_angle(base_pt, min_pt, max_pt)
        needle_angle = calculate_angle(base_pt, min_pt, tip_pt)
        gauge_value = min_val + (needle_angle / scale_angle) * (max_val - min_val)
        reading = f"{gauge_value:.2f}"

    draw_output_image(results, base_pt, min_pt, max_pt, tip_pt, reading, input_path, output_path)

    return render_template('index.html', result={
        'reading': reading,
        'filename': f"outputs/{filename}",
        'min_val': min_val,
        'max_val': max_val,
        'error': None
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)