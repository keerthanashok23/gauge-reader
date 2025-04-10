import sys
import math
import re
import cv2
import easyocr
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load model and OCR reader
model = YOLO("weights/best.pt")  # Adjust path if needed
reader = easyocr.Reader(['en'], gpu=False)

# === Helper Functions ===
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

def extract_boxes(results):
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

def run_ocr_on_region(img, bbox):
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    padding = 15
    x1 = max(0, int(x - w / 2) - padding)
    y1 = max(0, int(y - h / 2) - padding)
    x2 = min(img.shape[1], int(x + w / 2) + padding)
    y2 = min(img.shape[0], int(y + h / 2) + padding)
    crop = img[y1:y2, x1:x2]
    result = reader.readtext(crop, detail=0)
    for r in result:
        r_clean = r.replace(",", ".").strip()
        try:
            return float(r_clean)
        except ValueError:
            continue
    return None

# === Main Function ===
def main(image_path):
    results = model.predict(source=image_path, save=False, conf=0.5)[0]
    base, minimum, maximum, tip = extract_boxes(results)

    base_pt = calculate_midpoint(base)
    min_pt = calculate_midpoint(minimum)
    max_pt = calculate_midpoint(maximum)
    tip_pt = calculate_midpoint(tip)

    img = cv2.imread(image_path)

    # Try OCR first
    min_val = run_ocr_on_region(img, minimum) if minimum else None
    max_val = run_ocr_on_region(img, maximum) if maximum else None

    # Fallback to user input if OCR fails
    if min_val is None:
        try:
            min_val = float(input("OCR failed. Please enter the MIN value: "))
        except ValueError:
            min_val = None

    if max_val is None:
        try:
            max_val = float(input("OCR failed. Please enter the MAX value: "))
        except ValueError:
            max_val = None

    # Compute reading
    reading = "N/A"
    if all([base_pt, min_pt, max_pt, tip_pt, min_val is not None, max_val is not None]):
        scale_angle = calculate_angle(base_pt, min_pt, max_pt)
        needle_angle = calculate_angle(base_pt, min_pt, tip_pt)
        gauge_value = min_val + (needle_angle / scale_angle) * (max_val - min_val)
        reading = f"{gauge_value:.2f}"

    # Draw results
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        x, y, w, h = box.xywh[0].tolist()
        center = (int(x), int(y))
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        if label in ['minimum', 'maximum']:
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        elif label in ['tip', 'base']:
            cv2.circle(img, center, 6, (0, 0, 255), -1)

    # Print to console
    print("\n====== RESULTS ======")
    print(f"Minimum Value: {min_val}")
    print(f"Maximum Value: {max_val}")
    print(f"Gauge Reading: {reading}")

    # Show output
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Gauge Reading: {reading}")
    plt.axis("off")
    plt.show()

# CLI entry
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gauge_cli.py <image_path>")
    else:
        main(sys.argv[1])
