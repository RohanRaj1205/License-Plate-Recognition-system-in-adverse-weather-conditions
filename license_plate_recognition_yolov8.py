import os
import cv2
import numpy as np
from pathlib import Path
import easyocr
from ultralytics import YOLO
import torch

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Load YOLOv8 model (pre-trained on COCO)
model = YOLO('yolov8n.pt')  # Nano model for speed; use 'yolov8s.pt' for better accuracy

# Create directories for saving outputs
project_dir = Path("/content")  # Colab working directory
output_dir = project_dir / "output_plates"
output_dir.mkdir(exist_ok=True)
txt_output = project_dir / "license_plate_numbers.txt"

def process_image(image_path, output_dir, txt_output):
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error reading image: {image_path}")
        return

    # Perform YOLOv8 detection
    results = model(img, conf=0.5)  # Confidence threshold
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # Class IDs

    # Initialize list to store license plate numbers
    plate_numbers = []

    for box, score, cls in zip(boxes, scores, classes):
        # Using "car" (COCO class ID 2) as proxy; replace with license plate class if fine-tuned
        if int(cls) == 2:  # YOLOv8 COCO class 2 is "car"
            x1, y1, x2, y2 = map(int, box)
            plate_img = img[y1:y2, x1:x2]

            # Perform OCR
            ocr_result = reader.readtext(plate_img, detail=0)
            plate_text = "".join(ocr_result).strip()

            if plate_text:
                plate_numbers.append(plate_text)
                print(f"🚗 Extracted Plate from {image_path.name}: {plate_text}")

                # Draw bounding box and text on image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    plate_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                # Save cropped plate
                cropped_path = output_dir / f"cropped_{image_path.name}"
                cv2.imwrite(str(cropped_path), plate_img)
                print(f"✅ Cropped plate saved: {cropped_path}")

    # Save the annotated image
    output_image_path = output_dir / f"annotated_{image_path.name}"
    cv2.imwrite(str(output_image_path), img)

    # Write to text file
    with open(txt_output, "a", encoding="utf-8") as f:
        for plate in plate_numbers:
            f.write(f"{image_path.name}: {plate}\n")

def main():
    # Clear previous text file if exists
    if txt_output.exists():
        txt_output.unlink()

    # Process test images (adverse weather conditions)
    test_dir = Path("/content/online dataset 2/test/images")
    if not test_dir.exists():
        print(f"Test directory {test_dir} not found. Please unzip 'online dataset 2.zip' to '/content/online dataset 2' and ensure 'test/images' contains .jpg files.")
        return
    for img_path in test_dir.glob("*.jpg"):
        process_image(img_path, output_dir, txt_output)

    print(f"✅ License plate numbers saved to: {txt_output}")

if __name__ == "__main__":
    main()