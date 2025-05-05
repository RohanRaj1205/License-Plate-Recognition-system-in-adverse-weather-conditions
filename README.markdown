# License Plate Recognition with YOLOv8 and EasyOCR

## Overview
This project implements a license plate recognition system using **YOLOv8** for detecting cars and **EasyOCR** for extracting license plate text from images. It processes a dataset of images (e.g., `online dataset 2`) captured in adverse weather conditions, producing:
- **Annotated images** with bounding boxes and license plate text.
- **Cropped license plate images**.
- A **text file** (`license_plate_numbers.txt`) listing extracted plate numbers.

The project is designed to run in **Google Colab** with GPU acceleration, using **direct file uploads** (no Google Drive) for the dataset and script. It’s suitable for computer vision tasks, autonomous vehicle research, or learning about object detection and OCR.

## Prerequisites
- **Google Colab** account (free tier with GPU support).
- **Dataset**: A ZIP file (e.g., `online dataset 2.zip`) containing images in `test/images/` (e.g., `.jpg` files like `fog_effect12_jpg.rf...`).
- **Script**: `license_plate_recognition_yolov8.py` (provided in this repository).
- Basic familiarity with Python and Colab.

## Setup and Installation
Follow these steps in a new Google Colab notebook to set up and run the project.

1. **Open Google Colab**:
   - Go to [colab.research.google.com](https://colab.research.google.com/) and create a new notebook.

2. **Enable GPU**:
   - Go to **Runtime > Change runtime type**.
   - Select **GPU** under Hardware accelerator and click **Save**.

3. **Check CUDA and Python Version**:
   ```python
   !nvidia-smi
   !python --version
   ```
   - Note the CUDA version (e.g., 11.8 or 12.1) for Step 7.

4. **Upload the Dataset**:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
   - Select your dataset ZIP file (e.g., `online dataset 2.zip`) and upload.

5. **Unzip the Dataset**:
   ```python
   !unzip "online dataset 2.zip" -d "/content/online dataset 2"
   ```

6. **Verify Dataset**:
   ```python
   !ls "/content/online dataset 2/test/images"
   ```
   - Ensure `.jpg` files are listed (e.g., `fog_effect12_jpg.rf...`).

7. **Install Dependencies**:
   ```python
   # Uninstall conflicting packages
   !pip uninstall -y numpy torch torchvision torchaudio ultralytics easyocr opencv-python opencv-python-headless
   # Install compatible dependencies
   !pip install numpy==1.26.4 torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
   !pip install ultralytics==8.3.124 easyocr==1.7.2 opencv-python-headless==4.11.0.86 --force-reinstall
   ```
   - If CUDA 12.1 was shown in Step 3, use:
     ```python
     !pip uninstall -y numpy torch torchvision torchaudio ultralytics easyocr opencv-python opencv-python-headless
     !pip install numpy==1.26.4 torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
     !pip install ultralytics==8.3.124 easyocr==1.7.2 opencv-python-headless==4.11.0.86 --force-reinstall
     ```

8. **Verify Dependencies**:
   ```python
   import torch
   import torchvision
   import numpy as np
   from ultralytics import YOLO
   import easyocr
   import cv2
   print(np.__version__, torch.__version__, torchvision.__version__, cv2.__version__)
   ```
   - Expect: `1.26.4` (NumPy), `2.1.0` (PyTorch), `0.16.0` (torchvision), `4.11.0.86` (OpenCV).

9. **Upload and Save the Script**:
   - Download `license_plate_recognition_yolov8.py` from this repository.
   - Run:
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```
     - Upload `license_plate_recognition_yolov8.py`.
   - Alternatively, copy the script content into Colab:
     ```python
     %%writefile license_plate_recognition_yolov8.py
     # Paste the script content here (from this repository)
     ```

10. **Run the Script**:
    ```python
    !python license_plate_recognition_yolov8.py
    ```

11. **Download Outputs**:
    ```python
    !zip -r output_plates.zip /content/output_plates
    from google.colab import files
    files.download('output_plates.zip')
    files.download('/content/license_plate_numbers.txt')
    ```

## Outputs
- **Annotated Images**: `/content/output_plates/annotated_<image_name>.jpg` (images with bounding boxes and plate text).
- **Cropped Plates**: `/content/output_plates/cropped_<image_name>.jpg` (cropped license plate regions).
- **Text File**: `/content/license_plate_numbers.txt` (e.g., `fog_effect12_jpg.rf...: ABC123`).

## Notes
- **Car Detection**: The script uses YOLOv8’s pre-trained COCO model, detecting "car" (class ID 2) as a proxy. For better accuracy, fine-tune YOLOv8 on your dataset (e.g., `online dataset 2`) to detect license plates directly.
- **Dataset**: Ensure `online dataset 2.zip` unzips to `/content/online dataset 2/test/images` with `.jpg` files.
- **Video Processing**: Not included (no video provided). Contact for video support.
- **Colab Limits**: Free Colab may disconnect during long runs. Reconnect and re-upload files if needed.

## Troubleshooting
- **Dependency Errors**:
  - If imports fail (e.g., `AttributeError` in `torchvision`):
    ```python
    !pip uninstall -y numpy torch torchvision torchaudio ultralytics easyocr opencv-python opencv-python-headless
    !pip install numpy==1.26.4 torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
    !pip install ultralytics==8.3.124 easyocr==1.7.2 opencv-python-headless==4.11.0.86 --force-reinstall
    ```
    - Rerun Step 8.
  - Check versions:
    ```python
    !pip show torch torchvision numpy ultralytics easyocr opencv-python-headless
    ```
- **Dataset Not Found**:
  - If `Test directory /content/online dataset 2/test/images not found`:
    ```python
    !ls "/content/online dataset 2"
    ```
    - Re-upload and unzip:
      ```python
      from google.colab import files
      uploaded = files.upload()
      !unzip "online dataset 2.zip" -d "/content/online dataset 2"
      ```
- **No Images Processed**:
  - Verify `.jpg` files:
    ```python
    !ls "/content/online dataset 2/test/images"
    ```
  - Check dataset structure:
    ```python
    !ls -R "/content/online dataset 2"
    ```
- **Runtime Errors**:
  - Share the full error from `!python license_plate_recognition_yolov8.py`.

## Contributing
Feel free to fork this repository, submit issues, or contribute improvements (e.g., fine-tuning YOLOv8, video processing).

## License
This project is licensed under the MIT License.

## Contact
For questions or support, open an issue on GitHub.