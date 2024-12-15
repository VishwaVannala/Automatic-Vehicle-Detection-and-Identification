# Vehicle Detection and Counting with YOLOv8

This project uses the YOLOv8 object detection model to detect, track, and count vehicles in a video. It classifies vehicles (Car, Motorcycle, Bus, Truck) and tracks their movement to determine incoming and outgoing counts.

## Features
- Detect vehicles in a video using YOLOv8.
- Track vehicle movements across user-defined lines.
- Count incoming and outgoing vehicles for each category.
- Save a processed video with bounding boxes and vehicle counts.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv8 weights file:
    - [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)

## Usage

1. Place your input video in the appropriate directory.
2. Update the file paths in the script:
    - `video_path`: Path to your input video.
    - `output_path`: Path to save the processed output video.

3. Run the script:
    ```bash
    python vehicle_detection.py
    ```

4. View the results:
    - The processed video will be saved at the specified `output_path`.
    - Final incoming and outgoing vehicle counts will be printed to the console.

## Input and Output Videos

Sample input and output videos are available via the following Google Drive links:

- **Input Video**: [Download Input Video](https://drive.google.com/file/d/1_BaYK87G1KTYnzTxsFQpeoFkBecZvnxS/view?usp=sharing)
- **Output Video**: [Download Output Video](https://drive.google.com/file/d/1f9khINNcWw33eX_Zp0Y3-Bysqkzdt2ua/view?usp=sharing)

## Output

- **Processed Video**: Includes bounding boxes, labels, and counts displayed on each frame.
- **Console Output**: Total incoming and outgoing vehicle counts for each class.

## File Descriptions

- `vehicle_detection.py`: Main Python script for vehicle detection and counting.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: Documentation for the project.

## Dependencies

- Python  
- OpenCV  
- Ultralytics YOLOv8  
- Numpy  
