# ZumoPi - Hailo Object Detection

This repository contains the final project code for real-time object detection on an embedded robotic platform using a Raspberry Pi camera, classical OpenCV methods, YOLO models, and Hailo AI acceleration.

The project started with classical computer vision detectors and later moved to deep-learning-based detection using YOLO and Hailo inference. The main goal is to detect objects from the robot camera in real time and compare different implementation approaches: OpenCV-based detection, YOLO inference on the Raspberry Pi, and Hailo-accelerated inference.

---

## Project Overview

The system is designed for a ZumoPi / Raspberry Pi based robotic platform with a camera and a Hailo AI accelerator.

Main goals:

- Capture live images from the Raspberry Pi camera.
- Detect project objects such as cones, balls, and barcodes.
- Test classical OpenCV detectors.
- Test YOLO inference before Hailo deployment.
- Run a compiled Hailo `.hef` model on the Raspberry Pi.
- Display live detections and measure runtime behavior such as FPS and detection success.

---

## Repository Structure

```text
finalproj/
│
├── Opencv simulation/
│   ├── ball/
│   ├── barcode big/
│   ├── barcode small/
│   └── cone/
│
├── opencv pi/
│   ├── detect_ball.py
│   ├── detect_barcode.py
│   ├── detect_cones.py
│   ├── detect_small_barcode.py
│   ├── hub_detector.py
│   └── opencv_realtime_rpicam.py
│
├── YOLO/
│   ├── best.pt
│   └── yolo_realtime_test.py
│
└── Hailo/
    └── hailo_env/
        └── bin/
            ├── cone_yolov8_raw_heads_hailo8l.hef
            └── live_hailo_conr_rpicam_pipe.py
```

### Folder explanation

| Folder | Purpose |
|---|---|
| `Opencv simulation/` | Offline / simulation tests for the OpenCV detectors. Each object has its own folder. |
| `opencv pi/` | OpenCV detection code intended to run on the Raspberry Pi using the Pi camera. |
| `YOLO/` | YOLO model and real-time YOLO test script before Hailo deployment. |
| `Hailo/` | Hailo runtime script and compiled `.hef` model for hardware-accelerated inference. |

---

## Hardware Requirements

The project was developed for the following setup:

- Raspberry Pi 5 / ZumoPi platform
- Raspberry Pi camera
- Hailo AI accelerator, for example Hailo-8L / Hailo AI Kit
- Display or SSH connection to the Raspberry Pi
- Python environment on Raspberry Pi OS

---

## Software Requirements

The exact environment can depend on the Raspberry Pi image and Hailo installation, but the main requirements are:

- Python 3
- OpenCV
- NumPy
- Picamera2 / rpicam-apps
- Ultralytics YOLO
- HailoRT
- Hailo Python API / `hailo_platform`

Example Python packages:

```bash
pip install numpy opencv-python ultralytics
```

On Raspberry Pi, OpenCV and camera packages may also be installed through `apt`, depending on the OS image:

```bash
sudo apt update
sudo apt install python3-opencv python3-picamera2 rpicam-apps
```

For Hailo execution, install the official HailoRT package that matches the connected Hailo device and Raspberry Pi image.

---

## First-Time Setup

Clone the repository:

```bash
git clone https://github.com/maayanc5/ZumoPI---Hailo-Object-Detection.git
cd ZumoPI---Hailo-Object-Detection/finalproj
```

Create and activate a Python environment:

```bash
python3 -m venv hailo_env
source hailo_env/bin/activate
```

Install basic dependencies:

```bash
pip install numpy opencv-python ultralytics
```

Check that the Raspberry Pi camera works:

```bash
rpicam-still -o test.jpg
```

Check that the Hailo device is detected:

```bash
hailortcli scan
```

or:

```bash
hailortcli fw-control identify
```

---

## Running the OpenCV Version on Raspberry Pi

Go to the OpenCV Pi folder:

```bash
cd "opencv pi"
```

Run the real-time OpenCV wrapper with one of the detector scripts.

Example for cone detection:

```bash
python3 opencv_realtime_rpicam.py detect_cones.py   --source rpicam   --function find_cones   --expected '{"cone": 1}'   --class-name cone   --show
```

Example for ball detection:

```bash
python3 opencv_realtime_rpicam.py detect_ball.py   --source rpicam   --function find_ball   --expected '{"ball": 1}'   --class-name ball   --show
```

Example for small barcode detection:

```bash
python3 opencv_realtime_rpicam.py detect_small_barcode.py   --source rpicam   --function find_small_barcode   --expected '{"small_barcode": 1}'   --class-name small_barcode   --show
```

The script captures frames using `rpicam-still`, runs the selected detector, displays the result, and saves a CSV file with frame-by-frame detection results.

Stop the live preview by pressing `q` or using `Ctrl+C`.

---

## Running the YOLO Version

Go to the YOLO folder:

```bash
cd YOLO
```

Run the YOLO real-time test:

```bash
python3 yolo_realtime_test.py best.pt   --source picamera   --show   --expected 1   --conf 0.25
```

Optional label override:

```bash
python3 yolo_realtime_test.py best.pt   --source picamera   --show   --expected 1   --labels cone,ball,barcode
```

The YOLO script displays detections, calculates FPS, counts detected objects, and saves results to a CSV file.

---

## Running the Hailo Version

The Hailo version uses a compiled `.hef` file and runs inference through HailoRT.

Go to the Hailo script folder:

```bash
cd Hailo/hailo_env/bin
```

Activate the relevant environment if needed:

```bash
source ~/hailo_env/bin/activate
```

Run the Hailo live script:

```bash
python3 live_hailo_conr_rpicam_pipe.py
```

The script uses the following model path:

```python
HEF_PATH = "/home/pi/cone_yolov8_raw_heads_hailo8l.hef"
```

If the `.hef` file is located elsewhere, either copy it to the expected path:

```bash
cp cone_yolov8_raw_heads_hailo8l.hef /home/pi/cone_yolov8_raw_heads_hailo8l.hef
```

or edit `HEF_PATH` inside the script to point to the real location of the `.hef` file.

---

## Expected Outputs

Depending on the selected pipeline, the program may output:

- Live camera preview
- Bounding boxes around detected objects
- Detection confidence
- FPS measurement
- Detected object count
- CSV result files for later analysis

---

## Notes About the Hailo Model

The Hailo runtime uses a compiled `.hef` file. A `.hef` file is hardware-specific, so it must match the Hailo device architecture used on the Raspberry Pi.

For example:

- A model compiled for Hailo-8L should run on a Hailo-8L device.
- A model compiled for Hailo-8 may not be compatible with Hailo-8L, and vice versa.

If the script fails with a message such as:

```text
HAILO_OPEN_FILE_FAILURE
```

first check that the `.hef` file exists at the path used by the script:

```bash
ls -lh /home/pi/cone_yolov8_raw_heads_hailo8l.hef
```

If the file is missing, search for it:

```bash
find /home/pi -name "*.hef"
```

Then either copy it to the expected path or update `HEF_PATH`.

---

## Troubleshooting

### Camera does not open

Check that the camera works independently:

```bash
rpicam-still -o test.jpg
```

If this fails, the issue is probably camera configuration or hardware connection, not the detection code.

### YOLO cannot import `ultralytics`

Install Ultralytics:

```bash
pip install ultralytics
```

or activate the correct Python environment before running the script.

### Hailo cannot open the `.hef` file

Check that the file exists:

```bash
ls -lh /home/pi/cone_yolov8_raw_heads_hailo8l.hef
```

If it does not exist, find it:

```bash
find /home/pi -name "*.hef"
```

Then either copy it to `/home/pi/` or change the path in the script.

### Hailo device is not detected

Run:

```bash
hailortcli scan
```

If no device is detected, check that the Hailo driver, HailoRT, and hardware connection are properly installed.

---

## Development Flow

The project was developed in three main stages:

1. **Classical OpenCV detection**  
   Object-specific detectors were implemented and tested for cones, balls, and barcodes.

2. **YOLO model testing**  
   A YOLO model was trained and tested with a live Raspberry Pi camera stream before moving to Hailo.

3. **Hailo deployment**  
   The YOLO model was converted to a Hailo `.hef` model and executed on the Raspberry Pi using HailoRT for hardware-accelerated inference.

---

## Project Status

This repository contains the project code and model files used for real-time object detection experiments. The current code includes:

- OpenCV object detectors
- Raspberry Pi camera test scripts
- YOLO real-time testing
- Hailo inference script
- Hailo `.hef` model file

The project is intended as an experimental embedded AI pipeline and may require environment-specific adjustments, especially for camera configuration and HailoRT installation.

---

## Authors

Final project by:

- Maayan Cohen
- Dor Itzhaky

Electrical Engineering Final Project  
Tel Aviv University
