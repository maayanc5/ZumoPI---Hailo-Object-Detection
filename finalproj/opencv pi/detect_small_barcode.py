"""
Real-time OpenCV wrapper for small_barcode detection.

This file adapts the uploaded hub_detector.py logic for a generic real-time runner
such as:
python3 opencv_realtime_picam.py detect_small_barcode.py --source picam --expected '{"small_barcode": 1}' --class-name small_barcode --show
"""

import cv2
from hub_detector import detect_hubs


CLASS_NAME = "small_barcode"


def _hub_to_detection(hub, class_name=CLASS_NAME):
    cx, cy = hub["center"]
    r = hub["radius"]

    x1 = int(max(0, cx - r))
    y1 = int(max(0, cy - r))
    x2 = int(cx + r)
    y2 = int(cy + r)

    return {
        "class": class_name,
        "class_name": class_name,
        "label": class_name,
        "confidence": 1.0,
        "conf": 1.0,
        "bbox": [x1, y1, x2, y2],
        "box": [x1, y1, x2, y2],
        "center": [int(cx), int(cy)],
        "radius": int(r),
    }


def detect(frame_bgr, class_name=CLASS_NAME, **kwargs):
    """
    Input:
        frame_bgr: OpenCV camera frame in BGR format.

    Output:
        list of detections. Each detection includes class/label, bbox and confidence.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    hubs = detect_hubs(
        img_rgb,
        DEBUG=False,
        circle_diameter=100,          # uploaded barcodeDetector.py used 100
        expected_dark_fraction=0.85,
        min_blob_area=10,
    )

    return [_hub_to_detection(h, class_name=class_name) for h in hubs]


# Aliases, in case your runner searches for a more specific function name.
def detect_small_barcode(frame_bgr, **kwargs):
    return detect(frame_bgr, class_name=CLASS_NAME, **kwargs)


def detect_barcode(frame_bgr, **kwargs):
    return detect(frame_bgr, class_name=CLASS_NAME, **kwargs)
