#!/usr/bin/env python3
"""
Simple realtime YOLO test script for Raspberry Pi.

Goal:
    Test the original YOLO model before debugging Hailo.

This script gives:
    - Live camera preview
    - YOLO detections
    - Instant FPS
    - Average FPS
    - Detected / expected count
    - Success percent
    - CSV results

Supported sources:
    --source picamera
    --source 0
    --source 1
    --source video.mp4

Examples:
    python3 yolo_realtime_test.py best.pt --source picamera --show --expected 1

    python3 yolo_realtime_test.py best.pt --source picamera --show --expected 1 --conf 0.25

    python3 yolo_realtime_test.py best.pt --source picamera --show --expected 1 --labels barcode,cone,ball

Stop:
    Press q in the preview window.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:
    print("ERROR: Could not import ultralytics.")
    print("Try:")
    print("    source ~/hailo_env/bin/activate")
    print("or:")
    print("    pip install ultralytics")
    raise exc


class OpenCVCamera:
    """
    OpenCV camera or video-file reader.
    """

    def __init__(self, source, width, height):
        self.source = int(source) if str(source).isdigit() else source
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open source: {self.source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        return self

    def read(self):
        ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None

        frame = cv2.resize(frame, (self.width, self.height))
        return True, frame

    def close(self):
        if self.cap is not None:
            self.cap.release()


class PiCamera2Reader:
    """
    Picamera2 reader for Raspberry Pi Camera Module.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.picam2 = None

    def open(self):
        try:
            from picamera2 import Picamera2
        except Exception as exc:
            raise RuntimeError(
                "Picamera2 is not installed or cannot be imported."
            ) from exc

        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )

        self.picam2.configure(config)
        self.picam2.start()

        time.sleep(1.0)

        return self

    def read(self):
        frame_rgb = self.picam2.capture_array()

        if frame_rgb is None:
            return False, None

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def close(self):
        if self.picam2 is not None:
            self.picam2.stop()


def open_frame_source(source, width, height):
    """
    Open selected camera or video source.
    """
    if str(source).lower() in ["picamera", "picam", "libcamera", "rpi"]:
        print("Opening camera with Picamera2...")
        return PiCamera2Reader(width, height).open()

    print(f"Opening source with OpenCV: {source}")
    return OpenCVCamera(source, width, height).open()


def create_csv(csv_path):
    """
    Create CSV output file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    writer.writerow([
        "timestamp",
        "frame",
        "processing_time_ms",
        "instant_fps",
        "avg_fps",
        "detected_count",
        "expected_count",
        "success_percent",
        "avg_success_percent",
        "class_counts"
    ])

    return csv_file, writer


def draw_metrics(frame, fps, avg_fps, detected, expected, success, avg_success):
    """
    Draw metrics on frame.
    """
    lines = [
        f"FPS: {fps:.2f} | Avg FPS: {avg_fps:.2f}",
        f"Detected: {detected}/{expected}",
        f"Success: {success:.1f}% | Avg: {avg_success:.1f}%"
    ]

    y = 30

    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 30


def draw_yolo_boxes(frame, result, labels_override=None):
    """
    Draw YOLO boxes manually.
    """
    class_counts = {}

    if result.boxes is None:
        return frame, 0, class_counts

    names = result.names

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        if labels_override and cls_id < len(labels_override):
            class_name = labels_override[cls_id]
        else:
            class_name = names.get(cls_id, f"class_{cls_id}")

        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        x1, y1, x2, y2 = xyxy

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{class_name} {conf:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    detected_count = len(result.boxes)

    return frame, detected_count, class_counts


def main():
    parser = argparse.ArgumentParser(description="Realtime YOLO test before Hailo debugging")

    parser.add_argument("model", help="YOLO model path, for example best.pt")
    parser.add_argument("--source", default="picamera", help="picamera, 0, 1, or video path")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--expected", type=int, default=1, help="Expected object count")
    parser.add_argument("--labels", default="", help="Optional comma-separated label override")
    parser.add_argument("--csv", default="results/yolo_realtime.csv", help="CSV output path")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames")
    parser.add_argument("--save-video", default="", help="Optional annotated output video path")

    args = parser.parse_args()

    labels_override = [x.strip() for x in args.labels.split(",") if x.strip()]

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    print("Model class names:")
    print(model.names)

    reader = open_frame_source(args.source, args.width, args.height)

    csv_file, writer = create_csv(args.csv)

    video_writer = None

    if args.save_video:
        save_path = Path(args.save_video)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(save_path),
            fourcc,
            20.0,
            (args.width, args.height)
        )

    frame_count = 0
    success_sum = 0.0
    start_time = time.perf_counter()

    print("Running YOLO realtime test.")
    print("Press q to stop.")

    try:
        while True:
            ret, frame = reader.read()

            if not ret or frame is None:
                print("No frame received. Stopping.")
                break

            frame_count += 1

            t0 = time.perf_counter()

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )

            result = results[0]

            display = frame.copy()
            display, detected_count, class_counts = draw_yolo_boxes(
                display,
                result,
                labels_override=labels_override
            )

            t1 = time.perf_counter()

            processing_time = t1 - t0
            processing_time_ms = processing_time * 1000.0
            fps = 1.0 / processing_time if processing_time > 0 else 0.0

            elapsed = t1 - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0.0

            expected_count = max(args.expected, 1)
            success = min(detected_count / expected_count, 1.0) * 100.0

            success_sum += success
            avg_success = success_sum / frame_count

            draw_metrics(
                display,
                fps,
                avg_fps,
                detected_count,
                expected_count,
                success,
                avg_success
            )

            writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                frame_count,
                round(processing_time_ms, 3),
                round(fps, 3),
                round(avg_fps, 3),
                detected_count,
                expected_count,
                round(success, 3),
                round(avg_success, 3),
                str(class_counts)
            ])
            csv_file.flush()

            print(
                f"Frame {frame_count} | "
                f"FPS {fps:.2f} | "
                f"Avg FPS {avg_fps:.2f} | "
                f"Detected {detected_count}/{expected_count} | "
                f"Success {success:.1f}% | "
                f"Avg Success {avg_success:.1f}% | "
                f"Classes {class_counts}"
            )

            if video_writer is not None:
                video_writer.write(display)

            if args.show:
                cv2.imshow("YOLO Realtime Test", display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    finally:
        reader.close()
        csv_file.close()

        if video_writer is not None:
            video_writer.release()

        if args.show:
            cv2.destroyAllWindows()

    print("Finished.")
    print(f"CSV saved to: {args.csv}")

    if args.save_video:
        print(f"Video saved to: {args.save_video}")


if __name__ == "__main__":
    main()
