#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2


def load_python_module(script_path):
    script_path = Path(script_path).resolve()
    spec = importlib.util.spec_from_file_location(script_path.stem, str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_function(module, function_name):
    if function_name:
        if not hasattr(module, function_name):
            raise RuntimeError(f"Function '{function_name}' was not found in the detector file.")
        return getattr(module, function_name), function_name

    # Auto-detect a function with exactly two required positional args.
    import inspect
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except Exception:
            continue
        params = list(sig.parameters.values())
        positional = [
            p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
        if len(positional) == 2:
            return obj, name

    raise RuntimeError("No detection function with two arguments was found. Use --function.")


def capture_with_rpicam(output_path, width, height, timeout_ms):
    cmd = [
        "rpicam-still",
        "-n",
        "-t", str(timeout_ms),
        "--width", str(width),
        "--height", str(height),
        "-o", str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("rpicam-still failed:\n" + result.stderr)


def parse_expected(expected_str):
    if not expected_str:
        return {}
    return json.loads(expected_str)


def pass_fail(counts, expected):
    for key, val in expected.items():
        if counts.get(key, 0) != val:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Realtime-ish OpenCV runner using rpicam-still.")
    parser.add_argument("opencv_script", help="Detector file, e.g. detect_ball.py")
    parser.add_argument("--source", default="rpicam", help="Only rpicam is supported in this runner.")
    parser.add_argument("--function", default=None, help="Detector function name, e.g. find_ball")
    parser.add_argument("--expected", default="{}", help='Expected counts JSON, e.g. \'{"ball": 1}\'')
    parser.add_argument("--class-name", default="object", help="Class name for display/log")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--timeout-ms", type=int, default=80)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = run until q/Ctrl+C")
    args = parser.parse_args()

    if args.source != "rpicam":
        print("Warning: this runner uses rpicam-still regardless of --source.")

    expected = parse_expected(args.expected)

    module = load_python_module(args.opencv_script)
    detection_function, selected_function_name = get_function(module, args.function)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    input_path = results_dir / "_rpicam_input.jpg"
    output_path = results_dir / "_rpicam_output.jpg"
    csv_path = results_dir / "realtime_rpicam_results.csv"

    print(f"Running script: {args.opencv_script}")
    print(f"Running function: {selected_function_name}")
    print(f"Class name: {args.class_name}")
    print(f"Expected counts: {expected}")
    print("Using rpicam-still capture. Press q in the preview window or Ctrl+C to stop.")

    frame_idx = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp", "counts_json", "pass"])

        try:
            while True:
                frame_idx += 1
                t0 = time.time()

                capture_with_rpicam(input_path, args.width, args.height, args.timeout_ms)

                counts = detection_function(str(input_path), str(output_path))
                if counts is None:
                    counts = {}
                ok = pass_fail(counts, expected)

                writer.writerow([frame_idx, t0, json.dumps(counts), ok])
                f.flush()

                print(f"frame={frame_idx} counts={counts} pass={ok}")

                if args.show:
                    img = cv2.imread(str(output_path))
                    if img is not None:
                        cv2.imshow(f"rpicam realtime - {args.class_name}", img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break

                if args.max_frames and frame_idx >= args.max_frames:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    print("Finished.")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
