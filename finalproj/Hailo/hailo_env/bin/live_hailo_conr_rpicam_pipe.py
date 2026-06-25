#!/usr/bienv python3

import subprocess
import time
import cv2
import numpy as np

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    InferVStreams,
    FormatType,
    HailoStreamInterface,
)

# =========================
# Model / camera settings
# =========================

HEF_PATH = "/home/pi/cone_yolov8_raw_heads_hailo8l.hef"

WIDTH = 1280
HEIGHT = 720
FPS = 30

IMG_SIZE = 640
REG_MAX = 16

# =========================
# Best constants from geometry search
# =========================

CONF_THRES = 0.995
IOU_THRES = 0.15

TOPK = 60
MAX_DET = 3

MIN_AREA = 3000
MAX_AREA = 60000

MIN_RATIO = 0.10
MAX_RATIO = 0.80

MIN_HEIGHT = 70
MAX_CENTER_Y = 480


def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def letterbox_rgb(image, size=640):
    h, w = image.shape[:2]

    scale = min(size / w, size / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (size, size, 3),
        114,
        dtype=np.uint8,
    )

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def unletterbox_box(box, scale, pad_x, pad_y, orig_w, orig_h):
    x1, y1, x2, y2 = box

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    x1 = int(max(0, min(orig_w - 1, x1)))
    y1 = int(max(0, min(orig_h - 1, y1)))
    x2 = int(max(0, min(orig_w - 1, x2)))
    y2 = int(max(0, min(orig_h - 1, y2)))

    return x1, y1, x2, y2


def normalize_output(arr):
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3:
        if arr.shape[0] in (1, 64) and arr.shape[-1] not in (1, 64):
            arr = np.transpose(arr, (1, 2, 0))

    return arr.astype(np.float32)


def split_outputs(outputs):
    regs = {}
    clss = {}

    for name, value in outputs.items():
        arr = normalize_output(value)

        if arr.ndim != 3:
            print("Unexpected output:", name, arr.shape)
            continue

        h, w, c = arr.shape

        if c == 64:
            regs[(h, w)] = arr
        elif c == 1:
            clss[(h, w)] = arr
        else:
            print("Unknown output:", name, arr.shape)

    pairs = []

    for key in [(80, 80), (40, 40), (20, 20)]:
        if key in regs and key in clss:
            pairs.append((key, regs[key], clss[key]))

    return pairs


def decode_yolov8_1class(outputs, conf_thres):
    pairs = split_outputs(outputs)
    proj = np.arange(REG_MAX, dtype=np.float32)

    all_boxes = []
    all_scores = []

    for (h, w), reg, cls_logits in pairs:
        stride = IMG_SIZE / h

        reg = reg.reshape(h, w, 4, REG_MAX)
        reg = softmax(reg, axis=-1)
        dist = np.sum(reg * proj, axis=-1)

        scores = sigmoid(cls_logits.reshape(-1))

        ys, xs = np.meshgrid(
            np.arange(h),
            np.arange(w),
            indexing="ij",
        )

        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride

        x1 = cx - dist[:, :, 0] * stride
        y1 = cy - dist[:, :, 1] * stride
        x2 = cx + dist[:, :, 2] * stride
        y2 = cy + dist[:, :, 3] * stride

        boxes = np.stack(
            [x1, y1, x2, y2],
            axis=-1,
        ).reshape(-1, 4)

        idxs = np.where(scores >= conf_thres)[0]

        if idxs.size == 0:
            continue

        if idxs.size > TOPK:
            local_scores = scores[idxs]
            top_local = np.argpartition(local_scores, -TOPK)[-TOPK:]
            idxs = idxs[top_local]

        selected_boxes = boxes[idxs]
        selected_scores = scores[idxs]

        bw = np.maximum(0, selected_boxes[:, 2] - selected_boxes[:, 0])
        bh = np.maximum(0, selected_boxes[:, 3] - selected_boxes[:, 1])
        area = bw * bh
        ratio = bw / np.maximum(bh, 1)
        center_y = (selected_boxes[:, 1] + selected_boxes[:, 3]) / 2

        area_mask = (area >= MIN_AREA) & (area <= MAX_AREA)
        ratio_mask = (ratio >= MIN_RATIO) & (ratio <= MAX_RATIO)
        height_mask = bh >= MIN_HEIGHT
        position_mask = center_y <= MAX_CENTER_Y

        final_mask = area_mask & ratio_mask & height_mask & position_mask

        selected_boxes = selected_boxes[final_mask]
        selected_scores = selected_scores[final_mask]

        if len(selected_boxes) == 0:
            continue

        all_boxes.append(selected_boxes)
        all_scores.append(selected_scores)

    if not all_boxes:
        return np.empty((0, 4)), np.empty((0,))

    return np.concatenate(all_boxes), np.concatenate(all_scores)


def run_nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    xywh = []

    for x1, y1, x2, y2 in boxes:
        xywh.append(
            [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ]
        )

    keep = cv2.dnn.NMSBoxes(
        xywh,
        scores.astype(float).tolist(),
        score_threshold=0.0,
        nms_threshold=iou_thres,
    )

    if len(keep) == 0:
        return np.empty((0, 4)), np.empty((0,))

    keep = np.array(keep).reshape(-1)
    keep = keep[np.argsort(-scores[keep])]
    keep = keep[:MAX_DET]

    return boxes[keep], scores[keep]


def start_camera():
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--codec", "mjpeg",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FPS),
        "--nopreview",
        "-o", "-",
    ]

    print("Starting camera:")
    print(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    return proc


def main():
    hef = HEF(HEF_PATH)

    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()

    print("Input:")
    for info in input_infos:
        print(info.name, info.shape)

    print("Outputs:")
    for info in output_infos:
        print(info.name, info.shape)

    input_name = input_infos[0].name

    print()
    print("Detection constants:")
    print(f"CONF_THRES = {CONF_THRES}")
    print(f"MIN_AREA = {MIN_AREA}")
    print(f"MAX_AREA = {MAX_AREA}")
    print(f"MIN_RATIO = {MIN_RATIO}")
    print(f"MAX_RATIO = {MAX_RATIO}")
    print(f"MIN_HEIGHT = {MIN_HEIGHT}")
    print(f"MAX_CENTER_Y = {MAX_CENTER_Y}")
    print()

    configure_params = ConfigureParams.create_from_hef(
        hef,
        interface=HailoStreamInterface.PCIe,
    )

    proc = start_camera()
    buffer = b""

    with VDevice() as target:
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        input_params = InputVStreamParams.make(
            network_group,
            format_type=FormatType.UINT8,
        )

        output_params = OutputVStreamParams.make(
            network_group,
            format_type=FormatType.FLOAT32,
        )

        with InferVStreams(network_group, input_params, output_params) as pipe:
            with network_group.activate(network_group_params):
                prev_time = time.time()

                try:
                    while True:
                        chunk = proc.stdout.read(4096)

                        if not chunk:
                            continue

                        buffer += chunk

                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9")

                        if start == -1 or end == -1 or end <= start:
                            continue

                        jpg = buffer[start:end + 2]
                        buffer = buffer[end + 2:]

                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                        if frame_bgr is None:
                            continue

                        orig_h, orig_w = frame_bgr.shape[:2]

                        # rpicam-vid decoded by OpenCV gives BGR.
                        # Hailo input expects RGB.
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                        img_rgb, scale, pad_x, pad_y = letterbox_rgb(
                            frame_rgb,
                            IMG_SIZE,
                        )

                        input_data = np.expand_dims(
                            img_rgb,
                            axis=0,
                        ).astype(np.uint8)

                        outputs = pipe.infer({input_name: input_data})

                        boxes, scores = decode_yolov8_1class(
                            outputs,
                            CONF_THRES,
                        )

                        boxes, scores = run_nms(
                            boxes,
                            scores,
                            IOU_THRES,
                        )

                        now = time.time()
                        fps = 1.0 / max(now - prev_time, 1e-6)
                        prev_time = now

                        annotated = frame_bgr.copy()
                        best_conf = 0.0

                        for box, score in zip(boxes, scores):
                            score = float(score)
                            best_conf = max(best_conf, score)

                            x1, y1, x2, y2 = unletterbox_box(
                                box,
                                scale,
                                pad_x,
                                pad_y,
                                orig_w,
                                orig_h,
                            )

                            label = f"cone-yolo {score * 100:.1f}%"

                            cv2.rectangle(
                                annotated,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2,
                            )

                            cv2.putText(
                                annotated,
                                label,
                                (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                        cv2.putText(
                            annotated,
                            f"Hailo FPS: {fps:.1f} | Best: {best_conf * 100:.1f}%",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            2,
                        )

                        cv2.imshow("Hailo cone live", annotated)

                        key = cv2.waitKey(1) & 0xFF

                        if key == ord("q") or key == 27:
                            break

                except KeyboardInterrupt:
                    pass

                finally:
                    proc.terminate()
                    cv2.destroyAllWindows()
                    print("Stopped")


if __name__ == "__main__":
    main()
