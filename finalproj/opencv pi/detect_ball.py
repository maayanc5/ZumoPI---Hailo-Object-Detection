import cv2
import numpy as np

CLASS_NAME = "ball"


def find_ball(image_path, output_path):
    """
    Detector compatible with opencv_realtime_picam.py.

    Required signature:
        find_ball(image_path, output_path)

    It reads the current camera frame from image_path, detects a circular ball
    using OpenCV HoughCircles, draws the detection, saves output_path, and
    returns {"ball": count}.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return {CLASS_NAME: 0}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=50,
        maxRadius=0,
    )

    count = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Use the largest detected circle as the ball.
        best = max(circles[0, :], key=lambda c: c[2])
        x, y, r = int(best[0]), int(best[1]), int(best[2])

        count = 1

        # Draw center and circle.
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

        # Calculate average color inside the ball.
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        mean_val = cv2.mean(frame, mask=mask)
        color_text = f"B:{int(mean_val[0])} G:{int(mean_val[1])} R:{int(mean_val[2])}"

        diameter = 2 * r

        # Draw bounding box around the ball.
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(frame.shape[1] - 1, x + r)
        y2 = min(frame.shape[0] - 1, y + r)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Text overlay.
        cv2.putText(
            frame,
            f"{CLASS_NAME}: 1",
            (x1, max(20, y1 - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Diam: {diameter}px Center: ({x}, {y})",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Color {color_text}",
            (x1, min(frame.shape[0] - 10, y2 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            f"{CLASS_NAME}: 0",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, frame)
    return {CLASS_NAME: count}
