import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_hubs(img, DEBUG=False, thresholdLevel=0.4, circle_diameter=150,
                expected_dark_fraction=0.85, min_blob_area=10):  # Changed area to 10

    # --- 1. Resize if too large ---
    origH, origW = img.shape[:2]
    maxW, maxH = 1920, 1080
    if origW > maxW or origH > maxH:
        scale = min(maxW / origW, maxH / origH)
        img = cv2.resize(img, (int(origW * scale), int(origH * scale)))

    # --- 2. Pre-processing ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Binary Thresholding
    # bin_img: 255 (White) = Dark areas in original (The Big Black Circle)
    thresh_val = int(thresholdLevel * 255)
    _, bin_raw = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    bin_img = cv2.bitwise_not(bin_raw)

    # Blur for better shape detection
    blur = cv2.medianBlur(gray, 5)

    # --- 3. Detect Outer Circles (The Big Black Circle) ---
    # Adjusted radius range slightly wider
    min_radius = int((circle_diameter * 0.4) / 2)
    max_radius = int((circle_diameter * 1.6) / 2)

    # param2 lowered to 20 (more sensitive)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=circle_diameter,
                               param1=100, param2=20,
                               minRadius=min_radius, maxRadius=max_radius)

    hubs = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        rows, cols = bin_img.shape
        y_grid, x_grid = np.ogrid[0:rows, 0:cols]

        for (c_x, c_y, r) in circles[0, :]:
            # Create mask for the current circle
            circle_mask = (x_grid - c_x) ** 2 + (y_grid - c_y) ** 2 <= r ** 2

            # --- 4. Validate "Big Black Circle" ---
            total_pixels = np.sum(circle_mask)
            if total_pixels == 0: continue

            # Check if the circle is mostly dark
            dark_fraction = np.sum(bin_img[circle_mask] == 255) / total_pixels

            # Wide tolerance for dark fraction (0.4 to 1.0)
            if not (0.4 <= dark_fraction <= 1.0):
                continue

            # --- 5. Find the ONE White Circle Inside ---
            # ROI: Pixels inside the circle that are BRIGHT in original image (bin_img == 0)
            roi_mask = np.logical_and(circle_mask, bin_img == 0).astype(np.uint8) * 255

            # Find blobs in the bright regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)

            valid_blob = None
            valid_count = 0

            # Iterate blobs (skip background 0)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_blob_area:
                    continue

                bx, by = int(centroids[i][0]), int(centroids[i][1])
                bx, by = min(max(0, bx), cols - 1), min(max(0, by), rows - 1)

                pixel_rgb = img[by, bx]
                R, G, B = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])

                # RELAXED White criteria:
                # 1. Brightness: At least one channel > 80 (handles dim lighting)
                # 2. Color Balance: Channels shouldn't differ by more than 60
                if (max(R, G, B) > 80) and \
                        (abs(R - G) < 60 and abs(R - B) < 60 and abs(G - B) < 60):
                    valid_count += 1
                    valid_blob = {"center": (bx, by), "color": "white", "area": float(area)}

            # --- 6. Final Decision ---
            # We strictly want exactly ONE valid white blob inside
            if valid_count == 1:
                hubs.append({
                    "center": (float(c_x), float(c_y)),
                    "radius": float(r),
                    "inner_blob": valid_blob
                })
                if DEBUG:
                    cv2.circle(img, (c_x, c_y), r, (0, 255, 0), 2)
                    bx, by = int(valid_blob["center"][0]), int(valid_blob["center"][1])
                    cv2.circle(img, (bx, by), 5, (255, 0, 0), -1)

    if DEBUG:
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"Detected {len(hubs)} Hubs (Green=Outer, Blue=Inner)")
        plt.show()

    return hubs