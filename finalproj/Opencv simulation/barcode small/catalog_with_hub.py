import os
import cv2
# Note: Ensure hub_detector.py is in the same directory
from hub_detector import detect_hubs


def catalog_images(INPUT_FOLDER="input_images", OUTPUT_FOLDER="cataloged"):
    # Create base output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return

    print(f"Processing images from '{INPUT_FOLDER}'...")

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        full_path = os.path.join(INPUT_FOLDER, filename)

        # --- 1. Read Image ---
        # OpenCV reads in BGR format
        img_bgr = cv2.imread(full_path)
        if img_bgr is None:
            print(f"[SKIP] Unreadable file: {filename}")
            continue

        # Convert to RGB, as detect_hubs expects RGB format
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- 2. Detect Hubs ---
        # Ensure DEBUG=False so plots don't interrupt the loop
        hubs = detect_hubs(img_rgb, DEBUG=False)

        num_hubs_found = len(hubs)

        # --- 3. Classify and Annotate ---
        # Explicitly check if the number of hubs found is greater than 0
        if num_hubs_found > 0:
            cls = "hub"
            note = "(Annotated with circles)"

            # Draw circles on the RGB image
            for hub in hubs:
                # Draw Outer Circle (Green in RGB: 0, 255, 0)
                cx, cy = int(hub["center"][0]), int(hub["center"][1])
                r = int(hub["radius"])
                cv2.circle(img_rgb, (cx, cy), r, (0, 255, 0), 3)

                # Draw Inner Dot (Blue in RGB: 0, 0, 255)
                if "inner_blob" in hub and hub["inner_blob"]:
                    bx, by = int(hub["inner_blob"]["center"][0]), int(hub["inner_blob"]["center"][1])
                    cv2.circle(img_rgb, (bx, by), 5, (0, 0, 255), -1)
        else:
            # 0 hubs found
            cls = "unknown"
            note = "(Saved clean)"
            # Do NOT draw anything; keep the image clean.

        # --- 4. Save the Result ---
        dst_folder = os.path.join(OUTPUT_FOLDER, cls)
        os.makedirs(dst_folder, exist_ok=True)
        save_path = os.path.join(dst_folder, filename)

        # Convert RGB back to BGR before saving with OpenCV
        final_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_img_bgr)

        print(f"[OK] {filename} -> Class: '{cls}', Count: {num_hubs_found} {note}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    # Ensure directories exist for testing purposes if running this script directly
    # os.makedirs("input_images", exist_ok=True)
    catalog_images()