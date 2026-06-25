import cv2
import numpy as np
import os

# WSL Fallbacks
COLOR_BGR_HSV = getattr(cv2, 'COLOR_BGR_HSV', 40)

def detect_full_cone(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    # 1. Convert to HSV
    hsv = cv2.cvtColor(img, COLOR_BGR_HSV)

    # 2. Relaxed Red/Orange Range (Captures top, middle, and bottom bands)
    lower_red1 = np.array([0, 100, 70])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 100, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # 3. Clean up noise but don't over-process
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4. Find all red parts
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cone_parts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50: # Catch even small parts of the cone
            cone_parts.append(cnt)

    if cone_parts:
        # Combine all detected red points into one array
        all_points = np.concatenate(cone_parts)
        
        # Get one big bounding box for ALL parts combined
        x, y, w, h = cv2.boundingRect(all_points)

        # 5. Final check: Is it cone-shaped? (Tall enough?)
        aspect_ratio = h / float(w)
        if aspect_ratio > 0.8: 
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, "Full Cone", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Detected full cone in {os.path.basename(image_path)}")
    else:
        print(f"No cone parts found in {os.path.basename(image_path)}")

    cv2.imwrite(output_path, img)

def main():
    input_dir = 'conus_images'
    output_dir = 'output_results'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            detect_full_cone(os.path.join(input_dir, filename), 
                             os.path.join(output_dir, f"full_{filename}"))

if __name__ == "__main__":
    main()