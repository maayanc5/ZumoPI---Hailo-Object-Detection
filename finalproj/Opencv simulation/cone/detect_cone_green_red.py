import cv2
import numpy as np
import os

# WSL Fallback
COLOR_BGR_HSV = getattr(cv2, 'COLOR_BGR_HSV', 40)

def detect_gr_precision(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None: return
    hsv = cv2.cvtColor(img, COLOR_BGR_HSV)
    
    # 1. Create Masks
    mask_r = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255])) + \
             cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
    mask_g = cv2.inRange(hsv, np.array([35, 80, 60]), np.array([85, 255, 255]))
    
    # Bridge stripes vertically
    combined = cv2.bitwise_or(mask_r, mask_g)
    kernel = np.ones((40, 2), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        
        # Geometry Filter
        if h > 50 and 1.1 < aspect_ratio < 4.0:
            # 2. ALIGNMENT CHECK
            # We look for the horizontal center of the red pixels vs green pixels
            roi_r = cv2.inRange(hsv[y:y+h, x:x+w], np.array([0, 120, 50]), np.array([10, 255, 255]))
            roi_g = cv2.inRange(hsv[y:y+h, x:x+w], np.array([35, 60, 40]), np.array([85, 255, 255]))
            
            if cv2.countNonZero(roi_r) > 20 and cv2.countNonZero(roi_g) > 20:
                M_r = cv2.moments(roi_r)
                M_g = cv2.moments(roi_g)
                
                # Calculate X-centers
                cx_r = int(M_r["m10"] / M_r["m00"])
                cx_g = int(M_g["m10"] / M_g["m00"])
                
                # Real cones are perfectly aligned (centers are close)
                # Towels on a door are often staggered
                alignment_error = abs(cx_r - cx_g)
                
                if alignment_error < (w * 0.2): # 20% width tolerance
                    found_count += 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(img, "GR CONE", (x, y - 10), 2, 0.7, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"LOCKED: {os.path.basename(image_path)} ({found_count})")

def main():
    input_dir = 'conus_images'
    output_dir = 'output_results' # Returning to standard output dir
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            detect_gr_precision(os.path.join(input_dir, f), os.path.join(output_dir, f"precision_{f}"))

if __name__ == "__main__": main()