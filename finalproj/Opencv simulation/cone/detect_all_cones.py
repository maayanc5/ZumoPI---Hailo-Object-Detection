import cv2
import numpy as np
import os

# WSL Fallback
COLOR_BGR_HSV = getattr(cv2, 'COLOR_BGR_HSV', 40)

def find_cones(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None: return
    hsv = cv2.cvtColor(img, COLOR_BGR_HSV)
    h, w, _ = img.shape

    # TRACK 1: Green-White & Green-Red (Sensitive Green Track)
    # We use a large vertical dilation to merge green stripes into one block
    lower_green = np.array([35, 50, 40])
    upper_green = np.array([90, 255, 255])
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    
    # Physically bridge the white/red gaps vertically
    kernel_g = np.ones((50, 2), np.uint8) 
    dilated_g = cv2.dilate(mask_g, kernel_g, iterations=1)
    
    # TRACK 2: Red-White (Strict Red Track)
    # We keep this strict to ignore the towel
    mask_r = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255])) + \
             cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
    
    kernel_r = np.ones((40, 2), np.uint8)
    dilated_r = cv2.dilate(mask_r, kernel_r, iterations=1)

    # Combine detections
    combined = cv2.bitwise_or(dilated_g, dilated_r)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 150 < area < (h * w * 0.1): # Ignore tiny noise and giant towels
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = ch / float(cw)
            
            # THE SHAPE FILTER
            # 1. Must be taller than wide (aspect_ratio > 1.0)
            # 2. Cones are triangular, so they shouldn't fill the box (Extent < 0.8)
            extent = float(area) / (cw * ch)
            
            if 1.0 < aspect_ratio < 4.0 and extent < 0.8:
                # One last check: is it in the middle/bottom of the image? (Ignore ceiling glare)
                if y > h * 0.1:
                    cv2.rectangle(img, (x, y), (x + cw, y + ch), (0, 255, 0), 3)
                    cv2.putText(img, "CONE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

def main():
    input_dir, output_dir = 'conus_images', 'output_results'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            find_cones(os.path.join(input_dir, f), os.path.join(output_dir, f"track_{f}"))

if __name__ == "__main__": main()