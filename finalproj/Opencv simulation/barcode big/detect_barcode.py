import cv2
import numpy as np
import os

def detect_barcode_spatial(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None: return
    
    gray = cv2.cvtColor(img, 6) # COLOR_BGR_GRAY
    h, w = gray.shape

    # 1. CREATE A SPATIAL MASK
    # We ignore the right side and top of the image where the monitor/cables are.
    # This acts as a "Zone of Interest" for the desk surface.
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    # Define a polygon for the desk surface (Left side and center)
    roi_corners = np.array([[(0, h), (0, int(h*0.4)), (int(w*0.6), int(h*0.4)), (int(w*0.6), h)]], dtype=np.int32)
    cv2.fillPoly(mask_roi, roi_corners, 255)

    # 2. BLOB DETECTION (Restricted to ROI)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor, params.blobColor = True, 255
    params.filterByArea, params.minArea, params.maxArea = True, 10, 600
    params.filterByCircularity, params.minCircularity = True, 0.3
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Apply mask to the grayscale image before detecting
    masked_gray = cv2.bitwise_and(gray, mask_roi)
    blurred = cv2.medianBlur(masked_gray, 5)
    
    keypoints = detector.detect(blurred)
    centers = [kp.pt for kp in keypoints]

    # 3. DENSITY CLUSTERING
    final_cluster = None
    best_dist = float('inf')

    if len(centers) >= 4:
        for i, c1 in enumerate(centers):
            neighbors = [c1]
            distances = []
            for j, c2 in enumerate(centers):
                if i == j: continue
                d = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if d < 130: # Max distance between barcode dots
                    neighbors.append(c2)
                    distances.append(d)
            
            if len(neighbors) == 4:
                avg_d = sum(distances) / len(distances)
                if avg_d < best_dist:
                    best_dist = avg_d
                    final_cluster = neighbors

    if final_cluster:
        pts = np.array(final_cluster, dtype=np.int32)
        x, y, w_box, h_box = cv2.boundingRect(pts)
        cv2.rectangle(img, (x-20, y-20), (x+w_box+20, y+h_box+20), (0, 255, 0), 3)
        cv2.putText(img, "BARCODE DISK", (x-20, y-35), 2, 0.7, (0, 255, 0), 2)
        for pt in final_cluster:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

    cv2.imwrite(output_path, img)

def main():
    input_dir, output_dir = 'barcode_images', 'barcode_results'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            detect_barcode_spatial(os.path.join(input_dir, f), os.path.join(output_dir, f"final_{f}"))

if __name__ == "__main__": main()