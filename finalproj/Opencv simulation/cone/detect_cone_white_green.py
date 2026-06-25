import cv2
import numpy as np
import os

# WSL Fallback
COLOR_BGR_HSV = getattr(cv2, 'COLOR_BGR_HSV', 40)

def detect_wg_symmetry(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None: return
    hsv = cv2.cvtColor(img, COLOR_BGR_HSV)
    
    # 1. Mask for Green Stripes
    mask_g = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([85, 255, 255]))
    mask_g = cv2.medianBlur(mask_g, 5)
    
    # 2. Find individual stripes
    contours, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 30]

    # 3. Stack Grouping with SYMMETRY
    cones = []
    stripe_boxes.sort(key=lambda b: b[1]) # Top to bottom

    while len(stripe_boxes) > 0:
        bx, by, bw, bh = stripe_boxes.pop(0)
        current_cone = [(bx, by, bw, bh)]
        cx_target = bx + bw/2 # The vertical axis of the cone
        
        remaining = []
        for box in stripe_boxes:
            ox, oy, ow, oh = box
            cx_other = ox + ow/2
            
            # STRICT SYMMETRY: The center of the next stripe must be within 15px of the top one
            # This kills door frames and slanted clutter
            if abs(cx_target - cx_other) < 15 and (oy - (by + bh)) < 150:
                current_cone.append(box)
                # Update vertical bounds
                by, bh = min(by, oy), max(by + bh, oy + oh) - min(by, oy)
            else:
                remaining.append(box)
        stripe_boxes = remaining
        
        # 4. FINAL BOX EXPANSION
        if len(current_cone) >= 2:
            current_cone.sort(key=lambda b: b[1])
            top_w = current_cone[0][2]
            bot_w = current_cone[-1][2]
            
            # Taper check: Bottom must be wider than top
            if bot_w > (top_w * 1.1):
                all_pts = []
                for b in current_cone:
                    all_pts.extend([[b[0], b[1]], [b[0]+b[2], b[1]+b[3]]])
                
                x, y, w, h = cv2.boundingRect(np.array(all_pts))
                
                # Expand box by 15% vertically to catch the white tip and base
                v_pad = int(h * 0.15)
                h_pad = int(w * 0.1)
                
                y_final = max(0, y - v_pad)
                h_final = min(img.shape[0] - y_final, h + (2 * v_pad))
                x_final = max(0, x - h_pad)
                w_final = min(img.shape[1] - x_final, w + (2 * h_pad))

                cv2.rectangle(img, (x_final, y_final), (x_final + w_final, y_final + h_final), (0, 255, 0), 3)
                cv2.putText(img, "WG CONE", (x_final, y_final - 10), 2, 0.7, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"SYMMETRY CHECK COMPLETE: {os.path.basename(image_path)}")

def main():
    input_dir, output_dir = 'conus_images', 'output_results'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            detect_wg_symmetry(os.path.join(input_dir, f), os.path.join(output_dir, f"sym_{f}"))

if __name__ == "__main__": main()