import cv2
from hand_tracker import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1, detection_con=0.7)

counter = 0

# --- SEPARATED CALIBRATION THRESHOLDS ---
# Index Finger Settings
IDX_MIN = 300  # Curled resting position
IDX_MAX = 400  # Fully extended

# Pinky Finger Settings
PNK_MIN = 150  # Curled resting position
PNK_MAX = 250  # Adjust this if the pinky needs a different max than the index

index_locked = False
pinky_locked = False

print("Pro-Tuned Axis Tracker Active!")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue
    
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=True)
    lm_dict = detector.get_position(img)

    if all(id in lm_dict for id in [0, 8, 20]):
        
        wx, wy = lm_dict[0]   # Wrist
        ix, iy = lm_dict[8]   # Index Tip
        px, py = lm_dict[20]  # Pinky Tip

        idx_dx, idx_dy = abs(ix - wx), abs(iy - wy)
        pnk_dx, pnk_dy = abs(px - wx), abs(py - wy)

        idx_stretch = max(idx_dx, idx_dy)
        pnk_stretch = max(pnk_dx, pnk_dy)

        # --- MAP TO PERCENTAGES WITH INDEPENDENT MATH ---
        idx_pct = int((idx_stretch - IDX_MIN) * 100 / (IDX_MAX - IDX_MIN))
        idx_pct = max(0, min(100, idx_pct)) 

        pnk_pct = int((pnk_stretch - PNK_MIN) * 100 / (PNK_MAX - PNK_MIN))
        pnk_pct = max(0, min(100, pnk_pct))

        # --- INDEX LOGIC (+1) ---
        if idx_pct == 100 and not index_locked:
            counter += 1
            index_locked = True
        elif idx_pct < 40: # Must drop below 40% to reload
            index_locked = False

        # --- PINKY LOGIC (-1) ---
        if pnk_pct == 100 and not pinky_locked:
            counter -= 1
            pinky_locked = True
        elif pnk_pct < 40: # Must drop below 40% to reload
            pinky_locked = False

        # --- VISUAL FEEDBACK & CALIBRATION HUD ---
        idx_color = (0, 255, 0) if not index_locked else (255, 255, 255)
        pnk_color = (0, 0, 255) if not pinky_locked else (255, 255, 255)
        
        cv2.rectangle(img, (wx, wy), (ix, iy), idx_color, 2)
        cv2.rectangle(img, (wx, wy), (px, py), pnk_color, 2)

        cv2.putText(img, f"{idx_pct}%", (ix + 20, iy), cv2.FONT_HERSHEY_PLAIN, 2, idx_color, 2)
        cv2.putText(img, f"{pnk_pct}%", (px + 20, py), cv2.FONT_HERSHEY_PLAIN, 2, pnk_color, 2)

        # The HUD stays so you can keep an eye on the raw numbers
        cv2.putText(img, f"RAW IDX: {idx_stretch}", (20, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.putText(img, f"RAW PNK: {pnk_stretch}", (20, 170), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    cv2.putText(img, f"COUNTER: {counter}", (20, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)
    cv2.imshow("Gesture Controller", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()