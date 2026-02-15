import cv2
import numpy as np
from hand_tracker import HandDetector

# 1. Setup Main Camera and Brain
cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1, detection_con=0.7)
tip_ids = [8, 12, 16, 20]

# 2. Setup the Video Player
# This opens the video file so we can read it frame-by-frame
cap_vid = cv2.VideoCapture("peace.mp4")

# 3. Setup the Static Image
img_fist = cv2.imread("images/bowl.png")
if img_fist is not None:
    img_fist = cv2.resize(img_fist, (250, 250)) # Resize to a 250x250 square

print("Vibe Active: Peace sign plays video, Fist shows image!")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue
    
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=True)
    lm_list = detector.get_position(img)

    # Get your Mac camera's width and height (usually 1280x720)
    h, w, c = img.shape

    if len(lm_list) != 0:
        fingers = []
        for id in range(0, 4):
            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                fingers.append(1) 
            else:
                fingers.append(0) 
        
        # --- THE OVERLAY LOGIC ---
        # We define a 250x250 box in the Top Right Corner of the screen
        y1, y2 = 20, 270             # Top to Bottom
        x1, x2 = w - 270, w - 20     # Left to Right

        # 1. PEACE SIGN -> PLAY VIDEO
        if fingers == [1, 1, 0, 0]:
            # Grab the next frame of the video
            success_vid, vid_frame = cap_vid.read()
            
            # If the video ends, loop it back to the beginning!
            if not success_vid:
                cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success_vid, vid_frame = cap_vid.read()

            if success_vid:
                vid_frame = cv2.resize(vid_frame, (250, 250))
                # Paste the video frame onto your live camera feed
                img[y1:y2, x1:x2] = vid_frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3) # Blue border

        # 2. FIST -> SHOW STATIC IMAGE
        elif fingers == [0, 0, 0, 0]:
            if img_fist is not None:
                # Paste the static image onto your live camera feed
                img[y1:y2, x1:x2] = img_fist
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red border

    # Show EVERYTHING in just one single window
    cv2.imshow("Gesture Vibe Controller", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap_vid.release() # Release the video file too
cv2.destroyAllWindows()