import cv2
import numpy as np
from hand_tracker import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1, detection_con=0.7)
tip_ids = [8, 12, 16, 20]

# Video file to play on peace sign gesture
cap_vid = cv2.VideoCapture("peace.mp4")

img_fist = cv2.imread("images/bowl.png")
if img_fist is not None:
    img_fist = cv2.resize(img_fist, (250, 250))

print("Media Vibe Active: Peace sign plays video, Fist shows image!")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue

    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=True)
    lm_dict = detector.get_position(img)

    h, w, c = img.shape

    if len(lm_dict) != 0:
        fingers = []
        for id in range(0, 4):
            if lm_dict[tip_ids[id]][1] < lm_dict[tip_ids[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Overlay region in the top-right corner
        y1, y2 = 20, 270
        x1, x2 = w - 270, w - 20

        if fingers == [1, 1, 0, 0]:
            success_vid, vid_frame = cap_vid.read()
            if not success_vid:
                cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success_vid, vid_frame = cap_vid.read()
            if success_vid:
                vid_frame = cv2.resize(vid_frame, (250, 250))
                img[y1:y2, x1:x2] = vid_frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        elif fingers == [0, 0, 0, 0]:
            if img_fist is not None:
                img[y1:y2, x1:x2] = img_fist
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("Gesture Vibe Controller", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap_vid.release()
cv2.destroyAllWindows()
