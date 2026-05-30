import cv2
import numpy as np
import os
from hand_tracker import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1, detection_con=0.7)
tip_ids = [8, 12, 16, 20]

def load_vibe_image(image_path):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        return cv2.resize(img, (400, 400))
    else:
        blank = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(blank, "IMG MISSING", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return blank

img_fist = load_vibe_image("images/sushi.png")
img_peace = load_vibe_image("images/bowl.png")
img_open = load_vibe_image("images/fries.png")

img_waiting = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.putText(img_waiting, "WAITING...", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

print("Image Vibe Active: Show your gestures!")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue

    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=True)
    lm_dict = detector.get_position(img)

    current_output = img_waiting

    if len(lm_dict) != 0:
        fingers = []
        for id in range(0, 4):
            if lm_dict[tip_ids[id]][1] < lm_dict[tip_ids[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        if fingers == [0, 0, 0, 0]:
            current_output = img_fist
        elif fingers == [1, 1, 0, 0]:
            current_output = img_peace
        elif fingers == [1, 1, 1, 1]:
            current_output = img_open

    cv2.imshow("Camera Feed", img)
    cv2.imshow("Output Image", current_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
