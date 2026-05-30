import cv2
import pyautogui
from hand_tracker import HandDetector

pyautogui.PAUSE = 0

cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1, detection_con=0.7)
tip_ids = [8, 12, 16, 20]

last_gesture = "none"
tracked_gesture = "none"
gesture_frames = 0
REQUIRED_FRAMES = 4  # Hold for 4 frames to confirm gesture

print("Scroll Controller Active!")
print("Peace Sign = Scroll Up")
print("Index Finger = Scroll Down")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue

    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=True)
    lm_dict = detector.get_position(img)

    if len(lm_dict) != 0:
        fingers = []
        for id in range(0, 4):
            # Compare tip to knuckle for more reliable middle-finger detection
            if lm_dict[tip_ids[id]][1] < lm_dict[tip_ids[id] - 3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        raw_gesture = "none"
        if fingers == [1, 1, 0, 0]:
            raw_gesture = "peace"
        elif fingers == [1, 0, 0, 0]:
            raw_gesture = "index"

        if raw_gesture == tracked_gesture:
            gesture_frames += 1
        else:
            tracked_gesture = raw_gesture
            gesture_frames = 0

        if gesture_frames == REQUIRED_FRAMES:
            if tracked_gesture == "peace" and last_gesture != "peace":
                pyautogui.press('up')
                print("SWIPED UP")
                last_gesture = "peace"
            elif tracked_gesture == "index" and last_gesture != "index":
                pyautogui.press('down')
                print("SWIPED DOWN")
                last_gesture = "index"
            elif tracked_gesture == "none":
                last_gesture = "none"

        if last_gesture == "peace":
            cv2.putText(img, "PEACE: UP", (50, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        elif last_gesture == "index":
            cv2.putText(img, "INDEX: DOWN", (50, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        else:
            cv2.putText(img, "READY...", (50, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)

    cv2.imshow("Scroll Controller", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
