import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Set up the Hand Tracking model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Try camera index 0 first; on some Macs, you may need to try 1
cap = cv2.VideoCapture(0)

print("Vibe Check: Looking for your hand... Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Waiting for camera...")
        # If camera 0 fails, it might be busy. Retry index 1 if needed.
        continue

    # Flip the image horizontally for a selfie-view
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hands
    results = hands.process(rgb_image)

    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get specific coordinates (8 is Index Tip, 5 is Index Knuckle)
                index_tip = hand_landmarks.landmark[8]
                index_knuckle = hand_landmarks.landmark[5] 

                # In screen coordinates, higher Y means lower on the screen
                if index_tip.y > index_knuckle.y:
                    cv2.putText(image, "FIST DETECTED", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(image, "HAND OPEN", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                # Draw the lines so you can see the logic working
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Gesture Vibe Control', image)

    # Break loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()