# src/collect_data.py

import os
import cv2
import mediapipe as mp
import time

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

print("Starting data collection...")
print("Press a key (e.g., 'A', 'B', 'C') to start collecting images for that label.")
print("Press 'q' to quit.")

current_label = None
collecting = False
image_counter = 0
COLLECTION_PAUSE = 2  # seconds to get ready

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    display_frame = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display instructions and status
    status_text = f"Label: {current_label if current_label else 'None'} | Collecting: {collecting} | Count: {image_counter}"
    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display_frame, "Press a-z to set label, 's' to start/stop, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Data Collection', display_frame)

    key = cv2.waitKey(25) & 0xFF

    if key == ord('q'):
        break
    elif ord('a') <= key <= ord('z'):
        label_char = chr(key).upper()
        if current_label != label_char:
            current_label = label_char
            image_counter = 0
            label_dir = os.path.join(DATA_DIR, current_label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            print(f"Set label to: {current_label}. Directory created at {label_dir}")
            print(f"Get ready! Starting collection in {COLLECTION_PAUSE} seconds...")
            time.sleep(COLLECTION_PAUSE)
            collecting = True

    if collecting and current_label:
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                # Find bounding box
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Crop the hand region with a margin
                margin = 20
                x1 = max(0, int(x_min * w) - margin)
                y1 = max(0, int(y_min * h) - margin)
                x2 = min(w, int(x_max * w) + margin)
                y2 = min(h, int(y_max * h) + margin)
                
                hand_crop = frame[y1:y2, x1:x2]
                
                # *** THIS IS THE FIX ***
                # We now directly save the cropped image, no 'bg.jpg' needed.
                if hand_crop.size > 0:
                    img_path = os.path.join(DATA_DIR, current_label, f'{current_label}_{time.time()}.jpg')
                    cv2.imwrite(img_path, hand_crop)
                    image_counter += 1
                    print(f"Saved image: {img_path}")
                    time.sleep(0.1) # Small delay to avoid saving identical frames

cap.release()
cv2.destroyAllWindows()