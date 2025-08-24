# src/detect_sign.py
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import argparse
from gtts import gTTS
import os
# Argument parser for TTS flag
parser = argparse.ArgumentParser()
parser.add_argument('--tts', action='store_true', help='Enable Text-to-Speech output')
args = parser.parse_args()
# Load the trained model and label map
model = load_model('models/sign_model.h5')
with open('models/label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)
# Create a reverse map for easy lookup
reverse_label_map = {v: k for k, v in label_map.items()}
# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
last_prediction = None
prediction_confidence = 0
CONFIDENCE_THRESHOLD = 0.9  # Minimum confidence to consider a prediction stable
def speak(text):
    """Converts text to speech."""
    if text:
        tts = gTTS(text=text, lang='en')
        tts.save("speech.mp3")
        os.system("mpg321 speech.mp3") # You might need to install mpg321: sudo apt-get install mpg321
        os.remove("speech.mp3")
print("Starting real-time sign detection...")
print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    display_frame = frame.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Preprocess the hand region for prediction
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_coords) * w) - 20, int(min(y_coords) * h) - 20
            x2, y2 = int(max(x_coords) * w) + 20, int(max(y_coords) * h) + 20
            hand_crop = frame[y1:y2, x1:x2]
            
            if hand_crop.size > 0:
                from utils import preprocess_frame
                processed_hand = preprocess_frame(hand_crop)
                
                # Make a prediction
                prediction = model.predict(processed_hand)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                if confidence > CONFIDENCE_THRESHOLD:
                    predicted_label = reverse_label_map[predicted_class]
                    
                    # Display the prediction
                    cv2.rectangle(display_frame, (x1, y1 - 40), (x2, y1), (0, 255, 0), -1)
                    cv2.putText(display_frame, f'{predicted_label} ({confidence:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Text-to-speech output
                    if args.tts and predicted_label != last_prediction:
                        speak(predicted_label)
                        last_prediction = predicted_label
                else:
                    last_prediction = None
    cv2.imshow('Sign Language Detection', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()