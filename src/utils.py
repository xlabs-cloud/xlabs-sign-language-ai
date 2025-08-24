# src/utils.py

import os
import cv2
import numpy as np
import mediapipe as mp

IMG_SIZE = 64

def load_data(data_dir='data'):
    """Loads dataset from the data directory."""
    images = []
    labels = []
    label_map = {}
    current_label_id = 0

    print(f"Loading data from {data_dir}...")
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if os.path.isdir(label_dir):
            if label_name not in label_map:
                label_map[label_name] = current_label_id
                current_label_id += 1
            
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    # Preprocess and append
                    processed_img = preprocess_frame(img, for_model=False)
                    images.append(processed_img)
                    labels.append(label_map[label_name])
    
    print(f"Loaded {len(images)} images for {len(label_map)} labels.")
    return np.array(images), np.array(labels), label_map

def preprocess_frame(frame, for_model=True):
    """Resizes and normalizes a frame."""
    if frame is None:
        return None
    
    # Resize to a fixed square
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    
    if for_model:
        # Normalize and expand dimensions for model prediction
        img_normalized = img_resized / 255.0
        return np.expand_dims(img_normalized, axis=0)
        
    return img_resized


def draw_landmarks(frame, results):
    """Draws MediaPipe landmarks on a frame."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame