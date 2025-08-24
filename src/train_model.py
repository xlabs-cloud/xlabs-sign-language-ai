# src/train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from utils import load_data, IMG_SIZE
def create_cnn_model(num_classes):
    """Creates a CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def main():
    # Load data
    images, labels, label_map = load_data()
    
    if len(images) == 0:
        print("No data found. Please run collect_data.py first.")
        return
    # Normalize images
    images = images / 255.0
    
    # One-hot encode labels
    num_classes = len(label_map)
    labels_categorical = to_categorical(labels, num_classes=num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_categorical, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = create_cnn_model(num_classes)
    print("Starting model training...")
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)
    
    # Save the model and label map
    model.save('models/sign_model.h5')
    with open('models/label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)
        
    print("Model and label map saved to models/ directory.")
if __name__ == '__main__':
    main()