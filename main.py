import os
import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET


# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # Increase the number of epochs to allow the model to train more
NUM_CLASSES = 5  # Assuming 5 classes: hello, thanks, yes, no, iloveyou

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Find the 'object' element and then find the 'name' element within it
    object_elem = root.find('object')
    if object_elem is not None:
        name_elem = object_elem.find('name')
        if name_elem is not None:
            return name_elem.text
    print(f"Warning: Could not find label in {xml_file}")
    return None

def load_data(data_dir):
    images = []
    labels = []
    label_map = {'hello': 0, 'thanks': 1, 'yes': 2, 'no': 3, 'iloveyou': 4}
    
    print(f"Attempting to load data from directory: {data_dir}")
    print(f"Directory exists: {os.path.exists(data_dir)}")
    print(f"Directory contents: {os.listdir(data_dir)}")
    
    for file in os.listdir(data_dir):
        if file.lower().endswith('.xml'):
            xml_path = os.path.join(data_dir, file)
            img_file = file.replace('.xml', '.jpg')
            img_path = os.path.join(data_dir, img_file)
            
            print(f"Processing XML: {xml_path}")
            print(f"Looking for image: {img_path}")
            print(f"Image file exists: {os.path.exists(img_path)}")
            
            if os.path.exists(img_path):
                try:
                    label = parse_xml(xml_path)
                    if label is None or label not in label_map:
                        print(f"Unknown or missing label in file: {xml_path}")
                        continue
                    
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    labels.append(label_map[label])
                    print(f"Successfully processed image: {img_path} with label: {label}")
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
            else:
                print(f"Image file not found: {img_path}")
    
    print(f"Total images processed: {len(images)}")
    return np.array(images), np.array(labels)

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def train_model():
    # Load and preprocess data
    data_dir = r'P:\\SignLanguageDetection\\Tensorflow\\workspace\\images\\collectedimages'
    print(f"Full path to data directory: {os.path.abspath(data_dir)}")
    images, labels = load_data(data_dir)
    
    if len(images) == 0:
        raise ValueError("No valid images found in the specified directory.")
    
    # Split data into train and validation sets
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    
    # Create and compile the model
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val))
    
    # Save the model
    model.save('sign_language_model.h5')
    return model

def real_time_detection(model):
    cap = cv2.VideoCapture(0)
    label_map = {0: 'hello', 1: 'thanks', 2: 'yes', 3: 'no', 4: 'iloveyou'}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255.0
        
        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction[0])
        predicted_label = label_map[predicted_class]
        confidence = prediction[0][predicted_class]
        
        # Display result on frame
        cv2.putText(frame, f"{predicted_label} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Train the model (comment out if model is already trained)
    model = train_model()
    
    # For subsequent runs, load the saved model instead
    # model = tf.keras.models.load_model('sign_language_model.h5')
    
    # Start real-time detection
    real_time_detection(model)