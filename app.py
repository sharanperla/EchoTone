# Import necessary libraries 

import os 
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt


#load the dataset

def load_dataset(dataset_path):
    print("loading data.....")
    emotions = ['angry', 'disgust', 'happy', 'fear', 'neutral', 'sad']
    data, labels = [], []
    
    # Iterate through all folders in the dataset path
    for folder in os.listdir(dataset_path):
        for emotion in emotions:
            if emotion in folder:  # Check if the folder name contains the emotion
                emotion_path = os.path.join(dataset_path, folder)
                
                # Process only if it's a directory
                if os.path.isdir(emotion_path):  
                    for file in os.listdir(emotion_path):
                        if file.endswith('.wav'):
                            file_path = os.path.join(emotion_path, file)
                            
                            # Load audio file
                            audio, sr = librosa.load(file_path, sr=22050)
                            data.append(audio)
                            labels.append(emotion)  # Store only the emotion label

    return data, labels


# Step 2: Feature Extraction
def extract_features(data):
    print("extracting features.....")
    features = []
    for audio in data:
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        features.append(mfcc_scaled)
    return np.array(features)


def preprocess_data(dataset_path):
    print("preprocrssing data.....")
    audio_data,labels=load_dataset(dataset_path)
     # Extract features
    features = extract_features(audio_data)
    # Encode labels into numeric format
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
     # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, encoder


def build_model(input_shape):
    print("building model.....")
    model=tf.keras.Sequential(
        [
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation='softmax')

        ]
    )
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Save the trained model
def save_model(model, model_path="emotion_model.h5"):
    model.save(model_path)
    print(f"Model saved at {model_path}")

# Step 5: Train and Evaluate the Model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("training and evalvating model.....")
    model = build_model(X_train.shape[1])
    history = model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1
    )

    save_model(model)
    
    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()


# Step 6: Main Function
if __name__ == "__main__":
    # Path to the TESS dataset
    dataset_path = os.path.join(os.getcwd(), "Tess")  # Replace with the actual dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' not found!")

    # Preprocess the data
    X_train, X_test, y_train, y_test, encoder = preprocess_data(dataset_path)

    # Train and evaluate the model
    train_and_evaluate(X_train, X_test, y_train, y_test)


