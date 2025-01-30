from fastapi import FastAPI, UploadFile, File
import numpy as np
import librosa
import tensorflow as tf
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
MODEL_PATH = "emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Label Mapping
emotion_labels = ['angry', 'disgust', 'happy', 'fear', 'neutral', 'sad']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Extract Features
def extract_mfcc(audio_bytes):
    try:
        print(f"First 10 bytes: {audio_bytes[:10]}")  # Log first few bytes to inspect the file format
        y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return np.expand_dims(mfcc_scaled, axis=0)  # Reshape for model input
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise e



@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        features = extract_mfcc(audio_bytes)
        prediction = model.predict(features)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        return {"emotion": predicted_emotion}
    except Exception as e:
        return {"error": str(e)}
