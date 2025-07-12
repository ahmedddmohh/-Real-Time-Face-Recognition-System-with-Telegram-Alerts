import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime
import requests
import os

# Load the model
model = load_model("face_recognition_model.h5")

# Get number of output classes from the model
NUM_CLASSES = model.output_shape[-1]
print(f"✅ Model expects {NUM_CLASSES} classes")

# Try to load CLASS_NAMES dynamically from folder names
def get_class_names():
    folder = "Test_Subjects"
    if not os.path.isdir(folder):
        return [f"Person_{i}" for i in range(NUM_CLASSES)]
    
    names = sorted(os.listdir(folder))
    if len(names) != NUM_CLASSES:
        print("⚠️ CLASS_NAMES length mismatch. Using default names.")
        return [f"Person_{i}" for i in range(NUM_CLASSES)]
    
    return names

CLASS_NAMES = get_class_names()

EXCEL_FILE = "recognition_log.xlsx"
BOT_TOKEN = "8135129444:AAGamIhkDvoFYKuOrX7YKqchkmVMm8fthVU"
CHAT_ID = "1401565913"

def send_telegram_message(name):
    message = f"{name} has been recognized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message})
    except Exception as e:
        print("❌ Telegram error:", e)

def log_to_excel(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_entry = pd.DataFrame([[name, timestamp]], columns=["Name", "Timestamp"])
    try:
        existing = pd.read_excel(EXCEL_FILE)
        updated = pd.concat([existing, new_entry], ignore_index=True)
    except FileNotFoundError:
        updated = new_entry
    updated.to_excel(EXCEL_FILE, index=False)

def preprocess_frame(frame):
    face = cv2.resize(frame, (160, 160))  # ✅ fix: match model input size
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def run_recognition():
    cap = cv2.VideoCapture(0)
    recognized_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            continue

        input_tensor = preprocess_frame(frame)
        predictions = model.predict(input_tensor)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        print(f"🔍 Prediction vector: {predictions}")
        print(f"➡️ Predicted index: {predicted_index}, confidence: {confidence:.2f}")

        if predicted_index >= len(CLASS_NAMES):
            name = "Unknown"
        else:
            name = CLASS_NAMES[predicted_index]

        # Draw label
        cv2.putText(frame, f"Name: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)

        if name != "Unknown" and name not in recognized_names and confidence > 0.85:
            print(f"✅ Recognized: {name}")
            log_to_excel(name)
            send_telegram_message(name)
            recognized_names.add(name)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
