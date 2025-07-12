# This file contains your trained model logic
from deepface import DeepFace
import cv2
import os
import numpy as np
import pickle

# Assuming you already have the training and KNN model ready
data_dir = "Test_Subjects"
model_name = "Facenet"
detector_backend = "opencv"

print("[INFO] Loading images and training embeddings...")
embeddings = []
labels = []

for person in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        try:
            embedding_obj = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
                normalization="base"
            )
            embeddings.append(embedding_obj[0]["embedding"])
            labels.append(person)
        except:
            print(f"[WARN] Could not process {img_path}")

# Train simple KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(embeddings, labels)

print("[INFO] Recognition model is ready.")

# Inference function
def recognize_face(frame):
    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            target_size=(160, 160),
            detector_backend=detector_backend,
            enforce_detection=False
        )
        if not faces:
            return "No Face", 0.0

        for face in faces:
            face_img = face["face"]
            embedding = DeepFace.represent(
                img_path=face_img,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
                normalization="base"
            )[0]['embedding']

            prediction = knn.predict([embedding])[0]
            dist = knn.kneighbors([embedding])[0][0][0]
            confidence = max(0.0, 1 - dist / 20)  # Adjust scale
            if confidence > 0.5:
                return prediction, round(confidence, 2)
            else:
                return "Unknown", round(confidence, 2)
    except Exception as e:
        return str(e), 0.0