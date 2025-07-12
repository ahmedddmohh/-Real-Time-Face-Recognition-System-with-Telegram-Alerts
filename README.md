# -Real-Time-Face-Recognition-System-with-Telegram-Alerts
A real-time face recognition system using a custom CNN and KNN classifier. Sends Telegram alerts when someone is detected and logs every recognition in an Excel sheet. Built with OpenCV, ArcFace, and Python.

# Real-Time Face Recognition System with Telegram Alerts

This project implements a real-time face recognition system using a custom-trained CNN model, KNN classifier, and OpenCV. It integrates a Telegram bot to notify the user upon detection and logs every recognized face in an Excel sheet with a timestamp.

## 🔧 Features
- Custom CNN-based face recognition model
- Real-time webcam detection using OpenCV
- KNN classifier with ArcFace embeddings for improved accuracy
- Telegram bot sends a message when any known face is detected
- Excel logging with name, date, and time for each detection

## 🛠️ Technologies Used
- Python
- OpenCV
- FaceNet / ArcFace
- KNN Classifier (scikit-learn)
- Pandas + openpyxl (for Excel logging)
- Telegram Bot API

## ▶️ How to Run
1. Clone the repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Set up your Telegram bot and paste your token in the script
4. Run the system:  
   `python main.py`

## 📂 Folder Structure
- `model/`: CNN model and embeddings
- `images/`: Dataset (one folder per person)
- `logs/`: Excel log file
- `main.py`: Main script

#

## 🧠 Future Improvements
- Web interface with live video feed
- Face registration through UI
- Deploying the model in browser using TensorFlow.js

