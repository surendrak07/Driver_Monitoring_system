
import cv2
import numpy as np
import time
import streamlit as st
from scipy.spatial import distance
from tensorflow.keras.models import load_model
import pygame
from ultralytics import YOLO  
import mediapipe as mp
import threading

def stop_alarm_after_delay(channel, delay=5):
    time.sleep(delay)
    channel.stop()

def play_alarm():
    global last_alarm_time
    current_time = time.time()

    if not pygame.mixer.get_busy() and current_time - last_alarm_time > 4:
        last_alarm_time = current_time
        channel = alarm_sound.play()
        threading.Thread(target=stop_alarm_after_delay, args=(channel,), daemon=True).start()


# Load models
phone_model = YOLO("yolov8n.pt")

drowsiness_model = load_model(r"D:\main project\main project\code\drowsiness_detection_model.h5")

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Thresholds
EAR_THRESHOLD = 0.25
FRAME_CHECK = 20
drowsy_frames = 0
frame_count = 0
last_alarm_time = 0

# Initialize sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(r"D:\main project\main project\code\alarm.wav")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(frame):
    global drowsy_frames
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_eye = [(int(landmarks.landmark[i].x * frame.shape[1]),
                     int(landmarks.landmark[i].y * frame.shape[0])) for i in [362, 385, 387, 263, 373, 380]]
        right_eye = [(int(landmarks.landmark[i].x * frame.shape[1]),
                      int(landmarks.landmark[i].y * frame.shape[0])) for i in [33, 160, 158, 133, 153, 144]]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            drowsy_frames += 1
            if drowsy_frames >= FRAME_CHECK:
                cv2.putText(frame, "DROWSY ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_alarm()
        else:
            drowsy_frames = 0
    return frame

def detect_mobile_usage(frame):
    small_frame = cv2.resize(frame, (320, 240))
    results = phone_model(small_frame, verbose=False)
    h_ratio = frame.shape[0] / 240
    w_ratio = frame.shape[1] / 320

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 67 and conf > 0.5:  # 67 = phone
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * w_ratio)
                x2 = int(x2 * w_ratio)
                y1 = int(y1 * h_ratio)
                y2 = int(y2 * h_ratio)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "MOBILE DETECTED!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                play_alarm()
    return frame

def main():
    st.title("🚗 Driver Monitoring System")
    st.sidebar.header("⚙️ Settings")
    start_detection = st.sidebar.button("Start Detection")
    stop_detection = st.sidebar.button("Stop Detection")

    if start_detection:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        stframe = st.empty()

        global frame_count
        fps_start_time = time.time()
        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_detection:
                break

            frame_counter += 1
            frame_count += 1

            # Only detect mobile every 5 frames to reduce lag
            frame = detect_drowsiness(frame)
            if frame_count % 5 == 0:
                frame = detect_mobile_usage(frame)

            # Show FPS
            elapsed = time.time() - fps_start_time
            if elapsed > 1:
                fps = frame_counter / elapsed
                fps_text = f"FPS: {fps:.2f}"
                fps_start_time = time.time()
                frame_counter = 0
            else:
                fps_text = "Monitoring..."

            cv2.putText(frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
