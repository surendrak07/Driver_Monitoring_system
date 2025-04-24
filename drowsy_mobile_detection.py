#1
# import cv2
# import dlib
# import numpy as np
# import time
# from scipy.spatial import distance
# from tensorflow.keras.models import load_model
# import pygame
# from ultralytics import YOLO  

# phone_model = YOLO("yolov8n.pt")  

# drowsiness_model = load_model("drowsiness_detection_model.h5")

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# # Load face detector and facial landmarks predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Define threshold values
# EAR_THRESHOLD = 0.25
# FRAME_CHECK = 20  # Number of frames to confirm drowsiness

# drowsy_frames = 0

# # Initialize sound alerts
# pygame.mixer.init()
# alarm_sound = pygame.mixer.Sound("alarm.wav")

# def detect_drowsiness(frame):
#     global drowsy_frames
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     for face in faces:
#         landmarks = predictor(gray, face)
#         left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
#         right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         ear = (left_ear + right_ear) / 2.0
        
#         if ear < EAR_THRESHOLD:
#             drowsy_frames += 1
#             if drowsy_frames >= FRAME_CHECK:
#                 cv2.putText(frame, "DROWSY ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 alarm_sound.play()
#         else:
#             drowsy_frames = 0

#     return frame

# def detect_mobile_usage(frame):
#     results = phone_model(frame)
#     for result in results:
#         for box in result.boxes:
#             cls = int(box.cls[0])
#             if cls == 67:  # Class ID for cell phone in COCO dataset
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, "MOBILE DETECTED!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 alarm_sound.play()
#     return frame

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = detect_drowsiness(frame)
#     frame = detect_mobile_usage(frame)
#     cv2.imshow("Driver Monitoring System", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




#2

# import cv2
# import numpy as np
# import time
# from scipy.spatial import distance
# from tensorflow.keras.models import load_model
# import pygame
# from ultralytics import YOLO  
# import mediapipe as mp  # Replacing dlib with mediapipe

# phone_model = YOLO("yolov8n.pt")  

# # drowsiness_model = load_model("drowsiness_detection_model.h5")
# drowsiness_model = load_model(r"D:\main project\datasets\drowsiness_detection_model.h5")

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# # Define threshold values
# EAR_THRESHOLD = 0.25
# FRAME_CHECK = 20  # Number of frames to confirm drowsiness

# drowsy_frames = 0

# # Initialize sound alerts
# pygame.mixer.init()
# alarm_sound = pygame.mixer.Sound(r"D:\main project\datasets\alarm.wav")

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# def detect_drowsiness(frame):
#     global drowsy_frames
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             left_eye = [(face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]) for i in range(362, 368)]
#             right_eye = [(face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]) for i in range(33, 39)]
            
#             left_ear = eye_aspect_ratio(left_eye)
#             right_ear = eye_aspect_ratio(right_eye)
#             ear = (left_ear + right_ear) / 2.0
            
#             if ear < EAR_THRESHOLD:
#                 drowsy_frames += 1
#                 if drowsy_frames >= FRAME_CHECK:
#                     cv2.putText(frame, "DROWSY ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                     alarm_sound.play()
#             else:
#                 drowsy_frames = 0

#     return frame

# def detect_mobile_usage(frame):
#     results = phone_model(frame)
#     for result in results:
#         for box in result.boxes:
#             cls = int(box.cls[0])
#             if cls == 67:  # Class ID for cell phone in COCO dataset
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, "MOBILE DETECTED!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 alarm_sound.play()
#     return frame

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = detect_drowsiness(frame)
#     frame = detect_mobile_usage(frame)
#     cv2.imshow("Driver Monitoring System", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



#3
import cv2
import numpy as np
import time
from scipy.spatial import distance
from tensorflow.keras.models import load_model
import pygame
from ultralytics import YOLO  
import mediapipe as mp

# Load YOLO model for phone detection
phone_model = YOLO("yolov8n.pt")  

# Load drowsiness detection model

drowsiness_model = load_model(r"D:\main project\main project\code\drowsiness_detection_model.h5")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Define threshold values
EAR_THRESHOLD = 0.25
FRAME_CHECK = 20  # Number of frames to confirm drowsiness
drowsy_frames = 0

# Initialize sound alerts
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(r"D:\main project\main project\code\alarm.wav")

# Check if the alarm is playing
def play_alarm():
    if not pygame.mixer.get_busy():
        alarm_sound.play()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(frame):
    global drowsy_frames
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert only once
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]  # Use only the first detected face

        left_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                     int(face_landmarks.landmark[i].y * frame.shape[0])) for i in [362, 385, 387, 263, 373, 380]]
        right_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                      int(face_landmarks.landmark[i].y * frame.shape[0])) for i in [33, 160, 158, 133, 153, 144]]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < EAR_THRESHOLD:
            drowsy_frames += 1
            if drowsy_frames >= FRAME_CHECK:
                cv2.putText(frame, "DROWSY ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_alarm()
        else:
            drowsy_frames = 0  # Reset counter when eyes open

    return frame

def detect_mobile_usage(frame):
    results = phone_model(frame, verbose=False)
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == 67 and conf > 0.5:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "MOBILE DETECTED!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                play_alarm()
                
    return frame

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_drowsiness(frame)
    frame = detect_mobile_usage(frame)
    
    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
