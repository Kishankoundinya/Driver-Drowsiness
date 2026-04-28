import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv

# ================= SETTINGS =================
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CLOSED_TIME = 2  # seconds

ALARM_SOUND = "alarm.mp3"

# ================= INIT =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

cap = cv2.VideoCapture(0)

# ================= VARIABLES =================
start_time = None
alarm_on = False
yawn_count = 0
drowsy_count = 0

# ================= FUNCTIONS =================
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_EAR(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth):
    return euclidean(mouth[0], mouth[1]) / euclidean(mouth[2], mouth[3])

def play_alarm():
    os.system(f"afplay {ALARM_SOUND} &")  # macOS sound

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status_text = "Awake"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            left_eye, right_eye, mouth = [], [], []
            face_points = []

            # Collect all face points
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                face_points.append((x, y))

            # 🔲 FACE BOX
            x_coords = [p[0] for p in face_points]
            y_coords = [p[1] for p in face_points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            padding = 10
            cv2.rectangle(frame,
                          (x_min - padding, y_min - padding),
                          (x_max + padding, y_max + padding),
                          (255, 255, 0), 2)

            # Extract eyes
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))

            # Extract mouth
            for idx in MOUTH:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                mouth.append((x, y))

            # 🔲 LEFT EYE BOX
            lx = [p[0] for p in left_eye]
            ly = [p[1] for p in left_eye]
            cv2.rectangle(frame, (min(lx), min(ly)), (max(lx), max(ly)), (0, 255, 0), 2)

            # 🔲 RIGHT EYE BOX
            rx = [p[0] for p in right_eye]
            ry = [p[1] for p in right_eye]
            cv2.rectangle(frame, (min(rx), min(ry)), (max(rx), max(ry)), (0, 255, 0), 2)

            # 🔲 MOUTH BOX
            mx = [p[0] for p in mouth]
            my = [p[1] for p in mouth]
            cv2.rectangle(frame, (min(mx), min(my)), (max(mx), max(my)), (255, 0, 0), 2)

            # Calculations
            EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2
            MAR = calculate_MAR(mouth)

            # ================= DROWSINESS =================
            if EAR < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()

                elapsed = time.time() - start_time

                if elapsed > CLOSED_TIME:
                    status_text = "DROWSY 😴"
                    drowsy_count += 1

                    cv2.putText(frame, "DROWSY!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    if not alarm_on:
                        alarm_on = True
                        play_alarm()

            else:
                start_time = None
                alarm_on = False

            # ================= YAWN =================
            if MAR > MAR_THRESHOLD:
                status_text = "YAWNING 😮"
                yawn_count += 1

                cv2.putText(frame, "YAWNING!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Display status
    cv2.putText(frame, f"Status: {status_text}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# ================= SAVE REPORT =================
with open("report.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Drowsy Events", "Yawns"])
    writer.writerow([drowsy_count, yawn_count])

cap.release()
cv2.destroyAllWindows()