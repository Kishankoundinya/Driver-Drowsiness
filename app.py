# app.py - Main Streamlit Application
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import threading
import queue
import tempfile

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
    if os.path.exists(ALARM_SOUND):
        os.system(f"afplay {ALARM_SOUND} &")  # macOS sound
        # For Windows: os.system(f"start {ALARM_SOUND}")
        # For Linux: os.system(f"aplay {ALARM_SOUND}")

class DrowsinessDetector:
    def __init__(self):
        self.start_time = None
        self.alarm_on = False
        self.yawn_count = 0
        self.drowsy_count = 0
        self.ear_history = []
        self.mar_history = []
        self.status_history = []
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        
    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        
        status_text = "Awake"
        ear_value = 0
        mar_value = 0
        
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
                ear_value = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2
                mar_value = calculate_MAR(mouth)
                
                # Store history
                self.ear_history.append(ear_value)
                self.mar_history.append(mar_value)
                if len(self.ear_history) > 100:
                    self.ear_history.pop(0)
                    self.mar_history.pop(0)
                
                # ================= DROWSINESS =================
                if ear_value < EAR_THRESHOLD:
                    if self.start_time is None:
                        self.start_time = time.time()
                    
                    elapsed = time.time() - self.start_time
                    
                    if elapsed > CLOSED_TIME:
                        status_text = "DROWSY 😴"
                        if len(self.status_history) == 0 or self.status_history[-1] != "DROWSY":
                            self.drowsy_count += 1
                        
                        cv2.putText(frame, "DROWSY!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                        if not self.alarm_on:
                            self.alarm_on = True
                            play_alarm()
                else:
                    self.start_time = None
                    self.alarm_on = False
                
                # ================= YAWN =================
                if mar_value > MAR_THRESHOLD:
                    if len(self.status_history) == 0 or self.status_history[-1] != "YAWN":
                        status_text = "YAWNING 😮"
                        self.yawn_count += 1
                        cv2.putText(frame, "YAWNING!", (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        
        # Display status on frame
        cv2.putText(frame, f"Status: {status_text}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Display metrics
        cv2.putText(frame, f"EAR: {ear_value:.2f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar_value:.2f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, status_text, ear_value, mar_value

# ================= STREAMLIT UI =================
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
    }
    .title-text {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-text"><h1>🚗 Driver Drowsiness Detection System</h1><p>Real-time monitoring for driver safety</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    ear_threshold = st.slider("Eye Aspect Ratio (EAR) Threshold", 0.15, 0.35, EAR_THRESHOLD, 0.01)
    mar_threshold = st.slider("Mouth Aspect Ratio (MAR) Threshold", 0.4, 0.8, MAR_THRESHOLD, 0.05)
    closed_time = st.slider("Eye Closure Time (seconds)", 1, 5, CLOSED_TIME, 1)
    
    st.markdown("---")
    st.header("📊 Session Statistics")
    stats_placeholder = st.empty()
    
    st.markdown("---")
    st.header("ℹ️ How it works")
    st.info("""
    - **EAR (Eye Aspect Ratio)**: Detects when eyes are closed
    - **MAR (Mouth Aspect Ratio)**: Detects yawning
    - **Alert**: Continuous eye closure triggers alarm
    - **Report**: Session data saved automatically
    """)
    
    if st.button("📥 Download Report", use_container_width=True):
        if os.path.exists("report.csv"):
            with open("report.csv", "rb") as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name=f"drowsiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Live Video Feed")
    video_placeholder = st.empty()
    warning_placeholder = st.empty()

with col2:
    st.subheader("📈 Real-time Metrics")
    ear_placeholder = st.empty()
    mar_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("📊 Live Charts")
    chart_placeholder = st.empty()

# Initialize detector
detector = DrowsinessDetector()

# Update thresholds dynamically
EAR_THRESHOLD = ear_threshold
MAR_THRESHOLD = mar_threshold
CLOSED_TIME = closed_time

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Cannot access camera. Please check your camera connection.")
    st.stop()

# Create stop button
stop_button = st.button("🛑 Stop Monitoring", use_container_width=True)

# Main loop
while not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame")
        break
    
    # Process frame
    processed_frame, status, ear, mar = detector.process_frame(frame)
    
    # Update dynamic thresholds
    detector.start_time = None if ear >= ear_threshold else detector.start_time
    
    # Convert BGR to RGB for display
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Display video
    video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
    
    # Show warning if drowsy
    if status == "DROWSY 😴":
        warning_placeholder.error("⚠️ DROWSINESS DETECTED! PLEASE TAKE A BREAK! ⚠️")
    elif status == "YAWNING 😮":
        warning_placeholder.warning("😮 Yawn detected - you might be getting tired")
    else:
        warning_placeholder.success("✅ Driver is alert and attentive")
    
    # Update metrics
    with col2:
        ear_percentage = (1 - min(1, ear/ear_threshold)) * 100 if ear < ear_threshold else 0
        mar_percentage = min(100, (mar/mar_threshold) * 100) if mar > 0 else 0
        
        ear_placeholder.metric(
            "Eye Aspect Ratio (EAR)",
            f"{ear:.3f}",
            delta=f"{'⚠️ Low' if ear < ear_threshold else 'Normal'}",
            delta_color="inverse"
        )
        
        mar_placeholder.metric(
            "Mouth Aspect Ratio (MAR)",
            f"{mar:.3f}",
            delta=f"{'Yawning' if mar > mar_threshold else 'Normal'}",
            delta_color="inverse"
        )
    
    # Update charts
    if len(detector.ear_history) > 1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=detector.ear_history[-50:],
            mode='lines',
            name='EAR',
            line=dict(color='#00ff87', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=detector.mar_history[-50:],
            mode='lines',
            name='MAR',
            line=dict(color='#ff6b6b', width=2)
        ))
        
        fig.add_hline(y=ear_threshold, line_dash="dash", line_color="red", 
                     annotation_text="EAR Threshold")
        fig.add_hline(y=mar_threshold, line_dash="dash", line_color="orange",
                     annotation_text="MAR Threshold")
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Recent Frames",
            yaxis_title="Ratio Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Update sidebar stats
    with st.sidebar:
        stats_placeholder.metric("😴 Drowsy Events", detector.drowsy_count)
        stats_placeholder.metric("😮 Yawns Detected", detector.yawn_count)
        stats_placeholder.metric("🕒 Session Duration", f"{len(detector.ear_history)*0.033:.0f}s")
    
    # Short delay for smooth display
    time.sleep(0.033)

# Clean up
cap.release()
cv2.destroyAllWindows()

# Save report
with open("report.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Drowsy Events", "Yawns", "Session Duration (s)"])
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        detector.drowsy_count,
        detector.yawn_count,
        len(detector.ear_history) * 0.033
    ])

st.success("✅ Session ended! Report saved successfully!")
st.balloons()

# Display final report
st.markdown("---")
st.subheader("📋 Session Report")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Drowsy Events", detector.drowsy_count)
with col2:
    st.metric("Total Yawns", detector.yawn_count)
with col3:
    st.metric("Session Duration", f"{len(detector.ear_history)*0.033:.1f} seconds")

# Show data preview
if os.path.exists("report.csv"):
    df = pd.read_csv("report.csv")
    st.dataframe(df, use_container_width=True)