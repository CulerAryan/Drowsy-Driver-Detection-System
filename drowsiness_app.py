import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer
import streamlit as st
import tempfile
import time

# Initialize pygame mixer for alert sound
mixer.init()
mixer.music.load(r"C:\Users\aryan\OneDrive\Desktop\music.wav")

# Define EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load dlib models
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Users\aryan\Downloads\shape_predictor_68_face_landmarks.dat")

# Constants
thresh = 0.25
frame_check = 20

# Streamlit UI
st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title("😴 Drowsiness Detection System")
st.markdown("**Developed with OpenCV, Dlib, and Streamlit**")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
alert_placeholder = st.empty()
status = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    flag = 0
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Unable to access webcam.")
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftHull = cv2.convexHull(leftEye)
            rightHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                status.text(f"Eyes closed for {flag} frames")
                if flag >= frame_check:
                    alert_placeholder.warning("🚨 DROWSINESS DETECTED!")
                    cv2.putText(frame, "***** ALERT *****", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play(-1)
            else:
                flag = 0
                mixer.music.stop()
                alert_placeholder.empty()

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.03)

    cap.release()
    mixer.music.stop()
else:
    st.info("✅ Click the checkbox above to start the webcam.")
