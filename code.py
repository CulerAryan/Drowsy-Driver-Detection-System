import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer

# Initialize mixer and load alert sound
mixer.init()
mixer.music.load(r"C:\Users\aryan\OneDrive\Desktop\music.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold and frame check limits
thresh = 0.25
frame_check = 20
flag = 0

# Get facial landmark indexes for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

# Initialize face detector and landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
    r"C:\Users\aryan\Downloads\shape_predictor_68_face_landmarks.dat"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        ear = (leftEar + rightEar) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check for drowsiness
        if ear < thresh:
            flag += 1
            print(f"Eyes closed frames: {flag}")

            if flag >= frame_check:
                cv2.putText(frame, "***** ALERT *****", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Play alert sound continuously until eyes open
                if not mixer.music.get_busy():
                    mixer.music.play(-1)  # -1 means loop indefinitely
        else:
            flag = 0
            mixer.music.stop()

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
mixer.music.stop()
