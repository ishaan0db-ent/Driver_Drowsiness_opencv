import streamlit as st
import cv2
import mediapipe as mp
import math
import winsound

# Page Configuration
st.sidebar.title("System Information")

st.sidebar.write(
    "Model: MediaPipe FaceMesh"
)

st.sidebar.write(
    "Detection: Eye Aspect Ratio"
)

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="centered"
)

st.title(" Driver Drowsiness Detection System")

st.write(
    "Real-time drowsiness detection using "
    "MediaPipe FaceMesh and Eye Aspect Ratio (EAR)"
)


# Streamlit Checkbox to Start/Stop Camera

run = st.checkbox("Start Camera")


# Streamlit Placeholders for EAR and Alerts

FRAME_WINDOW = st.image([])

ear_placeholder = st.empty()

alert_placeholder = st.empty()

# MediaPipe FaceMesh Setup

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Camera Setup

camera = cv2.VideoCapture(0)

# Lower resolution = smoother performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


# Eye Landmark Indices for EAR Calculation

LEFT_EYE = [33, 160, 158, 133, 153, 144]

RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# Drowsiness Detection Parameters
closed_frames = 0
alarm_on = False 

EAR_THRESHOLD = 0.20

FRAME_THRESHOLD = 15


# Distance calculation between two points
def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# Calculation of EAR 

def calculate_EAR(eye, landmarks):

    vertical1 = distance(
        landmarks[eye[1]],
        landmarks[eye[5]]
    )

    vertical2 = distance(
        landmarks[eye[2]],
        landmarks[eye[4]]
    )

    horizontal = distance(
        landmarks[eye[0]],
        landmarks[eye[3]]
    )

    ear = (vertical1 + vertical2) / (
        2.0 * horizontal
    )

    return ear


# Main loop

while run:

    success, frame = camera.read()

    if not success:

        st.error("Camera not working")

        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process with MediaPipe
    results = face_mesh.process(rgb_frame)

    
    # Face Landmark Detection

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            
            # Draw eye landmarks on the frame
            
            for idx in LEFT_EYE + RIGHT_EYE:

                x = int(
                    landmarks[idx].x * frame.shape[1]
                )

                y = int(
                    landmarks[idx].y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0,255,0),
                    -1
                )

            
            # EAR Calculation
           
            left_ear = calculate_EAR(
                LEFT_EYE,
                landmarks
            )

            right_ear = calculate_EAR(
                RIGHT_EYE,
                landmarks
            )

            avg_ear = (
                left_ear + right_ear
            ) / 2

            
            # Drowsiness Logic
            if avg_ear < EAR_THRESHOLD:

                closed_frames += 1

            else:

                closed_frames = 0

            
            # Alert if eyes have been closed for too long
            
            if closed_frames > FRAME_THRESHOLD:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                alert_placeholder.error(
                    " DROWSINESS DETECTED!"
                )
                #Play Sound once 
                if not alarm_on:
                   winsound.Beep(1000, 500)
                alarm_on = True

            else:

                alert_placeholder.success(
                    " Driver Alert"
                )

            
            # Show the EAR value on the frame and in Streamlit
           
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2
            )

            ear_placeholder.info(
                f"EAR Value: {avg_ear:.2f}"
            )

    
    # Display the frame
    
    FRAME_WINDOW.image(
        frame,
        channels="BGR"
    )


# Release Camera

camera.release()