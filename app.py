import streamlit as st
import cv2


st.write("""
This project detects driver drowsiness in real time using
computer vision and OpenCV.
""")

closed_eyes = 0 
st.title("Driver Drowsiness Detection System")
st.sidebar.title("System Status")
st.sidebar.success("Camera Active")
st.sidebar.write(f"Closed Eye Frames: {closed_eyes}")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

face_cascade = cv2.CascadeClassifier(
    "data/haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    "data/haarcascade_eye.xml"
)

camera = cv2.VideoCapture(0)

closed_eyes = 0

while run:

    success, frame = camera.read()

    if not success:
        st.write("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            closed_eyes += 1
        else:
            closed_eyes = 0

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex+ew, ey+eh),
                (255,0,0),
                2
            )

    if closed_eyes > 20:
        
        
        


        cv2.putText(
            frame,
            "DROWSINESS ALERT!",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )
        st.error("DROWSINESS DETECTED!")
    else:
        st.success("Driver Alert")    

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)

camera.release()