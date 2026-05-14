import cv2
import time
import winsound
import os

print("STARTING DROWSINESS DETECTION...")

# PATH SETUP (VERY IMPORTANT)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

face_path = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(DATA_DIR, "haarcascade_eye.xml")

# LOAD CASCADES

face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)

if face_cascade.empty() or eye_cascade.empty():
    print("ERROR: Haarcascade files not loaded")
    exit()

# CAMERA SETUP

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera not opening")
    exit()

print("CAMERA OPENED")


# DROWSINESS PARAMETERS

EYE_CLOSED_THRESHOLD = 1.0  # seconds
eye_closed_start = None
alarm_on = False


# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        print("FRAME NOT RECEIVED")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        eyes_detected = len(eyes)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

   
    # DROWSINESS LOGIC
    
    if len(faces) > 0 and eyes_detected == 0:
        if eye_closed_start is None:
            eye_closed_start = time.time()
        else:
            elapsed = time.time() - eye_closed_start
            if elapsed >= EYE_CLOSED_THRESHOLD:
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )
                if not alarm_on:
                    winsound.Beep(1000, 1200)
                    alarm_on = True
    else:
        eye_closed_start = None
        alarm_on = False

    
    # DISPLAY
   
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Q PRESSED - EXITING")
        break


# CLEANUP

cap.release()
cv2.destroyAllWindows()
print("PROGRAM ENDED")
