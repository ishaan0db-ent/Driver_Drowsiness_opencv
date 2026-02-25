import cv2

print("OPENING CAMERA...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("FAILED TO OPEN CAMERA")
    exit()

print("CAMERA OPENED — PRESS Q TO EXIT")

while True:
    ret, frame = cap.read()

    if not ret:
        print("FRAME NOT READ")
        break

    cv2.imshow("LIVE CAMERA TEST", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Q PRESSED — EXITING")
        break

cap.release()
cv2.destroyAllWindows()
print("DONE")
