
import cv2
import platform

system = platform.system()

if system == "Windows":
    backend = cv2.CAP_DSHOW

elif system == "Linux":
    backend = cv2.CAP_V4L2

else:
    backend = cv2.CAP_ANY


camera = cv2.VideoCapture(0, backend)

if not camera.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

print("Camera connected successfully!")

while True:

    success, frame = camera.read()

    if not success:
        print("Cannot read camera frame")
        break

    cv2.imshow("Attendance Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
