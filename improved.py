import cv2
import numpy as np
from pyzbar.pyzbar import decode

# List of authorized names
authorized_names = ['Akshit Madan', 'Mira Singh', 'Aditya Dev', 'Sameer Bhat']

# Open the video file
cap = cv2.VideoCapture("close_qr.mp4")

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break


    scale_percent = 150
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)


    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)


    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    for qrcode in decode(blurred):
        codeData = qrcode.data.decode('utf-8')
        print(f"Detected QR Code Data: {codeData}")


        points = np.array(qrcode.polygon, dtype=np.int32)
        if len(points) == 4:
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img_resized, [points], True, (0, 255, 0), 5)


            x, y, w, h = qrcode.rect

            cv2.putText(img_resized, codeData, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    img_resized = cv2.resize(img_resized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


    cv2.imshow('QR Code Scanner', img_resized)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
