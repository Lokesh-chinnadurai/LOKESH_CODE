import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os

def detect_and_display_elephants(video_path, yolo_model_path, output_video_path):
    model = YOLO(yolo_model_path)
    zone = [(10, 12), (1911, 10), (1913, 1064), (8, 1060)]

    # Open the video
    cap = cv2.VideoCapture(video_path)
    output_folder = 'lab_factory1'
    os.makedirs(output_folder, exist_ok=True)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        yolomodel = model(frame)

        list1 = []
        for output in yolomodel:
            for detection in output.boxes:
                confi = detection.conf[0]

                class_name = model.names[0]

                if confi > 0.40 and class_name == "person":
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {confi:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    w, h = x2 - x1, y2 - y1
                    result = cv2.pointPolygonTest(np.array(zone, np.int32), (cx, cy), False)
                    if result >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                        cv2.putText(frame, f'person', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        list1.append(cx)

        person = len(list1)
        cv2.polylines(frame, [np.array(zone, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, f'Person_count:{person}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        re = cv2.resize(frame, (800, 800))
        cv2.imshow("frames", re)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_path = r'D:\vChanel\truck\labzone2video1.avi'
yolo_model_path = r"D:\vChanel\truck\re_trained_person.pt"
output_video_path = r'lab2videooutput.mp4'

elephant_count = detect_and_display_elephants(video_path, yolo_model_path, output_video_path)




