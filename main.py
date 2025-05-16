import cv2
import time
import numpy as np

COLORS = [(0,255,255), (255,255,0), (0,255,0), (255,0,0)]

class_names = []
with open("cocopt.names", "r", encoding='utf-8') as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture("animals.mp4")

net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)

model.setInputParams(size = (416,416), scale = 1/255)

while True:
    x, frame = cap.read()
    start = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.3)
    end = time.time()
    
    for (classid, scores, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid)%len(COLORS)]
        label = f"{class_names[classid]}:{scores}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0],box[1] -10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

    fps_label = f"FPS: {round((1.0/(end-start)),2)}"
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    frame = cv2.resize(frame, (1429, 768))
    cv2.imshow("detections", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()