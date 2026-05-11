# Import the necessary packages
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from EAR_calculator import *
from imutils.video import VideoStream
from matplotlib import style
import imutils
import mediapipe as mp
import time
import argparse
import os
# Fix for Raspberry Pi Wayland GUI crash
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import pygame
import time
import platform
from scipy.spatial import distance as dist
import os
import numpy as np
import pandas as pd
import threading

style.use('fivethirtyeight')

# Creating the dataset
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


pygame.mixer.init()

def play_warning_sounds(*sound_paths):
    global alarm_playing

    if alarm_playing:
        return

    def play_all():
        global alarm_playing
        try:
            for sound_path in sound_paths:
                if platform.system() == "Linux":
                    # Use native mpg123 player on Raspberry Pi
                    os.system(f"mpg123 -q '{sound_path}'")
                else:
                    # Use Pygame on Windows
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
        finally:
            alarm_playing = False

    alarm_playing = True
    threading.Thread(target=play_all, daemon=True).start()

# all eye and mouth aspect ratio with time
ear_list = []
mar_list = []
ts = []

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether raspberry pi camera shall be used or not")
args = vars(ap.parse_args())

EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 10
RESPONSE_FRAMES = 100
MAR_THRESHOLD = 0.6

BLINK_COUNT = 0
FRAME_COUNT = 0
mobile_usage_count = 0
response_count = 0
alarm_started = False
unconscious_detected = False
alarm_playing = False

print("[INFO] Loading the predictor.....")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MP_MAR_INDICES = [191, 13, 415, 312, 14, 324, 78, 308]

print("[INFO] Loading Camera.....")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2)

assure_path_exists("dataset/")
count_sleep = 0
count_yawn = 0

# Import YOLO for cell phone detection
net = cv2.dnn.readNet('dnn_model/yolov4-tiny.weights', 'dnn_model/yolov4-tiny.cfg')
classes = []
with open("dnn_model/coco.txt", "r") as f:
    classes = f.read().splitlines()
cell_phone_class_index = classes.index("cell phone")

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    frame = vs.read()

    if frame is None:
        continue

    cv2.putText(frame, "PRESS 'Q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    frame = imutils.resize(frame, width=640)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h_frame, w_frame, _ = frame.shape
            shape = np.array([[int(pt.x * w_frame), int(pt.y * h_frame)] for pt in face_landmarks.landmark])

            x_min = np.min(shape[:, 0])
            y_min = np.min(shape[:, 1])
            x_max = np.max(shape[:, 0])
            y_max = np.max(shape[:, 1])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "Driver", (x_min - 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            leftEye = shape[LEFT_EYE_INDICES]
            rightEye = shape[RIGHT_EYE_INDICES]
            mouth = shape[MP_MAR_INDICES]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        EAR = (leftEAR + rightEAR) / 2.0

        ear_list.append(EAR)
        ts.append(dt.datetime.now().strftime('%H:%M'))
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        MAR = mp_mouth_aspect_ratio(mouth)
        mar_list.append(MAR)

        # --- DROWSINESS CHECK (EYES) ---
        if EAR < EAR_THRESHOLD:
            FRAME_COUNT += 1
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if FRAME_COUNT >= CONSECUTIVE_FRAMES and not alarm_started:
                count_sleep += 1

                cv2.putText(frame, "DROWSINESS ALERT! (EYES CLOSED)", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                play_warning_sounds('sound files/alarm.wav', 'sound files/warning.wav')
                print("[INFO] Drowsiness detected. Waiting for driver response.")
                alarm_started = True
                response_count = 0

            if alarm_started:
                response_count += 1
                cv2.putText(frame, "WAKE UP! DON'T SLEEP WHILE DRIVING", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                if response_count >= RESPONSE_FRAMES and not unconscious_detected:
                    unconscious_detected = True
                    print("[EMERGENCY] Possible unconscious driver detected.")
                    print("[EMERGENCY] Slowing vehicle...")
                    print("[EMERGENCY] Stopping vehicle...")

                if unconscious_detected:
                    cv2.putText(frame, "UNCONSCIOUS DRIVER!", (55, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        else:
            FRAME_COUNT = 0
            response_count = 0
            alarm_started = False
            unconscious_detected = False

        # --- YAWN CHECK ---
        if MAR > MAR_THRESHOLD:
            count_yawn += 1

            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
            cv2.putText(frame, "DROWSINESS ALERT! (YAWN)", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
            play_warning_sounds('sound files/alarm.wav', 'sound files/warning_yawn.wav')

    # --- CELL PHONE DETECTION ---
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2 and class_id == cell_phone_class_index:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

            if label == "cell phone":
                mobile_usage_count += 1

                cv2.imwrite("dataset/frame_mobile_usage%d.jpg" % mobile_usage_count, frame)
                play_warning_sounds('sound files/mobwarn.wav')

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break