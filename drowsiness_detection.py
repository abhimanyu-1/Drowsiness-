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

# ==========================================================
# TWILIO SMS SETUP
# ==========================================================
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_ACCOUNT_SID = 'YOUR_ACCOUNT_SID'       # Replace with your Twilio SID
    TWILIO_AUTH_TOKEN  = 'YOUR_AUTH_TOKEN'         # Replace with your Twilio Auth Token
    TWILIO_FROM_NUMBER = '+1XXXXXXXXXX'            # Your Twilio phone number
    ALERT_TO_NUMBER    = '+91XXXXXXXXXX'           # Recipient phone number (e.g. family/fleet manager)
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    SMS_ENABLED = True
    print("[INFO] Twilio SMS ready.")
except Exception as e:
    print(f"[WARN] Twilio not configured. SMS disabled. ({e})")
    twilio_client = None
    SMS_ENABLED = False

sms_sent_flags = {
    'drowsiness': False,  # Reset each session
    'high_temp':  False,
}

# ==========================================================
# ARDUINO SERIAL SETUP
# ==========================================================
try:
    import serial
    # Change 'COM7' to the correct port (e.g., 'COM3', 'COM7' on Windows or '/dev/ttyUSB0' on Raspberry Pi)
    ARDUINO_PORT = '/dev/ttyUSB0' if platform.system() != "Windows" else 'COM7'
    BAUD_RATE = 115200
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to reboot after connection
    print(f"[INFO] Arduino connected on {ARDUINO_PORT}")
    HARDWARE_CONNECTED = True
except Exception as e:
    print(f"[WARN] Hardware NOT connected. Running in simulation mode. ({e})")
    arduino = None
    HARDWARE_CONNECTED = False

def send_to_arduino(command: str):
    """Send a single character command to the Arduino over Serial."""
    if HARDWARE_CONNECTED and arduino and arduino.is_open:
        try:
            arduino.write(command.encode())
            print(f"[ARDUINO] Sent: '{command}'")
        except Exception as e:
            print(f"[WARN] Failed to send to Arduino: {e}")
    else:
        print(f"[SIM] Arduino command: '{command}'")

# ==========================================================
# SMS ALERT FUNCTION
# ==========================================================
def send_sms(reason: str):
    """Send an SMS alert using Twilio. 'reason' is 'drowsiness' or 'high_temp'."""
    if not SMS_ENABLED:
        print(f"[SIM] SMS would be sent for reason: {reason}")
        return
    if sms_sent_flags.get(reason):
        return  # Avoid duplicate SMS for the same event

    messages = {
        'drowsiness': (
            "DRIVER ALERT: The driver has been detected drowsy multiple times. "
            "The vehicle is being automatically parked safely at the side of the road. "
            "Please assist the driver immediately."
        ),
        'high_temp': (
            "VEHICLE ALERT: The vehicle's engine temperature is critically high. "
            "The vehicle has been stopped for safety. "
            "Please check the engine and assist immediately."
        ),
    }

    def _send():
        try:
            twilio_client.messages.create(
                body=messages[reason],
                from_=TWILIO_FROM_NUMBER,
                to=ALERT_TO_NUMBER
            )
            sms_sent_flags[reason] = True
            print(f"[SMS] Alert sent successfully for: {reason}")
        except Exception as e:
            print(f"[WARN] SMS failed to send: {e}")

    threading.Thread(target=_send, daemon=True).start()

# ==========================================================
# ARDUINO INCOMING SERIAL READER (catches 'H' = High Temp)
# ==========================================================
def arduino_reader_thread():
    """Background thread: reads incoming serial data from Arduino.
       Triggers SMS if high temperature signal 'H' is received."""
    while True:
        try:
            if HARDWARE_CONNECTED and arduino and arduino.is_open and arduino.in_waiting > 0:
                incoming = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
                if 'H' in incoming:
                    print("[ARDUINO] High temperature signal received!")
                    send_sms('high_temp')
        except Exception:
            pass
        time.sleep(0.5)

threading.Thread(target=arduino_reader_thread, daemon=True).start()

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
                play_warning_sounds('sound files/alarm.mp3', 'sound files/warning.wav')
                alarm_started = True
                response_count = 0

                if count_sleep >= 2:
                    # === 2nd+ DROWSINESS: Escalate to PARKING MODE permanently ===
                    print(f"[EMERGENCY] Drowsiness detected {count_sleep} times! Triggering PARKING MODE.")
                    send_to_arduino('P')
                    unconscious_detected = True  # Prevent 'R' resume signal
                    # === SEND SMS ALERT to family / fleet manager ===
                    send_sms('drowsiness')
                else:
                    # === 1st DROWSINESS: Warn and stop temporarily ===
                    print(f"[INFO] Drowsiness detected (count: {count_sleep}). Waiting for driver response.")
                    send_to_arduino('S')

            if alarm_started:
                response_count += 1
                cv2.putText(frame, "WAKE UP! DON'T SLEEP WHILE DRIVING", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                if response_count >= RESPONSE_FRAMES and not unconscious_detected:
                    unconscious_detected = True
                    print("[EMERGENCY] Possible unconscious driver detected.")
                    print("[EMERGENCY] Triggering PARKING MODE on Arduino...")
                    # === TRIGGER ARDUINO: Send 'P' (PARK VEHICLE) ===
                    send_to_arduino('P')

                if unconscious_detected:
                    cv2.putText(frame, "UNCONSCIOUS DRIVER! PARKING...", (55, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        else:
            # Driver eyes are open - recover if we were alarming
            if alarm_started and not unconscious_detected:
                print("[INFO] Driver responded. Resuming vehicle.")
                # === TRIGGER ARDUINO: Send 'R' (RESUME / DRIVER AWAKE) ===
                send_to_arduino('R')
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
            play_warning_sounds('sound files/alarm.mp3', 'sound files/warning_yawn.mp3')
            # === TRIGGER ARDUINO: Send 'S' on repeated yawning (warning) ===
            if count_yawn % 3 == 0:  # Only trigger after every 3rd yawn to avoid spamming
                print("[INFO] Repeated yawning detected. Sending sleep warning to Arduino.")
                send_to_arduino('S')

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
                play_warning_sounds('sound files/Mobwarn.mp3')

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break