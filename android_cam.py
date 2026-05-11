import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import requests
import numpy as np
from EAR_calculator import *

from imutils.video import VideoStream 
import imutils 
import mediapipe as mp
import time 
import argparse 
import cv2 
import pandas as pd
import csv
import pygame
import threading
from scipy.spatial import distance as dist
import os 
from datetime import datetime

# Creating the dataset 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#all eye  and mouth aspect ratio with time
pygame.mixer.init()
alarm_playing = False

def play_warning_sounds(*sound_paths):
    global alarm_playing
    if alarm_playing:
        return
    def play_all():
        global alarm_playing
        try:
            for sound_path in sound_paths:
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        finally:
            alarm_playing = False
    alarm_playing = True
    threading.Thread(target=play_all, daemon=True).start()

ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
ts=[]
total_ts=[]

url = "http://192.168.64.134:8080/shot.jpg"

# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", required = True, help = "path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type = int, default = -1, help = "whether raspberry pi camera shall be used or not")
args = vars(ap.parse_args())

# Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
EAR_THRESHOLD = 0.2
# Declare another costant to hold the consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 5
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 0.6

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0 
# initialize moble use count
mobile_usage_count = 0

# Now, intialize the mediapipe face mesh
print("[INFO]Loading the predictor.....")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MP_MAR_INDICES = [191, 13, 415, 312, 14, 324, 78, 308]

# Now start the video stream and allow the camera to warm-up
print("[INFO]Loading Camera.....")
time.sleep(2) 

assure_path_exists("dataset_phonecam/")
count_sleep = 0
count_yawn = 0 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

# Import OpenCV and YOLO for cell phone detection
net = cv2.dnn.readNet('dnn_model/yolov4-tiny.weights', 'dnn_model/yolov4-tiny.cfg')
classes = []
with open("dnn_model/coco.txt", "r") as f:
    classes = f.read().splitlines()
cell_phone_class_index = classes.index("cell phone")
# Initialize the font and colors for drawing cell phone detection boxes
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True: 

	img_resp = requests.get(url)
	img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
	frame = cv2.imdecode(img_arr, -1)
	frame = imutils.resize(frame, width = 500)
	frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
	cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = face_mesh.process(rgb)

	# Now loop over all the face detections
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			h_frame, w_frame, _ = frame.shape
			shape = np.array([[int(pt.x * w_frame), int(pt.y * h_frame)] for pt in face_landmarks.landmark])

			x_min = np.min(shape[:, 0])
			y_min = np.min(shape[:, 1])
			x_max = np.max(shape[:, 0])
			y_max = np.max(shape[:, 1])
			cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)	
			# Put a number 
			cv2.putText(frame, "Driver", (x_min - 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			leftEye = shape[LEFT_EYE_INDICES]
			rightEye = shape[RIGHT_EYE_INDICES] 
			mouth = shape[MP_MAR_INDICES]
		# Compute the EAR for both the eyes 
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Take the average of both the EAR
		EAR = (leftEAR + rightEAR) / 2.0
		#live datawrite in csv
		ear_list.append(EAR)
		ts.append(dt.datetime.now().strftime('%H:%M'))
		
		# Compute the convex hull for both the eyes and then visualize it
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# Draw the contours 
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

		MAR = mp_mouth_aspect_ratio(mouth)

		mar_list.append(MAR)
		# Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
		# Thus, count the number of frames for which the eye remains closed 
		if EAR < EAR_THRESHOLD: 
			FRAME_COUNT += 1

			cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

			if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
				count_sleep += 1
				# Add the frame to the dataset ar a proof of drowsy driving
				cv2.imwrite("dataset_phonecam/frame_sleep%d.jpg" % count_sleep, frame)
				play_warning_sounds('sound files/alarm.mp3', 'sound files/warning.mp3')
				cv2.putText(frame, "DROWSINESS ALERT!", (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else: 				
			FRAME_COUNT = 0

		# Check if the person is yawning
		if MAR > MAR_THRESHOLD:
			count_yawn += 1
			cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
			cv2.putText(frame, "DROWSINESS ALERT!", (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.imwrite("dataset_phonecam/frame_yawn%d.jpg" % count_yawn, frame)
			play_warning_sounds('sound files/alarm.mp3', 'sound files/warning_yawn.mp3')
	
	# Cell phone detection code
	height, width, _ = frame.shape
	blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	output_layers_names = net.getUnconnectedOutLayersNames()  
	layerOutputs = net.forward(output_layers_names)
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
				confidences.append((float(confidence)))
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
                # Add the frame to the dataset ar a proof of mobile usage
				cv2.imwrite("dataset_phonecam/frame_mobile_usage%d.jpg" % mobile_usage_count, frame)
				play_warning_sounds('sound files/mobwarn.mp3')

	#total data collection for plotting	
	for i in ear_list:
		total_ear.append(i)
	for i in mar_list:
		total_mar.append(i)			
	for i in ts:
		total_ts.append(i)		

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF 
	if key == ord('q'):
		break

 # Check if it's time to update the CSV file
a = total_ear
b =total_mar
c = total_ts

df = pd.DataFrame({ "MAR":b,"EAR" : a,"TIME" : c})
df.to_csv("op_webcam.csv", index=False)
df=pd.read_csv("op_webcam.csv")

df.plot(x='TIME',y=['EAR','MAR'])
plt.xticks(rotation=450, ha='right')

plt.subplots_adjust(bottom=0.10)
plt.title('EAR & MAR calculation over time of webcam')
plt.ylabel('EAR & MAR')
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()

cv2.destroyAllWindows()

