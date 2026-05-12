import numpy as np
import cv2
from utils import *
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cfg', type=str, default='', help='Path to config file')
    parser.add_argument('--model_weights', type=str, default='', help='path to weights of model')
    parser.add_argument('--video', type=str, default='', help='path to video file')
    parser.add_argument('--src', type=int, default=0, help='source of the camera')
    parser.add_argument('--output_dir', type=str, default='', help='path to the output directory')
    args = parser.parse_args()

    frameWidth = 640
    frameHeight = 480

    # Load YOLO Model
    net = cv2.dnn.readNet(args.model_weights, args.model_cfg)
    with open("dnn_model/coco.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    font = cv2.FONT_HERSHEY_PLAIN
    frame_id = 0
    cameraFeed = False
    cameraNo = 1

    if cameraFeed:
        intialTracbarVals = [24, 55, 12, 100]  # wT,hT,wB,hB
        cap = cv2.VideoCapture(cameraNo)
        cap.set(3, frameWidth)
        cap.set(4, frameHeight)
    else:
        intialTracbarVals = [42, 63, 14, 87]   # wT,hT,wB,hB
        cap = cv2.VideoCapture(args.video)

    noOfArrayValues = 10
    arrayCounter = 0
    arrayCurve = np.zeros([noOfArrayValues])
    
    # Initialize trackbars for Perspective Warp
    initializeTrackbars(intialTracbarVals)

    starting_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            print('[i] ==> Done processing!!!')
            cv2.waitKey(1000)
            break

        if not cameraFeed:
            img = cv2.resize(img, (frameWidth, frameHeight), None)

        imgWarpPoints = img.copy()
        imgFinal = img.copy()

        # Phase 1: Lane Detection (Sliding Window)
        imgUndis = undistort(img)
        imgThres, imgCanny, imgColor = pipeline(imgUndis)
        
        src = valTrackbars()
        imgWarp = perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
        imgWarpPoints = drawPoints(imgWarpPoints, src)
        
        imgSliding, curves, lanes, ploty = sliding_window(imgWarp, draw_windows=True)

        lane_curve = 0
        try:
            curverad = get_curve(imgFinal, curves[0], curves[1])
            lane_curve = np.mean([curverad[0], curverad[1]])
            imgFinal = draw_lanes(imgFinal, curves[0], curves[1], frameWidth, frameHeight, src=src)

            # Curve averaging
            currentCurve = lane_curve // 50
            if int(np.sum(arrayCurve)) == 0:
                averageCurve = currentCurve
            else:
                averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
                
            if abs(averageCurve - currentCurve) > 200:
                arrayCurve[arrayCounter] = averageCurve
            else:
                arrayCurve[arrayCounter] = currentCurve
                
            arrayCounter += 1
            if arrayCounter >= noOfArrayValues:
                arrayCounter = 0
                
            cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth // 2 - 70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            pass

        imgFinal = drawLines(imgFinal, lane_curve)

        # Phase 2: Object Detection (YOLO)
        frame = imgFinal.copy()
        height, width, _ = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = "{}: {:.2f}%".format(classes[class_ids[i]], confidences[i] * 100)
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 10), font, 2, color, 2)

        frame_id += 1
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 30), font, 2, (0, 0, 0), 1)

        # Visualizations
        imgStacked = stackImages(0.7, ([imgUndis, frame],
                                       [imgColor, imgCanny],
                                       [imgWarp, imgSliding]))

        cv2.imshow("Pipeline Visualization", imgStacked)
        cv2.imshow("Final Result", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()