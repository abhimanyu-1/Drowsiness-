import cv2
import dlib
import numpy as np

print("dlib version:", dlib.__version__)
detector = dlib.get_frontal_face_detector()

try:
    frame = np.zeros((300, 450, 3), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    print("gray shape:", gray.shape, "dtype:", gray.dtype)
    rects = detector(gray, 1)
    print("Success with gray!")
except Exception as e:
    print("Error with gray:", e)

try:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("rgb shape:", rgb.shape, "dtype:", rgb.dtype)
    rects = detector(rgb, 1)
    print("Success with rgb!")
except Exception as e:
    print("Error with rgb:", e)
