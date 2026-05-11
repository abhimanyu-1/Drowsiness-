# Define the function for calculating the Eye Aspect Ratio(EAR)
from scipy.spatial import distance as dist 
def eye_aspect_ratio(eye):

	# Vertical eye landmarks
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4]) 

	# Horizontal eye landmarks 
	C = dist.euclidean(eye[0], eye[3])

	# The EAR Equation 
	EAR = (A + B) / (2.0 * C)
	return EAR

def mouth_aspect_ratio(mouth): 
	
	A = dist.euclidean(mouth[13], mouth[19])
	B = dist.euclidean(mouth[14], mouth[18])
	C = dist.euclidean(mouth[15], mouth[17])

	MAR = (A + B + C) / 3.0
	return MAR

def mp_mouth_aspect_ratio(mouth): 
	# For mediapipe, we pass an 8-element array containing the inner lip points directly
	A = dist.euclidean(mouth[0], mouth[5])
	B = dist.euclidean(mouth[1], mouth[4])
	C = dist.euclidean(mouth[2], mouth[3])
	# Horizontal distance
	D = dist.euclidean(mouth[6], mouth[7])
	
	if D == 0:
		D = 1
	
	# Normalize MAR (returns pure ratio, yawning is typically > 0.6)
	MAR = (A + B + C) / (3.0 * D)
	return MAR