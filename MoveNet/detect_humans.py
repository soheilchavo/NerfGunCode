import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
import RPi.GPIO as GPIO

#Variables
camera_index = 0
score_margin = 0.3
screen_size = (192, 192)

pin_h = 20 #Corresponds to pin 40 on the pi
pin_v = 21 #Corresponds to pin 38 on the pi

#Configure GPIO
GPIO.setmode(GPIO.BCM)

GPIO.setup(pin_h, GPIO.OUT)
GPIO.setup(pin_v, GPIO.OUT)

#Load MoveNet

interpreter = Interpreter(model_path="movenet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)

#Initialize Cap
cap = cv2.VideoCapture(camera_index)

while True:
	
	ret, frame = cap.read()
	if not ret:
		break

	image = cv2.resize(frame, screen_size)
	input_data = np.expand_dims(image, axis=0).astype(np.uint8)
	
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]
	
	h,w, _ = frame.shape
	
	x_avg = 0
	y_avg = 0
	
	avg_length = 0
	
	for kp in keypoints:
		y, x, conf = kp
		if conf > 0.3:
			cx = int(x*w)
			cy = int(y*h)
			
			x_avg += cx
			y_avg += cy
			
			avg_length += 1
			
			cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
	
	if avg_length > 0:
		x_avg = int(x_avg / avg_length)
		y_avg = int(y_avg / avg_length)	
		
		GPIO.output(pin_h, GPIO.HIGH if x_avg > int(w/2) else GPIO.LOW)
		GPIO.output(pin_v, GPIO.HIGH if y_avg > int(h/2) else GPIO.LOW)
		
		cv2.circle(frame, (x_avg, y_avg), 10, (255, 0, 0), -1)
		
	cv2.imshow("MoveNet Detection", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
