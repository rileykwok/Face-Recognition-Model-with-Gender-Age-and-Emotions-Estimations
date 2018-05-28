from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
from face_reco_image import FaceImage

USE_SMALL_FRAME = True
VISUALIZE_DATASET = False

face = FaceImage()

if VISUALIZE_DATASET:
    face.face_recognizer.visualize_dataset()

# starting video streaming
process_this_frame = True
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:

    bgr_image = video_capture.read()[1]
        
    if process_this_frame:
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        if USE_SMALL_FRAME:
            rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
        
        result_img = face.detect_face(rgb_image)
        
        if USE_SMALL_FRAME:
            result_img = cv2.resize(result_img, (0, 0), fx=2, fy=2)
        
        bgr_image = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        
    process_this_frame = not process_this_frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
