import mediapipe as mp
import cv2 as cv
import time
import os
import numpy as np



from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import text
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

from sensor_layer.camera import Camera


# Some mediapipe setup for the hands class
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

cam = Camera(source=0)
with HandLandmarker.create_from_options(options) as landmarker:
    while cam.cap.isOpened():

        frame = cam.get_frame()
        if frame is None:
            continue

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_ts = int(time.time() * 1000)

        results = landmarker.detect_for_video(image, frame_ts)
        hand_landmarks_list = results.hand_landmarks

        annotated = Camera.toBGR(frame)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
    # Draw the hand landmarks.
            mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())



        cv.imshow('MediaPipe Hands', cv.flip(annotated, 1))
        if cv.waitKey(1) == ord('q'):
            cam.release()
            break

        
    

