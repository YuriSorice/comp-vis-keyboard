import mediapipe as mp
import cv2 as cv
import time
import os
import numpy as np



from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import text


from sensor_layer.camera import Camera

class HandTracker:
    def __init__(self, model_path: str, max_hands: int=2):
        """Initializes the HandTracker and loads the Mediapipe model."""
        self.model_path = model_path
        self.max_hands = max_hands
        self.landmarker = self._build_landmarker()

        # Drawing Utils
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def _build_landmarker(self):
        """Configures and builds the HandLandmarker task object."""
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        options = HandLandmarkerOptions(
            base_options=BaseOptions(self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands
        )
        return HandLandmarker.create_from_options(options)
# Some mediapipe setup for the hands class


    def process_and_draw(self, frame, timestamp_ms: int):
        """Processes a frame and draws on the hand landmarks."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = self.landmarker.detect_for_video(image, timestamp_ms)

        annotated = Camera.toRGB(frame)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return annotated

    def run(self):
        """Initializes the camera, then draws the hand landmarks on 
        each frame and displays it."""
        cam = Camera(source=0)

        try:
            while cam.cap.isOpened():
                frame = cam.get_frame()
                if frame is None:
                    continue

                frame_ts = int(time.perf_counter() * 1000)

                annotated = self.process_and_draw(frame, frame_ts)

                cv.imshow('MediaPipe Hands', cv.flip(annotated, 1))

                if cv.waitKey(1) == ord('q'):
                    break
        finally:
            cam.release()

if __name__ == "__main__":
    # Get relative path for landmarker task   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')

    # Instantiate and run landmarking window
    tracker = HandTracker(model_path=model_path)
    tracker.run()

