import numpy as np
import cv2 as cv

# VideoCapture object required to capture video, argument
# is device index or name of a video file
class Camera:
    def __init__(self, source=0):
        """Initializes the opencv video capture."""  
        self.cap = cv.VideoCapture(0)

        self.fourcc = cv.VideoWriter_fourcc('m', 'p', 'v', '4')
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.out = cv.VideoWriter('output.mpv4', self.fourcc, 30.0, (self.width, self.height), isColor=False)

    def record(self):
        """Save the videoCapture with rgb coloring."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                print("can't receive frame (stream end?). Exiting ...")
                break
            rgb_frame = Camera.toRGB(frame)
            self.out.write(rgb_frame)
            cv.imshow('frame', rgb_frame)
            if cv.waitKey(1) == ord('q'):
                break

    def get_frame(self):
        """Reads a frame from the capture, 
        then converts it to rgb."""
        if self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("EMPTY FRAME...")
                return
            # frame.flags.writeable = False
            # convert to RGB since mediapipe reads as that and not BGR
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            return rgb_frame
    

    def release(self):
        """Releases the videoCapture and destroys all windows."""
        self.cap.release()
        self.out.release()
        cv.destroyAllWindows()

    @staticmethod
    def toRGB(frame: cv.typing.MatLike | None):
        if frame is None:
            return
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    @staticmethod
    def toBGR(frame: cv.typing.MatLike| None):
        if frame is None:
            return
        return cv.cvtColor(frame, cv.COLOR_RGB2BGR)