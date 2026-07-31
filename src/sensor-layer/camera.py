import numpy as np
import cv2 as cv

# VideoCapture object required to capture video, argument
# is device index or name of a video file

cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
out = cv.VideoWriter('output.avi', fourcc, 20.0, (width, height), isColor=False)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame, 0)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    out.write(gray_frame)
    cv.imshow('frame', gray_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()