# System Design
*See below for basic system architecture, tools used, and constraints.*
## System Architecture
1. Webcam
2. Video Ingestion: OpenCV (Open Source Computer Vision Library)
3. Feature Extraction: MediaPipe
4. Spatial Intersection Math
5. State Machine (MIDI Events)
6. MIDI Out (DAW)
## Dependencies
- opencv-python
- mediapipe
- numpy
- mido
- python-rtmidi
## Hardware Constraints

### Camera Sensor
#### Framerate
Using webcams with FPS of 30 introduces latency of ~33ms, its best to use higher framerates for lower latency between key presses.

#### Motion Blur
Fast piano playing will cause MediaPipe to lose track of fingertips.
