# RI_P2

Project Overview

This project implements a real-time perception pipeline for human face detection and tracking using classical computer vision algorithms. The system integrates three core tasks:
Background Subtraction: Isolates moving objects from the static scene using a Gaussian Mixture Model (GMM).
Face Detection: Detects faces using Viola-Jones (Haar Cascades) and Histogram of Oriented Gradients (HOG + SVM via dlib or OpenCV), allowing comparison of speed, accuracy, and robustness.
Face Tracking: Tracks detected faces efficiently using Shi-Tomasi corner features and Lucas-Kanade optical flow, with automatic re-detection when tracking is lost.
The final system is interactive, allowing the user to switch between visualization modes in real-time.

Features

Background/Foreground Mode (b)

Displays clean background plate and moving object masks.
Outputs: clean_background.png, foreground_mask.mp4.

Face Detection Mode (d)

Runs both Haar Cascades (blue bounding boxes) and HOG + SVM (green bounding boxes).
Outputs combined video in final_output.mp4.

Face Tracking Mode (t)

Tracks a detected face using optical flow (red bounding box).
Re-detection logic automatically re-acquires lost faces.

Controls

b – Switch to background/foreground view

d – Switch to face detection view

t – Switch to face tracking view

q – Quit application

Installation
1. Clone the repository:
git clone https://github.com/jjj7632/RI_P2.git

2. Install Python dependencies:
pip install opencv-python numpy

3. Install dlib for HOG + SVM face detection

pip install dlib-bin

4. Webcam / Video Input

By default, the script uses the webcam thats facing you when typing on your laptop:
cap = cv2.VideoCapture(1)

To use a video file, replace 0 with the file path:
cap = cv2.VideoCapture("path/to/video.mp4")

Usage

Run the main script:
python partD.py

The window will open your live webcam feed.
Press keys (b, d, t) to switch between modes.
Videos and background plate images will be automatically saved in the outputs/ folder.

Project Structure
proj2/
├─ partD.py              # Main integrated application
├─ outputs/              # Folder where results are saved
│  ├─ foreground_mask.mp4
│  ├─ clean_background.png
│  └─ final_output.mp4
├─ README.md             # This file
└─ [optional] videos/    # Sample input videos

Learning Outcomes

Implement and evaluate statistical background subtraction models.
Apply and compare Viola-Jones and HOG-based face detection algorithms.
Extract and track keypoint features for efficient object tracking.
Build a robust, real-time perception pipeline that intelligently switches between detection and tracking modes.
Analyze the performance, trade-offs, and failure modes of classical computer vision algorithms.
