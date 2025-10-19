import cv2
import numpy as np
import os

# SETUP
cap = cv2.VideoCapture(1)  # change to filename if using a video file
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

os.makedirs("outputs", exist_ok=True)

# Video writers for saving results
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_foreground = cv2.VideoWriter('outputs/foreground_mask.mp4', fourcc, fps, (width, height), False)
out_display = cv2.VideoWriter('outputs/final_output.mp4', fourcc, fps, (width * 2, height))

# Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
avg_background = None

# Face Detectors
haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# HOG + SVM face detector (classical style)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Optical Flow Params 
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variables
mode = 'd'
tracking = False
old_gray = None
p0 = None
bbox = None
original_point_count = 0

print("Controls: b = Background view | d = Detection view | t = Tracking view | q = Quit")

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()

    # ----- BACKGROUND / FOREGROUND -----
    fgmask = fgbg.apply(frame)
    out_foreground.write(fgmask)

    # Update background average for "clean plate"
    if avg_background is None:
        avg_background = np.float32(frame)
    cv2.accumulateWeighted(frame, avg_background, 0.01)
    clean_background = cv2.convertScaleAbs(avg_background)
    cv2.imwrite('outputs/clean_background.png', clean_background)

    # Compose side-by-side BG/FG display
    fg_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    bgfg_combined = np.hstack((clean_background, fg_bgr))

    # ----- MODE: BACKGROUND -----
    if mode == 'b':
        cv2.imshow("Mode: Background / Foreground", bgfg_combined)
        out_display.write(bgfg_combined)

    # ----- MODE: FACE DETECTION -----
    elif mode == 'd':
        # Haar (blue)
        faces_haar = haar_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces_haar:
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # HOG + SVM (green)
        rects, _ = hog.detectMultiScale(gray, winStride=(8, 8))
        for (x, y, w, h) in rects:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        combined = np.hstack((frame, display))
        cv2.imshow("Mode: Face Detection (Left: Original, Right: Detected)", combined)
        out_display.write(combined)

    # ----- MODE: TRACKING -----
    elif mode == 't':
        if not tracking:
            faces = haar_detector.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                bbox = (x, y, w, h)
                roi_gray = gray[y:y + h, x:x + w]
                p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

                if p0 is not None:
                    p0[:, 0, 0] += x
                    p0[:, 0, 1] += y
                    old_gray = gray.copy()
                    original_point_count = len(p0)
                    tracking = True
                    print("Tracking initialized with", len(p0), "features.")
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw points
                for pt in good_new:
                    cv2.circle(display, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

                # Compute median translation for stable motion
                movement = np.median(good_new - good_old, axis=0)
                if bbox is not None and not np.isnan(movement).any():
                    x, y, w, h = bbox
                    new_x = int(x + movement[0])
                    new_y = int(y + movement[1])
                    bbox = (new_x, new_y, w, h)
                    cv2.rectangle(display, (new_x, new_y), (new_x + w, new_y + h), (0, 0, 255), 2)

                old_gray = gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                # Re-detect if too few points remain
                if len(p0) < 0.5 * original_point_count:
                    print("Re-detecting face (lost too many features)")
                    tracking = False
                    p0 = None
            else:
                tracking = False
                print("Optical flow failed, re-detecting")

        combined = np.hstack((frame, display))
        cv2.imshow("Mode: Face Tracking (Left: Original, Right: Tracking)", combined)
        out_display.write(combined)

    # ----- KEY CONTROLS -----
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        mode = 'b'
        print("Switched to Background / Foreground mode")
    elif key == ord('d'):
        mode = 'd'
        print("Switched to Face Detection mode")
    elif key == ord('t'):
        mode = 't'
        print("Switched to Tracking mode")

# ========== CLEANUP ==========
cap.release()
out_foreground.release()
out_display.release()
cv2.destroyAllWindows()
print("All outputs saved in the 'outputs' folder.")
