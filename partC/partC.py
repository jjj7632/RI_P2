import cv2
import numpy as np

# === SETTINGS ===
input_video = "people_talking2.mp4"
output_video = "optical_flow_tracking.avi"

# Face detector (you can swap with your HOG detector if you prefer)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Lucas–Kanade optical flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Shi–Tomasi corner detection parameters
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# === INITIAL VARIABLES ===
face_detected = False
old_gray = None
p0 = None  # Tracked feature points
bbox = None  # Face bounding box
initial_num_points = 0
lost_threshold = 0.5  # Re-detect if fewer than 50% of original points remain

print("Processing video... (press 'q' to stop early)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not face_detected:
        # === FACE DETECTION ===
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            # Pick the largest face (you can adjust this)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            (x, y, w, h) = faces[0]
            bbox = (x, y, w, h)

            # Extract region of interest (face area)
            face_roi = gray[y:y+h, x:x+w]

            # === FEATURE DETECTION (Shi–Tomasi) ===
            p0 = cv2.goodFeaturesToTrack(face_roi, mask=None, **feature_params)
            if p0 is not None:
                # Offset feature coordinates to full-frame positions
                p0[:, 0, 0] += x
                p0[:, 0, 1] += y
                old_gray = gray.copy()
                initial_num_points = len(p0)
                face_detected = True
    else:
        # === FEATURE TRACKING (Lucas–Kanade) ===
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

        if p1 is None:
            face_detected = False
            continue

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw tracked points
        for pt in good_new:
            cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

        # === UPDATE BOUNDING BOX ===
        # Compute average movement (translation)
        if len(good_new) > 0:
            movement = np.mean(good_new - good_old, axis=0)
            dx, dy = movement.astype(int)
            x, y, w, h = bbox
            bbox = (x + dx, y + dy, w, h)

        # Draw updated bounding box in red
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # === CHECK FOR LOST TRACK ===
        if len(good_new) < lost_threshold * initial_num_points:
            face_detected = False  # Trigger re-detection
        else:
            # Prepare for next iteration
            old_gray = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    # === DISPLAY AND WRITE ===
    out.write(frame)
    cv2.imshow("Optical Flow Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Saved output as: {output_video}")
