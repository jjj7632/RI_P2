import cv2
import dlib
import numpy as np

# === INPUT/OUTPUT SETTINGS ===
input_video = "people_talking2.mp4"          # Change this to your input video
output_video = "face_detection_comparison.avi"

# === INITIALIZE DETECTORS ===
# Viola–Jones (Haar Cascade)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# HOG + SVM (Dlib)
hog_detector = dlib.get_frontal_face_detector()

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Output video: side-by-side comparison
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width * 2, frame_height))

print("Processing video...")

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Viola–Jones (Haar Cascade) ---
    faces_haar = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    haar_frame = frame.copy()
    for (x, y, w, h) in faces_haar:
        cv2.rectangle(haar_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box

    # --- HOG + SVM (Dlib) ---
    faces_hog = hog_detector(gray)
    hog_frame = frame.copy()
    for d in faces_hog:
        x, y, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        cv2.rectangle(hog_frame, (x, y), (x2, y2), (0, 255, 0), 2)  # Green box

    # --- Combine side-by-side ---
    combined = np.hstack((haar_frame, hog_frame))

    # Add labels
    cv2.putText(combined, "Haar Cascade (Viola-Jones)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(combined, "HOG + SVM", (frame_width + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Write to video
    out.write(combined)

    # Optional live preview (press 'q' to quit)
    cv2.imshow("Face Detection Comparison", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Saved comparison video as: {output_video}")
