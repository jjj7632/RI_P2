import cv2
import numpy as np

# === INPUT/OUTPUT FILE PATHS ===
input_video = "cpet347_background.mp4"       # Change this to your video filename
background_output = "clean_background.avi"
mask_output = "foreground_mask.avi"

# === VIDEO CAPTURE SETUP ===
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# === BACKGROUND SUBTRACTOR (MOG2) ===
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# === VIDEO WRITERS ===
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_mask = cv2.VideoWriter(mask_output, fourcc, fps, (frame_width, frame_height), False)
out_background = cv2.VideoWriter(background_output, fourcc, fps, (frame_width, frame_height), True)

# === PROCESS VIDEO ===
print("Processing video...")

# To estimate the background, we’ll average over all frames
background_accumulator = np.zeros((frame_height, frame_width, 3), np.float32)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Clean the mask (optional — removes noise)
    fgmask = cv2.medianBlur(fgmask, 5)

    # Accumulate background for clean plate generation
    background_accumulator += frame.astype(np.float32)
    frame_count += 1

    # Write binary mask frame (ensure single channel)
    out_mask.write(fgmask)

cap.release()

# === GENERATE CLEAN BACKGROUND (STATIC SCENE) ===
clean_background = (background_accumulator / frame_count).astype(np.uint8)
out_background.write(clean_background)

# Write the same background for each frame duration
for _ in range(int(frame_count)):
    out_background.write(clean_background)

out_background.release()
out_mask.release()

print("Done! Saved outputs:")
print(f"  Clean background: {background_output}")
print(f"  Foreground mask:  {mask_output}")
