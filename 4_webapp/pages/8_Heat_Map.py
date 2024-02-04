import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize frame storage, the heatmap, and determine the frame width and height
ret, frame1 = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    exit()

prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
heatmap = np.zeros_like(prev_frame, dtype=np.float32)

frame_height, frame_width = frame1.shape[:2]

# Initialize VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Compute difference between current frame and previous frame
    diff = cv2.absdiff(prev_frame, gray)
    diff_float = diff.astype(np.float32)  # Convert diff to float32

    # Accumulate the difference in the heatmap
    heatmap = cv2.addWeighted(heatmap, 0.9, diff_float, 0.1, 0)

    # Normalize the heatmap to fit [0, 255] as required for an 8-bit image representation
    norm_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    norm_heatmap = np.uint8(norm_heatmap)  # Convert to uint8 to apply color map
    heatmap_img = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)

    # Display the resulting frame
    cv2.imshow('Motion Heatmap', heatmap_img)

    # Write the frame to the video file
    out.write(heatmap_img)

    # Update previous frame
    prev_frame = gray.copy()

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
