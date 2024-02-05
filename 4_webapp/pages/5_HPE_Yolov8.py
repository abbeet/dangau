import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('./4_webapp/models/yolov8s-pose.pt')

# Define the RTSP URL
rtsp_url = "rtsp://admin:All3gr1t4@192.168.8.7:554/axis-media/media.amp?streamprofile=Quality"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    # Read a frame from the RTSP stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from RTSP stream.")
        break

    # Inference
    results = model(frame)

    # Iterate through each detection and draw it on the frame
    for det in results.xyxy[0]:  # detections for one image
        # Extract coordinates
        x1, y1, x2, y2, conf, cls = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], int(det[5])
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw label
        label = f'{results.names[cls]} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# results = model(source=0, show=True, conf=0.3)