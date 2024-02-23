import cv2
import urllib.request
import numpy as np

# Your camera URL
url = 'http://103.247.218.109:60001/cgi-bin/nphMotionJpeg?Resolution=320x240&Quality=Motion'

# Password manager to handle authentication
password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
password_mgr.add_password(None, url, 'admin', 'Budayasaya123')

handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
opener = urllib.request.build_opener(handler)
urllib.request.install_opener(opener)

# Use OpenCV to capture video frames
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Stream opened successfully.")

# Read and display the video frames
try:
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video Stream', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to retrieve frame.")
            break
except KeyboardInterrupt:
    print("Stream stopped.")

cap.release()
cv2.destroyAllWindows()
