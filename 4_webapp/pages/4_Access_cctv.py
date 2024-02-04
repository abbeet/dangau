import cv2
import requests
import numpy as np

# URL for the MJPEG stream (replace with your actual URL)
mjpeg_url = "http://103.247.218.109:60001/cgi-bin/nphMotionJpeg?Resolution=320x240&Quality=Motion"

while True:
    response = requests.get(mjpeg_url, auth=("admin", "Budayasaya123"), stream=True)
    if response.status_code == 200:
        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow('CCTV Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
cv2.destroyAllWindows()