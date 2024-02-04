import cv2
import mediapipe as mp
import numpy as np
import requests

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define your other functions (calculate_head_pose, get_image_points, draw_axes) here...

# Main function
def main():
    # Updated URL for the MJPEG stream with new resolution
    mjpeg_url = "http://103.247.218.109:60001/cgi-bin/nphMotionJpeg?Resolution=800x600&Quality=Motion"

    # Start a streaming session
    with requests.Session() as s:
        response = s.get(mjpeg_url, auth=("admin", "Budayasaya123"), stream=True)

        if response.status_code == 200:
            bytes_data = bytes()
            with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                while True:
                    for chunk in response.iter_content(chunk_size=1024):
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')  # JPEG start
                        b = bytes_data.find(b'\xff\xd9')  # JPEG end

                        # Extract a single frame from bytes
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b + 2]
                            bytes_data = bytes_data[b + 2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                            # Convert the BGR image to RGB and process it with MediaPipe
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = face_mesh.process(image)

                            # Draw face mesh
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            if results.multi_face_landmarks:
                                for face_landmarks in results.multi_face_landmarks:
                                    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                                    # Estimate head pose
                                    rotation_vector, translation_vector, image_points = calculate_head_pose(image, face_landmarks.landmark)

                                    # Draw axes
                                    image = draw_axes(image, rotation_vector, translation_vector, image_points)

                            cv2.imshow('Head Pose Estimation', image)
                            if cv2.waitKey(5) & 0xFF == 27:
                                break
        else:
            print("Failed to connect to the stream.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
