import cv2
import dlib
import numpy as np
import sys

face_predict_path = "C:/Users/barry/Documents/GitHub/dangau/4_webapp/models/shape_predictor_68_face_landmarks.dat"

# Initialize dlib's face detector and load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_predict_path)

# RTSP stream source
rtsp_url = "rtsp://admin:All3gr1t4@192.168.8.7:554/axis-media/media.amp?streamprofile=Quality"
cap = cv2.VideoCapture(rtsp_url)

# Predefined 3D model points of a generic face model
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corner
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 2D image points from the facial landmarks
        image_points = np.array([
                                    (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
                                    (landmarks.part(8).x, landmarks.part(8).y),       # Chin
                                    (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
                                    (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
                                    (landmarks.part(48).x, landmarks.part(48).y),     # Left Mouth corner
                                    (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
                                ], dtype="double")

        # Solve for pose
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, None)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255,0,0), 2)

    # Display the resulting frame
    cv2.imshow('Head Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()