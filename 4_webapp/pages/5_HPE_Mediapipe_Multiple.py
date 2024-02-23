import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display, and convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable to pass by reference.
    image.flags.writeable = False

    # Process the image to detect faces and facial landmarks
    results = face_mesh.process(image)

    # To improve performance, mark the image as writeable again.
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d = []  # Reset for each face
            face_3d = []  # Reset for each face

            img_h, img_w, img_c = image.shape

            for idx, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                # Consider specific landmarks for 2D and 3D points
                if idx in [33, 263, 1, 61, 291, 199]:
                    face_2d.append([x, y])
                    # For z coordinate, a scale factor is used to match the scale of x and y
                    face_3d.append([x, y, lm.z * 3000])

            # Only proceed if we have enough landmarks for the calculation
            if face_2d and face_3d:
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera matrix and distortion coefficients
                focal_length = img_w  # Approximate focal length
                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]], dtype=np.float64)
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # SolvePnP for head pose estimation
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                nose_3d_projection, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(face_2d[2][0]), int(face_2d[2][1]))  # Using the nose tip for drawing
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

    # Display FPS and image
    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()