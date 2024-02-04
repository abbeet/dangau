import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to calculate head pose
def calculate_head_pose(image, landmarks):
    # Define 3D model points of a generic head
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    # 2D image points from landmarks
    image_points = get_image_points(landmarks, image)

    # Camera internals
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1))

    # Solve PnP
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    return rotation_vector, translation_vector, image_points

# Function to extract image points from landmarks
def get_image_points(landmarks, image):
    return np.array([
        (landmarks[1].x * image.shape[1], landmarks[1].y * image.shape[0]), # Nose tip
        (landmarks[152].x * image.shape[1], landmarks[152].y * image.shape[0]), # Chin
        (landmarks[226].x * image.shape[1], landmarks[226].y * image.shape[0]), # Left eye left corner
        (landmarks[446].x * image.shape[1], landmarks[446].y * image.shape[0]), # Right eye right corner
        (landmarks[57].x * image.shape[1], landmarks[57].y * image.shape[0]), # Left mouth corner
        (landmarks[287].x * image.shape[1], landmarks[287].y * image.shape[0])  # Right mouth corner
    ], dtype=np.float64)

# Function to draw axes
def draw_axes(image, rotation_vector, translation_vector, image_points):
    camera_matrix = np.array(
        [[image.shape[1], 0, image.shape[1]/2],
         [0, image.shape[1], image.shape[0]/2],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4,1), dtype=np.float64)
    axis_length = 100

    # Project a 3D axis onto the image plane
    axis_points, _ = cv2.projectPoints(
        np.array([(axis_length, 0, 0), (0, axis_length, 0), (0, 0, axis_length)], dtype=np.float64), 
        rotation_vector, 
        translation_vector, 
        camera_matrix, 
        dist_coeffs)

    # Draw the axes
    origin = tuple(image_points[0].astype(int))
    image = cv2.line(image, origin, tuple(axis_points[0].ravel().astype(int)), (255, 0, 0), 3)
    image = cv2.line(image, origin, tuple(axis_points[1].ravel().astype(int)), (0, 255, 0), 3)
    image = cv2.line(image, origin, tuple(axis_points[2].ravel().astype(int)), (0, 0, 255), 3)

    return image

# Main function
def main():
    # Capture video from the first camera
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB and process it with MediaPipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw face mesh
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
#                    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                    # Estimate head pose
                    rotation_vector, translation_vector, image_points = calculate_head_pose(image, face_landmarks.landmark)

                    # Draw axes
                    image = draw_axes(image, rotation_vector, translation_vector, image_points)

            cv2.imshow('Head Pose Estimation', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

if __name__ == "__main__":
    main()
