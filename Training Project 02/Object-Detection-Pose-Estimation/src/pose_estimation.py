import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================
# MODEL PATH
# ==============================
MODEL_PATH = "pose_landmarker.task"


# ==============================
# CREATE POSE LANDMARKER
# ==============================
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=2
)

landmarker = vision.PoseLandmarker.create_from_options(options)


# ==============================
# OPEN WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()

print("Webcam started successfully.")
print("Press Q to quit.")


# ==============================
# POSE CONNECTIONS
# ==============================
connections = [
    (11, 12),  # shoulders
    (11, 13),  # left upper arm
    (13, 15),  # left lower arm
    (12, 14),  # right upper arm
    (14, 16),  # right lower arm

    (11, 23),  # left body
    (12, 24),  # right body
    (23, 24),  # hips

    (23, 25),  # left upper leg
    (25, 27),  # left lower leg
    (27, 29),  # left foot

    (24, 26),  # right upper leg
    (26, 28),  # right lower leg
    (28, 30),  # right foot
]


# ==============================
# VIDEO LOOP
# ==============================
frame_timestamp = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect pose
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    # ==============================
    # DRAW POSE
    # ==============================

    for pose_landmarks in result.pose_landmarks:

        h, w, _ = frame.shape

        points = []

        # Draw landmarks
        for landmark in pose_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw skeleton connections
        for start, end in connections:

            if start < len(points) and end < len(points):

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

    # Display
    cv2.imshow(
        "Human Pose Estimation",
        frame
    )

    # Increase timestamp
    frame_timestamp += 33

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Pose estimation stopped.")