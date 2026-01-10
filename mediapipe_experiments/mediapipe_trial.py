import cv2
import imageio
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to model and input
MODEL_PATH = "hand_landmarker.task"  # Replace with actual model path
GIF_PATH = "../sgsl_dataset/abuse/abuse.gif"  # Replace with your GIF path

# Load GIF and convert frames to RGB numpy arrays
gif_frames = imageio.mimread(GIF_PATH)
rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for frame in gif_frames]

# Create MediaPipe Image objects
mp_images = [mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) for frame in rgb_frames]

# Initialize the Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

results = []

with HandLandmarker.create_from_options(options) as landmarker:
    for idx, (mp_image, orig_frame) in enumerate(zip(mp_images, rgb_frames)):
        result = landmarker.detect(mp_image)
        print(f"Result:\n{result}")
        results.append(result)

        # Draw landmarks on original frame
        annotated = orig_frame.copy()

        for i, landmarks in enumerate(result.hand_landmarks):
            # Convert normalized landmarks to pixel coordinates
            h, w, _ = annotated.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Draw circles
            for x, y in points:
                cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

            # Draw handedness label
            print(f"Result.Handedness:\n{result.handedness[i]}")
            handed = result.handedness[i][0].category_name
            label = f"{handed} hand"
            cv2.putText(annotated, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show frame
        cv2.imshow("Hand Detection", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

cv2.destroyAllWindows()
