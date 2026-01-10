
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import cv2
import tempfile
import os

class FeatureExtractor:
    def __init__(self, model_path: str | None = None):
        """
        Initialize FeatureExtractor with HandLandmarker API.
        
        Args:
            model_path: Path to hand_landmarker.task model file. If None, searches common locations.
        """
        # Set default model path if not provided
        if model_path is None:
            # Try common locations
            possible_paths = [
                "./mediapipe_experiments/hand_landmarker.task",
                "./mediapipe/hand_landmarker.task",
                "mediapipe_experiments/hand_landmarker.task",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path is None:
                raise FileNotFoundError(
                    "Hand landmarker model file not found. Please specify model_path "
                    "or ensure hand_landmarker.task exists in mediapipe_experiments/ or mediapipe/"
                )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path

    def process_video(self, video_path: str, target_frames: int = 30, use_temporal_stats: bool = True) -> np.ndarray:
        """
        Reads a video, extracts hand landmarks, normalizes them, 
        resamples to target_frames, and computes feature vector.
        
        Args:
            video_path: Path to video file
            target_frames: Number of frames to resample to
            use_temporal_stats: If True, use compact temporal statistics (378 dims).
                               If False, use flattened vector (target_frames * 126 dims).
        
        Returns:
            np.ndarray: (D,) feature vector
        """
        frames, timestamps_ms = self._load_frames(video_path)
        if not frames:
            raise ValueError(f"Could not load frames from {video_path}")
            
        # 1. Extract Landmarks
        X_raw = self._run_mediapipe_hands(frames, timestamps_ms)
        
        # 2. Normalize
        X_norm = self._normalize_sequence(X_raw)
        
        # 3. Resample
        X_resampled = self._resample_sequence(X_norm, target_frames)
        
        # 4. Compute features
        if use_temporal_stats:
            return self._compute_temporal_stats(X_resampled)
        else:
            return X_resampled.flatten()
    
    def _compute_temporal_stats(self, X: np.ndarray) -> np.ndarray:
        """
        Compute temporal statistics from a sequence of landmarks.
        
        Args:
            X: (L, 126) landmark sequence
            
        Returns:
            np.ndarray: (378,) feature vector containing [mean, std, velocity_mean]
        """
        # Mean and std across time
        mean_vec = X.mean(axis=0)  # (126,)
        std_vec = X.std(axis=0)    # (126,)
        
        # Velocity (frame-to-frame differences)
        if X.shape[0] > 1:
            velocity = np.diff(X, axis=0)  # (L-1, 126)
            velocity_mean = velocity.mean(axis=0)  # (126,)
        else:
            velocity_mean = np.zeros(126, dtype=np.float32)
        
        # Concatenate: 126 + 126 + 126 = 378 dims
        return np.concatenate([mean_vec, std_vec, velocity_mean]).astype(np.float32)

    def _load_frames(self, video_path: str):
        """
        Load video frames with timestamps.
        
        Returns:
            frames: List of RGB numpy arrays
            timestamps_ms: List of timestamps in milliseconds for MediaPipe VIDEO mode
        """
        frames = []
        timestamps_ms = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []
        
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1e-3:
            src_fps = 30.0  # reasonable fallback
        
        frame_i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            # Calculate timestamp in milliseconds
            timestamp_ms = int((frame_i / src_fps) * 1000) if src_fps > 0 else frame_i * 33  # 33ms = ~30fps
            timestamps_ms.append(timestamp_ms)
            frame_i += 1
            
        cap.release()
        return frames, timestamps_ms

    def _run_mediapipe_hands(self, frames, timestamps_ms: list[int]):
        """
        Run MediaPipe HandLandmarker on a list of frames using VIDEO mode.
        
        Args:
            frames: List of RGB numpy arrays
            timestamps_ms: List of timestamps in milliseconds for VIDEO mode
            
        Returns: (L, 126) numpy array.
        """
        L = len(frames)
        X_raw = np.zeros((L, 126), dtype=np.float32)
        
        # Create HandLandmarker instance for this sequence
        BaseOptions = mp_tasks.BaseOptions
        HandLandmarker = mp_vision.HandLandmarker
        HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
        VisionRunningMode = mp_vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with HandLandmarker.create_from_options(options) as landmarker:
            for i, (frame, timestamp_ms) in enumerate(zip(frames, timestamps_ms)):
                # Convert numpy array to MediaPipe Image with explicit dimensions
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                # Process frame in VIDEO mode with timestamp
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                lh = np.zeros((21, 3), dtype=np.float32)
                rh = np.zeros((21, 3), dtype=np.float32)
                
                if result.hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                        # Extract handedness from new API structure
                        label = None
                        if hand_idx < len(result.handedness) and result.handedness[hand_idx]:
                            # New API: handedness[i] is iterable and contains category objects
                            # Access pattern: result.handedness[i][0].category_name
                            try:
                                handedness_cats = result.handedness[hand_idx]
                                if handedness_cats and len(handedness_cats) > 0:
                                    label = handedness_cats[0].category_name
                            except (IndexError, AttributeError, TypeError):
                                # Fallback: iterate if structure is different
                                for cat in result.handedness[hand_idx]:
                                    if hasattr(cat, 'category_name'):
                                        label = cat.category_name
                                        break
                        
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                        
                        if label == "Left":
                            lh = coords
                        elif label == "Right":
                            rh = coords
                        else:
                            # If handedness unknown, assign based on position (leftmost = left)
                            if np.all(lh == 0):
                                lh = coords
                            else:
                                rh = coords
                            
                vec = np.concatenate([lh.flatten(), rh.flatten()])
                X_raw[i] = vec
                
        return X_raw

    def _normalize_sequence(self, X_raw):
        """
        Normalize landmarks.
        Shape: (L, 126) -> (L, 126)
        """
        L = X_raw.shape[0]
        X_reshaped = X_raw.reshape(L, 2, 21, 3)
        X_norm = np.zeros_like(X_reshaped)
        
        WRIST = 0
        MIDDLE_MCP = 9
        
        for t in range(L):
            for h in range(2): 
                hand = X_reshaped[t, h]
                
                if np.all(hand == 0):
                    continue
                    
                wrist = hand[WRIST]
                middle = hand[MIDDLE_MCP]
                
                hand_centered = hand - wrist
                dist = np.linalg.norm(middle - wrist)
                scale = dist if dist > 1e-6 else 1.0
                
                X_norm[t, h] = hand_centered / scale
                
        return X_norm.reshape(L, 126)

    def _resample_sequence(self, X, target_len):
        """
        Resample sequence X of shape (L, D) to (target_len, D) using linear interpolation.
        """
        L, D = X.shape
        if L == target_len:
            return X
        if L == 0:
            return np.zeros((target_len, D), dtype=np.float32)
            
        x_old = np.linspace(0, 1, L)
        x_new = np.linspace(0, 1, target_len)
        
        X_new = np.zeros((target_len, D), dtype=np.float32)
        for d in range(D):
            X_new[:, d] = np.interp(x_new, x_old, X[:, d])
            
        return X_new

