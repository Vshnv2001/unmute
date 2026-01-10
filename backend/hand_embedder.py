# backend/hand_embedder.py
"""
Unified embedding pipeline for GIF dataset indexing and MP4/WebM query embedding.

Features:
- Consistent time normalization (FPS-aware sampling)
- Stable hand slot assignment (prevents left/right swaps)
- Idle frame trimming
- Missing detection interpolation
- EMA smoothing
- Wrist location features

Output: 388-dim embedding (378 temporal stats + 10 wrist location stats)
"""

from __future__ import annotations

import os
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


class HandEmbedder:
    """
    Shared pipeline for:
      - GIF dataset indexing (PIL decode, duration-aware resampling)
      - MP4/WebM query embedding (cv2 decode, FPS-aware sampling)

    Output embedding = temporal stats (mean/std/vel_mean of 126 dims) + wrist-location stats
    """

    def __init__(
        self,
        target_frames: int = 30,
        sample_fps: float = 15.0,
        max_seconds: float = 4.0,
        ema_alpha: float = 0.65,
        model_path: str | None = None,
    ):
        self.target_frames = target_frames
        self.sample_fps = sample_fps
        self.max_seconds = max_seconds
        self.ema_alpha = ema_alpha

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

        # Store model path for later use (we'll create landmarker per sequence)
        self.model_path = model_path

    # ----------------------------
    # Public API
    # ----------------------------
    def embed_gif(self, gif_path: str) -> np.ndarray:
        """Embed a GIF file. Returns (388,) float32 vector."""
        frames, times, timestamps_ms = self._load_gif_with_time(gif_path)
        return self._embed_from_frames(frames, times, timestamps_ms)

    def embed_video(self, video_path: str, flip: bool = False) -> np.ndarray:
        """
        Embed a video file (MP4/WebM). Returns (388,) float32 vector.
        
        Args:
            video_path: Path to video file
            flip: If True, horizontally flip frames (for mirror robustness)
        """
        frames, times, timestamps_ms = self._load_video_with_time(video_path, flip=flip)
        return self._embed_from_frames(frames, times, timestamps_ms)

    def close(self):
        """Release MediaPipe resources."""
        # HandLandmarker instances are created per sequence and cleaned up via context manager
        pass

    # ----------------------------
    # Loading + time base
    # ----------------------------
    def _load_gif_with_time(self, gif_path: str):
        """Load GIF frames with timing information from frame durations.
        
        Returns:
            frames: List of RGB numpy arrays
            times: Array of times in seconds (for compatibility)
            timestamps_ms: List of timestamps in milliseconds (for MediaPipe VIDEO mode)
        """
        frames = []
        times = []
        timestamps_ms = []
        t_ms = 0.0

        with Image.open(gif_path) as im:
            i = 0
            while True:
                try:
                    im.seek(i)
                    frames.append(np.array(im.convert("RGB")))
                    dur = float(im.info.get("duration", 100.0))  # ms
                    dur = max(dur, 1.0)
                    times.append(t_ms / 1000.0)
                    timestamps_ms.append(int(t_ms))
                    t_ms += dur
                    i += 1
                except EOFError:
                    break

        return frames, np.array(times, dtype=np.float32), timestamps_ms

    def _load_video_with_time(self, video_path: str, flip: bool = False):
        """Load video frames with FPS-aware sampling.
        
        Returns:
            frames: List of RGB numpy arrays
            times: Array of times in seconds (for compatibility)
            timestamps_ms: List of timestamps in milliseconds (for MediaPipe VIDEO mode)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], np.array([], dtype=np.float32), []

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1e-3:
            src_fps = 30.0  # reasonable fallback

        frames = []
        times = []
        timestamps_ms = []
        max_frames = int(self.sample_fps * self.max_seconds)
        step = max(int(round(src_fps / self.sample_fps)), 1)

        frame_i = 0
        kept = 0
        while kept < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Horizontal flip for mirror robustness
                if flip:
                    frame = np.ascontiguousarray(frame[:, ::-1, :])
                
                frames.append(frame)
                time_sec = kept / float(self.sample_fps)
                times.append(time_sec)
                timestamps_ms.append(int(time_sec * 1000))
                kept += 1
            frame_i += 1

        cap.release()
        return frames, np.array(times, dtype=np.float32), timestamps_ms

    # ----------------------------
    # Core pipeline
    # ----------------------------
    def _embed_from_frames(self, frames, times: np.ndarray, timestamps_ms: list[int]) -> np.ndarray:
        """Core embedding pipeline from frames.
        
        Args:
            frames: List of RGB numpy arrays
            times: Array of times in seconds (kept for compatibility)
            timestamps_ms: List of timestamps in milliseconds for MediaPipe VIDEO mode
        """
        if not frames:
            raise ValueError("No frames loaded")

        # 1) Landmarks + wrist xy (for optional location features)
        X_raw, present, wrist_xy = self._extract_landmarks(frames, timestamps_ms)

        # 2) Trim idle frames (reduces webcam lead-in/lead-out noise)
        X_raw, present, wrist_xy = self._trim_to_activity(X_raw, present, wrist_xy, pad=2)

        # Check if we have any valid frames after trimming
        if X_raw.shape[0] == 0:
            raise ValueError("No hand detected in any frame")

        # 3) Fill missing (interpolate) + smooth
        X_filled = self._interp_missing(X_raw, present)
        X_smooth = self._ema_smooth(X_filled, alpha=self.ema_alpha)

        # 4) Normalise (wrist-centre + scale by wrist->middle_mcp)
        X_norm = self._normalize_sequence(X_smooth)

        # 5) Resample to fixed length
        X_resampled = self._resample_sequence(X_norm, self.target_frames)

        # 6) Embedding = temporal stats + wrist stats
        emb_stats = self._compute_temporal_stats(X_resampled)  # 378 dims
        emb_wrist = self._wrist_location_stats(wrist_xy)       # 10 dims
        return np.concatenate([emb_stats, emb_wrist]).astype(np.float32)  # 388 dims

    # ----------------------------
    # Landmark extraction with stable slot assignment
    # ----------------------------
    def _extract_landmarks(self, frames, timestamps_ms: list[int]):
        """
        Extract landmarks with stable hand slot assignment using HandLandmarker API.
        
        Args:
            frames: List of RGB numpy arrays
            timestamps_ms: List of timestamps in milliseconds for VIDEO mode
        
        Returns:
            X: (L, 126) landmark array
            present: (L, 2) boolean array indicating hand presence
            wrist_xy: (L, 2, 2) wrist x,y positions (NaN when missing)
        """
        L = len(frames)
        X = np.zeros((L, 2, 21, 3), dtype=np.float32)
        present = np.zeros((L, 2), dtype=bool)
        wrist_xy = np.full((L, 2, 2), np.nan, dtype=np.float32)

        prev_wrist = [None, None]  # per slot: np.array([x,y]) in image-normalised coords

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
                
                hands = []

                if result.hand_landmarks:
                    for h_lm in result.hand_landmarks:
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lm], dtype=np.float32)
                        wrist = coords[0, :2].copy()
                        hands.append((coords, wrist))

                if len(hands) == 0:
                    continue

                # Assign to slots with continuity (prevents left/right swaps)
                if len(hands) == 1:
                    coords, wrist = hands[0]
                    slot = self._best_slot_for_single(wrist, prev_wrist)
                    X[i, slot] = coords
                    present[i, slot] = True
                    wrist_xy[i, slot] = wrist
                    prev_wrist[slot] = wrist
                else:
                    (c0, w0), (c1, w1) = hands[0], hands[1]

                    if prev_wrist[0] is None or prev_wrist[1] is None:
                        # fallback: leftmost wrist x -> slot0
                        if w0[0] <= w1[0]:
                            a0, a1 = (c0, w0), (c1, w1)
                        else:
                            a0, a1 = (c1, w1), (c0, w0)
                    else:
                        # choose assignment that minimises total wrist movement
                        cost_same = np.linalg.norm(w0 - prev_wrist[0]) + np.linalg.norm(w1 - prev_wrist[1])
                        cost_swap = np.linalg.norm(w0 - prev_wrist[1]) + np.linalg.norm(w1 - prev_wrist[0])
                        if cost_same <= cost_swap:
                            a0, a1 = (c0, w0), (c1, w1)
                        else:
                            a0, a1 = (c1, w1), (c0, w0)

                    (cA, wA), (cB, wB) = a0, a1
                    X[i, 0] = cA
                    present[i, 0] = True
                    wrist_xy[i, 0] = wA
                    prev_wrist[0] = wA
                    
                    X[i, 1] = cB
                    present[i, 1] = True
                    wrist_xy[i, 1] = wB
                    prev_wrist[1] = wB

        return X.reshape(L, 126), present, wrist_xy

    def _best_slot_for_single(self, wrist, prev_wrist):
        """Determine best slot for a single detected hand based on continuity."""
        if prev_wrist[0] is None and prev_wrist[1] is None:
            return 0 if wrist[0] <= 0.5 else 1
        if prev_wrist[0] is None:
            return 1
        if prev_wrist[1] is None:
            return 0
        d0 = np.linalg.norm(wrist - prev_wrist[0])
        d1 = np.linalg.norm(wrist - prev_wrist[1])
        return 0 if d0 <= d1 else 1

    # ----------------------------
    # Trim, fill, smooth
    # ----------------------------
    def _trim_to_activity(self, X_raw, present, wrist_xy, pad=2):
        """Trim sequence to frames with hand activity, with padding."""
        active = np.where(present.any(axis=1))[0]
        if len(active) == 0:
            return X_raw, present, wrist_xy

        a0, a1 = int(active[0]), int(active[-1])
        start = max(a0 - pad, 0)
        end = min(a1 + pad + 1, X_raw.shape[0])

        return X_raw[start:end], present[start:end], wrist_xy[start:end]

    def _interp_missing(self, X_raw, present):
        """Interpolate missing landmarks between detected frames."""
        L = X_raw.shape[0]
        X = X_raw.reshape(L, 2, 21, 3).copy()

        for h in range(2):
            idx = np.where(present[:, h])[0]
            if len(idx) == 0:
                continue

            first, last = idx[0], idx[-1]
            # Forward fill from first detection
            X[:first, h] = X[first, h]
            # Backward fill from last detection
            X[last + 1:, h] = X[last, h]

            # Linear interpolation for gaps
            for t in range(first, last):
                if present[t, h]:
                    continue
                t2 = idx[idx > t][0]
                t1 = idx[idx < t][-1]
                a = (t - t1) / float(t2 - t1)
                X[t, h] = (1 - a) * X[t1, h] + a * X[t2, h]

        return X.reshape(L, 126)

    def _ema_smooth(self, X, alpha=0.65):
        """Apply exponential moving average smoothing."""
        Y = X.copy()
        for t in range(1, Y.shape[0]):
            Y[t] = alpha * Y[t] + (1 - alpha) * Y[t - 1]
        return Y

    # ----------------------------
    # Normalise + resample
    # ----------------------------
    def _normalize_sequence(self, X_raw):
        """Normalize landmarks: wrist-centered, scaled by wrist->middle_mcp distance."""
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
        """Resample sequence to fixed length using linear interpolation."""
        L, D = X.shape
        if L == target_len:
            return X
        if L <= 1:
            return np.repeat(X[:1], target_len, axis=0) if L == 1 else np.zeros((target_len, D), np.float32)

        x_old = np.linspace(0, 1, L)
        x_new = np.linspace(0, 1, target_len)
        X_new = np.zeros((target_len, D), dtype=np.float32)
        for d in range(D):
            X_new[:, d] = np.interp(x_new, x_old, X[:, d])
        return X_new

    # ----------------------------
    # Embedding
    # ----------------------------
    def _compute_temporal_stats(self, X):
        """Compute temporal statistics: mean, std, velocity_mean."""
        mean_vec = X.mean(axis=0)
        std_vec = X.std(axis=0)
        if X.shape[0] > 1:
            velocity_mean = np.diff(X, axis=0).mean(axis=0)
        else:
            velocity_mean = np.zeros(X.shape[1], dtype=np.float32)
        return np.concatenate([mean_vec, std_vec, velocity_mean]).astype(np.float32)

    def _wrist_location_stats(self, wrist_xy):
        """
        Compute wrist location statistics.
        
        Args:
            wrist_xy: (L, 2, 2) with NaNs when missing.
            
        Returns:
            10-dim vector:
              - per slot: mean(x,y), std(x,y) => 8 dims
              - inter-wrist distance mean/std (when both available) => 2 dims
        """
        feats = []
        for h in range(2):
            w = wrist_xy[:, h, :]  # (L, 2)
            valid = ~np.isnan(w).any(axis=1)
            if valid.any():
                feats.extend(np.nanmean(w[valid], axis=0).tolist())
                feats.extend(np.nanstd(w[valid], axis=0).tolist())
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])

        # Inter-wrist distance when both hands visible
        both = (~np.isnan(wrist_xy[:, 0, :]).any(axis=1)) & (~np.isnan(wrist_xy[:, 1, :]).any(axis=1))
        if both.any():
            d = np.linalg.norm(wrist_xy[both, 0, :] - wrist_xy[both, 1, :], axis=1)
            feats.append(float(d.mean()))
            feats.append(float(d.std()))
        else:
            feats.extend([0.0, 0.0])

        return np.array(feats, dtype=np.float32)

    # ----------------------------
    # Utility: Hand coverage check
    # ----------------------------
    def check_hand_coverage(self, video_path: str, threshold: float = 0.3) -> bool:
        """
        Check if video has sufficient hand visibility.
        
        Args:
            video_path: Path to video file
            threshold: Minimum fraction of frames with hands detected
            
        Returns:
            True if coverage is sufficient, False otherwise
        """
        frames, _, timestamps_ms = self._load_video_with_time(video_path)
        if not frames:
            return False
        
        _, present, _ = self._extract_landmarks(frames, timestamps_ms)
        coverage = present.any(axis=1).mean()
        return coverage >= threshold
