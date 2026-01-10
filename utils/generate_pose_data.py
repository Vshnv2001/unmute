#!/usr/bin/env python3
"""
generate_pose_data.py
"""

import os
import cv2
import glob
import json  # NEW: for report
import pickle
import imageio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "./mediapipe/hand_landmarker.task"
DATASET_DIR = "./sgsl_dataset"
OUT_DIR = None  # None → write pose.pkl inside each sign folder; else path
OVERWRITE = True
MAX_HANDS = 2
EMA_ALPHA = 0.35
CONTINUITY_LAMBDA = 80
DEBUG_VIS = False
DEBUG_EVERY_N = 0
FPS_FALLBACK = 30.0

# How many frames with any detected hand are required to consider the export valid
REQUIRE_MIN_DETECTIONS = 5  # NEW: tweak for your GIF lengths
REPORT_FILENAME = "generation_report.json"  # NEW

# ----------------------------
# MediaPipe Hand landmark names (0..20) — use ALL landmarks
# ----------------------------
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Convenience indices used only for normalization/scale
WRIST = 0
MIDDLE_MCP = 9

# ----------------------------
# Dataclasses
# ----------------------------
@dataclass
class HandStream:
    xyz: np.ndarray              # (T, 21, 3) all landmarks
    world: Optional[np.ndarray]  # (T, 21, 3)
    present: np.ndarray          # (T,)
    prob_smoothed: np.ndarray    # (T,)

@dataclass
class SequencePack:
    sign: str
    num_frames: int
    image_size: Tuple[int, int]
    fps: float
    handedness_strategy: str
    left: HandStream
    right: HandStream
    origin_xy: np.ndarray        # (T, 2)
    scale: np.ndarray            # (T,)
    left_norm: np.ndarray        # (T, 21, 3)
    right_norm: np.ndarray       # (T, 21, 3)
    left_bbox: np.ndarray        # (T, 4)
    right_bbox: np.ndarray       # (T, 4)

# ----------------------------
# Utils
# ----------------------------
def load_gif_rgb_frames(path: str) -> Tuple[List[np.ndarray], List[float]]:
    reader = imageio.get_reader(path)
    frames, durations = [], []
    try:
        for i, frame in enumerate(reader):
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frames.append(frame)
            meta = reader.get_meta_data(index=i)
            dur = meta.get("duration", None)
            if dur is None:
                dur = reader.get_meta_data().get("duration", None)
            durations.append(float(dur) if dur is not None else np.nan)
    finally:
        reader.close()
    if np.any(np.isnan(durations)):
        finite = [d for d in durations if not np.isnan(d)]
        durations = [float(np.median(finite)) if np.isnan(d) else d for d in durations] if finite else [1000.0 / FPS_FALLBACK] * len(frames)
    return frames, durations

def to_mp_image(frame_rgb: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

def _bbox_from_landmarks(lms: np.ndarray) -> np.ndarray:
    xs, ys = lms[:, 0], lms[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

def _palm_size(lms: np.ndarray) -> float:
    return float(np.linalg.norm(lms[MIDDLE_MCP, :2] - lms[WRIST, :2]) + 1e-8)

def _origin_from_two_hands(lmsL: np.ndarray, lmsR: np.ndarray) -> np.ndarray:
    hasL = not np.isnan(lmsL).any()
    hasR = not np.isnan(lmsR).any()
    if hasL and hasR:
        return 0.5 * (lmsL[WRIST, :2] + lmsR[WRIST, :2])
    elif hasL:
        return lmsL[WRIST, :2]
    elif hasR:
        return lmsR[WRIST, :2]
    return np.array([np.nan, np.nan], dtype=np.float32)

def _normalize(lms: np.ndarray, origin_xy: np.ndarray, scale: float) -> np.ndarray:
    out = lms.copy()
    if not np.isnan(lms).any() and not np.isnan(origin_xy).any() and scale > 0:
        out[:, :2] = (out[:, :2] - origin_xy[None, :]) / scale
    else:
        out[:] = np.nan
    return out

def _ema(prev: Optional[float], new: float, alpha: float) -> float:
    return new if prev is None else alpha * new + (1.0 - alpha) * prev

# ----------------------------
# Handedness resolver
# ----------------------------
class HandednessResolver:
    def __init__(self, img_w: int, img_h: int, ema_alpha: float = EMA_ALPHA, continuity_lambda: float = CONTINUITY_LAMBDA):
        self.img_w = img_w; self.img_h = img_h
        self.ema_alpha = ema_alpha; self.cont_lambda = continuity_lambda
        self.prev_left_wrist = None; self.prev_right_wrist = None
        self.prev_left_prob = None; self.prev_right_prob = None

    def assign(self, det_xy: List[np.ndarray], probs_left: List[float], probs_right: List[float]) -> Tuple[Optional[int], Optional[int], float, float]:
        n = len(det_xy)
        if n == 0:
            return None, None, self.prev_left_prob or 0.0, self.prev_right_prob or 0.0
        if n == 1:
            xw, yw = det_xy[0][WRIST, 0] * self.img_w, det_xy[0][WRIST, 1] * self.img_h
            cost_left = -probs_left[0]; cost_right = -probs_right[0]
            if self.prev_left_wrist is not None:
                cost_left += np.linalg.norm(np.array([xw, yw]) - np.array(self.prev_left_wrist)) / self.cont_lambda
            if self.prev_right_wrist is not None:
                cost_right += np.linalg.norm(np.array([xw, yw]) - np.array(self.prev_right_wrist)) / self.cont_lambda
            if cost_left <= cost_right:
                idx_left, idx_right = 0, None; pL, pR = probs_left[0], 1.0 - probs_left[0]
            else:
                idx_left, idx_right = None, 0; pL, pR = 1.0 - probs_right[0], probs_right[0]
        else:
            a, b = 0, 1
            wrists = []
            for k in (a, b):
                wrists.append((det_xy[k][WRIST, 0] * self.img_w, det_xy[k][WRIST, 1] * self.img_h))
            score_ab = probs_left[a] + probs_right[b]
            score_ba = probs_left[b] + probs_right[a]
            pen_ab = 0.0; pen_ba = 0.0
            if self.prev_left_wrist is not None:
                pen_ab += np.linalg.norm(np.array(wrists[a]) - np.array(self.prev_left_wrist)) / self.cont_lambda
                pen_ba += np.linalg.norm(np.array(wrists[b]) - np.array(self.prev_left_wrist)) / self.cont_lambda
            if self.prev_right_wrist is not None:
                pen_ab += np.linalg.norm(np.array(wrists[b]) - np.array(self.prev_right_wrist)) / self.cont_lambda
                pen_ba += np.linalg.norm(np.array(wrists[a]) - np.array(self.prev_right_wrist)) / self.cont_lambda
            if (score_ab - pen_ab) >= (score_ba - pen_ba):
                idx_left, idx_right = a, b; pL, pR = probs_left[a], probs_right[b]
            else:
                idx_left, idx_right = b, a; pL, pR = probs_left[b], probs_right[a]
        pL_s = _ema(self.prev_left_prob, pL, self.ema_alpha)
        pR_s = _ema(self.prev_right_prob, pR, self.ema_alpha)
        self.prev_left_prob, self.prev_right_prob = pL_s, pR_s
        if idx_left is not None:
            self.prev_left_wrist = (det_xy[idx_left][WRIST, 0] * self.img_w, det_xy[idx_left][WRIST, 1] * self.img_h)
        if idx_right is not None:
            self.prev_right_wrist = (det_xy[idx_right][WRIST, 0] * self.img_w, det_xy[idx_right][WRIST, 1] * self.img_h)
        return idx_left, idx_right, pL_s, pR_s

# ----------------------------
# Core extraction (uses ALL 21 landmarks)
# ----------------------------
def extract_from_frames(frames: List[np.ndarray], durations_ms: List[float], model_path: str) -> SequencePack:
    H, W, _ = frames[0].shape
    fps = 1000.0 / float(np.median(durations_ms)) if len(durations_ms) else FPS_FALLBACK
    T = len(frames)

    def alloc_stream():
        return (np.full((T, 21, 3), np.nan, dtype=np.float32),
                np.full((T,), False, dtype=bool),
                np.full((T, 21, 3), np.nan, dtype=np.float32))
    left_xyz, left_present, left_world = alloc_stream()
    right_xyz, right_present, right_world = alloc_stream()
    left_p_sm = np.zeros((T,), dtype=np.float32); right_p_sm = np.zeros((T,), dtype=np.float32)
    left_bbox = np.full((T, 4), np.nan, dtype=np.float32); right_bbox = np.full((T, 4), np.nan, dtype=np.float32)

    BaseOptions = mp_python.BaseOptions
    HandLandmarker = mp_vision.HandLandmarker
    HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    resolver = HandednessResolver(img_w=W, img_h=H)

    with HandLandmarker.create_from_options(options) as landmarker:
        ts_ms = 0.0
        for t, (frame, dur) in enumerate(zip(frames, durations_ms)):
            result = landmarker.detect_for_video(to_mp_image(frame), int(ts_ms))
            ts_ms += dur

            det_xy, det_xyz, det_world, p_left, p_right = [], [], [], [], []
            if result and result.hand_landmarks:
                for i, landmarks in enumerate(result.hand_landmarks):
                    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (21,3)
                    det_xyz.append(arr); det_xy.append(arr[:, :2])
                    if result.hand_world_landmarks and len(result.hand_world_landmarks) > i:
                        warr = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[i]], dtype=np.float32)
                    else:
                        warr = np.full((21, 3), np.nan, dtype=np.float32)
                    det_world.append(warr)
                for hd in result.handedness:
                    pl = pr = 0.0
                    for cat in hd:
                        name = cat.category_name.lower()
                        if name.startswith("left"):
                            pl = float(cat.score)
                        elif name.startswith("right"):
                            pr = float(cat.score)
                    if (pl + pr) > 1.0:
                        s = pl + pr; pl, pr = pl / s, pr / s
                    p_left.append(pl); p_right.append(pr)

            li, ri, pL_s, pR_s = resolver.assign(det_xy, p_left, p_right)
            left_p_sm[t], right_p_sm[t] = pL_s, pR_s
            if li is not None:
                left_xyz[t] = det_xyz[li]; left_world[t] = det_world[li]; left_present[t] = True
                left_bbox[t] = _bbox_from_landmarks(det_xyz[li][:, :2])
            if ri is not None:
                right_xyz[t] = det_xyz[ri]; right_world[t] = det_world[ri]; right_present[t] = True
                right_bbox[t] = _bbox_from_landmarks(det_xyz[ri][:, :2])

    origin_xy = np.full((T, 2), np.nan, dtype=np.float32)
    scale = np.full((T,), np.nan, dtype=np.float32)
    left_norm = np.full((T, 21, 3), np.nan, dtype=np.float32)
    right_norm = np.full((T, 21, 3), np.nan, dtype=np.float32)

    for t in range(T):
        l_ok, r_ok = left_present[t], right_present[t]
        lmsL = left_xyz[t] if l_ok else np.full((21, 3), np.nan, dtype=np.float32)
        lmsR = right_xyz[t] if r_ok else np.full((21, 3), np.nan, dtype=np.float32)
        o = _origin_from_two_hands(lmsL, lmsR); origin_xy[t] = o
        scales = []
        if l_ok: scales.append(_palm_size(lmsL))
        if r_ok: scales.append(_palm_size(lmsR))
        s = float(np.mean(scales)) if len(scales) else np.nan; scale[t] = s
        left_norm[t] = _normalize(lmsL, o, s); right_norm[t] = _normalize(lmsR, o, s)

    return SequencePack(
        sign="",
        num_frames=T,
        image_size=(H, W),
        fps=float(fps),
        handedness_strategy="video_smoothing_continuity",
        left=HandStream(xyz=left_xyz, world=left_world, present=left_present, prob_smoothed=left_p_sm),
        right=HandStream(xyz=right_xyz, world=right_world, present=right_present, prob_smoothed=right_p_sm),
        origin_xy=origin_xy, scale=scale, left_norm=left_norm, right_norm=right_norm,
        left_bbox=left_bbox, right_bbox=right_bbox,
    )

# ----------------------------
# Dataset driver with reporting
# ----------------------------
def process_dataset(dataset_dir: str = DATASET_DIR, model_path: str = MODEL_PATH, out_dir: Optional[str] = OUT_DIR, overwrite: bool = OVERWRITE):
    sign_dirs = sorted([p for p in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(p)])
    os.makedirs(out_dir, exist_ok=True) if out_dir else None

    report = {
        "generated": [],
        "skipped_exists": [],
        "missing_gif": [],
        "too_few_detections": [],
        "failed_exception": [],
    }

    total = len(sign_dirs)
    for idx, sdir in enumerate(sign_dirs, 1):
        sign = os.path.basename(sdir)
        gif_path = os.path.join(sdir, f"{sign}.gif")
        if not os.path.isfile(gif_path):
            cand = glob.glob(os.path.join(sdir, "*.gif"))
            if not cand:
                print(f"[{idx}/{total}] SKIP {sign}: no GIF found")
                report["missing_gif"].append(sign)
                continue
            gif_path = cand[0]

        out_path = (os.path.join(out_dir, f"{sign}.pkl") if out_dir else os.path.join(sdir, "pose.pkl"))
        if (not overwrite) and os.path.exists(out_path):
            print(f"[{idx}/{total}] SKIP {sign}: pose.pkl exists")
            report["skipped_exists"].append(sign)
            continue

        try:
            print(f"[{idx}/{total}] {sign}: loading {os.path.relpath(gif_path)}")
            frames, durs = load_gif_rgb_frames(gif_path)
            if len(frames) == 0:
                print(f"    ! empty GIF → skip")
                report["too_few_detections"].append(sign)
                continue

            pack = extract_from_frames(frames, durs, model_path)
            pack.sign = sign

            # Require at least N frames with any hand present
            n_present = int(np.count_nonzero(pack.left.present | pack.right.present))
            if n_present < REQUIRE_MIN_DETECTIONS:
                print(f"    ! too few detections ({n_present}) → skip")
                report["too_few_detections"].append(sign)
                continue

            # Optional quick visual check
            if DEBUG_VIS and (DEBUG_EVERY_N > 0) and (idx % DEBUG_EVERY_N == 0):
                vis_path = os.path.join(out_dir or sdir, f"{sign}_vis.mp4")
                _save_debug_vis(vis_path, frames, pack)

            # Save pose.pkl (ALL 21 landmarks + names in meta)
            blob = {
                "meta": {
                    "sign": pack.sign,
                    "num_frames": pack.num_frames,
                    "image_size": pack.image_size,
                    "fps": pack.fps,
                    "handedness_strategy": pack.handedness_strategy,
                    "landmark_names": LANDMARK_NAMES,  # NEW
                },
                "left": {
                    "xyz": pack.left.xyz,
                    "world": pack.left.world,
                    "present": pack.left.present,
                    "prob_smoothed": pack.left.prob_smoothed,
                    "bbox": pack.left_bbox,
                    "norm": pack.left_norm,
                },
                "right": {
                    "xyz": pack.right.xyz,
                    "world": pack.right.world,
                    "present": pack.right.present,
                    "prob_smoothed": pack.right.prob_smoothed,
                    "bbox": pack.right_bbox,
                    "norm": pack.right_norm,
                },
                "norm_meta": {
                    "origin_xy": pack.origin_xy,
                    "scale": pack.scale,
                },
            }
            with open(out_path, "wb") as f:
                pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"    → wrote {os.path.relpath(out_path)}")
            report["generated"].append(sign)

        except Exception as e:
            print(f"    ! ERROR processing {sign}: {e}")
            report["failed_exception"].append({"sign": sign, "error": str(e)})

    # -------- Summary / Report --------
    print("\n=== Generation Summary ===")
    print(f"Generated: {len(report['generated'])}/{total}")
    print(f"Skipped (exists): {len(report['skipped_exists'])}")
    print(f"Missing GIF: {len(report['missing_gif'])}")
    print(f"Too few detections (<{REQUIRE_MIN_DETECTIONS} frames): {len(report['too_few_detections'])}")
    print(f"Errors: {len(report['failed_exception'])}")

    # Show which signs weren't generated
    not_generated = sorted(set([os.path.basename(p) for p in sign_dirs]) - set(report["generated"]))
    if not_generated:
        print("\nSigns NOT generated:")
        for s in not_generated:
            print(" -", s)

    # Save JSON report next to dataset root (or OUT_DIR root if set)
    report_path = os.path.join(OUT_DIR or DATASET_DIR, REPORT_FILENAME)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {os.path.relpath(report_path)}")

# ----------------------------
# Optional: simple visualizer (unchanged)
# ----------------------------
def _save_debug_vis(path: str, frames: List[np.ndarray], pack: SequencePack, fps: Optional[float] = None):
    H, W = pack.image_size
    fps = fps or pack.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

    def draw_hand(img, pts01, color):
        for x, y in pts01:
            cv2.circle(img, (int(x * W), int(y * H)), 3, color, -1)

    for t, frame in enumerate(frames):
        img = frame.copy()
        if pack.left.present[t]:
            draw_hand(img, pack.left.xyz[t][:, :2], (0, 255, 0))
            cv2.putText(img, f"L p={pack.left.prob_smoothed[t]:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)
        if pack.right.present[t]:
            draw_hand(img, pack.right.xyz[t][:, :2], (255, 0, 0))
            cv2.putText(img, f"R p={pack.right.prob_smoothed[t]:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"    (debug) wrote {os.path.relpath(path)}")

if __name__ == "__main__":
    process_dataset(DATASET_DIR, MODEL_PATH, OUT_DIR, OVERWRITE)
