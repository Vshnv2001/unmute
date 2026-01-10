import os
import glob
import json
import pickle
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from datetime import datetime

def discover_sign_folders(dataset_root):
    """List all subdirectories in dataset_root."""
    return [
        d for d in glob.glob(os.path.join(dataset_root, "*"))
        if os.path.isdir(d)
    ]

def find_gif_and_json(sign_dir):
    """Find the first .gif and .json file in the directory."""
    gifs = glob.glob(os.path.join(sign_dir, "*.gif"))
    jsons = glob.glob(os.path.join(sign_dir, "*.json"))
    
    gif_path = gifs[0] if gifs else None
    json_path = jsons[0] if jsons else None
    
    return gif_path, json_path

def load_gif_frames(gif_path):
    """Load GIF frames as list of RGB numpy arrays with timestamps.
    
    Returns:
        frames: List of RGB numpy arrays
        timestamps_ms: List of timestamps in milliseconds for MediaPipe VIDEO mode
    """
    frames = []
    timestamps_ms = []
    t_ms = 0.0
    try:
        with Image.open(gif_path) as im:
            index = 0
            while True:
                try:
                    im.seek(index)
                    # Convert to RGB (handle palettes/transparency)
                    frame = im.convert('RGB')
                    frames.append(np.array(frame))
                    # Get frame duration in milliseconds
                    dur = float(im.info.get("duration", 100.0))  # ms
                    dur = max(dur, 1.0)
                    timestamps_ms.append(int(t_ms))
                    t_ms += dur
                    index += 1
                except EOFError:
                    break
    except Exception as e:
        print(f"Error loading {gif_path}: {e}")
        return [], []
    return frames, timestamps_ms

def run_mediapipe_hands(frames, timestamps_ms: list[int], model_path: str):
    """
    Run MediaPipe HandLandmarker on a list of frames using VIDEO mode.
    
    Args:
        frames: List of RGB numpy arrays
        timestamps_ms: List of timestamps in milliseconds for VIDEO mode
        model_path: Path to hand_landmarker.task model file
    
    Returns: (L, 126) numpy array.
             L = number of frames
             126 = 2 hands * 21 landmarks * 3 coords
    """
    L = len(frames)
    X_raw = np.zeros((L, 126), dtype=np.float32)
    
    # Create HandLandmarker instance for this sequence
    BaseOptions = mp_tasks.BaseOptions
    HandLandmarker = mp_vision.HandLandmarker
    HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
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
            
            # We need to map left/right consistently.
            # MediaPipe hand_landmarks list corresponds to handedness.
            # We want [Left 21x3, Right 21x3] flattened.
            
            # Default zero
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
                    
                    # Extract coords (x,y,z)
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
                        
            # Flatten and store: [lh_x, lh_y, lh_z, ... rh_x, rh_y, rh_z ...]
            # 21*3 = 63. 63*2 = 126.
            vec = np.concatenate([lh.flatten(), rh.flatten()])
            X_raw[i] = vec
        
    return X_raw

def normalize_sequence(X_raw):
    """
    Normalize landmarks for avatar replay.
    Shape: (L, 126) -> (L, 126)
    
    For each hand:
      - Translate: Wrist (idx 0) -> (0,0,0)
      - Scale: Dist(Wrist, MiddleMCP(9)) -> 1.0 (approx)
    """
    # X_raw is (L, 126) -> splits into (L, 63) left, (L, 63) right
    # Reshape to (L, 2, 21, 3) for easier math
    L = X_raw.shape[0]
    X_reshaped = X_raw.reshape(L, 2, 21, 3)
    
    X_norm = np.zeros_like(X_reshaped)
    
    # Indices
    WRIST = 0
    MIDDLE_MCP = 9
    
    for t in range(L):
        for h in range(2): # 0=Left, 1=Right
            hand = X_reshaped[t, h] # (21, 3)
            
            # Check if hand is present (not all zeros)
            if np.all(hand == 0):
                continue
                
            wrist = hand[WRIST]
            middle = hand[MIDDLE_MCP]
            
            # Translation
            hand_centered = hand - wrist
            
            # Scale
            dist = np.linalg.norm(middle - wrist)
            scale = dist if dist > 1e-6 else 1.0
            
            X_norm[t, h] = hand_centered / scale
            
    return X_norm.reshape(L, 126)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sgsl_dataset", help="Path to input dataset")
    parser.add_argument("--output", default="sgsl_processed", help="Path to output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max signs to process (for testing)")
    parser.add_argument("--model_path", default=None, help="Path to hand_landmarker.task model file")
    args = parser.parse_args()
    
    # Set default model path if not provided
    model_path = args.model_path
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
    
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Hand landmarker model file not found. Please specify --model_path "
            "or ensure hand_landmarker.task exists in mediapipe_experiments/ or mediapipe/"
        )
    
    # Setup Output
    os.makedirs(os.path.join(args.output, "landmarks_pkl"), exist_ok=True)
    
    # Discovery
    t0 = datetime.now()
    sign_folders = discover_sign_folders(args.dataset)
    print(f"Found {len(sign_folders)} potential sign folders.")
    
    if args.limit:
        sign_folders = sign_folders[:args.limit]
        print(f"Limiting to first {args.limit} folders.")
        
    # Pass 1: Compute L_max (skip actual processing for now, just load gifs?)
    # Generating L_max is best done if we know distribution. 
    # For MVP, loading all GIFs twice is slow. 
    # Strategy: Process and store, track max length. THEN pad in a second pass or just save L_max in meta and pad at runtime? 
    # Task says "Resample/pad each sequence to L_max". 
    # Let's do a quick pass of opening GIFs for length.
    
    print("Pass 1: Computing L_max...")
    max_len = 0
    valid_folders = []
    
    for sf in sign_folders:
        gif_path, _ = find_gif_and_json(sf)
        if not gif_path:
            continue
            
        try:
            with Image.open(gif_path) as im:
                # Pillow property for frame count
                n_frames = getattr(im, 'n_frames', 1)
                max_len = max(max_len, n_frames)
                valid_folders.append((sf, gif_path))
        except Exception:
            pass
            
    print(f"Global L_max: {max_len}")
    if max_len == 0:
        print("No valid GIFs found.")
        return

    # Pass 2: Process
    print("Pass 2: Processing...")
    print(f"Using model: {model_path}")
    
    processed_count = 0
    
    meta_record = {
        "L_max": max_len,
        "processed_at": str(datetime.now()),
        "signs": []
    }
    
    for sf, gif_path in valid_folders:
        sign_name = os.path.basename(sf)
        _, json_path = find_gif_and_json(sf)
        
        # Load JSON meta
        meta_data = {}
        if json_path:
            try:
                with open(json_path, 'r') as f:
                    meta_data = json.load(f)
            except:
                pass
                
        # Load Frames
        frames, timestamps_ms = load_gif_frames(gif_path)
        if not frames:
            continue
            
        # Run MediaPipe
        X_raw = run_mediapipe_hands(frames, timestamps_ms, model_path)
        
        # Normalize
        X_norm = normalize_sequence(X_raw)
        
        # Pad to L_max
        L_curr = X_norm.shape[0]
        if L_curr < max_len:
            padding = np.zeros((max_len - L_curr, 126), dtype=np.float32)
            X_final = np.concatenate([X_norm, padding], axis=0)
        else:
            X_final = X_norm[:max_len]
            
        # Save PKL
        out_path = os.path.join(args.output, "landmarks_pkl", f"{sign_name}.pkl")
        payload = {
            "sign": sign_name,
            "X": X_final,
            "L_orig": L_curr,
            "L_max": max_len,
            "meta": meta_data
        }
        
        with open(out_path, 'wb') as f:
            pickle.dump(payload, f)
            
        meta_record["signs"].append(sign_name)
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} signs...")
    
    # Save Global Meta
    with open(os.path.join(args.output, "meta.json"), 'w') as f:
        json.dump(meta_record, f, indent=2)
        
    print(f"Done. Processed {processed_count} signs. Outputs in {args.output}")

if __name__ == "__main__":
    main()
