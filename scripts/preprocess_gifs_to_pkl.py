import os
import glob
import json
import pickle
import argparse
import numpy as np
import mediapipe as mp
from PIL import Image
from datetime import datetime
from multiprocessing import Pool, cpu_count

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
    """Load GIF frames as list of RGB numpy arrays."""
    frames = []
    try:
        with Image.open(gif_path) as im:
            index = 0
            while True:
                try:
                    im.seek(index)
                    # Convert to RGB (handle palettes/transparency)
                    frame = im.convert('RGB')
                    frames.append(np.array(frame))
                    index += 1
                except EOFError:
                    break
    except Exception as e:
        print(f"Error loading {gif_path}: {e}")
        return []
    return frames

def run_mediapipe_hands(frames, hands_solution):
    """
    Run MediaPipe Hands on a list of frames.
    Returns: (L, 126) numpy array.
             L = number of frames
             126 = 2 hands * 21 landmarks * 3 coords
    """
    L = len(frames)
    X_raw = np.zeros((L, 126), dtype=np.float32)
    
    for i, frame in enumerate(frames):
        results = hands_solution.process(frame)
        
        # We need to map left/right consistently.
        # MediaPipe 'multi_hand_landmarks' list corresponds to 'multi_handedness'.
        # We want [Left 21x3, Right 21x3] flattend.
        
        # Default zero
        lh = np.zeros((21, 3), dtype=np.float32)
        rh = np.zeros((21, 3), dtype=np.float32)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Check label
                label = results.multi_handedness[hand_idx].classification[0].label
                # MediaPipe: "Left" matches left hand in camera view (which is user's right if mirrored?)
                # Usually standard input is mirrored. "Left" label -> user's left hand if not mirrored.
                # We'll trust the label.
                
                # Extract coords (x,y,z)
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                
                if label == "Left":
                    lh = coords
                else:
                    rh = coords
                    
        # Flatten and store: [lh_x, lh_y, lh_z, ... rh_x, rh_y, rh_z ...]
        # 21*3 = 63. 63*2 = 126.
        vec = np.concatenate([lh.flatten(), rh.flatten()])
        X_raw[i] = vec
        
    return X_raw

def run_mediapipe_pose(frames, pose_solution):
    """
    Run MediaPipe Pose on a list of frames.
    Returns: (L, 99) numpy array with raw coordinates.
             L = number of frames
             99 = 33 landmarks * 3 coords (x, y, z)
    """
    L = len(frames)
    X_raw = np.zeros((L, 99), dtype=np.float32)
    
    for i, frame in enumerate(frames):
        results = pose_solution.process(frame)
        
        # Default zero array for pose landmarks
        pose_landmarks = np.zeros((33, 3), dtype=np.float32)
        
        if results.pose_landmarks:
            # Extract coords (x, y, z) for all 33 pose landmarks
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        
        # Flatten and store: [x0, y0, z0, x1, y1, z1, ... x32, y32, z32]
        # 33*3 = 99
        vec = pose_landmarks.flatten()
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

def process_single_sign(args_tuple):
    """
    Worker function to process a single sign.
    Creates MediaPipe solutions in this process and processes the sign.
    
    Args:
        args_tuple: (sign_folder, gif_path, output_dir, max_len)
    
    Returns:
        sign_name if successful, None otherwise
    """
    sf, gif_path, output_dir, max_len = args_tuple
    sign_name = os.path.basename(sf)
    
    try:
        # Create MediaPipe solutions in this worker process
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Find JSON file
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
        frames = load_gif_frames(gif_path)
        if not frames:
            mp_hands.close()
            mp_pose.close()
            return None
        
        # Run MediaPipe Hands
        X_raw = run_mediapipe_hands(frames, mp_hands)
        
        # Normalize
        X_norm = normalize_sequence(X_raw)
        
        # Pad to L_max
        L_curr = X_norm.shape[0]
        if L_curr < max_len:
            padding = np.zeros((max_len - L_curr, 126), dtype=np.float32)
            X_final = np.concatenate([X_norm, padding], axis=0)
        else:
            X_final = X_norm[:max_len]
        
        # Save hand landmarks PKL
        out_path = os.path.join(output_dir, "landmarks_pkl", f"{sign_name}.pkl")
        payload = {
            "sign": sign_name,
            "X": X_final,
            "L_orig": L_curr,
            "L_max": max_len,
            "meta": meta_data
        }
        
        with open(out_path, 'wb') as f:
            pickle.dump(payload, f)
        
        # Run MediaPipe Pose for full body
        X_pose_raw = run_mediapipe_pose(frames, mp_pose)
        
        # Pad pose data to L_max (raw coordinates, no normalization)
        L_pose_curr = X_pose_raw.shape[0]
        if L_pose_curr < max_len:
            pose_padding = np.zeros((max_len - L_pose_curr, 99), dtype=np.float32)
            X_pose_final = np.concatenate([X_pose_raw, pose_padding], axis=0)
        else:
            X_pose_final = X_pose_raw[:max_len]
        
        # Save full body pose PKL
        pose_out_path = os.path.join(output_dir, "landmarks_pkl", f"{sign_name}_full_body_pose.pkl")
        pose_payload = {
            "sign": sign_name,
            "X": X_pose_final,
            "L_orig": L_pose_curr,
            "L_max": max_len,
            "meta": meta_data
        }
        
        with open(pose_out_path, 'wb') as f:
            pickle.dump(pose_payload, f)
        
        # Cleanup
        mp_hands.close()
        mp_pose.close()
        
        return sign_name
        
    except Exception as e:
        print(f"Error processing {sign_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sgsl_dataset", help="Path to input dataset")
    parser.add_argument("--output", default="sgsl_processed", help="Path to output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max signs to process (for testing)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()
    
    # Determine number of workers
    num_workers = args.workers if args.workers else cpu_count()
    print(f"Using {num_workers} worker processes")
    
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

    # Pass 2: Process with multiprocessing
    print("Pass 2: Processing...")
    
    # Prepare arguments for worker function
    worker_args = [
        (sf, gif_path, args.output, max_len)
        for sf, gif_path in valid_folders
    ]
    
    meta_record = {
        "L_max": max_len,
        "processed_at": str(datetime.now()),
        "signs": []
    }
    
    # Process signs in parallel
    processed_count = 0
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for progress tracking
        results = pool.imap_unordered(process_single_sign, worker_args)
        
        for sign_name in results:
            if sign_name is not None:
                meta_record["signs"].append(sign_name)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(valid_folders)} signs...")
    
    # Sort signs for consistent output
    meta_record["signs"].sort()
    
    # Save Global Meta
    with open(os.path.join(args.output, "meta.json"), 'w') as f:
        json.dump(meta_record, f, indent=2)
        
    print(f"Done. Processed {processed_count} signs. Outputs in {args.output}")

if __name__ == "__main__":
    main()
