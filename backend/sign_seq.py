import os
import pickle
import numpy as np

class SignSequenceManager:
    def __init__(self, pkl_dir: str = None):
        if pkl_dir is None:
            # Get absolute path relative to this file's location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            pkl_dir = os.path.join(parent_dir, "sgsl_processed", "landmarks_pkl")
        self.pkl_dir = pkl_dir
        print(f"[SignSequenceManager] PKL directory: {self.pkl_dir}")

    def get_sign_frames(self, sign_name: str):
        """
        Load frames for a given sign.
        Returns: {
            "frames": [
                { "left": [[x,y,z]...], "right": [[x,y,z]...] },
                ...
            ],
            "L_orig": int,
            "L_max": int
        }
        """
        pkl_path = os.path.join(self.pkl_dir, f"{sign_name}.pkl")
        print(f"PKL path: {pkl_path}")
        print(f"PKL path exists: {os.path.exists(pkl_path)}")
        if not os.path.exists(pkl_path):
            print(f"Sign data not found for {sign_name}")
            print(f"Checked path: {os.path.abspath(pkl_path)}")
            return None
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        # Data X is (L, 126). 
        # 0-62 = Left flattened. 63-125 = Right flattened.
        X = data["X"]
        L, D = X.shape
        
        # Filter out zero-padded frames - only keep frames with actual data
        frames_out = []
        for t in range(L):
            row = X[t]
            
            # Skip frames that are entirely zeros
            if not np.any(row != 0):
                continue
                
            lh_flat = row[:63]
            rh_flat = row[63:]
            
            # Reshape (21, 3)
            lh = np.round(lh_flat.reshape(21, 3), 4).tolist()
            rh = np.round(rh_flat.reshape(21, 3), 4).tolist()
            
            frames_out.append({
                "left": lh,
                "right": rh
            })
        
        print(f"[get_sign_frames] {sign_name}: {len(frames_out)} non-zero frames out of {L} total")
            
        return {
            "frames": frames_out,
            "L_orig": data.get("L_orig", L),
            "L_max": data.get("L_max", L)
        }

    def get_sign_pose_frames(self, sign_name: str):
        """
        Load full body pose frames for a given sign.
        Checks for {sign_name}.pkl and converts hand data to pose-like format.
        Returns: {
            "frames": [
                { "pose": [[x,y,z]...] },  # landmarks per frame
                ...
            ],
            "L_orig": int,
            "L_max": int
        }
        """
        pkl_path = os.path.join(self.pkl_dir, f"{sign_name}.pkl")
        print(f"Pose PKL path: {pkl_path}")
        print(f"Pose PKL path exists: {os.path.exists(pkl_path)}")
        if not os.path.exists(pkl_path):
            print(f"Pose sign data not found for {sign_name}")
            print(f"Checked path: {os.path.abspath(pkl_path)}")
            return None
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        X = data["X"]
        L, D = X.shape
        print(f"Data shape: ({L}, {D})")
        
        # Check data format based on dimension
        if D == 99:
            # Full body pose data: 33 landmarks × 3 coordinates
            frames_out = []
            for t in range(L):
                row = X[t]
                pose = np.round(row.reshape(33, 3), 4).tolist()
                frames_out.append({"pose": pose})
            
            return {
                "frames": frames_out,
                "L_orig": data.get("L_orig", L),
                "L_max": data.get("L_max", L)
            }
        elif D == 126:
            # Hand-only data: 21 landmarks × 3 coordinates × 2 hands
            # Convert to a format with left and right hand landmarks
            print(f"Converting hand data (126 elements) to pose format")
            frames_out = []
            for t in range(L):
                row = X[t]
                
                # Skip frames that are entirely zeros (padding)
                if not np.any(row != 0):
                    continue
                
                lh_flat = row[:63]
                rh_flat = row[63:]
                
                # Reshape to (21, 3) for each hand
                lh = np.round(lh_flat.reshape(21, 3), 4).tolist()
                rh = np.round(rh_flat.reshape(21, 3), 4).tolist()
                
                frames_out.append({
                    "left_hand": lh,
                    "right_hand": rh
                })
            
            print(f"[get_sign_pose_frames] {sign_name}: {len(frames_out)} non-zero frames out of {L} total")
            
            return {
                "frames": frames_out,
                "L_orig": data.get("L_orig", L),
                "L_max": data.get("L_max", L),
                "format": "hands"  # Indicate this is hand data
            }
        else:
            print(f"Unknown data format with {D} elements")
            return None
