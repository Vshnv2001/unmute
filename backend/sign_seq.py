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
        
        # Filter out zero-padded frames first
        non_zero_frames = []
        for t in range(L):
            row = X[t]
            if np.any(row != 0):
                non_zero_frames.append(row)
        
        if len(non_zero_frames) == 0:
            print(f"[get_sign_frames] {sign_name}: No non-zero frames found")
            return None
        
        # Stack into array for normalization
        X_filtered = np.array(non_zero_frames)
        
        # Normalize to 0-1 range
        # Find global min/max across all non-zero values
        X_nonzero = X_filtered[X_filtered != 0]
        if len(X_nonzero) > 0:
            x_min = X_nonzero.min()
            x_max = X_nonzero.max()
            print(f"[get_sign_frames] {sign_name}: Data range [{x_min:.4f}, {x_max:.4f}]")
            
            # Normalize: (x - min) / (max - min)
            X_normalized = np.zeros_like(X_filtered)
            mask = X_filtered != 0
            X_normalized[mask] = (X_filtered[mask] - x_min) / (x_max - x_min)
        else:
            X_normalized = X_filtered
        
        # Convert to frames
        frames_out = []
        for row in X_normalized:
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
            
            # Filter out zero-padded frames first
            non_zero_frames = []
            for t in range(L):
                row = X[t]
                if np.any(row != 0):
                    non_zero_frames.append(row)
            
            if len(non_zero_frames) == 0:
                print(f"[get_sign_pose_frames] {sign_name}: No non-zero frames found")
                return None
            
            # Stack into array for normalization
            X_filtered = np.array(non_zero_frames)
            
            # Normalize to 0-1 range
            X_nonzero = X_filtered[X_filtered != 0]
            if len(X_nonzero) > 0:
                x_min = X_nonzero.min()
                x_max = X_nonzero.max()
                print(f"[get_sign_pose_frames] {sign_name}: Data range [{x_min:.4f}, {x_max:.4f}]")
                
                # Normalize: (x - min) / (max - min)
                X_normalized = np.zeros_like(X_filtered)
                mask = X_filtered != 0
                X_normalized[mask] = (X_filtered[mask] - x_min) / (x_max - x_min)
            else:
                X_normalized = X_filtered
            
            # Convert to frames
            frames_out = []
            for row in X_normalized:
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
