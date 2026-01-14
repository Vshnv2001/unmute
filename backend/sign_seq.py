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
        
        frames_out = []
        for t in range(L):
            row = X[t]
            lh_flat = row[:63]
            rh_flat = row[63:]
            
            # Reshape (21, 3)
            # We assume non-zero means present? Or just send all.
            # Frontend can handle zeros.
            
            # To reduce JSON size, maybe round floats?
            # 3 decimal places is enough for normalized coords?
            lh = np.round(lh_flat.reshape(21, 3), 4).tolist()
            rh = np.round(rh_flat.reshape(21, 3), 4).tolist()
            
            frames_out.append({
                "left": lh,
                "right": rh
            })
            
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
        
        frames_out = []
        
        if D == 99:
            # Pose data: 33 landmarks * 3 coordinates
            for t in range(L):
                row = X[t]
                pose = np.round(row.reshape(33, 3), 4).tolist()
                frames_out.append({"pose": pose})
        elif D == 126:
            # Hand data: 2 hands * 21 landmarks * 3 coordinates
            # Convert to pose-like format: create 33 "landmarks" 
            # where we map hands to approximate body positions
            for t in range(L):
                row = X[t]
                lh_flat = row[:63]  # Left hand
                rh_flat = row[63:]  # Right hand
                
                lh = np.round(lh_flat.reshape(21, 3), 4)
                rh = np.round(rh_flat.reshape(21, 3), 4)
                
                # Create a 33-landmark "pose" array
                # Map hands to approximate shoulder/arm positions
                # Landmarks 0-10: face (fill with zeros or center)
                # Landmarks 11-16: arms (use hand wrist as reference)
                # Landmarks 17-22: hands left (use left hand landmarks)
                # Landmarks 23-32: lower body (fill with zeros)
                
                pose = np.zeros((33, 3))
                
                # Use wrist (landmark 0) as arm endpoint
                if np.any(lh[0] != 0):
                    pose[15] = lh[0]  # Left wrist
                    pose[13] = lh[0] + [0, -0.1, 0]  # Left elbow (approximate)
                    pose[11] = lh[0] + [0.1, -0.2, 0]  # Left shoulder (approximate)
                    
                if np.any(rh[0] != 0):
                    pose[16] = rh[0]  # Right wrist
                    pose[14] = rh[0] + [0, -0.1, 0]  # Right elbow (approximate)
                    pose[12] = rh[0] + [-0.1, -0.2, 0]  # Right shoulder (approximate)
                
                # Add some hand finger tips for visibility
                pose[17:21] = lh[4:8]  # Left fingertips
                pose[21:25] = rh[4:8]  # Right fingertips
                
                frames_out.append({"pose": np.round(pose, 4).tolist()})
        else:
            print(f"Unknown data shape: {D}")
            return None
            
        return {
            "frames": frames_out,
            "L_orig": data.get("L_orig", L),
            "L_max": data.get("L_max", L)
        }
