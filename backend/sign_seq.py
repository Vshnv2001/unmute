import os
import pickle
import numpy as np

class SignSequenceManager:
    def __init__(self, pkl_dir: str = "sgsl_processed/landmarks_pkl"):
        self.pkl_dir = pkl_dir

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
        if not os.path.exists(pkl_path):
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
        Returns: {
            "frames": [
                { "pose": [[x,y,z]...] },  # 33 landmarks per frame
                ...
            ],
            "L_orig": int,
            "L_max": int
        }
        """
        pkl_path = os.path.join(self.pkl_dir, f"{sign_name}_full_body_pose.pkl")
        if not os.path.exists(pkl_path):
            return None
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        # Data X is (L, 99). 
        # 99 = 33 landmarks * 3 coordinates (x, y, z)
        # Flattened as [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]
        X = data["X"]
        L, D = X.shape
        
        frames_out = []
        for t in range(L):
            row = X[t]
            
            # Reshape (33, 3) - 33 landmarks with x, y, z coordinates
            # Round to 4 decimal places to reduce JSON size
            pose = np.round(row.reshape(33, 3), 4).tolist()
            
            frames_out.append({
                "pose": pose
            })
            
        return {
            "frames": frames_out,
            "L_orig": data.get("L_orig", L),
            "L_max": data.get("L_max", L)
        }
