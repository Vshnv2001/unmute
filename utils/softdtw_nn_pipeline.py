#!/usr/bin/env python3
"""
softdtw_nn_pipeline.py
Build soft-DTW NN prototypes with strong augmentations and classify pose.pkl clips.

- Compatible with generate_pose_data.py outputs (includes meta.landmark_names)
- Uses ALL 21 MediaPipe landmarks per hand

Usage:
  python softdtw_nn_pipeline.py build-index --dataset ./sgsl_dataset --out ./processed/prototypes --n-aug 200
  python softdtw_nn_pipeline.py classify --protos ./processed/prototypes --query ./sgsl_dataset/abuse/pose.pkl --k 5
"""
import os, glob, json, pickle, argparse, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# ---- Optional soft-DTW; fallback to fastdtw ----
HAVE_SOFTDTW = True
try:
    from tslearn.metrics import soft_dtw
    from tslearn.barycenters import softdtw_barycenter
except Exception:
    HAVE_SOFTDTW = False
    from fastdtw import fastdtw

# ----------------- I/O helpers -----------------
def load_pkl(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_npy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True
               ) if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)) else None
    np.save(path, arr)

def load_npy(path: str) -> np.ndarray:
    return np.load(path)

# ----------------- Landmark indexing -----------------
DEFAULT_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_FINGER_MCP","INDEX_FINGER_PIP","INDEX_FINGER_DIP","INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP","MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP","RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]
MIN_T = 4  # minimum frames we require for a sequence


def get_index(names: List[str], key: str, default: int) -> int:
    try:
        return names.index(key)
    except Exception:
        return default

# ----------------- Feature building -----------------
def _fill_time_nans(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Fill NaNs over time by nearest valid neighbor per joint.
    If a joint has no valid frames at all, fill the entire joint with fill_value.
    arr: (T, J, 3)
    """
    x = arr.copy()
    T, J, D = x.shape
    for j in range(J):
        valid = ~np.any(np.isnan(x[:, j, :]), axis=1)  # (T,)
        if valid.any():
            idx = np.where(valid)[0]
            vals = x[valid, j, :]                      # (#valid, 3)
            # nearest-neighbor fill over time
            for t in np.where(~valid)[0]:
                k = idx[np.argmin(np.abs(idx - t))]
                x[t, j, :] = vals[np.argmin(np.abs(idx - t))]
        else:
            # Entire joint is missing: hard-fill with zeros (normalized space)
            x[:, j, :] = fill_value
    return x

def _safe_diff(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=0, prepend=x[:1])

def _relational_feats(L: np.ndarray, R: np.ndarray, wrist_idx: int) -> np.ndarray:
    """
    Left/right wrist relational: dx, dy, dist, sinθ, cosθ. Shape (T,5).
    """
    lw = L[:, wrist_idx, :2]; rw = R[:, wrist_idx, :2]
    v = rw - lw
    dist = np.linalg.norm(v, axis=1, keepdims=True)
    ang = np.arctan2(v[:, 1], v[:, 0])
    sa = np.sin(ang)[:, None]; ca = np.cos(ang)[:, None]
    return np.concatenate([v, dist, sa, ca], axis=1).astype(np.float32)

def make_features_from_blob(blob: Dict) -> np.ndarray:
    """
    Build (T, D) features from pose.pkl using ALL 21 landmarks per hand.
      - positions (L+R): 21*3*2 = 126 dims
      - velocity & acceleration: +126 +126
      - relational wrist features (dx,dy,dist,sinθ,cosθ): +5
      => D = 126*3 + 5 = 383
    """
    meta = blob.get("meta", {})
    names = meta.get("landmark_names", DEFAULT_LANDMARK_NAMES)
    wrist_idx = get_index(names, "WRIST", 0)

    L = blob["left"]["norm"].astype(np.float32)   # (T,21,3) with NaNs possible
    R = blob["right"]["norm"].astype(np.float32)
    T = L.shape[0]

    Lf = _fill_time_nans(L).reshape(T, -1)  # (T, 21*3)
    Rf = _fill_time_nans(R).reshape(T, -1)  # (T, 21*3)
    X = np.concatenate([Lf, Rf], axis=1)    # (T, 126)

    V = _safe_diff(X)
    A = _safe_diff(V)
    rel = _relational_feats(_fill_time_nans(L), _fill_time_nans(R), wrist_idx)  # (T,5)

    feats = np.concatenate([X, V, A, rel], axis=1).astype(np.float32)
    return feats

# ----------------- Landmark-level augmentations (work on 21 joints) -----------------
def seq_resample_landmarks(seq: np.ndarray, newT: int) -> np.ndarray:
    """seq: (T, J, D). Linear resample to new length."""
    T = seq.shape[0]
    if newT == T or T < 2:
        return seq.copy()
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, newT)
    out = []
    for j in range(seq.shape[1]):
        out_j = []
        for d in range(seq.shape[2]):
            out_j.append(np.interp(t_new, t_old, seq[:, j, d]))
        out.append(np.stack(out_j, axis=1))
    return np.stack(out, axis=1).transpose(1,0,2)  # (newT,J,D)

def seq_resample_features(seq: np.ndarray, newT: int) -> np.ndarray:
    """Resample (T,D) feature sequences in time."""
    T, D = seq.shape
    if newT == T or T < 2:
        return seq.copy()
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, newT)
    return np.stack([np.interp(t_new, t_old, seq[:, d]) for d in range(D)], axis=1)

def global_speed(L: np.ndarray, R: np.ndarray, speed: float) -> Tuple[np.ndarray, np.ndarray]:
    # newT = max(4, int(round(L.shape[0] / speed)))
    newT = max(MIN_T, int(round(L.shape[0]/speed)))
    return seq_resample_landmarks(L, newT), seq_resample_landmarks(R, newT)

def elastic_timewarp(L: np.ndarray, R: np.ndarray, mag=0.15, knots=6):
    T = L.shape[0]
    anchors = np.linspace(0, T-1, knots).astype(int)
    shifts = np.random.uniform(-mag, mag, size=knots) * (T-1)
    warp = np.interp(np.arange(T), anchors, anchors + shifts)
    warp = np.clip(warp, 0, T-1)
    def warp_seq(S):
        out = []
        for j in range(S.shape[1]):
            out_j = []
            for d in range(S.shape[2]):
                out_j.append(np.interp(warp, np.arange(T), S[:, j, d]))
            out.append(np.stack(out_j, axis=1))
        return np.stack(out, axis=1).transpose(1,0,2)
    return warp_seq(L), warp_seq(R)

def random_trim(L: np.ndarray, R: np.ndarray, max_frac=0.1):
    """
    Trim start/end but never go below MIN_T frames. If the sequence is already
    short, return as-is.
    """
    T = L.shape[0]
    if T <= MIN_T:
        return L, R

    max_drop = int(max_frac * T)
    # Choose a and b so that T - (a+b) >= MIN_T
    a = np.random.randint(0, max_drop + 1) if max_drop > 0 else 0
    b_max = max(0, min(max_drop, T - MIN_T - a))
    b = np.random.randint(0, b_max + 1) if b_max > 0 else 0

    return L[a:T-b], R[a:T-b]


def frame_drop(L: np.ndarray, R: np.ndarray, drop_p=0.05):
    """
    Randomly drop frames independently with prob=drop_p, but ensure at least
    MIN_T frames remain. If T <= MIN_T, skip dropping entirely.
    """
    T = L.shape[0]
    if T <= MIN_T:
        return L, R

    keep = np.random.rand(T) > drop_p
    k = int(keep.sum())

    if k < MIN_T:
        # Force-keep exactly min(MIN_T, T) frames (no replacement issue)
        n_keep = min(MIN_T, T)
        idx = np.random.choice(T, size=n_keep, replace=False)
        keep[:] = False
        keep[idx] = True

    return L[keep], R[keep]


def spatial_transform(L: np.ndarray, R: np.ndarray, trans_std=0.02, scale_std=0.05, rot_deg=10):
    dx, dy = np.random.normal(0, trans_std, size=2)
    s = 1.0 + np.random.normal(0, scale_std)
    theta = np.deg2rad(np.random.uniform(-rot_deg, rot_deg))
    Rm = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    def xform(S):
        X = S.copy()
        xy = X[..., :2]
        xy = (xy @ Rm.T) * s + np.array([dx, dy], dtype=np.float32)
        X[..., :2] = xy
        return X
    return xform(L), xform(R)

def jitter(L: np.ndarray, R: np.ndarray, sigma=0.01):
    return (L + np.random.normal(0, sigma, L.shape).astype(np.float32),
            R + np.random.normal(0, sigma, R.shape).astype(np.float32))

def occlude(L: np.ndarray, R: np.ndarray, max_len_frac=0.1, which="rand"):
    T = L.shape[0]
    span = max(1, int(np.random.uniform(0.02, max_len_frac) * T))
    start = np.random.randint(0, T - span)
    target = which if which != "rand" else ("left" if np.random.rand() < 0.5 else "right")
    if target == "left":
        L[start:start+span] = np.nan
    else:
        R[start:start+span] = np.nan
    return L, R

def maybe_mirror(L: np.ndarray, R: np.ndarray, enable=True):
    if not enable:
        return L, R
    Lm, Rm = L.copy(), R.copy()
    Lm[..., 0] = -Lm[..., 0]
    Rm[..., 0] = -Rm[..., 0]
    return Rm, Lm  # swap

def build_features_with_aug(blob: Dict, allow_mirror=True) -> np.ndarray:
    """Apply augs to normalized landmarks then build features from ALL joints."""
    L = blob["left"]["norm"].astype(np.float32)
    R = blob["right"]["norm"].astype(np.float32)

    # Temporal augs
    if np.random.rand() < 0.9:
        L, R = global_speed(L, R, speed=np.random.uniform(0.7, 1.3))
    if np.random.rand() < 0.8:
        L, R = elastic_timewarp(L, R, mag=np.random.uniform(0.05, 0.2))
    if np.random.rand() < 0.6:
        L, R = random_trim(L, R, max_frac=0.12)
    if np.random.rand() < 0.6:
        L, R = frame_drop(L, R, drop_p=np.random.uniform(0.02, 0.08))

    # Spatial augs
    if np.random.rand() < 0.8:
        L, R = spatial_transform(L, R, trans_std=0.02, scale_std=0.05, rot_deg=10)
    if np.random.rand() < 0.8:
        L, R = jitter(L, R, sigma=np.random.uniform(0.003, 0.015))
    if np.random.rand() < 0.4:
        L, R = occlude(L, R, max_len_frac=0.12, which="rand")
    if np.random.rand() < 0.5:
        L, R = maybe_mirror(L, R, enable=allow_mirror)

    fake = {"left": {"norm": L}, "right": {"norm": R}, "meta": blob.get("meta", {})}
    return make_features_from_blob(fake)

# ----------------- Distance & prototypes -----------------
def zscore(seq: np.ndarray) -> np.ndarray:
    # Replace NaN/Inf first to avoid degenerate stats
    x = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    z = (x - mu) / sd
    return z


def softdtw_distance(a: np.ndarray, b: np.ndarray, gamma=0.1) -> float:
    a = zscore(a); b = zscore(b)
    # Final safety net
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    if HAVE_SOFTDTW:
        return float(soft_dtw(a, b, gamma=gamma))
    dist, _ = fastdtw(a, b, dist=lambda x, y: np.linalg.norm(x - y))
    return float(dist)


@dataclass
class ClassIndex:
    label: str
    proto_paths: List[str]      # .npy paths for each prototype OR single barycenter
    use_barycenter: bool

class SoftDTWNN:
    def __init__(self, proto_root: str, tau: float = 10.0, gamma: float = 0.1):
        self.proto_root = proto_root
        self.tau = tau
        self.gamma = gamma
        with open(os.path.join(proto_root, "index.json"), "r") as f:
            meta = json.load(f)
        self.labels = []
        self.classes: List[ClassIndex] = []
        for entry in meta["classes"]:
            self.labels.append(entry["label"])
            self.classes.append(ClassIndex(entry["label"], entry["proto_paths"], entry["use_barycenter"]))

    def topk(self, query_blob: Dict, k=5) -> List[Tuple[str, float]]:
        Q = make_features_from_blob(query_blob)
        dists = []
        for c in self.classes:
            if c.use_barycenter:
                P = load_npy(c.proto_paths[0])
                d = softdtw_distance(Q, P, gamma=self.gamma)
            else:
                ds = []
                for p in c.proto_paths:
                    P = load_npy(p)
                    ds.append(softdtw_distance(Q, P, gamma=self.gamma))
                d = float(np.min(ds))
            dists.append(d)
        d = np.array(dists, dtype=np.float32)
        scores = -d / self.tau
        scores -= scores.max()
        logp = scores - np.log(np.exp(scores).sum())
        idx = np.argsort(-logp)[:k]
        return [(self.labels[i], float(logp[i])) for i in idx]

# ----------------- Build index -----------------
def build_index(dataset_dir: str,
                out_dir: str,
                n_aug: int = 200,
                mirror_all: bool = True,
                use_barycenter: bool = False,
                seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    classes_meta = []
    signs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)])
    for sdir in signs:
        label = os.path.basename(sdir)
        pkl = os.path.join(sdir, "pose.pkl")
        if not os.path.exists(pkl):
            print(f"[skip] {label}: no pose.pkl")
            continue
        print(f"[build] {label}")
        blob = load_pkl(pkl)

        # Generate augmented features
        feats = []
        for _ in range(n_aug):
            feats.append(build_features_with_aug(blob, allow_mirror=mirror_all))
        # Always include the original (no aug) as a prototype
        feats.append(make_features_from_blob(blob))

        # Save prototypes
        class_dir = os.path.join(out_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        proto_paths = []
        if use_barycenter and HAVE_SOFTDTW:
            base = min(128, int(np.median([f.shape[0] for f in feats])))
            normed = [zscore(seq_resample_features(f, base)) for f in feats]
            bary = softdtw_barycenter(np.stack(normed, axis=0), gamma=0.1)  # (base, D)
            npy_path = os.path.join(class_dir, "barycenter.npy")
            save_npy(npy_path, bary.astype(np.float32))
            proto_paths = [npy_path]
            use_bc = True
        else:
            for i, f in enumerate(feats):
                npy_path = os.path.join(class_dir, f"proto_{i:04d}.npy")
                save_npy(npy_path, f.astype(np.float32))
                proto_paths.append(npy_path)
            use_bc = False

        classes_meta.append({"label": label, "proto_paths": proto_paths, "use_barycenter": use_bc})

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"classes": classes_meta}, f, indent=2)
    print(f"Done. Indexed {len(classes_meta)} classes into {out_dir}")

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index")
    b.add_argument("--dataset", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-aug", type=int, default=200)
    b.add_argument("--no-mirror", action="store_true", help="Disable mirroring")
    b.add_argument("--barycenter", action="store_true", help="Use one soft-DTW barycenter per class (requires tslearn)")
    b.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("classify")
    c.add_argument("--protos", required=True)
    c.add_argument("--query", required=True, help="path to pose.pkl")
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--tau", type=float, default=10.0)
    c.add_argument("--gamma", type=float, default=0.1)

    args = ap.parse_args()

    if args.cmd == "build-index":
        build_index(args.dataset, args.out, n_aug=args.n_aug,
                    mirror_all=not args.no_mirror, use_barycenter=args.barycenter, seed=args.seed)

    elif args.cmd == "classify":
        nn = SoftDTWNN(args.protos, tau=args.tau, gamma=args.gamma)
        blob = load_pkl(args.query)
        topk = nn.topk(blob, k=args.k)
        print("Top-k:")
        for gloss, logp in topk:
            print(f"  {gloss:>20s}   logp={logp: .3f}")

if __name__ == "__main__":
    main()
