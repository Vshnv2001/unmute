
import os
import faiss
import json
import numpy as np
from typing import List, Dict
from backend.hand_embedder import HandEmbedder


class RetrievalService:
    """
    Sign retrieval service using FAISS index and HandEmbedder.
    
    Features:
    - Dual-search (normal + flipped) for mirror robustness
    - Confidence threshold filtering
    - Margin threshold for ambiguous match rejection
    - Result aggregation by label
    """
    
    # Confidence threshold for cosine similarity (0 to 1, higher = more similar)
    CONFIDENCE_THRESHOLD = 0.55
    
    # Minimum margin between top-1 and top-2 similarity (reject if too ambiguous)
    MARGIN_THRESHOLD = 0.03
    
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self.index = None
        self.index_dim = None  # Expected embedding dimension
        self.labels = []
        self.video_paths = []
        self.embedder = HandEmbedder(target_frames=30, sample_fps=15.0, max_seconds=4.0)
        self._load_artifacts()

    def _load_artifacts(self):
        index_path = os.path.join(self.artifacts_dir, "faiss.index")
        labels_path = os.path.join(self.artifacts_dir, "labels.json")
        paths_path = os.path.join(self.artifacts_dir, "video_paths.json")

        if os.path.exists(index_path):
            print(f"Loading index from {index_path}")
            self.index = faiss.read_index(index_path)
            self.index_dim = self.index.d  # Store expected dimension
            print(f"  Index has {self.index.ntotal} vectors, dim={self.index_dim}")
            
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
                
        if os.path.exists(paths_path):
            with open(paths_path, 'r') as f:
                self.video_paths = json.load(f)

    def search(self, video_path: str, k: int = 20) -> List[Dict]:
        """
        Process a query video and retrieve top-k similar signs.
        
        Uses dual-search (normal + flipped) for mirror robustness,
        with confidence threshold and margin filtering.
        
        Args:
            video_path: Path to query video (MP4/WebM)
            k: Number of candidates to retrieve per search
            
        Returns:
            List of result dicts with label, similarity, confidence, video_path
        """
        if not self.index:
            print("No index loaded")
            return []
        
        if self.index_dim is None:
            print("Index dimension not initialized")
            return []

        try:
            # Helper function to validate and reshape embedding
            def prepare_query_vector(embedding: np.ndarray) -> np.ndarray:
                """Validate and reshape embedding to (1, index_dim) for FAISS search."""
                # Ensure float32
                embedding = embedding.astype(np.float32)
                
                # Flatten to 1D if needed
                embedding_flat = embedding.flatten()
                
                # Validate dimension
                if embedding_flat.shape[0] != self.index_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.index_dim}, "
                        f"got {embedding_flat.shape[0]}. "
                        f"Embedding shape: {embedding.shape}"
                    )
                
                # Reshape to (1, index_dim) for FAISS
                return np.reshape(embedding_flat, (1, self.index_dim))
            
            # 1) Query embedding (normal orientation)
            emb1 = self.embedder.embed_video(video_path, flip=False)
            q1 = prepare_query_vector(emb1)
            faiss.normalize_L2(q1)
            sims1, idxs1 = self.index.search(q1, k)

            # 2) Query embedding (flipped for mirror robustness)
            emb2 = self.embedder.embed_video(video_path, flip=True)
            q2 = prepare_query_vector(emb2)
            faiss.normalize_L2(q2)
            sims2, idxs2 = self.index.search(q2, k)

            # 3) Merge candidates and aggregate by label (keep max similarity)
            candidates = []
            for sims, idxs in [(sims1, idxs1), (sims2, idxs2)]:
                for j in range(k):
                    idx = int(idxs[0][j])
                    sim = float(sims[0][j])
                    if idx < 0 or idx >= len(self.labels):
                        continue
                    candidates.append((self.labels[idx], sim, idx))

            # Aggregate by label, keeping best similarity
            by_label = {}
            for label, sim, idx in candidates:
                if (label not in by_label) or (sim > by_label[label]["similarity"]):
                    by_label[label] = {
                        "label": label,
                        "similarity": sim,
                        "video_path": self.video_paths[idx] if idx < len(self.video_paths) else None,
                    }

            results = sorted(by_label.values(), key=lambda x: x["similarity"], reverse=True)

            # Debug logging
            print(f"\n{'='*50}")
            print(f"RETRIEVAL DEBUG (threshold: {self.CONFIDENCE_THRESHOLD}, margin: {self.MARGIN_THRESHOLD})")
            print(f"{'='*50}")
            for i, r in enumerate(results[:10]):
                status = "✓ PASS" if r["similarity"] >= self.CONFIDENCE_THRESHOLD else "✗ FILTERED"
                print(f"  [{i+1}] {r['label']:20s} | sim: {r['similarity']:.4f} | {status}")
            print(f"{'='*50}")

            # 4) Apply confidence threshold
            if not results:
                print("No results found")
                return []

            if results[0]["similarity"] < self.CONFIDENCE_THRESHOLD:
                print(f"Top result below threshold ({results[0]['similarity']:.4f} < {self.CONFIDENCE_THRESHOLD})")
                return []

            # 5) Apply margin threshold (reject if too ambiguous)
            if len(results) >= 2:
                margin = results[0]["similarity"] - results[1]["similarity"]
                if margin < self.MARGIN_THRESHOLD:
                    print(f"Result too ambiguous (margin: {margin:.4f} < {self.MARGIN_THRESHOLD})")
                    return []

            # Format top-5 results
            out = []
            for r in results[:5]:
                if r["similarity"] >= self.CONFIDENCE_THRESHOLD:
                    out.append({
                        "label": r["label"],
                        "similarity": r["similarity"],
                        "confidence": f"{r['similarity'] * 100:.1f}%",
                        "video_path": r["video_path"],
                    })
            
            print(f"Returning {len(out)} results")
            print(f"{'='*50}\n")
            return out

        except ValueError as e:
            # Dimension mismatch or validation errors
            print(f"Dimension validation error: {e}")
            print(f"  Index dimension: {self.index_dim}")
            if self.index:
                print(f"  Index type: {type(self.index).__name__}")
            import traceback
            traceback.print_exc()
            return []
        except Exception as e:
            print(f"Search error: {e}")
            print(f"  Index dimension: {self.index_dim}")
            if self.index:
                print(f"  Index vectors: {self.index.ntotal}, dim: {self.index.d}")
            import traceback
            traceback.print_exc()
            return []

    def close(self):
        """Release embedder resources."""
        self.embedder.close()
