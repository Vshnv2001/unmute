
import os
import sys
import json
import numpy as np
import faiss
import glob

# Add root to path to allow importing backend modules
sys.path.append(os.getcwd())

from backend.hand_embedder import HandEmbedder


def main():
    dataset_root = "sgsl_dataset"
    output_dir = "sgsl_processed/retrieval_artifacts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate GIFs directly (removes PKL drift, ensures consistency)
    gif_paths = glob.glob(os.path.join(dataset_root, "*", "*.gif"))
    if not gif_paths:
        print("No GIFs found in dataset. Please ensure sgsl_dataset/<sign>/*.gif exists.")
        return
    
    print(f"Found {len(gif_paths)} GIF files. Building index with HandEmbedder...")
    
    embedder = HandEmbedder(target_frames=30, sample_fps=15.0, max_seconds=4.0)
    
    vectors = []
    labels = []
    video_paths = []
    
    errors = 0
    for i, gif_path in enumerate(gif_paths):
        sign_name = os.path.basename(os.path.dirname(gif_path))
        
        try:
            vec = embedder.embed_gif(gif_path)  # (388,)
            vectors.append(vec)
            labels.append(sign_name)
            video_paths.append(os.path.abspath(gif_path))
            
        except Exception as e:
            print(f"Error embedding {gif_path}: {e}")
            errors += 1
            continue
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(gif_paths)} GIFs...")
    
    embedder.close()
    
    if not vectors:
        print("No valid vectors extracted.")
        return
    
    print(f"\nProcessed {len(vectors)} GIFs successfully ({errors} errors)")
    
    vectors = np.array(vectors, dtype=np.float32)
    print(f"Embeddings shape: {vectors.shape}")
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)
    print("Vectors normalized for cosine similarity.")
    
    # Build FAISS Index using Inner Product (cosine sim for normalized vectors)
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    print(f"Index built with {index.ntotal} vectors, dim={d}")
    
    # Save Artifacts
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    np.save(os.path.join(output_dir, "embeddings.npy"), vectors)
    
    with open(os.path.join(output_dir, "labels.json"), 'w') as f:
        json.dump(labels, f)
        
    with open(os.path.join(output_dir, "video_paths.json"), 'w') as f:
        json.dump(video_paths, f)
        
    print(f"\nArtifacts saved to {output_dir}/")
    print(f"  - faiss.index ({index.ntotal} vectors, {d} dims)")
    print(f"  - embeddings.npy")
    print(f"  - labels.json ({len(labels)} labels)")
    print(f"  - video_paths.json ({len(video_paths)} paths)")


if __name__ == "__main__":
    main()
