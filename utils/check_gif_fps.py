import sys
from pathlib import Path
from PIL import Image

def get_gif_fps(gif_path: Path) -> float:
    """Return the frames‑per‑second (FPS) of a GIF.

    The FPS is calculated as ``num_frames / total_duration`` where the total
    duration is the sum of the ``duration`` attribute of each frame (in
    milliseconds).  If the GIF does not contain duration information we fall
    back to a default of 100 ms per frame (10 FPS).
    """
    with Image.open(gif_path) as im:
        if im.format != "GIF":
            raise ValueError(f"{gif_path} is not a GIF image")
        num_frames = im.n_frames
        # Some GIFs store the per‑frame delay in ``info['duration']`` (ms).
        # Pillow exposes it via ``im.info['duration']`` for the current frame.
        total_ms = 0
        for frame in range(num_frames):
            im.seek(frame)
            frame_duration = im.info.get('duration', 100)  # default 100 ms
            total_ms += frame_duration
        total_seconds = total_ms / 1000.0
        if total_seconds == 0:
            return 0.0
        return num_frames / total_seconds

def main(argv=None):
    argv = argv or sys.argv[1:]
    
    # Default to sgsl_dataset folder in the current directory
    dataset_path = Path("../sgsl_dataset")
    
    # Allow user to specify a different dataset path
    if argv:
        dataset_path = Path(argv[0])
    
    if not dataset_path.is_dir():
        print(f"Dataset directory not found: {dataset_path}")
        return 1
    
    print(f"Scanning dataset: {dataset_path}")
    print("-" * 60)
    
    fps_values = []
    errors = []
    
    # Iterate through all subdirectories
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    total_dirs = len(subdirs)
    
    for idx, folder in enumerate(subdirs, 1):
        folder_name = folder.name
        gif_path = folder / f"{folder_name}.gif"
        
        if gif_path.is_file():
            try:
                fps = get_gif_fps(gif_path)
                fps_values.append(fps)
                
                # Show progress every 50 items or for the last item
                if idx % 50 == 0 or idx == total_dirs:
                    print(f"Processed {idx}/{total_dirs} folders...")
            except Exception as e:
                errors.append((folder_name, str(e)))
    
    print("-" * 60)
    
    if not fps_values:
        print("No valid GIF files found in the dataset.")
        return 1
    
    # Calculate statistics
    avg_fps = sum(fps_values) / len(fps_values)
    min_fps = min(fps_values)
    max_fps = max(fps_values)
    
    print(f"\nDataset Statistics:")
    print(f"  Total GIFs processed: {len(fps_values)}")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Min FPS: {min_fps:.2f}")
    print(f"  Max FPS: {max_fps:.2f}")
    
    if errors:
        print(f"\n  Errors encountered: {len(errors)}")
        print(f"  First few errors:")
        for folder_name, error in errors[:5]:
            print(f"    - {folder_name}: {error}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
