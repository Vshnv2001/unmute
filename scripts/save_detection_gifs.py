#!/usr/bin/env python3
"""
Script to save hand and body detection results from a GIF to two separate output GIFs.
Based on preprocess_gifs_to_pkl.py for detection logic.
"""

import os
import sys
import argparse
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw

# Note: We define our own load_gif_with_durations function to preserve frame timings

# MediaPipe Pose landmark connections for drawing
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
    (9, 10),  # Mouth
    # Upper body
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Shoulder to hip
    (23, 24),  # Hips
    # Lower body (may not be visible)
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]

# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]

def draw_hand_landmarks(frame, hand_results):
    """
    Draw only hand landmarks on a frame.
    Returns annotated frame as numpy array.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    
    # Draw Hand landmarks
    if hand_results and hand_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Get hand label
            label = hand_results.multi_handedness[hand_idx].classification[0].label
            hand_color = (255, 255, 0) if label == "Left" else (255, 165, 0)  # Yellow or Orange
            
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks.landmark[start_idx]
                end = hand_landmarks.landmark[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                draw.line([(x1, y1), (x2, y2)], fill=hand_color, width=2)
            
            # Draw landmarks
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                draw.ellipse([x-2, y-2, x+2, y+2], fill=hand_color)
            
            # Draw label
            if hand_landmarks.landmark:
                wrist = hand_landmarks.landmark[0]
                label_x, label_y = int(wrist.x * w), int(wrist.y * h) - 20
                try:
                    draw.text((label_x, label_y), label, fill=hand_color)
                except:
                    pass  # Font not available
    
    return np.array(img)

def draw_pose_landmarks(frame, pose_results):
    """
    Draw only pose/body landmarks on a frame.
    Returns annotated frame as numpy array.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    
    # Draw Pose landmarks
    if pose_results and pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks.landmark
        
        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]
                
                # Only draw if both landmarks are visible
                if start.visibility > 0.5 and end.visibility > 0.5:
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=2)
        
        # Draw pose landmarks
        for idx, landmark in enumerate(pose_landmarks):
            if landmark.visibility > 0.5:
                x, y = int(landmark.x * w), int(landmark.y * h)
                # Color code: head (0-10) = blue, upper body (11-16) = green, rest = red
                if idx <= 10:
                    color = (0, 0, 255)  # Blue for head
                elif idx <= 16:
                    color = (0, 255, 0)  # Green for upper body
                else:
                    color = (255, 0, 0)  # Red for lower body
                draw.ellipse([x-3, y-3, x+3, y+3], fill=color)
    
    return np.array(img)

def load_gif_with_durations(gif_path):
    """
    Load GIF frames with their durations.
    Returns: (frames, durations) where durations is a list of frame durations in milliseconds
    """
    frames = []
    durations = []
    try:
        with Image.open(gif_path) as im:
            index = 0
            while True:
                try:
                    im.seek(index)
                    # Convert to RGB (handle palettes/transparency)
                    frame = im.convert('RGB')
                    frames.append(np.array(frame))
                    
                    # Get frame duration (default to 100ms if not available)
                    duration = im.info.get('duration', 100)
                    durations.append(duration)
                    
                    index += 1
                except EOFError:
                    break
    except Exception as e:
        print(f"Error loading {gif_path}: {e}")
        return [], []
    return frames, durations

def save_gif(frames, output_path, durations=None, default_duration=100):
    """
    Save a list of frames as a GIF.
    
    Args:
        frames: List of numpy arrays (RGB images)
        output_path: Path to save the GIF
        durations: List of frame durations in milliseconds (optional)
        default_duration: Default duration if durations not provided
    """
    if not frames:
        print(f"Warning: No frames to save for {output_path}")
        return
    
    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    
    # Use provided durations or default
    if durations and len(durations) == len(frames):
        # Save with individual frame durations
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0
        )
    else:
        # Save with uniform duration
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=default_duration,
            loop=0
        )

def process_gif_to_detection_gifs(input_gif_path, output_hand_path, output_body_path, duration_ms=None):
    """
    Process a GIF and save hand and body detection results to two separate GIFs.
    
    Args:
        input_gif_path: Path to input GIF file
        output_hand_path: Path to save hand detection GIF
        output_body_path: Path to save body detection GIF
        duration_ms: Frame duration in milliseconds (None to use original durations)
    """
    print(f"Loading GIF: {input_gif_path}")
    frames, durations = load_gif_with_durations(input_gif_path)
    
    if not frames:
        print("Error: Could not load frames from GIF")
        return False
    
    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")
    
    # Use provided duration or original durations
    if duration_ms is not None:
        durations = [duration_ms] * len(frames)
    
    # Initialize MediaPipe solutions
    print("Initializing MediaPipe...")
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
    
    # Process frames
    print("Processing frames...")
    hand_frames = []
    body_frames = []
    
    for i, frame in enumerate(frames):
        # Process with MediaPipe
        hand_results = mp_hands.process(frame)
        pose_results = mp_pose.process(frame)
        
        # Draw hand landmarks only
        hand_frame = draw_hand_landmarks(frame, hand_results)
        hand_frames.append(hand_frame)
        
        # Draw pose landmarks only
        body_frame = draw_pose_landmarks(frame, pose_results)
        body_frames.append(body_frame)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames...")
    
    mp_hands.close()
    mp_pose.close()
    
    # Save hand detection GIF
    print(f"\nSaving hand detection GIF to: {output_hand_path}")
    save_gif(hand_frames, output_hand_path, durations=durations, default_duration=100)
    
    # Save body detection GIF
    print(f"Saving body detection GIF to: {output_body_path}")
    save_gif(body_frames, output_body_path, durations=durations, default_duration=100)
    
    print("\n✅ Done! Both GIFs saved successfully.")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Save hand and body detection results from a GIF to two separate output GIFs"
    )
    parser.add_argument(
        "input_gif",
        help="Path to input GIF file"
    )
    parser.add_argument(
        "--output-hand",
        "-oh",
        default=None,
        help="Path to save hand detection GIF (default: <input_name>_hands.gif)"
    )
    parser.add_argument(
        "--output-body",
        "-ob",
        default=None,
        help="Path to save body detection GIF (default: <input_name>_body.gif)"
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=None,
        help="Frame duration in milliseconds (default: use original GIF frame durations)"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.input_gif):
        print(f"Error: GIF file not found: {args.input_gif}")
        sys.exit(1)
    
    # Generate default output paths if not provided
    input_dir = os.path.dirname(args.input_gif)
    input_basename = os.path.splitext(os.path.basename(args.input_gif))[0]
    
    if args.output_hand is None:
        args.output_hand = os.path.join(input_dir, f"{input_basename}_hands.gif")
    
    if args.output_body is None:
        args.output_body = os.path.join(input_dir, f"{input_basename}_body.gif")
    
    # Create output directory if needed
    for output_path in [args.output_hand, args.output_body]:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Process the GIF
    success = process_gif_to_detection_gifs(
        args.input_gif,
        args.output_hand,
        args.output_body,
        args.duration
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
