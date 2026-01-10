#!/usr/bin/env python3
"""
Test script to visualize MediaPipe Hands and Pose detections on a single GIF.
Based on the methods from preprocess_gifs_to_pkl.py
"""

import os
import sys
import argparse
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import functions from preprocess script
sys.path.insert(0, os.path.dirname(__file__))
from preprocess_gifs_to_pkl import (
    load_gif_frames,
    normalize_sequence
)

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

def draw_landmarks_on_frame(frame, hand_results, pose_results, frame_idx):
    """
    Draw MediaPipe landmarks on a frame.
    Returns annotated frame as numpy array.
    """
    # Convert to PIL Image for drawing
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

def visualize_detections(gif_path, output_path=None, max_frames=None):
    """
    Load a GIF, process it with MediaPipe, and visualize the detections.
    """
    print(f"Loading GIF: {gif_path}")
    frames = load_gif_frames(gif_path)
    
    if not frames:
        print("Error: Could not load frames from GIF")
        return
    
    if max_frames:
        frames = frames[:max_frames]
    
    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")
    
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
    
    # Process frames and extract data simultaneously
    print("Processing frames...")
    annotated_frames = []
    hand_results_list = []
    pose_results_list = []
    
    # Also collect raw data for statistics
    X_hands_raw = np.zeros((len(frames), 126), dtype=np.float32)
    X_pose_raw = np.zeros((len(frames), 99), dtype=np.float32)
    
    for i, frame in enumerate(frames):
        # Process with MediaPipe
        hand_results = mp_hands.process(frame)
        pose_results = mp_pose.process(frame)
        
        hand_results_list.append(hand_results)
        pose_results_list.append(pose_results)
        
        # Extract hand data (same logic as preprocess script)
        lh = np.zeros((21, 3), dtype=np.float32)
        rh = np.zeros((21, 3), dtype=np.float32)
        
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                label = hand_results.multi_handedness[hand_idx].classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                if label == "Left":
                    lh = coords
                else:
                    rh = coords
        
        X_hands_raw[i] = np.concatenate([lh.flatten(), rh.flatten()])
        
        # Extract pose data
        pose_landmarks = np.zeros((33, 3), dtype=np.float32)
        if pose_results.pose_landmarks:
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark], dtype=np.float32)
        X_pose_raw[i] = pose_landmarks.flatten()
        
        # Draw landmarks
        annotated_frame = draw_landmarks_on_frame(frame, hand_results, pose_results, i)
        annotated_frames.append(annotated_frame)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames...")
    
    mp_hands.close()
    mp_pose.close()
    
    # Normalize hand data
    print("\nNormalizing hand data...")
    X_hands_norm = normalize_sequence(X_hands_raw)
    
    # Print statistics
    print("\n=== Detection Statistics ===")
    print(f"Total frames: {len(frames)}")
    
    # Hand detection stats
    hands_detected = sum(1 for r in hand_results_list if r.multi_hand_landmarks)
    print(f"Frames with hands detected: {hands_detected}/{len(frames)} ({100*hands_detected/len(frames):.1f}%)")
    
    # Pose detection stats
    pose_detected = sum(1 for r in pose_results_list if r.pose_landmarks)
    print(f"Frames with pose detected: {pose_detected}/{len(frames)} ({100*pose_detected/len(frames):.1f}%)")
    
    # Data shapes
    print(f"\nHand landmarks shape: {X_hands_raw.shape} (raw), {X_hands_norm.shape} (normalized)")
    print(f"Pose landmarks shape: {X_pose_raw.shape}")
    
    # Visualize
    print("\nCreating visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Show first few annotated frames
    n_show = min(6, len(annotated_frames))
    for i in range(n_show):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(annotated_frames[i])
        ax.set_title(f"Frame {i}")
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    # Create animated visualization
    print("\nCreating animated visualization...")
    fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
    ax_anim.axis('off')
    
    im = ax_anim.imshow(annotated_frames[0])
    ax_anim.set_title("Animated Detection Visualization")
    
    def animate(frame_idx):
        im.set_array(annotated_frames[frame_idx])
        ax_anim.set_title(f"Frame {frame_idx}/{len(annotated_frames)}")
        return [im]
    
    anim = animation.FuncAnimation(
        fig_anim, animate, frames=len(annotated_frames),
        interval=100, blit=True, repeat=True
    )
    
    print("Displaying animated visualization (close window to continue)...")
    plt.show()
    
    # Plot landmark trajectories
    print("\nPlotting landmark trajectories...")
    plot_landmark_trajectories(X_hands_raw, X_pose_raw, frames[0].shape)
    
    print("\nDone!")

def plot_landmark_trajectories(X_hands, X_pose, frame_shape):
    """
    Plot trajectories of key landmarks over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    h, w = frame_shape[:2]
    n_frames = X_hands.shape[0]
    
    # Left hand wrist trajectory (landmark 0, indices 0-2)
    ax = axes[0, 0]
    if np.any(X_hands[:, :3] != 0):
        wrist_x = X_hands[:, 0] * w
        wrist_y = X_hands[:, 1] * h
        ax.plot(wrist_x, wrist_y, 'b-', label='Left Wrist', linewidth=2)
        ax.scatter(wrist_x[0], wrist_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(wrist_x[-1], wrist_y[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Left Hand Wrist Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Invert Y to match image coordinates
    
    # Right hand wrist trajectory (landmark 0, indices 63-65)
    ax = axes[0, 1]
    if np.any(X_hands[:, 63:66] != 0):
        wrist_x = X_hands[:, 63] * w
        wrist_y = X_hands[:, 64] * h
        ax.plot(wrist_x, wrist_y, 'r-', label='Right Wrist', linewidth=2)
        ax.scatter(wrist_x[0], wrist_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(wrist_x[-1], wrist_y[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Right Hand Wrist Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # Pose: Left shoulder (landmark 11, indices 33-35) and Right shoulder (12, 36-38)
    ax = axes[1, 0]
    if np.any(X_pose[:, 33:36] != 0):
        shoulder_x = X_pose[:, 33] * w
        shoulder_y = X_pose[:, 34] * h
        ax.plot(shoulder_x, shoulder_y, 'g-', label='Left Shoulder', linewidth=2)
    if np.any(X_pose[:, 36:39] != 0):
        shoulder_x = X_pose[:, 36] * w
        shoulder_y = X_pose[:, 37] * h
        ax.plot(shoulder_x, shoulder_y, 'm-', label='Right Shoulder', linewidth=2)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Shoulder Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # Pose: Nose trajectory (landmark 0, indices 0-2)
    ax = axes[1, 1]
    if np.any(X_pose[:, :3] != 0):
        nose_x = X_pose[:, 0] * w
        nose_y = X_pose[:, 1] * h
        ax.plot(nose_x, nose_y, 'c-', label='Nose', linewidth=2)
        ax.scatter(nose_x[0], nose_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(nose_x[-1], nose_y[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Nose Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Test and visualize MediaPipe detections on a single GIF"
    )
    parser.add_argument(
        "gif_path",
        help="Path to input GIF file"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save visualization image (optional)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process (for testing)"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.gif_path):
        print(f"Error: GIF file not found: {args.gif_path}")
        print("\nExample usage:")
        print("  python scripts/test_preprocess_visualize.py sgsl_dataset/abuse/abuse.gif")
        sys.exit(1)
    
    visualize_detections(args.gif_path, args.output, args.max_frames)

if __name__ == "__main__":
    main()
