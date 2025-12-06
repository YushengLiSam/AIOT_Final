"""
Visualize RTMPose keypoints from .pkl files.

Features:
- --pkl_file can be a single .pkl file OR a directory.
- When a directory is given, all .pkl files under it are processed (recursively).
- By default, no window is shown: everything is rendered and saved directly.
- Optional mapping from PKL tree to video tree via --video_root.

Usage examples (three typical cases):

1) Single PKL file -> pose visualization video (no original RGB video, blank background):
    python demo/visualize_pose.py \
        --pkl_file data/WLASL/rtmpose_format/sample.pkl

2) Directory of PKL files -> pose visualization videos (blank background for each sequence):
    python demo/visualize_pose.py \
        --pkl_file data/WLASL/rtmpose_format

   This will recursively find all .pkl files under data/WLASL/rtmpose_format
   and save "<pkl_stem>_pose.mp4" next to each PKL file.

3) Directory of PKL files + matching RGB videos -> overlay pose on RGB and save to a custom output dir:
    python demo/visualize_pose.py \
        --pkl_file  data/WLASL/rtmpose_format \
        --video_root data/WLASL/rgb_format \
        --output_dir outputs/wlasl_pose

   For each "rtmpose_format/subdir/name.pkl" the script will try to load
   "rgb_format/subdir/name.mp4" as the video, draw the pose on it, and save
   "outputs/wlasl_pose/subdir/name_pose.mp4".
"""

import argparse
import pickle
import cv2
import numpy as np
import os
from pathlib import Path


# RTMPose Wholebody keypoint connections (OpenPose 133 format)
SKELETON_CONNECTIONS = {
    # Body (COCO 17 keypoints)
    'body': [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ],
    # Left hand (21 keypoints, indices 91-111)
    'left_hand': [(i, i + 1) for i in range(91, 111)],
    # Right hand (21 keypoints, indices 112-132)
    'right_hand': [(i, i + 1) for i in range(112, 132)],
    # Face (68 keypoints, indices 23-90)
    'face': [
        # Jaw line
        *[(i, i + 1) for i in range(23, 39)],
        # Eyebrows
        *[(i, i + 1) for i in range(40, 45)],
        *[(i, i + 1) for i in range(46, 51)],
        # Nose
        *[(i, i + 1) for i in range(52, 56)],
        (56, 57), (57, 58), (58, 59),
        # Eyes
        *[(i, i + 1) for i in range(60, 67)], (67, 60),
        *[(i, i + 1) for i in range(68, 75)], (75, 68),
        # Mouth
        *[(i, i + 1) for i in range(76, 87)], (87, 76),
    ]
}

# Color scheme for different body parts (BGR)
COLORS = {
    'body': (0, 255, 0),       # Green
    'left_hand': (255, 0, 0),  # Blue
    'right_hand': (0, 0, 255), # Red
    'face': (255, 255, 0)      # Cyan
}


def load_pose_data(pkl_file):
    """Load pose data dictionary from a pickle file."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def draw_keypoints_on_frame(frame, keypoints, scores, confidence_threshold=0.3):
    """
    Draw keypoints and skeleton connections on a frame.

    Args:
        frame: H x W x 3 BGR image.
        keypoints: array of shape (K, 2), normalized coordinates in [0, 1].
        scores: array of shape (K,), confidence per keypoint.
        confidence_threshold: minimum score to show a keypoint/connection.

    Returns:
        The frame with keypoints and skeleton drawn.
    """
    H, W = frame.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    keypoints_pixel = keypoints * np.array([W, H])

    # Draw skeleton connections
    for part_name, connections in SKELETON_CONNECTIONS.items():
        color = COLORS.get(part_name, (255, 255, 255))

        for start_idx, end_idx in connections:
            if start_idx < len(keypoints_pixel) and end_idx < len(keypoints_pixel):
                if scores[start_idx] > confidence_threshold and scores[end_idx] > confidence_threshold:
                    start_point = tuple(keypoints_pixel[start_idx].astype(int))
                    end_point = tuple(keypoints_pixel[end_idx].astype(int))
                    cv2.line(frame, start_point, end_point, color, 2)

    # Draw individual keypoints
    for i, (kpt, score) in enumerate(zip(keypoints_pixel, scores)):
        if score > confidence_threshold:
            x, y = int(kpt[0]), int(kpt[1])

            # Assign color based on index range
            if i < 17:  # Body keypoints
                color = COLORS['body']
            elif 91 <= i <= 111:  # Left hand keypoints
                color = COLORS['left_hand']
            elif 112 <= i <= 132:  # Right hand keypoints
                color = COLORS['right_hand']
            else:  # Face or others
                color = COLORS['face']

            cv2.circle(frame, (x, y), 3, color, -1)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)

    return frame


def visualize_pose_video(pkl_file, video_file=None, output_file=None,
                         confidence_threshold=0.3, show_window=False):
    """
    Visualize pose sequence as a video.

    By default, this function only saves the output to a video file and does not show any window.

    Args:
        pkl_file: path to pose .pkl file.
        video_file: optional path to original video; if None or invalid, a blank canvas is used.
        output_file: path to save the output video; if None, uses "<pkl_stem>_pose.mp4" next to the PKL.
        confidence_threshold: threshold for keypoint visibility.
        show_window: if True, show frames in a window (useful for debugging, not recommended for batch runs).
    """
    # Load pose data
    print(f"Loading pose data from {pkl_file}...")
    pose_data = load_pose_data(pkl_file)
    keypoints_list = pose_data['keypoints']  # list of (1, K, 2)
    scores_list = pose_data['scores']        # list of (1, K)

    print(f"Loaded {len(keypoints_list)} frames with {keypoints_list[0].shape[1]} keypoints each")

    # Try to load original video
    if video_file and os.path.exists(video_file):
        print(f"Loading video from {video_file}...")
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        # Read all frames into memory
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        print(f"Loaded {len(frames)} video frames")
    else:
        # No valid video; create blank frames for visualization
        print("No valid video file provided, creating blank canvas...")
        fps = 30
        width, height = 1280, 720
        frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in keypoints_list]

    # Determine output path
    if output_file is None:
        p = Path(pkl_file)
        output_file = str(p.with_name(p.stem + "_pose.mp4"))

    out_dir = os.path.dirname(output_file) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    print(f"Saving output to {output_file}...")

    # Number of frames to process
    num_frames = min(len(frames), len(keypoints_list))

    for i in range(num_frames):
        frame = frames[i].copy()
        keypoints = keypoints_list[i][0]  # (K, 2)
        scores = scores_list[i][0]        # (K,)

        # Draw pose on the frame
        frame_with_pose = draw_keypoints_on_frame(frame, keypoints, scores, confidence_threshold)

        # Optionally annotate frame index
        cv2.putText(
            frame_with_pose,
            f"Frame: {i + 1}/{num_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Optional visualization window (disabled by default)
        if show_window:
            cv2.imshow('Pose Visualization', frame_with_pose)
            key = cv2.waitKey(int(1000 / fps))
            if key == ord('q'):
                break

        # Write frame to output video
        writer.write(frame_with_pose)

    writer.release()
    print(f"Video saved to {output_file}")

    if show_window:
        cv2.destroyAllWindows()


def visualize_single_frame(pkl_file, frame_idx=0, output_image=None):
    """
    Visualize a single frame from pose data and save as an image.

    Args:
        pkl_file: path to pose .pkl file.
        frame_idx: index of the frame to visualize.
        output_image: path to save the image; if None, uses "<pkl_stem>_frameXXXX.jpg".
    """
    pose_data = load_pose_data(pkl_file)
    keypoints_list = pose_data['keypoints']
    scores_list = pose_data['scores']

    if frame_idx >= len(keypoints_list):
        print(f"Frame index {frame_idx} out of range (max: {len(keypoints_list) - 1})")
        return

    keypoints = keypoints_list[frame_idx][0]
    scores = scores_list[frame_idx][0]

    # Create blank canvas (no video)
    width, height = 1280, 720
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw pose on the blank canvas
    frame_with_pose = draw_keypoints_on_frame(frame, keypoints, scores)

    # Add text label with frame index
    cv2.putText(
        frame_with_pose,
        f"Frame {frame_idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Determine output path
    if output_image is None:
        p = Path(pkl_file)
        output_image = str(p.with_name(f"{p.stem}_frame{frame_idx:04d}.jpg"))

    out_dir = os.path.dirname(output_image) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Save image to disk (no OpenCV window)
    cv2.imwrite(output_image, frame_with_pose)
    print(f"Image saved to {output_image}")


def collect_pkl_files(root):
    """
    Collect all .pkl files under a directory (recursively).

    Args:
        root: root directory.

    Returns:
        Sorted list of absolute paths to .pkl files.
    """
    pkl_files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".pkl"):
                pkl_files.append(os.path.join(dirpath, name))
    pkl_files.sort()
    return pkl_files


def main():
    parser = argparse.ArgumentParser(
        description='Visualize RTMPose keypoints from .pkl files (single file or directory).'
    )
    parser.add_argument(
        '--pkl_file',
        type=str,
        required=True,
        help='Path to a .pkl file OR a directory containing .pkl files.'
    )
    parser.add_argument(
        '--video_file',
        type=str,
        default=None,
        help='Path to original video file (used only in single-file mode).'
    )
    parser.add_argument(
        '--video_root',
        type=str,
        default=None,
        help='If set and --pkl_file is a directory, root directory of videos '
             'mirroring the PKL directory structure.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save the output video (single-file mode).'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save outputs when --pkl_file is a directory. '
             'If not set, outputs are placed next to each PKL file.'
    )
    parser.add_argument(
        '--output_image',
        type=str,
        default=None,
        help='Path to save output image (single-frame + single-file mode).'
    )
    parser.add_argument(
        '--frame_idx',
        type=int,
        default=0,
        help='Frame index to visualize (for single-frame mode).'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.3,
        help='Confidence threshold for displaying keypoints.'
    )
    parser.add_argument(
        '--single_frame',
        action='store_true',
        help='If set, visualize only a single frame instead of a full video.'
    )

    args = parser.parse_args()

    if not os.path.exists(args.pkl_file):
        print(f"Error: path not found: {args.pkl_file}")
        return

    # -------- Directory mode: process ALL .pkl files -------- #
    if os.path.isdir(args.pkl_file):
        pkl_files = collect_pkl_files(args.pkl_file)
        if not pkl_files:
            print(f"No .pkl files found under {args.pkl_file}")
            return

        print(f"Found {len(pkl_files)} .pkl files under {args.pkl_file}")

        for pkl_path in pkl_files:
            p = Path(pkl_path)

            # Determine output directory for this specific PKL file
            if args.output_dir is not None:
                # Keep the same relative subdirectory structure
                rel = os.path.relpath(pkl_path, args.pkl_file)
                rel_parent = os.path.dirname(rel)
                out_dir = os.path.join(args.output_dir, rel_parent)
            else:
                out_dir = str(p.parent)

            os.makedirs(out_dir, exist_ok=True)

            if args.single_frame:
                # For each PKL, output a single frame as a JPG
                output_image = os.path.join(out_dir, f"{p.stem}_frame{args.frame_idx:04d}.jpg")
                visualize_single_frame(pkl_path, args.frame_idx, output_image)
            else:
                # Try to find matching video file (optional)
                if args.video_root is not None:
                    rel = os.path.relpath(pkl_path, args.pkl_file)
                    video_candidate = Path(args.video_root) / Path(rel).with_suffix(".mp4")
                    video_file = str(video_candidate)
                    if not os.path.exists(video_file):
                        print(
                            f"[WARN] Video not found for {pkl_path}, expected {video_file}. "
                            f"Using blank canvas."
                        )
                        video_file = None
                else:
                    video_file = None

                output_file = os.path.join(out_dir, f"{p.stem}_pose.mp4")
                visualize_pose_video(
                    pkl_path,
                    video_file=video_file,
                    output_file=output_file,
                    confidence_threshold=args.confidence_threshold,
                    show_window=False  # absolutely no windows in batch mode
                )

    # -------- Single-file mode: still headless, but uses CLI paths -------- #
    else:
        pkl_path = args.pkl_file

        if args.single_frame:
            output_image = args.output_image
            if output_image is None:
                p = Path(pkl_path)
                output_image = str(p.with_name(f"{p.stem}_frame{args.frame_idx:04d}.jpg"))
            visualize_single_frame(pkl_path, args.frame_idx, output_image)
        else:
            # Optional video file
            video_file = None
            if args.video_file is not None and os.path.exists(args.video_file):
                video_file = args.video_file
            elif args.video_file is not None:
                print(
                    f"[WARN] Given --video_file does not exist: {args.video_file}. "
                    f"Using blank canvas."
                )

            # Output video path
            output_file = args.output_file
            if output_file is None:
                p = Path(pkl_path)
                output_file = str(p.with_name(p.stem + "_pose.mp4"))

            visualize_pose_video(
                pkl_path,
                video_file=video_file,
                output_file=output_file,
                confidence_threshold=args.confidence_threshold,
                show_window=False  # no window in single-file CLI mode either
            )


if __name__ == '__main__':
    main()
