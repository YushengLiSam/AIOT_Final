import argparse
import os
import cv2
import pickle
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rtmlib import Custom


def process_image(img_path, src_root, tgt_root, head, overwrite=False):
    """
    Process a single image: run head, extract head keypoints and save to target path.

    Args:
        img_path: path to the input image
        src_root: source root directory for computing relative paths
        tgt_root: root output directory (will preserve relative structure)
        head: model callable
        overwrite: whether to overwrite existing output

    Returns:
        (img_path, saved_path or None, status)
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return img_path, None, "read_failed"

        h, w = img.shape[:2]

        keypoints, scores = head(np.uint8(img))

        # keypoints expected shape: (num_people, num_joints, 2)
        # scores expected shape: (num_people, num_joints) or (num_people,)
        if keypoints is None or len(keypoints) == 0:
            # no detections
            head_kpts = np.array([])
            head_scores = np.array([])
        else:
            # choose the first detected person and save all keypoints
            person_kpts = keypoints[0]
            person_scores = scores[0] if scores is not None and len(scores) > 0 else None

            head_kpts = person_kpts.astype(float)
            # normalize by width/height (x by w, y by h)
            head_kpts[:, 0] = head_kpts[:, 0] / float(w)
            head_kpts[:, 1] = head_kpts[:, 1] / float(h)

            if person_scores is not None:
                try:
                    head_scores = np.array(person_scores)
                except Exception:
                    head_scores = np.array([])
            else:
                head_scores = np.array([])

        # Build target path preserving relative structure
        rel_path = os.path.relpath(img_path, start=src_root)
        rel_dir = os.path.dirname(rel_path)
        tgt_dir = os.path.join(tgt_root, rel_dir)
        os.makedirs(tgt_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(tgt_dir, base_name + '.pkl')

        if os.path.exists(out_path) and not overwrite:
            return img_path, out_path, "exists"

        data = {
            'head_keypoints': head_kpts,
            'head_scores': head_scores,
            'image_size': [w, h]
        }

        with open(out_path, 'wb') as f:
            pickle.dump(data, f)

        return img_path, out_path, "saved"

    except Exception as e:
        return img_path, None, f"error:{e}"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", default="data/raw/affectnet", help="source image directory")
    parser.add_argument("--tgt_dir", default="data/landmarks/affectnet", help="target landmarks directory")

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    parser.add_argument("--openpose_skeleton", action="store_true", help="use openpose format")
    parser.add_argument("--mode", default="lightweight", choices=["performance", "lightweight", "balanced"])

    parser.add_argument("--image_extensions", nargs='+', default=["jpg", "jpeg", "png"])
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.tgt_dir, exist_ok=True)

    # ================== Face Detection Models (106 keypoints) ==================
    # RTMPose Face6 models (trained on 6 datasets: COCO-Wholebody-Face, WFLW, 300W, COFW, Halpe, LaPa)
    # - rtmpose-t: NME=1.67, fastest
    # - rtmpose-s: NME=1.59, balanced
    # - rtmpose-m: NME=1.44, most accurate
    
    # RTMPose-t Face6:
    # pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-face6_pt-in1k_120e-256x256-df79d9a5_20230529.zip'
    # pose_input_size=(256, 256)
    
    # RTMPose-s Face6:
    # pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-face6_pt-in1k_120e-256x256-d779fdef_20230529.zip'
    # pose_input_size=(256, 256)
    
    # RTMPose-m Face6 (CURRENTLY USED):
    # pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.zip'
    # pose_input_size=(256, 256)

    head = Custom(to_openpose=args.openpose_skeleton,
                det_class='YOLOX',
                det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
                det_input_size=(416, 416),
                pose_class='RTMPose',
                pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.zip',
                pose_input_size=(256, 256),
                backend=args.backend,
                device=args.device)

    # Check if src_dir contains subdirectories or files directly
    subdirs = [d for d in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, d))]
    
    all_stats = []
    overall_images = 0
    overall_time = 0.0
    
    if subdirs:
        # Process each subdirectory
        print(f"Found {len(subdirs)} subdirectories")
        
        for subdir in subdirs:
            src_subdir = os.path.join(args.src_dir, subdir)
            tgt_subdir = os.path.join(args.tgt_dir, subdir)
            os.makedirs(tgt_subdir, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Processing subdirectory: {subdir}")
            print(f"{'='*60}")
            
            # Collect images in this subdirectory
            img_files = []
            for root, _, files in os.walk(src_subdir):
                for f in files:
                    if any(f.lower().endswith('.' + ext.lower()) for ext in args.image_extensions):
                        img_files.append(os.path.join(root, f))
            
            print(f"Found {len(img_files)} images in {subdir}")
            
            if len(img_files) == 0:
                continue
            
            saved = 0
            failed = 0
            exists = 0
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(process_image, img, args.src_dir, args.tgt_dir, head, args.overwrite) for img in img_files]
                for f in tqdm(futures, desc=f"[{subdir}] Processing images", total=len(futures)):
                    img_path, out_path, status = f.result()
                    if status == 'saved':
                        saved += 1
                    elif status == 'exists':
                        exists += 1
                    else:
                        failed += 1
            
            processing_time = time.time() - start_time
            total_images = len(img_files)
            
            print(f"[{subdir}] Completed: saved={saved}, exists={exists}, failed={failed}, time={processing_time:.2f}s")
            
            # Record statistics for this subdirectory
            avg_time_per_image = processing_time / total_images if total_images > 0 else 0
            subdir_stats = (
                f"\n========== Statistics for {subdir} ==========\n"
                f"Total images: {total_images}\n"
                f"Successfully processed: {saved}\n"
                f"Already existed: {exists}\n"
                f"Failed: {failed}\n"
                f"Total processing time: {processing_time:.2f} seconds\n"
                f"Average time per image: {avg_time_per_image:.4f} seconds\n"
                f"Average images/sec: {1.0/avg_time_per_image:.2f}\n"
                f"{'='*50}\n"
            )
            print(subdir_stats)
            all_stats.append(subdir_stats)
            overall_images += total_images
            overall_time += processing_time
    else:
        # Process files directly in src_dir (no subdirectories)
        print("Processing images directly in source directory")
        
        img_files = []
        for root, _, files in os.walk(args.src_dir):
            for f in files:
                if any(f.lower().endswith('.' + ext.lower()) for ext in args.image_extensions):
                    img_files.append(os.path.join(root, f))

        print(f"Found {len(img_files)} images")

        if len(img_files) > 0:
            saved = 0
            failed = 0
            exists = 0
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(process_image, img, args.src_dir, args.tgt_dir, head, args.overwrite) for img in img_files]
                for f in tqdm(futures, desc="Processing images", total=len(futures)):
                    img_path, out_path, status = f.result()
                    if status == 'saved':
                        saved += 1
                    elif status == 'exists':
                        exists += 1
                    else:
                        failed += 1
            
            processing_time = time.time() - start_time
            print(f"Completed: saved={saved}, exists={exists}, failed={failed}, time={processing_time:.2f}s")
            
            overall_images = len(img_files)
            overall_time = processing_time
    
    # Write overall statistics
    if overall_images > 0:
        avg_time_per_image = overall_time / overall_images
        overall_stats = (
            f"\n{'='*60}\n"
            f"========== OVERALL Statistics ==========\n"
            f"Total images: {overall_images}\n"
            f"Total processing time: {overall_time:.2f} seconds\n"
            f"Average time per image: {avg_time_per_image:.4f} seconds\n"
            f"Average images/sec: {1.0/avg_time_per_image:.2f}\n"
            f"{'='*60}\n"
        )
        print(overall_stats)
        
        # Write statistics to log file
        log_path = os.path.join(args.tgt_dir, "log.txt")
        with open(log_path, 'w') as f:
            for stats in all_stats:
                f.write(stats)
            f.write(overall_stats)
        print(f"Statistics saved to {log_path}")


if __name__ == "__main__":
    main()