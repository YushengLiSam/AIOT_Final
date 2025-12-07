#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
export_and_pack_yolox_face_for_rtmlib.py

End-to-end helper script for YOLOX face detector + rtmlib.

It can:
1) Call MMDeploy's deploy.py to export a YOLOX detector to ONNX (onnxruntime_dynamic),
2) Find the MMDeploy onnx_sdk directory that contains:
       - end2end.onnx
       - deploy.json
       - pipeline.json
       - detail.json
3) Pack that directory into a .zip file that can be directly used by rtmlib.

You can also use it in "pack-only" mode if you already ran deploy.py yourself.

Typical YOLOX-face usage
------------------------
Example (export + pack in one go):

    python export_and_pack_yolox_face_for_rtmlib.py \
        --deploy-cfg demo/mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py \
        --model-cfg demo/mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
        --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
        --input-img demo/mmpose/demo/resources/demo_face.jpg \
        --work-dir pretrained_weight/yolox_face_onnx \
        --output pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c_rtmlib.zip

After that, you can use the generated zip in rtmlib:

    from rtmlib import Custom

    det_zip = 'yolo-x_8xb8-300e_coco-face_13274d7c_rtmlib.zip'

    detector = Custom(
        det_class='YOLOX',
        det=det_zip,
        det_input_size=(640, 640),
        pose_class='RTMPose',
        pose='xxx_rtmpose.zip',
        pose_input_size=(256, 256),
        backend='onnxruntime',
        device='cuda',
    )

Pack-only mode
--------------
If you already ran MMDeploy and have an onnx_sdk directory:

    python export_and_pack_yolox_face_for_rtmlib.py \
        --sdk-dir work_dirs/yolox_face_onnx/20231205/yolox_onnx/yolo-x_8xb8-300e_coco-face_13274d7c \
        --output yolo-x_8xb8-300e_coco-face_13274d7c_rtmlib.zip
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Set
import zipfile
import subprocess


# Files that must exist in the onnx_sdk leaf directory
REQUIRED_FILES: Set[str] = {
    "end2end.onnx",
    "deploy.json",
    "pipeline.json",
    "detail.json",
}


def find_leaf_sdk_dir(root: Path) -> Optional[Path]:
    """Recursively search under `root` for a directory that contains all REQUIRED_FILES.

    - If `root` itself is a directory and already contains all required files, return `root`.
    - Otherwise, do a DFS under `root` and return the first directory that contains all required files.
    """
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a valid directory")

    # First check the root itself
    names = {p.name for p in root.iterdir() if p.is_file()}
    if REQUIRED_FILES.issubset(names):
        return root

    # Then recursively search child directories
    for sub in root.rglob("*"):
        if not sub.is_dir():
            continue
        sub_names = {p.name for p in sub.iterdir() if p.is_file()}
        if REQUIRED_FILES.issubset(sub_names):
            return sub

    return None


def make_rtmlib_zip(
    sdk_leaf_dir: Path,
    output_zip: Path,
    keep_parent_levels: int = 2,
) -> None:
    """Pack `sdk_leaf_dir` into a rtmlib-ready zip file.

    Parameters
    ----------
    sdk_leaf_dir : Path
        Directory that contains end2end.onnx + the three json files.
    output_zip : Path
        Path to the output zip file.
    keep_parent_levels : int
        How many levels of parent directories above `sdk_leaf_dir` to keep in the zip.

        For example, if:
            sdk_leaf_dir = /.../20231205/yolox_onnx/yolo-x_face/

        and keep_parent_levels = 2, then the zip will contain:
            20231205/yolox_onnx/yolo-x_face/...

        This makes the structure closer to official onnx_sdk zips.
    """
    sdk_leaf_dir = sdk_leaf_dir.resolve()

    # Check required files exist
    names = {p.name for p in sdk_leaf_dir.iterdir() if p.is_file()}
    missing = REQUIRED_FILES - names
    if missing:
        raise RuntimeError(
            f"{sdk_leaf_dir} is missing required files: {', '.join(sorted(missing))}"
        )

    # Decide which directory to treat as the zip root (include some parent levels)
    root = sdk_leaf_dir
    for _ in range(keep_parent_levels):
        if root.parent == root:
            break
        root = root.parent

    root = root.resolve()

    print(f"[INFO] rtmlib zip root: {root}")
    print(f"[INFO] onnx_sdk leaf dir: {sdk_leaf_dir}")
    print(f"[INFO] output zip: {output_zip}")

    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # Create zip file
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(root)
            zf.write(path, rel.as_posix())
            # Optional debug print:
            # print(f"  add {rel}")

    print(f"[OK] Generated rtmlib model package: {output_zip}")


def run_mmdeploy_export(
    deploy_cfg: Path,
    model_cfg: Path,
    checkpoint: Path,
    input_img: Path,
    work_dir: Path,
    device: str = "cuda:0",
    mmdeploy_root: Path = Path("mmdeploy"),
    show: bool = False,
) -> None:
    """Call MMDeploy's deploy.py to export a YOLOX detector to ONNX.

    This is equivalent to running:

        python mmdeploy/tools/deploy.py \
            <deploy_cfg> \
            <model_cfg> \
            <checkpoint> \
            <input_img> \
            --work-dir <work_dir> \
            --device <device> \
            [--show]

    Parameters
    ----------
    deploy_cfg : Path
        Path to deploy config, e.g.:
        mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py
    model_cfg : Path
        Path to model config, e.g.:
        mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py
    checkpoint : Path
        Path to YOLOX checkpoint (.pth).
    input_img : Path
        Path to an input image for MMDeploy demo.
    work_dir : Path
        Directory where MMDeploy will write export results.
    device : str
        Device passed to MMDeploy, e.g. "cuda:0" or "cpu".
    mmdeploy_root : Path
        Root directory of the mmdeploy repo (where tools/deploy.py lives).
    show : bool
        Whether to pass --show to deploy.py.
    """
    deploy_py = mmdeploy_root / "tools" / "deploy.py"
    if not deploy_py.is_file():
        raise FileNotFoundError(
            f"Could not find mmdeploy/tools/deploy.py at: {deploy_py}. "
            "Please check --mmdeploy-root."
        )

    cmd = [
        sys.executable,
        str(deploy_py),
        str(deploy_cfg),
        str(model_cfg),
        str(checkpoint),
        str(input_img),
        "--work-dir",
        str(work_dir),
        "--device",
        device,
        "--dump-info",
    ]

    if show:
        cmd.append("--show")

    print("[INFO] Running MMDeploy export:")
    print("       " + " ".join(cmd))

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(cmd, check=True)
    print("[OK] MMDeploy export finished.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export a YOLOX model with MMDeploy and pack the resulting onnx_sdk "
            "into a rtmlib-ready zip, or pack an existing onnx_sdk directory."
        )
    )

    # Mode 1: pack-only (already have onnx_sdk)
    parser.add_argument(
        "--sdk-dir",
        type=str,
        default=None,
        help=(
            "Directory that contains end2end.onnx / deploy.json / pipeline.json / detail.json, "
            "or a parent directory of it (the script will recursively search downward). "
            "If this is provided, export step is skipped and we only pack."
        ),
    )

    # Mode 2: export + pack (run mmdeploy/tools/deploy.py first)
    parser.add_argument(
        "--deploy-cfg",
        type=str,
        default="demo/mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py",
        help=(
            "Path to MMDeploy deploy config, e.g. "
            "mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py"
        ),
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="demo/mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py",
        help=(
            "Path to model config, e.g. "
            "mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth",
        help="Path to YOLOX checkpoint (.pth), e.g. yolo-x_8xb8-300e_coco-face_13274d7c.pth",
    )
    parser.add_argument(
        "--input-img",
        type=str,
        default="demo/mmpose/demo/resources/demo_face.jpg",
        help="Path to an input image to feed MMDeploy demo, e.g. demo.jpg",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="pretrained_weight/yolox_face_onnx",
        help=(
            "Working directory for MMDeploy export, e.g. pretrained_weight/yolox_face_onnx. "
            "Used only in export + pack mode."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device for MMDeploy export, e.g. "cuda:0" or "cpu". Default: cuda:0',
    )
    parser.add_argument(
        "--mmdeploy-root",
        type=str,
        default="demo/mmdeploy",
        help=(
            "Root directory of the mmdeploy repo, where tools/deploy.py is located. "
            "Default: demo/mmdeploy"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Pass --show to MMDeploy deploy.py (optional).",
    )

    # Common options
    parser.add_argument(
        "--output",
        type=str,
        required=None,
        default="pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c_rtmlib.zip",
        help=(
            "Path to the output zip file, e.g. "
            "pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c_rtmlib.zip"
        ),
    )
    parser.add_argument(
        "--keep-parent-levels",
        type=int,
        default=2,
        help=(
            "Number of parent directory levels above the onnx_sdk leaf to keep "
            "in the zip structure. Default is 2."
        ),
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    output_zip = Path(args.output)

    # Mode 1: pack-only
    if args.sdk_dir is not None:
        sdk_root = Path(args.sdk_dir)
        if not sdk_root.exists():
            raise SystemExit(f"[ERROR] sdk-dir does not exist: {sdk_root}")

        leaf = find_leaf_sdk_dir(sdk_root)
        if leaf is None:
            raise SystemExit(
                f"[ERROR] Could not find a directory under {sdk_root} that contains "
                f"{', '.join(sorted(REQUIRED_FILES))}. Please check your MMDeploy export."
            )

        make_rtmlib_zip(
            leaf,
            output_zip,
            keep_parent_levels=args.keep_parent_levels,
        )
        return

    # Mode 2: export + pack
    needed = [args.deploy_cfg, args.model_cfg, args.checkpoint, args.input_img, args.work_dir]
    if any(v is None for v in needed):
        raise SystemExit(
            "[ERROR] Either provide --sdk-dir for pack-only mode, "
            "or provide all of --deploy-cfg / --model-cfg / --checkpoint / --input-img / --work-dir "
            "for export + pack mode."
        )

    deploy_cfg = Path(args.deploy_cfg)
    model_cfg = Path(args.model_cfg)
    checkpoint = Path(args.checkpoint)
    input_img = Path(args.input_img)
    work_dir = Path(args.work_dir)

    # Run MMDeploy export
    run_mmdeploy_export(
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        input_img=input_img,
        work_dir=work_dir,
        device=args.device,
        mmdeploy_root=Path(args.mmdeploy_root),
        show=args.show,
    )

    # Find generated onnx_sdk leaf dir under work_dir
    leaf = find_leaf_sdk_dir(work_dir)
    if leaf is None:
        raise SystemExit(
            f"[ERROR] After MMDeploy export, could not find a directory under {work_dir} "
            f"that contains {', '.join(sorted(REQUIRED_FILES))}. "
            "Please check the MMDeploy output structure."
        )

    # Pack into rtmlib zip
    make_rtmlib_zip(
        leaf,
        output_zip,
        keep_parent_levels=args.keep_parent_levels,
    )


if __name__ == "__main__":
    main()
