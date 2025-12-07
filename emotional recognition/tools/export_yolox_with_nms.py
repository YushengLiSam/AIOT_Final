#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 MMDeploy 正确导出包含 NMS 的 YOLOX face 检测模型
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Export YOLOX face model with NMS using MMDeploy')
    parser.add_argument('--checkpoint', type=str, 
                       default='pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth',
                       help='YOLOX checkpoint path')
    parser.add_argument('--output-dir', type=str,
                       default='pretrained_weight/yolox_face_with_nms',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda:0)')
    
    args = parser.parse_args()
    
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint.exists():
        print(f"Error: Checkpoint not found: {checkpoint}")
        return 1
    
    try:
        from mmdeploy.apis import torch2onnx
        from mmdeploy.utils import get_input_shape, load_config
    except ImportError:
        print("Error: MMDeploy not installed or not in path")
        print("Please install: pip install mmdeploy mmdeploy-runtime")
        return 1
    
    # 使用本地包中的配置文件
    script_dir = Path(__file__).parent.parent
    model_cfg = script_dir / 'demo/packages/mmpose/.mim/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py'
    deploy_cfg = script_dir / 'demo/packages/mmdeploy/.mim/configs/mmdet/detection/detection_onnxruntime_dynamic.py'
    
    if not model_cfg.exists():
        print(f"Error: Model config not found: {model_cfg}")
        return 1
    
    if not deploy_cfg.exists():
        print(f"Error: Deploy config not found: {deploy_cfg}")
        return 1
    
    print("="*70)
    print("Export YOLOX Face Model with NMS")
    print("="*70)
    print(f"Checkpoint: {checkpoint}")
    print(f"Model config: {model_cfg.name}")
    print(f"Deploy config: {deploy_cfg.name}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # 使用 MMDeploy API 导出
    try:
        from mmdeploy.apis.utils import build_task_processor
        from mmdeploy.utils import get_input_shape, load_config
        
        deploy_cfg_dict = load_config(str(deploy_cfg))[0]
        model_cfg_dict = load_config(str(model_cfg))[0]
        
        # 修改部署配置以包含 NMS
        deploy_cfg_dict['codebase_config'] = {
            'type': 'mmdet',
            'task': 'ObjectDetection',
            'model_type': 'end2end',  # 包含 NMS 后处理
            'post_processing': {
                'score_threshold': 0.05,
                'iou_threshold': 0.5,
                'max_output_boxes_per_class': 200,
                'pre_top_k': 5000,
                'keep_top_k': 100,
                'background_label_id': -1,
            }
        }
        
        deploy_cfg_dict['onnx_config']['output_names'] = ['dets', 'labels']
        deploy_cfg_dict['onnx_config']['input_shape'] = [640, 640]
        
        print("\nStarting export...")
        
        # 构建任务处理器
        task_processor = build_task_processor(model_cfg_dict, deploy_cfg_dict, args.device)
        
        # 导出
        torch2onnx(
            img=None,
            work_dir=str(output_dir),
            save_file=str(output_dir / 'end2end.onnx'),
            deploy_cfg=deploy_cfg_dict,
            model_cfg=model_cfg_dict,
            model_checkpoint=str(checkpoint),
            device=args.device
        )
        
        print(f"\n✓ Export completed!")
        print(f"  ONNX file: {output_dir / 'end2end.onnx'}")
        
        # 验证输出
        import onnxruntime as ort
        session = ort.InferenceSession(str(output_dir / 'end2end.onnx'))
        
        print(f"\nModel verification:")
        print(f"  Inputs:")
        for inp in session.get_inputs():
            print(f"    {inp.name}: {inp.shape}")
        print(f"  Outputs:")
        for out in session.get_outputs():
            print(f"    {out.name}: {out.shape}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during export: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*70)
        print("FALLBACK: Using simplified export")
        print("="*70)
        print("\nMMDeploy export failed. This usually means:")
        print("1. MMDeploy is not properly installed")
        print("2. Config files are not compatible")
        print("\nRecommendation: Use the pre-built online model instead:")
        print("  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip")
        print("\nThis model is fully tested and compatible with rtmlib.")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
