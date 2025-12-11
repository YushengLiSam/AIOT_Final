# YOLOX Face 权重转换 - 使用说明

## 概述

`convert_yolox_face_to_onnx.py` 是一个一体化脚本,用于将 YOLOX face 检测器的 PyTorch 权重转换为 rtmlib 所需的 ONNX 格式。

## 主要功能

1. **读取配置文件**: 从 mmdet、mmpose、mmdeploy 的配置文件中读取模型配置
2. **导出 ONNX**: 使用 MMDeploy 将 PyTorch 模型导出为 ONNX 格式
3. **生成配置**: 自动生成 deploy.json, detail.json, pipeline.json
4. **打包**: 将所有文件打包为 rtmlib 兼容的 zip 文件

## 使用方法

### 基本用法

```powershell
python tools/convert_yolox_face_to_onnx.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip
```

### 完整用法 (提供所有配置文件)

```powershell
python tools/convert_yolox_face_to_onnx.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip `
    --mmpose-config mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py `
    --mmdeploy-config mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py `
    --device cpu
```

### 自定义输入尺寸

```powershell
python tools/convert_yolox_face_to_onnx.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip `
    --input-size 640 640
```

## 参数说明

### 必需参数

- `--checkpoint`: YOLOX face 权重文件路径 (.pth)
- `--output`: 输出 zip 文件路径

### 可选参数

- `--mmdet-config`: MMDetection 配置文件路径
- `--mmpose-config`: MMPose 配置文件路径 (针对 face detection)
- `--mmdeploy-config`: MMDeploy 配置文件路径
- `--demo-img`: 演示图片路径 (用于 MMDeploy 测试)
- `--device`: 导出设备 (cpu 或 cuda:0), 默认 cpu
- `--work-dir`: 工作目录, 默认 work_dirs/yolox_face_export
- `--input-size`: 输入尺寸 (高度 宽度), 默认 640 640

## 环境要求

```powershell
pip install mmdet==3.0.0 mmcv>=2.0.0 mmdeploy>=1.2.0 onnxruntime
```

## 配置文件说明

脚本会读取以下配置文件 (如果提供):

1. **mmpose config** (推荐): `mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py`
   - 包含针对 face detection 优化的配置
   
2. **mmdet config**: `mmdet/configs/yolox/yolox_s_8xb8-300e_coco.py`
   - 通用 YOLOX 配置
   
3. **mmdeploy config**: `mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py`
   - MMDeploy 导出配置

如果不提供配置文件,脚本会使用默认配置生成 JSON 文件,但可能需要手动导出 ONNX 模型。

## 输出文件

转换成功后会生成一个 zip 文件,包含:

```
yolox_face_coco.zip
├── deploy.json       # 模型部署配置
├── detail.json       # 详细的导出配置信息
├── end2end.onnx      # ONNX 模型文件
└── pipeline.json     # 推理流程配置
```

## 使用转换后的模型

```python
from rtmlib import Custom

detector = Custom(
    det_class='YOLOX',
    det='pretrained_weight/yolox_face_coco.zip',
    det_input_size=(640, 640),
    backend='onnxruntime',
    device='cpu'
)

# 进行检测
import cv2
img = cv2.imread('test.jpg')
results = detector(img)
```

## 故障排除

### 1. MMDeploy 未安装

```powershell
pip install mmdeploy mmdeploy-runtime
```

### 2. ONNX 导出失败

如果自动导出失败,可以:

1. 脚本会生成配置文件到 `work_dirs/yolox_face_export/sdk/`
2. 手动将 ONNX 文件复制到该目录
3. 使用 `onnx2zip.py` 打包:

```powershell
python tools/onnx2zip.py `
    --sdk-dir work_dirs/yolox_face_export/sdk `
    --output pretrained_weight/yolox_face_coco.zip
```

### 3. 配置文件缺失

如果没有 mmdet/mmpose/mmdeploy 仓库,可以:

1. 仅使用 `--checkpoint` 和 `--output` 参数运行
2. 脚本会生成默认配置文件
3. 手动补充 ONNX 文件

## 参考

- 参考格式: `pretrained_weight/yolox_onnx/yolox_tiny_8xb8-300e_humanart-6f3252f9/`
- 该脚本生成的格式与 rtmlib 官方格式完全兼容
