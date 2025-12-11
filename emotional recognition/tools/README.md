# 权重转换工具

本目录包含用于将预训练权重转换为 rtmlib 兼容格式的工具。

## 工具列表

### convert_yolox_face_to_onnx.py

一键式转换工具,用于将 YOLOX face 检测器转换为 rtmlib 格式。

**功能:**
- 自动读取 mmdet、mmpose、mmdeploy 配置文件
- 使用 MMDeploy 导出 ONNX 模型
- 生成配置文件 (deploy.json, detail.json, pipeline.json)
- 打包为 rtmlib 可用的 zip 文件

**使用示例:**

```powershell
# 完整导出流程 (推荐)
python tools/convert_yolox_face_to_onnx.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip `
    --mmpose-config mmpose/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py `
    --mmdeploy-config mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py `
    --device cpu

# 如果没有配置文件,仅生成JSON配置
python tools/convert_yolox_face_to_onnx.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip `
    --input-size 640 640
```

### onnx2zip.py

将包含 ONNX 模型和配置文件的目录打包为 rtmlib 格式的 zip 文件。

**使用示例:**

```powershell
# 打包已有的 ONNX 模型目录
python tools/onnx2zip.py `
    --sdk-dir work_dirs/yolox_face_export/sdk `
    --output pretrained_weight/yolox_face_coco.zip
```

## 快速开始

### 环境准备

```powershell
# 安装必需的包
pip install mmdet==3.0.0 mmcv>=2.0.0 mmdeploy>=1.2.0 onnxruntime
```

### 转换 YOLOX Face 权重

```powershell
# 一键导出 (如果有完整环境)
python tools/convert_yolox_face_to_onnx.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip
```

### 使用转换后的模型

```python
from rtmlib import Custom

detector = Custom(
    det_class='YOLOX',
    det='pretrained_weight/yolox_face_coco.zip',
    det_input_size=(640, 640),
    backend='onnxruntime',
    device='cpu'
)
```

## 参考文档

详细的转换指南请参考: [YOLOX Face 转换指南](../docs/YOLOX_FACE_CONVERSION_GUIDE.md)

## 参考格式

生成的格式与以下参考目录兼容:

```
pretrained_weight/yolox_onnx/yolox_tiny_8xb8-300e_humanart-6f3252f9/
```

将包含 ONNX 模型和配置文件的目录打包为 rtmlib 格式的 zip 文件。

**功能:**
- 验证必需文件存在 (end2end.onnx, deploy.json, detail.json, pipeline.json)
- 打包为 rtmlib 兼容的 zip 格式
- 支持自定义目录层级

**使用示例:**

```powershell
# 打包已有的 ONNX 模型目录
python tools/onnx2zip.py `
    --sdk-dir pretrained_weight/yolox_onnx/yolox_face_coco `
    --output pretrained_weight/yolox_face_coco.zip `
    --keep-parent-levels 2
```

### 3. `convert_yolox_face_to_onnx.py`
备用转换脚本,提供更多手动控制选项。

**使用示例:**

```powershell
python tools/convert_yolox_face_to_onnx.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output-dir pretrained_weight/yolox_onnx/yolox_face_coco `
    --input-size 640 640
```

## 快速开始

### 环境准备

```powershell
# 安装必需的包
pip install mmdet==3.0.0
pip install mmcv>=2.0.0
pip install mmdeploy>=1.2.0
pip install onnxruntime
```

### 转换 YOLOX Face 权重

```powershell
# 方式 1: 一键导出 (推荐)
python tools/export_yolox_face.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip

# 方式 2: 手动两步走
# 步骤 1: 生成配置文件
python tools/export_yolox_face.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output yolox_face.zip `
    --skip-export

# 步骤 2: 手动导出 ONNX 后打包
python tools/onnx2zip.py `
    --sdk-dir work_dirs/yolox_face_export/yolox_face_onnx `
    --output pretrained_weight/yolox_face_coco.zip
```

### 使用转换后的模型

```python
from rtmlib import Custom
import cv2

# 初始化检测器
detector = Custom(
    det_class='YOLOX',
    det='pretrained_weight/yolox_face_coco.zip',
    det_input_size=(640, 640),
    backend='onnxruntime',
    device='cpu'
)

# 加载图像并检测
img = cv2.imread('test.jpg')
results = detector(img)
print(f"检测到 {len(results)} 个目标")
```

## 文件结构说明

转换后的 rtmlib 格式包含以下文件:

```
yolox_face_coco.zip
├── deploy.json       # 模型部署配置
├── detail.json       # 详细的导出配置信息
├── end2end.onnx     # ONNX 模型文件
└── pipeline.json    # 推理流程配置
```

### deploy.json
定义模型的基本信息,包括模型名称、后端、精度等。

### detail.json
包含详细的导出配置,包括:
- Codebase 信息 (MMDetection 版本等)
- ONNX 导出配置
- 后处理参数 (NMS 阈值等)

### pipeline.json
定义完整的推理流程,包括:
- 预处理步骤 (Resize, Pad, Normalize 等)
- 模型推理
- 后处理 (ResizeBBox, NMS 等)

### end2end.onnx
导出的 ONNX 模型文件,包含完整的推理图。

## 故障排除

### 问题 1: 找不到 MMDeploy

**错误信息:**
```
[ERROR] 未找到 MMDeploy,请确保已正确安装
```

**解决方案:**
```powershell
pip install mmdeploy mmdeploy-runtime
```

### 问题 2: ONNX 导出失败

**可能原因:**
- MMDetection 版本不兼容
- CUDA 版本问题
- 模型配置错误

**解决方案:**
1. 检查依赖版本:
   ```powershell
   pip list | findstr -i "mmdet mmcv mmdeploy"
   ```

2. 使用 `--skip-export` 跳过自动导出,手动完成

3. 尝试使用 CPU 设备:
   ```powershell
   python tools/export_yolox_face.py ... --device cpu
   ```

### 问题 3: 缺少必需文件

**错误信息:**
```
[ERROR] 缺少文件: end2end.onnx
```

**解决方案:**
确保 ONNX 导出成功,并且文件在正确的位置。检查 `work_dirs/` 目录下的输出。

## 参考文档

详细的转换指南请参考: [YOLOX Face 转换指南](../docs/YOLOX_FACE_CONVERSION_GUIDE.md)

## 参考格式

本工具生成的格式与以下参考目录兼容:
```
pretrained_weight/yolox_onnx/yolox_tiny_8xb8-300e_humanart-6f3252f9/
```

该目录包含标准的 rtmlib ONNX 模型格式,可作为格式参考。
