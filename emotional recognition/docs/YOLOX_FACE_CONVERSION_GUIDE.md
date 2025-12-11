# YOLOX Face 权重转换指南

本指南说明如何将 `yolo-x_8xb8-300e_coco-face_13274d7c.pth` 转换为 rtmlib 可用的格式。

## 方法一: 使用自动脚本 (推荐)

### 1. 安装依赖

```powershell
# 安装 MMDetection 和 MMDeploy
pip install mmdet==3.0.0
pip install mmcv>=2.0.0
pip install mmdeploy>=1.2.0
pip install mmdeploy-runtime
pip install onnxruntime
```

### 2. 运行导出脚本

```powershell
python tools/export_yolox_face.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip `
    --input-size 640 640 `
    --device cpu
```

参数说明:
- `--checkpoint`: PyTorch 权重文件路径
- `--output`: 输出的 zip 文件路径
- `--input-size`: 输入图像尺寸 (高度 宽度)
- `--device`: 导出设备 (cpu 或 cuda:0)

### 3. 使用生成的模型

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

---

## 方法二: 手动导出 (如果自动脚本失败)

### 步骤 1: 生成配置文件

```powershell
python tools/export_yolox_face.py `
    --checkpoint pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    --output pretrained_weight/yolox_face_coco.zip `
    --skip-export
```

这会在 `work_dirs/yolox_face_export/yolox_face_onnx/` 生成:
- `deploy.json`
- `detail.json`
- `pipeline.json`

### 步骤 2: 手动导出 ONNX

如果你有完整的 MMDeploy 环境:

```powershell
python -m mmdeploy.tools.deploy `
    work_dirs/yolox_face_export/temp_configs/mmdeploy_config.py `
    work_dirs/yolox_face_export/temp_configs/yolox_face_config.py `
    pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.pth `
    work_dirs/yolox_face_export/demo.jpg `
    --work-dir work_dirs/yolox_face_export/mmdeploy_output `
    --device cpu `
    --dump-info
```

### 步骤 3: 复制 ONNX 文件

将导出的 `end2end.onnx` 复制到:
```
work_dirs/yolox_face_export/yolox_face_onnx/end2end.onnx
```

### 步骤 4: 打包为 zip

```powershell
python tools/onnx2zip.py `
    --sdk-dir work_dirs/yolox_face_export/yolox_face_onnx `
    --output pretrained_weight/yolox_face_coco.zip
```

---

## 方法三: 使用已有的 ONNX 模型

如果你已经有 ONNX 模型文件,只需创建配置文件并打包:

### 1. 创建目录结构

```powershell
mkdir pretrained_weight\yolox_onnx\yolox_face_coco
```

### 2. 复制 ONNX 文件

将你的 `end2end.onnx` 复制到该目录。

### 3. 创建配置文件

手动创建以下文件 (参考 `yolox_tiny_8xb8-300e_humanart-6f3252f9` 中的格式):

**deploy.json**:
```json
{
    "version": "1.2.0",
    "task": "Detector",
    "models": [
        {
            "name": "yolox",
            "net": "end2end.onnx",
            "weights": "",
            "backend": "onnxruntime",
            "precision": "FP32",
            "batch_size": 1,
            "dynamic_shape": false
        }
    ],
    "customs": []
}
```

**detail.json** 和 **pipeline.json**: 参考参考文件夹中的内容,修改相应的尺寸参数。

### 4. 打包

```powershell
python tools/onnx2zip.py `
    --sdk-dir pretrained_weight\yolox_onnx\yolox_face_coco `
    --output pretrained_weight\yolox_face_coco.zip
```

---

## 故障排除

### 问题 1: MMDeploy 安装失败

尝试使用 conda 安装:
```powershell
conda install -c conda-forge mmdeploy
```

或者跳过自动导出,使用方法二手动导出。

### 问题 2: ONNX 导出失败

检查依赖版本:
```powershell
pip list | findstr -i "mmdet mmcv mmdeploy"
```

确保版本兼容:
- mmdet == 3.0.0
- mmcv >= 2.0.0
- mmdeploy >= 1.2.0

### 问题 3: 导出的模型无法使用

确认 rtmlib 版本与 MMDetection 版本兼容。参考 `yolox_tiny_8xb8-300e_humanart` 使用的版本。

---

## 验证导出结果

### 检查 zip 文件内容

```powershell
python -c "import zipfile; zf = zipfile.ZipFile('pretrained_weight/yolox_face_coco.zip'); print('ZIP 内容:'); [print(f'  {name}') for name in sorted(zf.namelist())]"
```

应该包含:
- `deploy.json`
- `detail.json`
- `end2end.onnx`
- `pipeline.json`

### 测试模型

```python
from rtmlib import Custom
import cv2

# 加载模型
detector = Custom(
    det_class='YOLOX',
    det='pretrained_weight/yolox_face_coco.zip',
    det_input_size=(640, 640),
    backend='onnxruntime',
    device='cpu'
)

# 测试图像
img = cv2.imread('test_image.jpg')
results = detector(img)
print(f"检测到 {len(results)} 个人脸")
```

---

## 参考

- [MMDeploy 文档](https://github.com/open-mmlab/mmdeploy)
- [MMDetection 文档](https://github.com/open-mmlab/mmdetection)
- [rtmlib 文档](https://github.com/Tau-J/rtmlib)
- 参考格式: `pretrained_weight/yolox_onnx/yolox_tiny_8xb8-300e_humanart-6f3252f9/`
