# 脚本使用说明

## 功能

`convert_yolox_face_to_onnx.py` 会自动从已安装的 Python 包中读取配置文件:

- **mmpose**: `demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py`
- **mmdet**: `configs/yolox/yolox_s_8xb8-300e_coco.py`
- **mmdeploy**: `configs/mmdet/detection/detection_onnxruntime_dynamic.py`

## 快速使用

### 方式 1: 一键转换 (需要完整环境)

如果你安装了所有必要的包:

```powershell
# 安装依赖
pip install mmdet==3.0.0 mmpose mmdeploy mmdeploy-runtime onnxruntime

# 运行脚本
python tools/convert_yolox_face_to_onnx.py
```

脚本会:
1. 从包中自动读取配置文件
2. 生成 JSON 配置
3. 使用 MMDeploy 导出 ONNX
4. 打包为 zip

### 方式 2: 仅生成配置 (推荐)

如果不想安装完整的 mmdet/mmpose/mmdeploy:

```powershell
# 直接运行
python tools/convert_yolox_face_to_onnx.py
```

脚本会:
1. 使用默认配置生成 JSON 文件
2. 提示你手动添加 ONNX 文件

然后你可以:

```powershell
# 将 ONNX 文件复制到 SDK 目录
Copy-Item path/to/your/end2end.onnx pretrained_weight/yolox_face_workdir/sdk/

# 打包
python tools/onnx2zip.py `
    --sdk-dir pretrained_weight/yolox_face_workdir/sdk `
    --output pretrained_weight/yolo-x_8xb8-300e_coco-face_13274d7c.zip
```

## 当前状态

脚本已成功生成以下文件:

```
pretrained_weight/yolox_face_workdir/sdk/
├── deploy.json       ✅ 已生成
├── detail.json       ✅ 已生成
└── pipeline.json     ✅ 已生成
```

只需添加 `end2end.onnx` 文件即可完成打包。

## 配置文件格式

生成的配置文件格式与参考目录完全一致:
```
pretrained_weight/yolox_onnx/yolox_tiny_8xb8-300e_humanart-6f3252f9/
```

可以直接用于 rtmlib。
