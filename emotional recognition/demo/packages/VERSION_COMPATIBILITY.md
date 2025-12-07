# 包版本兼容性说明

本目录中的包版本已经过测试，彼此兼容。

## 已安装的包版本

| 包名 | 版本 | 用途 |
|------|------|------|
| mmcv | 2.0.1 | OpenMMLab 计算机视觉基础库 |
| mmdet | 3.0.0 | 目标检测工具箱 |
| mmdeploy | 1.2.0 | 模型部署工具箱 |
| mmpose | 1.3.2 | 姿态估计工具箱 |

## 兼容性矩阵

这些版本是根据 OpenMMLab 官方兼容性要求选择的：

- **mmdet 3.0.0** 需要:
  - mmcv >= 2.0.0rc4
  - mmengine >= 0.7.0

- **mmpose 1.3.2** 需要:
  - mmcv >= 2.0.0
  - mmdet >= 3.0.0 (可选，用于检测器)
  - mmengine >= 0.7.0

- **mmdeploy 1.2.0** 支持:
  - mmcv 2.0.x
  - mmdet 3.0.x
  - mmpose 1.x

## Python 版本要求

- Python >= 3.7
- 当前测试环境: Python 3.9

## 依赖说明

这些包还依赖以下库（已包含在系统环境中）:
- numpy
- opencv-python
- matplotlib
- scipy
- torch
- torchvision
- onnxruntime (用于 ONNX 推理)

## 验证安装

运行以下命令验证所有包是否正确安装:

```bash
cd demo
python test_local_packages.py
```

## 更新建议

如需更新包版本，请参考以下兼容性组合：

### 推荐组合 1 (当前使用)
- mmcv: 2.0.1
- mmdet: 3.0.0
- mmdeploy: 1.2.0
- mmpose: 1.3.2

### 推荐组合 2 (更新版本)
- mmcv: 2.1.0+
- mmdet: 3.2.0+
- mmdeploy: 1.3.0+
- mmpose: 1.3.2+

注意：更新前请查阅 OpenMMLab 官方文档确认版本兼容性。
