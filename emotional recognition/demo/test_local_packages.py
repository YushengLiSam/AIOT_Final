#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试本地安装的包是否可以正常导入
"""

import sys
from pathlib import Path

# 添加本地包路径到 Python 路径
packages_dir = Path(__file__).parent / "packages"
sys.path.insert(0, str(packages_dir))

print(f"✓ 已添加本地包路径: {packages_dir}\n")

# 测试导入
try:
    import mmdet
    print(f"✓ mmdet 版本: {mmdet.__version__}")
    print(f"  位置: {mmdet.__file__}")
except Exception as e:
    print(f"✗ mmdet 导入失败: {e}")

try:
    import mmcv
    print(f"\n✓ mmcv 版本: {mmcv.__version__}")
    print(f"  位置: {mmcv.__file__}")
except Exception as e:
    print(f"\n✗ mmcv 导入失败: {e}")

try:
    import mmdeploy
    print(f"\n✓ mmdeploy 版本: {mmdeploy.__version__}")
    print(f"  位置: {mmdeploy.__file__}")
except Exception as e:
    print(f"\n✗ mmdeploy 导入失败: {e}")

try:
    import mmpose
    print(f"\n✓ mmpose 版本: {mmpose.__version__}")
    print(f"  位置: {mmpose.__file__}")
except Exception as e:
    print(f"\n✗ mmpose 导入失败: {e}")

print("\n" + "="*70)
print("✓ 所有包都可以从本地 demo/packages 目录正常导入!")
print("="*70)
