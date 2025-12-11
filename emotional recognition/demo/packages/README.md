# 本地 Python 包

这个目录包含了项目所需的 Python 包,无需全局安装即可使用。

## 包含的包

- **mmdet** (3.0.0) - OpenMMLab Detection Toolbox
- **mmcv** (2.0.1) - OpenMMLab Computer Vision Foundation Library  
- **mmdeploy** (1.2.0) - OpenMMLab Model Deployment Toolbox
- **mmpose** (1.3.2) - OpenMMLab Pose Estimation Toolbox

## 使用方法

在您的 Python 脚本中,添加以下代码来使用本地包:

```python
import sys
from pathlib import Path

# 添加本地包路径
packages_dir = Path(__file__).parent / "packages"
sys.path.insert(0, str(packages_dir))

# 现在可以导入了
import mmdet
import mmcv
import mmdeploy
import mmpose
```

## 示例

查看 `test_local_packages.py` 以获取完整的使用示例。

## 磁盘使用

- mmdet: ~17MB
- mmcv: ~3.7MB
- mmdeploy: ~18MB
- mmpose: ~7.9MB
- 总计: ~47MB

## 更新包

如果需要更新这些包,可以运行:

```bash
# 从系统安装目录复制最新版本
pip install --upgrade mmdet==3.0.0 mmcv==2.0.1 mmdeploy==1.2.0 mmpose==1.3.2
cp -r $(pip show mmdet | grep Location | cut -d' ' -f2)/mmdet ./packages/
cp -r $(pip show mmcv | grep Location | cut -d' ' -f2)/mmcv ./packages/
cp -r $(pip show mmdeploy | grep Location | cut -d' ' -f2)/mmdeploy ./packages/
cp -r $(pip show mmpose | grep Location | cut -d' ' -f2)/mmpose ./packages/
```
