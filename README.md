这是一个为你定制的 **README.md** 文档。你可以直接把它放在你的 GitHub 仓库或者项目文件夹里。

这份文档不仅包含了如何使用代码，还详细记录了我们讨论过的**硬件接线逻辑**和**参数校准方法**，这对于你的 Final Project 报告和演示（Demo）非常有帮助，显得非常专业。

-----

# Desktop Pet Robot - Precision Chassis Driver

# 桌面宠物小车 - 高精度步进电机底盘驱动

本项目是哥伦比亚大学 CE 研究生课程 Final Project 的一部分。这是一个基于 **树莓派 (Raspberry Pi)** 和 **Python** 的底层驱动库，用于控制由 **4个步进电机** 驱动的坦克式（差速驱动）小车。

该驱动采用了 **八拍半步模式 (8-beat Half-step)** 励磁算法，能够实现厘米级精度的直线行驶和度数级精度的原地旋转（坦克掉头）。

## 🛠 硬件架构 (Hardware Architecture)

由于 L298N 驱动步进电机的特性，本项目采用了 **双板四电机并联** 的方案：

  * **控制器**: Raspberry Pi 4B (或任意支持 GPIO 的型号)
  * **驱动模块**: 2x L298N 双桥驱动板
  * **执行器**: 4x 步进电机 (推荐 28BYJ-48 或 NEMA17)
  * **电源**: 7V - 12V 锂电池组 (共地连接)

### 接线逻辑 (Critical Wiring)

为了利用两块驱动板控制四轮差速运动，接线必须遵循以下**并联规则**：

1.  **左侧驱动 (Left Board)**:

      * **L298N 输出 A & B**: 同时连接 **左前 (FL)** 和 **左后 (BL)** 电机的线圈。
      * **GPIO 信号**: GPIO 17, 18, 27, 22 $\rightarrow$ IN1, IN2, IN3, IN4

2.  **右侧驱动 (Right Board)**:

      * **L298N 输出 A & B**: 同时连接 **右前 (FR)** 和 **右后 (BR)** 电机的线圈。
      * **GPIO 信号**: GPIO 23, 24, 25, 5 $\rightarrow$ IN1, IN2, IN3, IN4
      * *注意：右侧电机线序需与左侧完全一致（代码已处理镜像安装的逻辑）。*

## 🔌 引脚定义 (Pinout Configuration)

| 驱动板功能 | L298N 引脚 | 树莓派 GPIO (BCM) | 物理引脚 (Board) |
| :--- | :--- | :--- | :--- |
| **左轮控制** | IN1 | **17** | 11 |
| | IN2 | **18** | 12 |
| | IN3 | **27** | 13 |
| | IN4 | **22** | 15 |
| **右轮控制** | IN1 | **23** | 16 |
| | IN2 | **24** | 18 |
| | IN3 | **25** | 22 |
| | IN4 | **5** | 29 |

> **⚠️ 重要提示**: 树莓派的 GND 必须与 L298N 的电源地（GND）相连，否则无法控制。

## ⚙️ 参数校准 (Calibration)

在运行代码前，必须在 `driver.py` 顶部根据实际测量值修改以下常数，以确保运动精度：

```python
# 1. 轮子直径 (cm) - 决定直线距离的精度
WHEEL_DIAMETER = 6.0   

# 2. 轮距 (cm) - 左右轮中心距，决定旋转角度的精度
TRACK_WIDTH = 15.0     

# 3. 电机步数 - 28BYJ-48通常为4096，NEMA17通常为200
STEPS_PER_REV = 4096   
```

## 🚀 快速开始 (Quick Start)

### 1\. 依赖安装

需要安装树莓派 GPIO 库：

```bash
sudo apt-get update
sudo apt-get install python3-rpi.gpio
```

### 2\. 代码示例

创建一个 `main.py` 并调用驱动类：

```python
from mecanum_driver import PrecisionTank
import time

# 初始化小车
car = PrecisionTank()

try:
    print("1. 前进 20 厘米")
    car.move_cm(20)
    time.sleep(1)

    print("2. 原地右转 90 度")
    car.turn_degrees(90)
    time.sleep(1)

    print("3. 后退 10 厘米")
    car.move_cm(-10)

finally:
    # 程序退出时自动释放 GPIO 资源
    car.cleanup()
```

## 🧠 技术原理 (Technical Details)

### 1\. 八拍半步驱动 (Half-step Excitation)

本驱动不使用粗糙的 4 拍模式，而是使用 8 拍时序 (`A -> AB -> B -> BC...`)。

  * **优势**: 消除低速抖动，防止宠物小车在桌面上打滑。
  * **精度**: 步进角分辨率提高一倍，旋转定位更精准。

### 2\. 运动学解算 (Kinematics)

  * **直线**: $Steps = \frac{Distance}{\pi \times D_{wheel}} \times Steps_{rev}$
  * **旋转**: 利用差速原理，将目标角度转换为弧长，再转换为左右轮的反向步数。
      * $Arc = \frac{Angle}{360} \times (\pi \times Width_{track})$

## 📋 常见问题 (FAQ)

**Q: 为什么我叫它前进，它却原地打转？**
A: 右侧电机的物理接线可能反了。不要改接线，请在代码 `move_cm` 函数中，将右轮的 `-idx` 改为 `idx` 即可。

**Q: 为什么转 90 度实际只转了 80 度？**
A: `TRACK_WIDTH` (轮距) 测量偏小，或者轮子在地毯上打滑。请增大代码中的 `TRACK_WIDTH` 值进行微调。

-----

*Created for Columbia University EC Project - Fall 2025*