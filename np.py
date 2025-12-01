import RPi.GPIO as GPIO
import time

# ================== 核心参数 (必须修改) ==================
# 1. 轮子直径 (厘米)
WHEEL_DIAMETER = 6.0   
# 2. 轮距: 左轮中心到右轮中心的距离 (厘米)
TRACK_WIDTH = 15.0     
# 3. 步进电机转一圈需要的步数
#    - 如果是 28BYJ-48 (那种小的圆的)，填 4096
#    - 如果是 NEMA17 (那种方的)，填 200 (或者根据细分设置填 400/800/1600)
STEPS_PER_REV = 4096   
# ========================================================

# === 引脚定义 (BCM编码) ===
# 确认这4个脚接的是左边的板子 (控制左边两个并联电机)
LEFT_PINS = [17, 18, 27, 22] 
# 确认这4个脚接的是右边的板子 (控制右边两个并联电机)
RIGHT_PINS = [23, 24, 25, 5]

class PrecisionTank:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.pins = LEFT_PINS + RIGHT_PINS
        
        # 初始化引脚，全部拉低防止发热
        for p in self.pins:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, 0)
            
        # 八拍模式 (Half-step): 精度高，扭矩大，动作平滑
        # 对应 L298N 的 IN1-IN4 顺序
        self.seq = [
            [1,0,0,0], [1,1,0,0], [0,1,0,0], [0,1,1,0],
            [0,0,1,0], [0,0,1,1], [0,0,0,1], [1,0,0,1]
        ]

    def _apply_step(self, pins, step_index):
        """给指定的4个引脚发送时序信号"""
        seq_val = self.seq[step_index % 8] # 取余数循环
        for pin, val in zip(pins, seq_val):
            GPIO.output(pin, val)

    def turn_degrees(self, degrees, speed_delay=0.002):
        """
        精确旋转指定角度
        :param degrees: 正数向右转，负数向左转
        """
        # 1. 计算小车需要转过的弧长 (Arc Length)
        #    弧长 = (角度 / 360) * 车身周长
        arc_length = (abs(degrees) / 360.0) * (3.1415926 * TRACK_WIDTH)
        
        # 2. 将弧长换算成轮子需要转的步数
        #    步数 = (弧长 / 轮子周长) * 一圈步数
        steps_needed = int((arc_length / (3.1415926 * WHEEL_DIAMETER)) * STEPS_PER_REV)
        
        print(f"指令: 旋转 {degrees}度 | 计算: 需行走 {steps_needed} 步")

        # 3. 开始步进
        for i in range(steps_needed):
            if degrees > 0: 
                # === 向右转 (Right Turn) ===
                # 左轮前进 (正序 0->7)
                self._apply_step(LEFT_PINS, i)
                # 右轮后退 (也是正序，但因为电机安装方向相反，物理上是反转)
                # *注意*: 如果发现轮子转反了，把这里的 i 改成 -i
                self._apply_step(RIGHT_PINS, i)
            else:
                # === 向左转 (Left Turn) ===
                # 左轮后退 (逆序 7->0)
                self._apply_step(LEFT_PINS, -i)
                # 右轮前进 (逆序)
                self._apply_step(RIGHT_PINS, -i)
            
            # 步间延迟，控制速度
            time.sleep(speed_delay)
            
        self.stop()

    def move_cm(self, distance, speed_delay=0.002):
        """直线移动"""
        steps = int((abs(distance) / (3.1415926 * WHEEL_DIAMETER)) * STEPS_PER_REV)
        direction = 1 if distance > 0 else -1
        
        print(f"指令: 移动 {distance}cm | 计算: 需行走 {steps} 步")
        
        for i in range(steps):
            idx = i if direction == 1 else -i
            # 直行时，左右轮在代码逻辑上通常是一正一反（因为安装是镜像的）
            # 你需要根据实际情况调试这里的正负号
            self._apply_step(LEFT_PINS, idx)
            self._apply_step(RIGHT_PINS, -idx) 
            time.sleep(speed_delay)
        self.stop()

    def stop(self):
        """停止并释放线圈，防止电机过热"""
        for p in self.pins:
            GPIO.output(p, 0)
            
    def cleanup(self):
        self.stop()
        GPIO.cleanup()

# ================= 测试入口 =================
if __name__ == "__main__":
    car = PrecisionTank()
    try:
        # 测试1: 向右转 90 度
        car.turn_degrees(90)
        time.sleep(1)
        
        # 测试2: 向左转 90 度 (回到原位)
        car.turn_degrees(-90)
        
    except KeyboardInterrupt:
        pass
    finally:
        car.cleanup()