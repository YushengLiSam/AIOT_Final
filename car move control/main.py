from machine import Pin, PWM
import time
import network
import socket
import json # 引入JSON库，方便给大模型回话

# ================= 0. Wi-Fi 配置 =================
WIFI_SSID = ""
WIFI_PASS = ""

# ================= 1. 核心配置 =================
SPEED = 200
# 转向校准参数
TIME_FOR_90 = 0.75
SEC_PER_DEG = TIME_FOR_90 / 90.0
STARTUP_COMPENSATION = 0.05

# ================= 2. 引脚定义 =================
# 右边 (Right)
R_PINS = [14, 32, 15, 33]
# 左边 (Left)
L_PINS = [21, 19, 5, 4]

# ================= 3. 电机初始化 =================
def make_pwm(pin_num):
    p = PWM(Pin(pin_num))
    p.freq(1000)
    p.duty(0)
    return p

rp = [make_pwm(p) for p in R_PINS]
lp = [make_pwm(p) for p in L_PINS]

# ================= 4. 底层动作 =================
def stop():
    for p in rp + lp: p.duty(0)

def set_motor(pwm_pin1, pwm_pin2, speed):
    pwm_pin1.duty(speed)
    pwm_pin2.duty(0)

def move_forward_raw():
    set_motor(rp[0], rp[1], SPEED); set_motor(rp[2], rp[3], SPEED)
    set_motor(lp[0], lp[1], SPEED); set_motor(lp[2], lp[3], SPEED)

def move_backward_raw():
    set_motor(rp[1], rp[0], SPEED); set_motor(rp[3], rp[2], SPEED)
    set_motor(lp[1], lp[0], SPEED); set_motor(lp[3], lp[2], SPEED)

def turn_right_raw():
    set_motor(lp[0], lp[1], SPEED); set_motor(lp[2], lp[3], SPEED)
    set_motor(rp[1], rp[0], SPEED); set_motor(rp[3], rp[2], SPEED)

def turn_left_raw():
    set_motor(lp[1], lp[0], SPEED); set_motor(lp[3], lp[2], SPEED)
    set_motor(rp[0], rp[1], SPEED); set_motor(rp[2], rp[3], SPEED)

def turn(angle):
    """
    输入任意角度，自动计算时间
    angle: 正数=右转, 负数=左转 (例如: 15, 30, 45, 90, 180...)
    """
    if angle == 0: return
    
    target_angle = abs(angle)
    
    # --- 核心算法 ---
    # 1. 基础时间 = 角度 * 每度时间
    duration = target_angle * SEC_PER_DEG
    
    # 2. 智能补偿 (针对小角度)
    # 如果角度小于 45度，电机可能还没跑起来就停了，所以加一点点时间
    if target_angle < 45:
        duration += STARTUP_COMPENSATION
        print(f"[补偿] 小角度起步，增加 {STARTUP_COMPENSATION}s")

    direction = "右" if angle > 0 else "左"
    print(f">> 指令: 向{direction}转 {target_angle}° | 时间: {duration:.4f}s")

    # 执行
    if angle > 0:
        turn_right_raw()
    else:
        turn_left_raw()
        
    time.sleep(duration)
    
    # 刹车 & 消除惯性
    stop()
    time.sleep(0.5)
  

# ================= 5. API 业务逻辑 (带自动停止) =================

def api_forward():
    print(">> 执行: 前进 1秒")
    move_forward_raw()
    time.sleep(1)
    stop()
    return "Moved Forward 1s."

def api_backward():
    print(">> 执行: 后退 1秒")
    move_backward_raw()
    time.sleep(1)
    stop()
    return "Moved Backward 1s."

def api_turn(angle):
    turn(angle)
    return f"Turned {angle} degrees"

# ================= 6. 网络连接 =================
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('Connecting to WiFi...')
        wlan.connect(WIFI_SSID, WIFI_PASS)
        retry = 0
        while not wlan.isconnected() and retry < 15:
            retry += 1
            time.sleep(1)
            
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f'\n>>> API Ready at: http://{ip}')
        return ip
    else:
        print('WiFi Connection Failed')
        return None

# ================= 7. Server 主循环 (含参数解析) =================
def run_server():
    ip = connect_wifi()
    if not ip: return

    addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(addr)
    s.listen(1)

    print("等待 API 调用...")
    stop()

    while True:
        try:
            cl, addr = s.accept()
            # 读取 Request (第一行通常包含 URL)
            request = cl.recv(1024).decode('utf-8')
            
            # 获取请求行，例如 "GET /turn?angle=90 HTTP/1.1"
            request_line = request.split('\n')[0]
            
            msg = "Invalid Command"
            status = "error"
            
            # === 路由逻辑 ===
            
            if 'GET /forward' in request_line:
                msg = api_forward()
                status = "success"
                
            elif 'GET /backward' in request_line:
                msg = api_backward()
                status = "success"
                
            elif 'GET /turn' in request_line:
                # 解析参数: /turn?angle=90
                try:
                    # 提取 URL 部分 "/turn?angle=90"
                    path = request_line.split(' ')[1] 
                    if 'angle=' in path:
                        # 提取数字
                        angle_str = path.split('angle=')[1]
                        angle = int(angle_str)
                        msg = api_turn(angle)
                        status = "success"
                    else:
                        msg = "Missing 'angle' parameter"
                except:
                    msg = "Parameter Parsing Error"
            
            # 返回 JSON
            response = {
                "status": status, 
                "result": msg
            }
            
            cl.send('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n')
            cl.send(json.dumps(response))
            cl.close() 
            
        except OSError:
            cl.close()
        except KeyboardInterrupt:
            s.close()
            stop()
            break

# 启动
try:
    run_server()
except KeyboardInterrupt:
    stop()