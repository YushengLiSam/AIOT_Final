from machine import Pin, PWM, I2C, ADC
import time
import network
import socket
import json
import uselect
import bmp280
import urequests


# ================= 0. wifi config =================
WIFI_SSID = "Jacksroom"
WIFI_PASS = "Gtj030912"
UDP_TARGET_IP = "192.168.0.172"
UDP_PORT = 3056
STATIC_IP_CONF = ('192.168.0.242', '255.255.255.0', '192.168.0.1', '8.8.8.8')


# computer config
PC_IP = "192.168.0.172"
PC_PORT = 3056  # FastAPI default port
HTTP_TRIGGER_URL = f"http://{PC_IP}:{PC_PORT}/touch" 

# udp brocast
UDP_TARGET_IP = PC_IP
UDP_PORT_NUM = 3056

SPEED = 200
TIME_FOR_45 = 0.75
SEC_PER_DEG = TIME_FOR_45 / 45.0
STARTUP_COMPENSATION = 0.05
LIGHT_THRESHOLD = 30
TRIGGER_TIME_MS = 500

R_PINS = [14, 32, 15, 33]
L_PINS = [21, 19, 5, 4]

try:
    i2c = I2C(0, scl=Pin(20), sda=Pin(22), freq=100000)
    sensor = bmp280.BMP280(i2c, addr=0x76)
    print("[Init] BMP280 Ready")
except:
    print("[Init] BMP280 Not Found")
    sensor = None

# Light Sensor (GPIO 37 -> SP, Input Only)

touch_pin = ADC(Pin(37))
touch_pin.atten(ADC.ATTN_11DB)
touch_pin.width(ADC.WIDTH_12BIT)

def make_pwm(pin_num):
    p = PWM(Pin(pin_num))
    p.freq(1000)
    p.duty(0)
    return p

rp = [make_pwm(p) for p in R_PINS]
lp = [make_pwm(p) for p in L_PINS]

def stop():
    for p in rp + lp: p.duty(0)

def set_motor(pwm_pin1, pwm_pin2, speed):
    pwm_pin1.duty(speed)
    pwm_pin2.duty(0)

def move_forward_raw():
    set_motor(lp[0], lp[1], SPEED); set_motor(lp[2], lp[3], SPEED)
    set_motor(rp[0], rp[1], SPEED); set_motor(rp[2], rp[3], SPEED)

def move_backward_raw():
    set_motor(lp[1], lp[0], SPEED); set_motor(lp[3], lp[2], SPEED)
    set_motor(rp[1], rp[0], SPEED); set_motor(rp[3], rp[2], SPEED)

def turn_right_raw():
    set_motor(lp[0], lp[1], SPEED); set_motor(lp[2], lp[3], SPEED)
    set_motor(rp[1], rp[0], SPEED); set_motor(rp[3], rp[2], SPEED)

def turn_left_raw():
    set_motor(lp[1], lp[0], SPEED); set_motor(lp[3], lp[2], SPEED)
    set_motor(rp[0], rp[1], SPEED); set_motor(rp[2], rp[3], SPEED)

def turn(angle):
    if angle == 0: return
    target_angle = abs(angle)
    duration = target_angle * SEC_PER_DEG
    if target_angle < 45: duration += STARTUP_COMPENSATION
    
    if angle > 0: turn_right_raw()
    else: turn_left_raw()
    
    time.sleep(duration)
    stop()
    time.sleep(0.5)

def api_forward():
    move_forward_raw(); time.sleep(1); stop()
    return "Moved Forward 1s"

def api_backward():
    move_backward_raw(); time.sleep(1); stop()
    return "Moved Backward 1s"

def api_turn(angle):
    turn(angle)
    return f"Turned {angle} degrees"

def api_temp():
    if sensor: return round(sensor.temperature, 2)
    return 0.0

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('Connecting WiFi...')
        wlan.connect(WIFI_SSID, WIFI_PASS)
        retry = 0
        while not wlan.isconnected() and retry < 20:
            retry += 1; time.sleep(1)
    if wlan.isconnected():
        print(f'\n>>> API Ready: http://{wlan.ifconfig()[0]}')
        return wlan.ifconfig()[0]
    return None

def start_system():
    ip = connect_wifi()
    if not ip: return

    # 1. API Server
    addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(addr)
    server_socket.listen(1)
    
    # 2. UDP Socket 
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 3. Poll
    poll = uselect.poll()
    poll.register(server_socket, uselect.POLLIN)

    print(">>> Service is ready: Awaiting instructions...")
    stop()
    
    is_triggered = False
    dark_start_time = 0
    is_dark = False

    while True:
        try:
            curr_val = touch_pin.read()
            
            if curr_val < LIGHT_THRESHOLD:
                # Fade to black (being touched)
                if not is_dark:
                    dark_start_time = time.ticks_ms()
                    is_dark = True
                else:
                    duration = time.ticks_diff(time.ticks_ms(), dark_start_time)
                    
                    #  Obstruction lasting > 500ms and no previous trigger
                    if duration > TRIGGER_TIME_MS and not is_triggered:
                        print(f"[Trigger] Continuous{duration}ms | Value:{curr_val}")
                    
                        try:
                            print(f"Request computer:{HTTP_TRIGGER_URL}")
                            r = urequests.get(HTTP_TRIGGER_URL)
                            print(f"computer response: {r.status_code}")
                            r.close()
                        except Exception as req_err:
                            print(f"request failure: {req_err}")

                        is_triggered = True
            else:
                # Brighten (Remove Hand)
                if is_triggered:
                    print("[Reset] Hands off!")
                is_dark = False
                is_triggered = False
                dark_start_time = 0
                
        except Exception as e:
            print("Loop Err:", e)

        events = poll.poll(0) 
        
        if events:
            try:
                cl, addr = server_socket.accept()
                request = cl.recv(1024).decode('utf-8')
                req_line = request.split('\n')[0]
                
                msg = "Invalid"; status = "error"; temp = 0
                
                if 'GET /forward' in req_line:
                    msg = api_forward(); status = "success"
                elif 'GET /backward' in req_line:
                    msg = api_backward(); status = "success"
                elif 'GET /turn' in req_line:
                    try:
                        val = req_line.split('angle=')[1].split(' ')[0]
                        msg = api_turn(int(val)); status = "success"
                    except: msg = "Arg Error"
                elif 'GET /status' in req_line:
                    msg = "Status OK"; status = "success"; temp = api_temp()

                response = {
                    "status": status, "result": msg, 
                    "temp": temp, "device": "esp32-pet"
                }
                cl.send('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n')
                cl.send(json.dumps(response))
                cl.close()
            except Exception as e:
                print("API Err:", e)
                try: cl.close()
                except: pass
        
        time.sleep(0.05)

try:
    start_system()
except KeyboardInterrupt:
    stop()