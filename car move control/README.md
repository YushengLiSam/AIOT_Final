# LLM-Ready ESP32 Robot Car API

**A MicroPython-based REST API firmware for 4WD DC motor robot cars.**

This project turns an ESP32-based robot car into an **IoT device** controllable via HTTP requests. It is specifically designed to be operated by **AI Agents (LLMs)** like ChatGPT, Claude, or local models, featuring precise time-based turning algorithms and JSON-formatted responses.

-----

## ✨ Features

  * **REST API Control**: Control the car using simple HTTP GET requests.
  * **LLM Integration**: Returns standardized JSON responses (`status`, `result`) optimized for AI tool calling.
  * **Smart Turning Algorithm**: Automatically calculates rotation time based on input degrees (e.g., `turn(90)` vs `turn(45)`).
  * **PWM Speed Control**: Soft-start and speed limiting to protect gearboxes and ensure stability.
  * **No-Code Operation**: Once flashed, the car connects to Wi-Fi and waits for commands.

-----

## 🛠️ Hardware Requirements

  * **Controller**: ESP32 Development Board (e.g., Adafruit Feather V2, DevKit V1).
  * **Motors**: 4x DC Gear Motors (Yellow "TT" motors).
  * **Drivers**: 2x L298N Dual H-Bridge Motor Drivers.
  * **Power**: 12V Battery Pack (High current, \>3A recommended).
  * **Chassis**: 4WD Car Chassis.

-----

## Wiring & Pinout

This firmware uses **8 GPIOs** to control 4 motors independently (via PWM).

### Power Connections

  * **Battery (+)** $\to$ L298N 12V Input (Both drivers).
  * **Battery (-)** $\to$ L298N GND **AND** ESP32 GND (**Common Ground is critical**).
  * **ESP32 Power** $\to$ USB via Raspberry Pi or Step-down converter (5V).

### GPIO Mapping (Configured in `main.py`)

| Motor Group | L298N Pin | ESP32 Pin (GPIO) | Description |
| :--- | :--- | :--- | :--- |
| **Right Side** | IN1 | **14** | Right Front/Back Forward |
| | IN2 | **32** | Right Front/Back Backward |
| | IN3 | **15** | Right Front/Back Forward |
| | IN4 | **33** | Right Front/Back Backward |
| **Left Side** | IN1 | **21** | Left Front/Back Forward |
| | IN2 | **19** | Left Front/Back Backward |
| | IN3 | **5** | Left Front/Back Forward |
| | IN4 | **4** | Left Front/Back Backward |

> **Note**: If your wheels spin in the wrong direction, swap the two wires connecting the motor to the L298N green terminal. Do not change the code.

-----

## 🚀 Installation & Setup

1.  **Flash MicroPython**: Ensure your ESP32 is flashed with the latest MicroPython firmware.
2.  **Configuration**: Open `main.py` and edit the Wi-Fi credentials:
    ```python
    WIFI_SSID = "Your_WiFi_Name"
    WIFI_PASS = "Your_WiFi_Password"
    ```
3.  **Upload**: Use **Thonny IDE** to save the file as `main.py` directly to the ESP32.
4.  **Run**: Reset the board. Watch the console for the IP address:
    ```text
    >>> API Ready at: http://192.168.1.X
    ```

-----

## 📡 API Documentation

The car runs a lightweight web server on port **80**.

### 1\. Move Forward

  * **Endpoint**: `GET /forward`
  * **Action**: Moves forward for **1 second**, then stops automatically.
  * **Response**: `{"status": "success", "result": "Moved Forward 1s."}`

### 2\. Move Backward

  * **Endpoint**: `GET /backward`
  * **Action**: Moves backward for **1 second**, then stops automatically.
  * **Response**: `{"status": "success", "result": "Moved Backward 1s."}`

### 3\. Turn (Precise Angle)

  * **Endpoint**: `GET /turn?angle=<degrees>`
  * **Parameters**:
      * `angle` (int):
          * **Positive (+)**: Turn Right (Clockwise).
          * **Negative (-)**: Turn Left (Counter-Clockwise).
  * **Examples**:
      * `/turn?angle=90` (Right turn)
      * `/turn?angle=-45` (Left turn)
      * `/turn?angle=180` (U-Turn)
  * **Response**: `{"status": "success", "result": "Turned 90 degrees"}`

-----

## ⚙️ Calibration (Optional)

If the car turns too much or too little, adjust these constants at the top of `main.py`:

```python
# Speed (0-1023): Adjust motor power
SPEED = 200 

# Calibration: How many seconds does it take to turn 90 degrees?
# Increase if it turns < 90, Decrease if it turns > 90.
TIME_FOR_90 = 0.75 
```

-----

## 🤖 LLM System Prompt (Integration Guide)

To control this car using an AI Agent (like ChatGPT or LangChain), copy and paste the following system prompt into your LLM:

```markdown
# Role
You are an AI agent controlling a physical robot car via HTTP APIs.
The car accepts commands at: http://[INSERT_CAR_IP_HERE]

# API Tools
1. **Move Forward**: `GET /forward` (Moves 1s)
2. **Move Backward**: `GET /backward` (Moves 1s)
3. **Turn**: `GET /turn?angle={degrees}`
   - Positive angle = Right Turn.
   - Negative angle = Left Turn.

# Instructions
- To move in a square: Forward -> Turn 90 -> Forward -> Turn 90...
- Always wait for the HTTP response before sending the next command.
- Output a Python script using the `requests` library to perform the user's task.
```

-----

## 📝 License

This project is open-source. Feel free to modify and distribute.