# 🚗 AIoT Smart Pet Car (ESP32 Firmware)

This repository contains the MicroPython firmware for the **AIoT Final Project**. The system controls a 4-wheel drive robot car powered by an ESP32, featuring HTTP API control, environmental sensing (Temperature), and interactive touch feedback via a light sensor.

## ✨ Features

* **RESTful API Control**: Control movement (Forward, Backward, Turn) via HTTP GET requests.
* **Environmental Sensing**: Real-time temperature monitoring using the **BMP280** sensor.
* **Interactive "Petting" Mode**: Uses a light sensor (Photoresistor) to detect touch. When "pet" (covered) for 500ms, it triggers a callback to a host computer.
* **Non-blocking Server**: Uses `uselect` polling to handle HTTP requests while simultaneously monitoring sensors.
* **WiFi Connectivity**: Auto-connects to configured WiFi and broadcasts its IP.

## 🛠 Hardware Configuration

### Pin Mapping

| Component | ESP32 GPIO Pins | Description |
| :--- | :--- | :--- |
| **Right Motors** | 14, 32, 15, 33 | PWM Channels for Right Wheels |
| **Left Motors** | 21, 19, 5, 4 | PWM Channels for Left Wheels |
| **I2C Bus** | SDA: 22, SCL: 20 | For BMP280 Sensor |
| **Light Sensor** | 37 (ADC1) | Analog input for touch detection |

### Bill of Materials (BOM)
* ESP32 Development Board
* L298N (or similar) Motor Driver
* 4x DC Motors & Wheels
* BMP280 Barometric Pressure & Temp Sensor
* Photoresistor (Light Sensor)
* Power Supply (Battery Pack)

## 🚀 Installation & Setup

1.  **Flash MicroPython**: Ensure your ESP32 is flashed with the latest MicroPython firmware.
2.  **Upload Dependencies**:
    * Upload `bmp280.py` (Driver library) to the root of the ESP32.
    * Upload `main.py` (The code in this repo).
3.  **Configure Network**:
    Open `main.py` and update the following variables:
    ```python
    WIFI_SSID = "Your_WiFi_Name"
    WIFI_PASS = "Your_WiFi_Password"
    
    # IP of the computer running the emotional recognition/controller backend
    PC_IP = "192.168.0.172" 
    ```
4.  **Run**: Reset the board. The serial monitor will output:
    ```text
    [Init] BMP280 Ready
    Connecting WiFi...
    >>> API Ready: [http://192.168.0.xxx](http://192.168.0.xxx)
    ```

## 📡 API Reference

The car runs a lightweight HTTP server on **Port 80**.

### 1. Movement Control
| Endpoint | Method | Params | Description |
| :--- | :--- | :--- | :--- |
| `/forward` | `GET` | None | Moves forward for 1 second. |
| `/backward` | `GET` | None | Moves backward for 1 second. |
| `/turn` | `GET` | `angle` (int) | Turns left (negative) or right (positive).<br>Example: `/turn?angle=90` |

### 2. System Status
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/status` | `GET` | Returns JSON with status and temperature. |

**Response Example:**
```json
{
    "status": "success",
    "result": "Status OK",
    "temp": 24.50,
    "device": "esp32-pet"
}