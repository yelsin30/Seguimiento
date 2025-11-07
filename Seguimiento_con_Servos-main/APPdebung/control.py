import serial
import time

ESP32_PORT = "COM8"
BAUD_RATE = 115200

try:
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"✅ Conectado a {ESP32_PORT}")
except Exception as e:
    print(f"❌ No se pudo abrir {ESP32_PORT}: {e}")
    ser = None

def mover_servo(pan=None, tilt=None):
    if ser is None:
        print("⚠️ No hay conexión serial activa.")
        return
    try:
        comando = f"x:{int(pan)},y:{int(tilt)}\n"
        ser.write(comando.encode('utf-8'))
    except Exception as e:
        print("⚠️ Error al enviar comando por Serial:", e)
