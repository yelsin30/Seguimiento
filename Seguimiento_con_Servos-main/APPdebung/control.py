import requests

ESP32_IP = "http://10.187.208.231"  # cambia por tu IP real de la ESP32-CAM

def mover_servo(pan=None, tilt=None):
    """Envía comandos a la ESP32-CAM para mover los servos"""
    try:
        if pan is not None:
            requests.get(f"{ESP32_IP}/?PAN={int(pan)}", timeout=0.3)
        if tilt is not None:
            requests.get(f"{ESP32_IP}/?TILT={int(tilt)}", timeout=0.3)
    except Exception as e:
        print("⚠️ Error al enviar comando a servos:", e)

