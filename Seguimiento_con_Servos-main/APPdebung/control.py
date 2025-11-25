import requests

# ✅ IP actual de tu ESP32-CAM
ESP32_IP = "http://10.18.122.231"

def mover_servo(pan=None, tilt=None):
    """Envía comandos a la ESP32-CAM para mover los servos por WiFi."""
    if pan is None and tilt is None:
        return

    try:
        # Limitar valores entre 0° y 180°
        if pan is not None:
            pan = max(0, min(180, int(pan)))
        if tilt is not None:
            tilt = max(0, min(180, int(tilt)))

        # Construir URL
        url = f"{ESP32_IP}/?"
        params = []
        if pan is not None:
            params.append(f"PAN={pan}")
        if tilt is not None:
            params.append(f"TILT={tilt}")
        url += "&".join(params)

        # Enviar petición
        r = requests.get(url, timeout=0.5)
        if r.status_code == 200:
            print(f"✅ Servos movidos → {url}")
        else:
            print(f"⚠️ ESP32 respondió con error: {r.status_code}")

    except requests.exceptions.ConnectTimeout:
        print("⚠️ Tiempo de espera agotado: no hay conexión con la ESP32.")
    except Exception as e:
        print("⚠️ Error al enviar comando a servos:", e)
