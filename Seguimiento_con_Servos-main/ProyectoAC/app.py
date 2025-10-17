#!/usr/bin/env python3
"""
app_debug.py
Versión del app con:
 - Debug de clases del modelo
 - Mostrar probabilidades de predicción
 - Guardar imágenes+landmarks con tecla 's' para recopilar datos de reentreno
 - Opción --swap-actions para invertir la acción asociada a cada etiqueta
"""
import os
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import time
import argparse
import traceback
from pathlib import Path
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", help="ruta a modelo_gestos.pkl")
parser.add_argument("--camera-url", "-c", default="http://10.187.208.231:81/stream")
parser.add_argument("--swap-actions", action="store_true",
                    help="Intercambia acciones: interpretar 'derecha' como retroceder y 'izquierda' como avanzar (temporal)")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent
# carga cámara
ESP32_URL = args.camera_url if args.camera_url != "0" else 0
cap = cv2.VideoCapture(ESP32_URL)
time.sleep(0.5)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)
if not cap.isOpened():
    raise SystemExit("No se pudo abrir cámara")

# cargar modelo (soporta payload dict con 'model' y 'scaler')
model_path = args.model or BASE_DIR / "modelo_gestos.pkl"
model_path = Path(model_path)
if not model_path.exists():
    raise SystemExit(f"No se encontró modelo en {model_path}")

with open(model_path, "rb") as f:
    payload = pickle.load(f)

if isinstance(payload, dict) and "model" in payload:
    clf = payload["model"]
    scaler = payload.get("scaler", None)
    print("Payload cargado: contiene model + optional scaler")
else:
    clf = payload
    scaler = None

print("model.classes_:", getattr(clf, "classes_", None))
has_proba = hasattr(clf, "predict_proba")
print("predict_proba available:", has_proba)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.5)

# carpeta para salvar ejemplos nuevos (puede revisar y mover luego)
SAVE_DIR = BASE_DIR / "dataset" / "images" / "Puño_new"
SAVE_LM_DIR = BASE_DIR / "dataset" / "landmarks" / "Puño_new"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_LM_DIR.mkdir(parents=True, exist_ok=True)
save_count = 0

# mapeo de accion -> tecla
action_map = {
    "derecha": "up",
    "izquierda": "down",
}
# si swap-actions activo invierte el map
if args.swap_actions:
    action_map = {"derecha":"down", "izquierda":"up"}
    print("Actions swapped temporally (derecha->down, izquierda->up)")

def save_landmarks_csv(lm, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark_index","x","y","z"])
        for i, l in enumerate(lm.landmark):
            writer.writerow([i, float(l.x), float(l.y), float(l.z)])

pTime = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        img = frame.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            # preprocess identical al entrenamiento: centra en wrist
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
            coords = coords - coords[0]
            datos = coords.flatten().reshape(1, -1)

            if scaler is not None:
                try:
                    datos_s = scaler.transform(datos)
                except Exception as e:
                    print("Error aplicando scaler:", e)
                    datos_s = datos
            else:
                datos_s = datos

            # predicción
            try:
                if has_proba:
                    probs = clf.predict_proba(datos_s)[0]
                    classes = clf.classes_
                    idx = np.argmax(probs)
                    pred = classes[idx]
                    prob = probs[idx]
                    # Mostrar top2
                    top2_idx = probs.argsort()[-2:][::-1]
                    top2 = [(classes[i], float(probs[i])) for i in top2_idx]
                else:
                    pred = clf.predict(datos_s)[0]
                    prob = None
                    top2 = [(pred, None)]
            except Exception as e:
                print("Error en predicción:", e)
                pred = "ERROR"
                prob = None
                top2 = []

            # imprimir y mostrar en pantalla
            text = f"Pred: {pred}"
            if prob is not None:
                text += f" ({prob:.2f})"
            cv2.putText(img, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
            # top2
            y0=80
            for cls, p in top2:
                if p is None:
                    cv2.putText(img, f"{cls}", (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0),1)
                else:
                    cv2.putText(img, f"{cls}: {p:.2f}", (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0),1)
                y0 += 22

            # acción (solo si confianza razonable)
            if prob is None or prob >= 0.55:
                act = action_map.get(str(pred).strip().lower(), None)
                if act:
                    # no enviar pulsación constantemente: solo cuando prob pasa de <umbral a >=umbral
                    # por simplicidad, dejamos que el usuario decida; comentar la linea siguiente para desactivar:
                    pyautogui.press(act)
                    cv2.putText(img, f"Accion: {act}", (10, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            else:
                cv2.putText(img, "Confianza baja - sin accion", (10, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255),2)

        else:
            cv2.putText(img, "Mano NO detectada", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)

        # FPS
        cTime = time.time()
        fps = 1/(cTime-pTime) if pTime>0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),1)

        cv2.imshow("Debug Gestos", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('s') and res.multi_hand_landmarks:
            # guardar ejemplo (imagen + csv) para reentreno
            ts = int(time.time()*1000)
            img_name = f"puño_sample_{ts}.jpg"
            csv_name = f"puño_sample_{ts}.csv"
            cv2.imwrite(str(SAVE_DIR / img_name), frame)
            save_landmarks_csv(res.multi_hand_landmarks[0], str(SAVE_LM_DIR / csv_name))
            save_count += 1
            print("Guardado ejemplo:", img_name, csv_name)

except Exception as e:
    print("Error inesperado:", e)
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()