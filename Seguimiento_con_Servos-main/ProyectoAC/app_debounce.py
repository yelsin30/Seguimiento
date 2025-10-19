#!/usr/bin/env python3
"""
app_solo_cls.py
Versión simplificada: solo usa clasificador de gestos (sin handedness de MediaPipe).
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import time
import argparse
from pathlib import Path
import sys

# ---------------- ARGUMENTOS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", default="modelo_gestos.pkl", help="Ruta al modelo de gestos")
parser.add_argument("--camera", "-c", default="http://10.187.208.231:81/stream", help="Fuente de la cámara")
parser.add_argument("--threshold", type=float, default=0.6, help="Probabilidad mínima para aceptar predicción")
parser.add_argument("--consec", type=int, default=3, help="Frames consecutivos requeridos")
parser.add_argument("--cooldown", type=float, default=0.8, help="Cooldown entre acciones")
parser.add_argument("--next-key", default="right", help="Tecla para avanzar")
parser.add_argument("--prev-key", default="left", help="Tecla para retroceder")
parser.add_argument("--mirror", action="store_true", help="Voltear horizontalmente el frame")
parser.add_argument("--no-press", action="store_true", help="No enviar teclas (modo debug)")
args = parser.parse_args()

# ---------------- FUNCIONES AUXILIARES ----------------
def find_model_path(candidate):
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = []
    if candidate:
        candidates.append(Path(candidate))
    candidates += [
        script_dir / "modelo_gestos.pkl",
        script_dir / "models" / "modelo_gestos.pkl",
        script_dir / "dataset" / "modelo_gestos.pkl",
        cwd / "modelo_gestos.pkl"
    ]
    for p in candidates:
        if p.exists():
            return p
    print("❌ No se encontró modelo_gestos.pkl.")
    sys.exit(1)

def load_payload(path):
    with open(path, "rb") as f:
        p = pickle.load(f)
    if isinstance(p, dict) and "model" in p:
        return p["model"], p.get("scaler", None), p.get("classes", None)
    else:
        return p, None, getattr(p, "classes_", None)

def preprocess_hand_landmarks(hand_landmarks, use_z=True):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z] if use_z else [lm.x, lm.y])
    arr = np.array(coords, dtype=np.float32)
    arr = arr - arr[0]
    vec = arr.flatten().reshape(1, -1)
    return vec

def map_action_key_from_label_text(label_text, next_key, prev_key):
    if label_text is None:
        return None
    lab = str(label_text).strip().lower()
    if lab in {"derecha", "right"}:
        return next_key
    if lab in {"izquierda", "puno", "cerrado", "left"}:
        return prev_key
    return None

# ---------------- CARGAR MODELO ----------------
model_path = find_model_path(args.model)
print("📂 Cargando modelo desde:", model_path)
clf, scaler, classes = load_payload(model_path)
print("✅ Modelo cargado. Clases:", classes)

n_feat = getattr(clf, "n_features_in_", None)
use_z = (n_feat == 63)

# ---------------- CÁMARA ----------------
cam_src = args.camera
if cam_src == "0":
    cam_src = 0
cap = cv2.VideoCapture(cam_src)
time.sleep(0.3)
if not cap.isOpened():
    print("⚠ No se pudo abrir cámara. Intentando cámara local...")
    cap.release()
    cap = cv2.VideoCapture(0)
    time.sleep(0.3)
    if not cap.isOpened():
        print("❌ No se pudo abrir ninguna cámara.")
        sys.exit(1)

cap.set(3, 640)
cap.set(4, 480)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.65, min_tracking_confidence=0.6)

last_label = None
consec_count = 0
last_action_time = 0

print(f"🚀 Inicio app_solo_cls (threshold={args.threshold}, consec={args.consec}, cooldown={args.cooldown})")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if args.mirror:
            frame = cv2.flip(frame, 1)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        display = frame.copy()

        classifier_label = None
        classifier_prob = None

        if results and results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Predicción con clasificador
            datos = preprocess_hand_landmarks(hand_lm, use_z=use_z)
            if scaler is not None:
                try:
                    datos = scaler.transform(datos)
                except Exception as e:
                    print("Warning: fallo aplicando scaler:", e)
            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(datos)[0]
                    idx = int(np.argmax(probs))
                    classifier_prob = float(probs[idx])
                    if classes is not None:
                        classifier_label = classes[idx]
                    else:
                        classifier_label = str(clf.classes_[idx])
                else:
                    pred = clf.predict(datos)[0]
                    classifier_label = str(pred)
                    classifier_prob = 1.0
            except Exception as e:
                print("❌ Error predicción:", e)

            # Mostrar info en pantalla
            if classifier_label is not None:
                cv2.putText(display, f"Cls: {classifier_label} ({classifier_prob:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Determinar acción
            final_action = None
            if classifier_label is not None and classifier_prob >= args.threshold:
                final_action = map_action_key_from_label_text(
                    classifier_label, args.next_key, args.prev_key
                )

            # Debounce
            if final_action is None:
                last_label = None
                consec_count = 0
            else:
                if last_label == final_action:
                    consec_count += 1
                else:
                    last_label = final_action
                    consec_count = 1

                cv2.putText(display, f"Consec: {consec_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

                now = time.time()
                if consec_count >= args.consec and (now - last_action_time) >= args.cooldown:
                    if final_action and not args.no_press:
                        pyautogui.press(final_action)
                        print(f"[ACTION] {final_action} (cls={classifier_label}, prob={classifier_prob})")
                    last_action_time = now
                    consec_count = 0

        else:
            cv2.putText(display, "Mano NO detectada", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            last_label = None
            consec_count = 0

        cv2.imshow("Gestos - SOLO CLS", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
