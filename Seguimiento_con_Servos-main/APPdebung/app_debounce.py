#!/usr/bin/env python3
"""
app_solo_cls_pose.py
Versión final optimizada: usa clasificador de gestos y agrega seguimiento de persona (MediaPipe Pose + control de servos por WiFi).
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # asegurar import local

from control import mover_servo  # ✅ versión por WiFi

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import time
import argparse
from pathlib import Path

# ---------------- ARGUMENTOS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", default="modelo_gestos.pkl", help="Ruta al modelo de gestos")

# 🔴 AQUÍ ESTABA EL PROBLEMA: antes era http://10.18.122.231:81/stream
parser.add_argument(
    "--camera", "-c",
    default="http://10.18.122.231/stream",
    help="Fuente de la cámara (ESP32-CAM)"
)

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
    for p in [Path(candidate),
              script_dir / "modelo_gestos.pkl",
              script_dir / "models" / "modelo_gestos.pkl",
              script_dir / "dataset" / "modelo_gestos.pkl",
              cwd / "modelo_gestos.pkl"]:
        if p.exists():
            return p
    print("❌ No se encontró modelo_gestos.pkl.")
    sys.exit(1)

def load_payload(path):
    with open(path, "rb") as f:
        p = pickle.load(f)
    if isinstance(p, dict) and "model" in p:
        return p["model"], p.get("scaler"), p.get("classes")
    else:
        return p, None, getattr(p, "classes_", None)

def preprocess_hand_landmarks(hand_landmarks, use_z=True):
    coords = np.array([[lm.x, lm.y, lm.z] if use_z else [lm.x, lm.y]
                       for lm in hand_landmarks.landmark], dtype=np.float32)
    coords -= coords[0]  # normaliza respecto al primer punto
    return coords.flatten().reshape(1, -1)

def map_action_key_from_label_text(label_text, next_key, prev_key):
    if not label_text:
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
# Si pones -c 0, usa webcam local; si es URL, usa stream de la ESP32
cap = cv2.VideoCapture(0 if cam_src == "0" else cam_src)
time.sleep(0.3)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    sys.exit(1)
cap.set(3, 640)
cap.set(4, 480)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.65, min_tracking_confidence=0.6)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------- VARIABLES ----------------
last_label = None
consec_count = 0
last_action_time = 0

pan, tilt = 90, 90
last_move_time = 0
move_delay = 0.25  # segundos entre actualizaciones

print(f"🚀 Inicio app_solo_cls_pose (threshold={args.threshold}, consec={args.consec}, cooldown={args.cooldown})")

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
        results_pose = pose.process(img_rgb)
        display = frame.copy()

        # === SEGUIMIENTO DE PERSONA ===
        if results_pose.pose_landmarks:
            h, w, _ = frame.shape
            nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            x_nose, y_nose = int(nose.x * w), int(nose.y * h)
            cv2.circle(display, (x_nose, y_nose), 10, (0, 255, 255), -1)
            cv2.putText(display, "Persona detectada", (10, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Control de servos
            now = time.time()
            if now - last_move_time >= move_delay:
                cx, cy = x_nose, y_nose
                frame_h, frame_w = frame.shape[:2]
                pan = int(np.clip(90 + (cx - frame_w // 2) // 10, 0, 180))
                tilt = int(np.clip(90 - (cy - frame_h // 2) // 10, 0, 180))
                mover_servo(pan=pan, tilt=tilt)
                last_move_time = now
        else:
            cv2.putText(display, "Persona NO detectada", (10, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # === DETECCIÓN DE MANO Y GESTOS ===
        classifier_label, classifier_prob = None, None
        if results and results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display, hand_lm, mp_hands.HAND_CONNECTIONS)

            datos = preprocess_hand_landmarks(hand_lm, use_z)
            if scaler is not None:
                try:
                    datos = scaler.transform(datos)
                except Exception:
                    pass

            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(datos)[0]
                    idx = int(np.argmax(probs))
                    classifier_prob = float(probs[idx])
                    classifier_label = classes[idx] if classes is not None else str(clf.classes_[idx])
                else:
                    classifier_label = str(clf.predict(datos)[0])
                    classifier_prob = 1.0
            except Exception as e:
                print("❌ Error predicción:", e)

            if classifier_label is not None:
                cv2.putText(display, f"Cls: {classifier_label} ({classifier_prob:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Acciones con debounce
            final_action = None
            if classifier_label and classifier_prob >= args.threshold:
                final_action = map_action_key_from_label_text(classifier_label, args.next_key, args.prev_key)

            if final_action:
                if last_label == final_action:
                    consec_count += 1
                else:
                    last_label = final_action
                    consec_count = 1

                now = time.time()
                if consec_count >= args.consec and (now - last_action_time) >= args.cooldown:
                    if not args.no_press:
                        pyautogui.press(final_action)
                        print(f"[ACTION] {final_action} (cls={classifier_label}, prob={classifier_prob})")
                    last_action_time = now
                    consec_count = 0
            else:
                last_label = None
                consec_count = 0
        else:
            cv2.putText(display, "Mano NO detectada", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            last_label = None
            consec_count = 0

        # Mostrar
        cv2.imshow("Gestos + Seguimiento Persona (WiFi)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrumpido por el usuario.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
