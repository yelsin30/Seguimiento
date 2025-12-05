#!/usr/bin/env python3
"""
app_solo_cls_pose.py
Versión optimizada: clasificador de gestos + seguimiento de CUERPO COMPLETO (MediaPipe Pose + control de servos por WiFi).
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
parser.add_argument(
    "--camera", "-c",
    default="http://10.18.122.142/stream",
    help="Fuente de la cámara (ESP32-CAM)"
)
parser.add_argument("--threshold", type=float, default=0.7, help="Probabilidad mínima para aceptar predicción")
parser.add_argument("--consec", type=int, default=5, help="Frames consecutivos requeridos")
parser.add_argument("--cooldown", type=float, default=1.5, help="Cooldown entre acciones")
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
    coords -= coords[0]
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

def calculate_body_center(landmarks, w, h):
    """
    Calcula el centro del cuerpo usando el promedio de hombros y caderas.
    Esto proporciona un seguimiento más estable que solo la nariz.
    """
    try:
        # Obtener puntos clave del torso
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calcular centro del torso (promedio de 4 puntos)
        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        # Convertir a coordenadas de píxeles
        cx = int(center_x * w)
        cy = int(center_y * h)
        
        # Verificar visibilidad (opcional: solo usar si los puntos son visibles)
        avg_visibility = (left_shoulder.visibility + right_shoulder.visibility + 
                         left_hip.visibility + right_hip.visibility) / 4
        
        return cx, cy, avg_visibility
    except Exception as e:
        print(f"⚠️ Error calculando centro del cuerpo: {e}")
        return None, None, 0.0

# ---------------- CARGAR MODELO ----------------
model_path = find_model_path(args.model)
print("📂 Cargando modelo desde:", model_path)
clf, scaler, classes = load_payload(model_path)
print("✅ Modelo cargado. Clases:", classes)

n_feat = getattr(clf, "n_features_in_", None)
use_z = (n_feat == 63)

# ---------------- CÁMARA ----------------
cam_src = args.camera
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
move_delay = 0.15  # ⚡ Más rápido para seguimiento fluido (antes 0.25)

# Suavizado de movimiento (filtro simple)
smoothing_factor = 0.7  # 0 = sin suavizado, 1 = máximo suavizado
last_pan, last_tilt = 90, 90

print(f"🚀 Inicio app_solo_cls_pose (threshold={args.threshold}, consec={args.consec}, cooldown={args.cooldown})")
print(f"📹 Seguimiento por CUERPO COMPLETO activado")

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

        # === SEGUIMIENTO DE CUERPO COMPLETO ===
        if results_pose.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = results_pose.pose_landmarks.landmark
            
            # Dibujar esqueleto completo
            mp_draw.draw_landmarks(
                display, 
                results_pose.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
            )
            
            # Calcular centro del cuerpo
            cx, cy, visibility = calculate_body_center(landmarks, w, h)
            
            if cx is not None and cy is not None and visibility > 0.5:
                # Dibujar punto de seguimiento
                cv2.circle(display, (cx, cy), 15, (0, 255, 0), -1)
                cv2.circle(display, (cx, cy), 20, (255, 255, 255), 2)
                cv2.putText(display, "CENTRO", (cx - 30, cy - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostrar coordenadas del centro
                cv2.putText(display, f"Centro: ({cx}, {cy})", (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Control de servos con suavizado
                now = time.time()
                if now - last_move_time >= move_delay:
                    frame_h, frame_w = frame.shape[:2]
                    
                    # Calcular nuevas posiciones (invertidas para seguimiento natural)
                    target_pan = 90 + (cx - frame_w // 2) // 8  # Ajuste más sensible
                    target_tilt = 90 - (cy - frame_h // 2) // 8
                    
                    # Aplicar suavizado exponencial
                    pan = int(smoothing_factor * last_pan + (1 - smoothing_factor) * target_pan)
                    tilt = int(smoothing_factor * last_tilt + (1 - smoothing_factor) * target_tilt)
                    
                    # Limitar rangos
                    pan = np.clip(pan, 0, 180)
                    tilt = np.clip(tilt, 0, 180)
                    
                    # Enviar comandos solo si hay cambio significativo
                    if abs(pan - last_pan) > 2 or abs(tilt - last_tilt) > 2:
                        mover_servo(pan=pan, tilt=tilt)
                        last_pan, last_tilt = pan, tilt
                    
                    last_move_time = now
                
                # Indicador visual de seguimiento activo
                cv2.putText(display, "SEGUIMIENTO ACTIVO", (10, 430),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"Pan: {pan}° Tilt: {tilt}°", (10, 490),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(display, "Cuerpo parcialmente visible", (10, 430),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        else:
            cv2.putText(display, "CUERPO NO DETECTADO", (10, 430),
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
                cv2.putText(display, f"Gesto: {classifier_label} ({classifier_prob:.2f})",
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
                        print(f"[ACTION] {final_action} (cls={classifier_label}, prob={classifier_prob:.2f})")
                    last_action_time = now
                    consec_count = 0
            else:
                last_label = None
                consec_count = 0
        else:
            cv2.putText(display, "Mano NO detectada", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            last_label = None
            consec_count = 0

        # Mostrar modo de seguimiento
        frame_h = display.shape[0]
        cv2.putText(display, f"Modo: Seguimiento Cuerpo Completo", (10, frame_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Gestos + Seguimiento de Cuerpo (WiFi)", display)
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
    print("✅ Programa finalizado correctamente")