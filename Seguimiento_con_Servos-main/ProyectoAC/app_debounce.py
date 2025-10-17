#!/usr/bin/env python3
"""
app_debounce.py (busca modelo en rutas razonables)

Busca modelo_gestos.pkl en, por este orden:
 - ruta pasada con --model (absoluta o relativa)
 - carpeta del script (same folder as this file)
 - carpeta del script/models, /dataset, /models
 - directorio de trabajo actual (cwd)

Si lo encuentra lo carga; si no, imprime la lista de rutas buscadas.
"""
import argparse
import pickle
import time
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

def find_model_path(candidate):
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = []
    if candidate:
        candidates.append(Path(candidate))
    # Look near script
    candidates += [
        script_dir / "modelo_gestos.pkl",
        script_dir / "models" / "modelo_gestos.pkl",
        script_dir / "dataset" / "modelo_gestos.pkl",
        script_dir / "ProyectoAC" / "modelo_gestos.pkl",
        cwd / "modelo_gestos.pkl"
    ]
    # deduplicate while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        p = p.resolve() if p.exists() else p
        if str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    for p in uniq:
        try:
            if p.exists():
                return p
        except Exception:
            # skip invalid path objects
            continue
    return uniq  # return list of attempted paths if none found

def load_payload(path):
    with open(path, "rb") as f:
        p = pickle.load(f)
    if isinstance(p, dict) and "model" in p:
        return p["model"], p.get("scaler", None), p.get("classes", None)
    else:
        return p, None, getattr(p, "classes_", None)

# (el resto del script queda igual que tu app_debounce original, pero usando find_model_path)
def parse_camera_arg(s):
    if s == "0":
        return 0
    try:
        return int(s)
    except Exception:
        return s

def map_action(label, swap, classes_list):
    lab = str(label).strip()
    if swap:
        if lab.lower() == "derecha":
            return "down"
        if lab.lower() in ("puno", "izquierda", "cerrado", "puno"):
            return "up"
    else:
        if lab.lower() == "derecha":
            return "up"
        if lab.lower() in ("puno", "izquierda", "cerrado"):
            return "down"
    return None

def preprocess_hand_landmarks(hand_landmarks, use_z=True):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z] if use_z else [lm.x, lm.y])
    arr = np.array(coords, dtype=np.float32)
    arr = arr - arr[0]
    vec = arr.flatten().reshape(1, -1)
    return vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="modelo_gestos.pkl")
    parser.add_argument("--camera", "-c", default="http://10.187.208.231:81/stream")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--consec", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=0.8)
    parser.add_argument("--swap-actions", action="store_true")
    parser.add_argument("--no-press", action="store_true", help="No enviar pulsaciones (solo debug)")
    args = parser.parse_args()

    model_candidate = args.model
    found = find_model_path(model_candidate)
    if isinstance(found, list):
        print("No se encontró modelo_gestos.pkl. Rutas intentadas:")
        for p in found:
            print("  -", p)
        print("Coloca modelo_gestos.pkl en alguna de esas rutas o pásalo con --model RUTA")
        sys.exit(1)
    model_path = Path(found)
    print("Cargando modelo desde:", model_path)
    clf, scaler, classes = load_payload(model_path)
    print("Modelo cargado. classes (payload):", classes)
    n_feat = getattr(clf, "n_features_in_", None)
    print("Modelo n_features_in_:", n_feat)

    use_z = True
    if n_feat == 42:
        use_z = False
    elif n_feat == 63:
        use_z = True

    cam_src = parse_camera_arg(args.camera)
    cap = cv2.VideoCapture(cam_src)
    time.sleep(0.3)
    if not cap.isOpened():
        print("No se pudo abrir cámara:", cam_src, "intentando 0")
        cap.release()
        cap = cv2.VideoCapture(0)
        time.sleep(0.3)
        if not cap.isOpened():
            print("No se pudo abrir ninguna cámara. Salir.")
            sys.exit(1)
    cap.set(3, 640)
    cap.set(4, 480)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.65, min_tracking_confidence=0.6)

    last_label = None
    consec_count = 0
    last_action_time = 0

    print(f"Inicio app_debounce (threshold={args.threshold}, consec={args.consec}, cooldown={args.cooldown}, swap={args.swap_actions})")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            img = frame.copy()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            display = img.copy()

            if results and results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(display, hand_lm, mp_hands.HAND_CONNECTIONS)

                datos = preprocess_hand_landmarks(hand_lm, use_z=use_z)
                datos_a = datos
                if scaler is not None:
                    try:
                        datos_a = scaler.transform(datos)
                    except Exception as e:
                        print("Warning: fallo aplicando scaler:", e)
                        datos_a = datos

                prob = None
                pred_label = None
                try:
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(datos_a)[0]
                        idx = int(np.argmax(probs))
                        prob = float(probs[idx])
                        if classes is not None:
                            pred_label = classes[idx]
                        else:
                            pred_label = str(clf.classes_[idx])
                    else:
                        pred = clf.predict(datos_a)[0]
                        pred_label = str(pred)
                        prob = None
                except Exception as e:
                    print("Error predicción:", e)
                    pred_label = "ERROR"
                    prob = None

                display_text = f"Pred: {pred_label}"
                if prob is not None:
                    display_text += f" ({prob:.2f})"
                cv2.putText(display, display_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                if pred_label == last_label:
                    consec_count += 1
                else:
                    last_label = pred_label
                    consec_count = 1

                cv2.putText(display, f"Consec: {consec_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)

                now = time.time()
                ok_prob = True if prob is None else prob >= args.threshold

                if ok_prob and consec_count >= args.consec and (now - last_action_time) >= args.cooldown:
                    action = map_action(pred_label, args.swap_actions, classes)
                    if action and not args.no_press:
                        try:
                            pyautogui.press(action)
                            print(f"[ACTION] {pred_label} -> {action} (prob={prob})")
                        except Exception as e:
                            print("Error enviando tecla:", e)
                    else:
                        print(f"[INFO] Pred:{pred_label} no mapeada a acción or no-press set. (prob={prob})")
                    last_action_time = now
                    consec_count = 0

            else:
                cv2.putText(display, "Mano NO detectada", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                last_label = None
                consec_count = 0

            cv2.imshow("Gestos - debounce", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()