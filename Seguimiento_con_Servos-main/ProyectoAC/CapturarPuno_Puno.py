#!/usr/bin/env python3
"""
CapturarPuno_Puno.py

Versión lista para guardar en dataset/images/Puno (sin interacción con 'ñ').
Presiona 'p' para guardar imagen (+ CSV si usas --save-landmarks).
"""
import argparse
from pathlib import Path
import cv2
import mediapipe as mp
import time
import csv
import sys

def ensure_dir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo crear la carpeta {p}: {e}")
        return False

def timestamp_name(prefix: str):
    return f"{prefix}_{int(time.time()*1000)}"

def save_image(img, path: Path):
    try:
        ok = cv2.imwrite(str(path), img)
        if not ok:
            raise IOError("cv2.imwrite returned False")
        return True, None
    except Exception as e:
        return False, str(e)

def save_landmarks_csv(hand_landmarks, path: Path):
    try:
        with open(str(path), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["landmark_index", "x", "y", "z"])
            for i, lm in enumerate(hand_landmarks.landmark):
                writer.writerow([i, float(lm.x), float(lm.y), float(lm.z)])
        return True, None
    except Exception as e:
        return False, str(e)

def parse_camera(src):
    if src == "0":
        return 0
    try:
        ival = int(src)
        return ival
    except Exception:
        return src

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", default="http://10.187.208.231:81/stream")
    parser.add_argument("--label", "-l", default="Puno")  # default changed to "Puno"
    parser.add_argument("--output-base", "-o", default="dataset")
    parser.add_argument("--save-landmarks", action="store_true")
    parser.add_argument("--max-save", type=int, default=0)
    args = parser.parse_args()

    base = Path(args.output_base).resolve()
    label = args.label
    images_dir = base / "images" / label
    landmarks_dir = base / "landmarks" / label

    print(f"[INFO] Base de salida: {base.resolve()}")
    print(f"[INFO] Carpeta imágenes: {images_dir.resolve()}")
    if args.save_landmarks:
        print(f"[INFO] Carpeta landmarks: {landmarks_dir.resolve()}")

    if not ensure_dir(images_dir):
        print("[FATAL] No se pudo crear la carpeta de imágenes. Salir.")
        sys.exit(1)
    if args.save_landmarks and not ensure_dir(landmarks_dir):
        print("[FATAL] No se pudo crear la carpeta de landmarks. Salir.")
        sys.exit(1)

    cam_source = parse_camera(args.camera)
    cap = cv2.VideoCapture(cam_source)
    time.sleep(0.5)
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(args.camera))
            time.sleep(0.5)
        except Exception:
            pass

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara con source={args.camera}.")
        sys.exit(1)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    show_landmarks = True
    saved_count = 0
    prefix = label.replace(" ", "_")

    print("Instrucciones: Presiona 'p' para guardar (imagen + landmarks), 'v' toggle landmarks, 'q' o ESC salir.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            display = frame.copy()
            detected = False
            hand_lm = None
            if results and results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                detected = True
                if show_landmarks:
                    mp_drawing.draw_landmarks(display, hand_lm, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                              mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))

            txt = f"Label: {label}  Saved: {saved_count}"
            cv2.putText(display, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if not detected:
                cv2.putText(display, "No hand detected", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Capturar Puño (Puno) - presiona 'p' para guardar", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                print("[INFO] Salir solicitado por usuario.")
                break
            elif key == ord('v'):
                show_landmarks = not show_landmarks
            elif key == ord('p'):
                if not detected:
                    print("[WARN] No se detectó mano: no se guarda la imagen.")
                    continue
                name = timestamp_name(prefix)
                img_path = images_dir / (name + ".jpg")
                ok_img, err_img = save_image(frame, img_path)
                if ok_img:
                    print(f"[SAVED] Imagen guardada en: {img_path.resolve()}")
                else:
                    print(f"[ERROR] No se pudo guardar imagen en {img_path.resolve()}: {err_img}")

                if args.save_landmarks and hand_lm is not None:
                    csv_path = landmarks_dir / (name + ".csv")
                    ok_csv, err_csv = save_landmarks_csv(hand_lm, csv_path)
                    if ok_csv:
                        print(f"[SAVED] Landmarks guardados en: {csv_path.resolve()}")
                    else:
                        print(f"[ERROR] No se pudo guardar CSV en {csv_path.resolve()}: {err_csv}")

                saved_count += 1
                if args.max_save > 0 and saved_count >= args.max_save:
                    print(f"[INFO] Alcanzado max-save={args.max_save}. Saliendo.")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print(f"[DONE] Total guardadas: {saved_count}")
        print(f"[DONE] Revisa carpeta: {images_dir.resolve()}")
        if args.save_landmarks:
            print(f"[DONE] Revisa carpeta landmarks: {landmarks_dir.resolve()}")

if __name__ == "__main__":
    main()