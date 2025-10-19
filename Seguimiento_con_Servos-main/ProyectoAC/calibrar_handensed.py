#!/usr/bin/env python3
"""
calibrar_handedness.py

Muestra la cámara y permite calibrar la inversión de handedness de MediaPipe.
Instrucciones:
 - Muestra tu mano DERECHA en la cámara y presiona 'r'.
 - Muestra tu mano IZQUIERDA en la cámara y presiona 'l' (opcional, refuerzo).
El script imprimirá si MediaPipe reportó Left/Right y te dirá si conviene usar --flip-handedness.
Salida recomendada: "USE --flip-handedness" o "NO FLIP NEEDED".
"""
import cv2
import mediapipe as mp
import time

def main():
    cam = 0  # cambia a URL si usas ESP32, p.ej. "http://.../stream"
    cap = cv2.VideoCapture(cam)
    time.sleep(0.2)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Prueba con --camera o revisa la conexión.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("Calibración handedness:")
    print(" - Muestra TU MANO DERECHA y presiona 'r' (right).")
    print(" - Opcional: muestra la mano IZQUIERDA y presiona 'l' (left) para confirmar.")
    print(" - Presiona ESC o 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            img = cv2.flip(frame, 1)  # opcional: espejo si tu preview es espejo; si no, comentar
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            if res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, "Press 'r' for RIGHT, 'l' for LEFT, ESC to exit", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Calibrar Handedness (press r/l)", img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break
            if k == ord('r') or k == ord('l'):
                desired = 'Right' if k == ord('r') else 'Left'
                if res and res.multi_handedness:
                    mp_label = res.multi_handedness[0].classification[0].label
                    mp_score = res.multi_handedness[0].classification[0].score
                    print(f"You pressed: {desired}. MediaPipe reports: {mp_label} (score={mp_score:.2f})")
                    if mp_label.lower() != desired.lower():
                        print("Recommendation: USE --flip-handedness (MediaPipe is inverted vs your input).")
                    else:
                        print("Recommendation: NO flip needed.")
                else:
                    print("No handedness detected at this frame. Try again ensuring the hand is visible.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == '__main__':
    main()