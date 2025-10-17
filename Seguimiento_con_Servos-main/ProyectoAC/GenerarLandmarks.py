import cv2
import mediapipe as mp
import os
import csv

# Carpetas de entrada y salida
carpetas = ["Derecha", "Izquierda"]
images_dir = "dataset/images"
landmarks_dir = "dataset/landmarks"

mp_hands = mp.solutions.hands

for clase in carpetas:
    img_folder = os.path.join(images_dir, clase)
    lmk_folder = os.path.join(landmarks_dir, clase)
    os.makedirs(lmk_folder, exist_ok=True)

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    for img_file in os.listdir(img_folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(img_folder, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"No se pudo leer {img_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            lmk_path = os.path.join(lmk_folder, img_file.replace('.jpg', '.csv').replace('.jpeg', '.csv').replace('.png', '.csv'))
            with open(lmk_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["landmark_index", "x", "y", "z"])
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        writer.writerow([idx, lm.x, lm.y, lm.z])
                else:
                    print(f"No se detectó mano en {img_file}, landmark vacío.")
    hands.close()

print("✅ Landmarks generados para todas las imágenes")