import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import time

# ---------------- CONFIGURACIÓN ----------------
ESP32_URL = "http://10.187.208.231:81/stream"
TIMEOUT_CONEXION = 5  # segundos
RESOLUCION = (640, 480)

# ---------------- CONEXIÓN A ESP32-CAM ----------------
print("🔄 Conectando a ESP32-CAM...")
cap = cv2.VideoCapture(ESP32_URL)
inicio = time.time()
conectado = False

while time.time() - inicio < TIMEOUT_CONEXION:
    ret, _ = cap.read()
    if ret:
        conectado = True
        break
    time.sleep(0.5)

if not conectado:
    print(f"⚠ No se pudo conectar a la ESP32-CAM en {TIMEOUT_CONEXION} segundos. Cambiando a cámara local...")
    cap.release()
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo acceder a ninguna cámara.")
    exit()

cap.set(3, RESOLUCION[0])
cap.set(4, RESOLUCION[1])

# ---------------- CARGAR MODELO DE GESTOS ----------------
try:
    with open("modelo_gestos.pkl", "rb") as f:
        modelo = pickle.load(f)
    print("✅ Modelo de gestos cargado correctamente.")
except Exception as e:
    print(f"❌ Error cargando modelo_gestos.pkl: {e}")
    exit()

# ---------------- CONFIGURAR MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=0
)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- FUNCIÓN DE PROCESAMIENTO ----------------
def preprocess_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    coords -= coords[0]  # normalizar respecto al primer landmark (muñeca)
    return coords.flatten().reshape(1, -1)

# ---------------- LOOP PRINCIPAL ----------------
pTime = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ No se reciben frames. Verifica conexión con la cámara.")
            time.sleep(1)
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # -------- DETECCIÓN DE PERSONA (nariz) --------
        results_pose = pose.process(image_rgb)
        if results_pose.pose_landmarks:
            centro = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            x_centro = int(centro.x * w)
            y_centro = int(centro.y * h)
            cv2.circle(frame, (x_centro, y_centro), 10, (0, 255, 255), -1)
            cv2.putText(frame, 'Persona Detectada', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, 'Persona NO detectada', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # -------- DETECCIÓN DE MANO Y GESTO --------
        results_hands = hands.process(image_rgb)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                datos = preprocess_landmarks(hand_landmarks)
                if hasattr(modelo, "n_features_in_") and datos.shape[1] != modelo.n_features_in_:
                    print("⚠ Landmarks insuficientes para predicción, omitiendo frame.")
                    continue

                try:
                    prediccion = modelo.predict(datos)[0]
                    if prediccion == "Derecha":
                        pyautogui.press('right')
                        cv2.putText(frame, '👉 Gesto Derecha', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    elif prediccion == "Izquierda":
                        pyautogui.press('left')
                        cv2.putText(frame, '👈 Gesto Izquierda', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                except Exception as e:
                    print("❌ Error en la predicción:", e)
        else:
            cv2.putText(frame, 'Mano NO detectada', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # -------- FPS --------
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # -------- MOSTRAR --------
        cv2.imshow("Control por Gestos + Detección Persona", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

except Exception as e:
    import traceback
    print("❌ Error inesperado:", e)
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
