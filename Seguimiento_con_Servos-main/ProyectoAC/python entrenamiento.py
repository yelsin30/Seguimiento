#!/usr/bin/env python3
"""
entrenamiento.py

Versión robusta que:
 - Carga X_landmarks.npy e y_labels.npy (usa allow_pickle=True para y).
 - Asegura que y sea un array de strings.
 - Usa LabelEncoder para convertir etiquetas a valores numéricos internamente.
 - Escala con StandardScaler y entrena un SVC (probability=True).
 - Guarda payload = {"model": clf, "scaler": scaler, "classes": list_of_labels}
 - Imprime un resumen de datos y el classification_report.
"""
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import sys

BASE = Path(__file__).resolve().parent
Xf = BASE / "X_landmarks.npy"
yf = BASE / "y_labels.npy"
OUT_MODEL = BASE / "modelo_gestos.pkl"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Comprobación de archivos
if not Xf.exists() or not yf.exists():
    print(f"[ERROR] Faltan X_landmarks.npy y/o y_labels.npy en {BASE}")
    sys.exit(1)

# Cargar X
X = np.load(str(Xf))
print(f"[INFO] X cargado: shape = {X.shape}, dtype = {X.dtype}")

# Cargar y con allow_pickle=True para evitar ValueError de object arrays
try:
    y = np.load(str(yf), allow_pickle=True)
except Exception as e:
    print(f"[ERROR] al cargar y_labels.npy: {e}")
    sys.exit(1)

# Asegurar formato de y: array 1D de strings
try:
    # si viene como array de bytes, convertir a str
    y = np.array([str(v) for v in y], dtype=object)
except Exception:
    y = np.ravel(y).astype(str)

print(f"[INFO] y cargado: N={y.shape[0]}, ejemplo etiquetas: {np.unique(y)[:10]}")

if X.shape[0] != y.shape[0]:
    print(f"[ERROR] Número de muestras en X ({X.shape[0]}) y y ({y.shape[0]}) no coincide.")
    sys.exit(1)

# Codificar etiquetas con LabelEncoder (mantener lista de clases texto)
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = list(le.classes_)
print(f"[INFO] Etiquetas codificadas. Clases ({len(classes)}): {classes}")

# Dividir dataset
if len(classes) < 2:
    print("[ERROR] Necesitas al menos 2 clases para entrenar un clasificador.")
    sys.exit(1)

X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=TEST_SIZE,
                                                  random_state=RANDOM_SEED, stratify=y_enc)

# Escalar
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# Entrenar SVC
clf = SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=RANDOM_SEED)
print("[INFO] Entrenando SVC...")
clf.fit(X_train_s, y_train)

# Evaluación
y_pred = clf.predict(X_val_s)
print("\n=== Evaluación en validación ===")
print(classification_report(y_val, y_pred, zero_division=0, target_names=classes))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))

# Guardar payload
payload = {
    "model": clf,
    "scaler": scaler,
    "classes": classes  # lista de etiquetas en texto, en el mismo orden que LabelEncoder
}
try:
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(payload, f)
    print(f"[OK] Modelo guardado en: {OUT_MODEL}")
except Exception as e:
    print(f"[ERROR] al guardar el modelo: {e}")
    sys.exit(1)