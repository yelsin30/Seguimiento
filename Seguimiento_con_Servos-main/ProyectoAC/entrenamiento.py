import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import pickle

landmarks_dir = "dataset/landmarks"
X, y = [], []
for clase in ["Derecha", "Izquierda"]:
    clase_dir = os.path.join(landmarks_dir, clase)
    for archivo in os.listdir(clase_dir):
        if archivo.endswith(".csv"):
            df = pd.read_csv(os.path.join(clase_dir, archivo))
            datos = df[["x", "y", "z"]].values.flatten()
            # ✅ Verificar que tenga el tamaño esperado
            if len(datos) == 63:   # 21 puntos * 3 coordenadas
                X.append(datos)
                y.append(clase)
            else:
                print(f"Archivo {archivo} descartado: tamaño {len(datos)}")

X = np.array(X)
y = np.array(y)

modelo = SVC()
modelo.fit(X, y)
with open("modelo_gestos.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("✅ Modelo entrenado y guardado como modelo_gestos.pkl")
