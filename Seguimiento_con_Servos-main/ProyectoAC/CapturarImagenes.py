import os
import numpy as np
import pandas as pd

landmarks_dir = "dataset/landmarks"
classes = ["Derecha", "Izquierda"]   # carpetas existentes
EXPECTED_LANDMARKS = 21
DIM = EXPECTED_LANDMARKS * 3  # x,y,z

# Ajustes:
PAD_MISSING = False   # True = rellenar con ceros; False = descartar archivos con missing landmarks
CANONIZE_TO_RIGHT = True  # True = convertir manos izquierdas a sistema "derecha" (mirror X)
USE_Z = True  # si quieres ignorar Z pon False

def load_and_process_file(csv_path, infer_side_from_folder=None):
    """
    Lee csv que tiene columnas: landmark_index, x, y, z  (o similar)
    - Ordena por landmark_index
    - Comprueba tamaño (21)
    - Infiera hand side si infer_side_from_folder es None (usa index vs wrist)
    - Si CANONIZE_TO_RIGHT y mano==left => refleja X: x = 1 - x
    - Normaliza centrando por wrist (landmark 0) -> coords - wrist_coord
    - Devuelve vector 1xDIM
    """
    try:
        df = pd.read_csv(csv_path, comment='#', header=0)
    except Exception as e:
        print("ERROR leyendo", csv_path, e)
        return None, None

    # Asegurar que exista landmark_index, x,y (z opcional)
    cols = [c.lower() for c in df.columns]
    # buscar columnas esperadas:
    # admitir formatos con/ sin header: si no hay 'landmark_index' asumimos que primera columna es index
    if 'landmark_index' not in cols:
        # intentar si primera columna es el index numérico
        if df.shape[1] >= 3:
            df = df.rename(columns={df.columns[0]:'landmark_index', df.columns[1]:'x', df.columns[2]:'y'})
            if df.shape[1] >= 4:
                df = df.rename(columns={df.columns[3]:'z'})
        else:
            print("CSV con formato inesperado:", csv_path)
            return None, None

    # normalizar nombres minúsculas
    df.columns = [c.lower() for c in df.columns]

    # ordenar por index
    df_sorted = df.sort_values(by='landmark_index').reset_index(drop=True)

    # tomar solo las primeras EXPECTED_LANDMARKS filas (si hay más)
    if len(df_sorted) < EXPECTED_LANDMARKS:
        if PAD_MISSING:
            # rellenar con ceros
            pad_needed = EXPECTED_LANDMARKS - len(df_sorted)
            pad_df = pd.DataFrame({
                'landmark_index': list(range(len(df_sorted), EXPECTED_LANDMARKS)),
                'x': [0.0]*pad_needed,
                'y': [0.0]*pad_needed,
            })
            if 'z' in df_sorted.columns and USE_Z:
                pad_df['z'] = [0.0]*pad_needed
            df_sorted = pd.concat([df_sorted, pad_df], ignore_index=True)
        else:
            # descartar
            return None, None
    elif len(df_sorted) > EXPECTED_LANDMARKS:
        df_sorted = df_sorted.iloc[:EXPECTED_LANDMARKS]

    # extraer coords
    if USE_Z and 'z' in df_sorted.columns:
        coords = df_sorted[['x','y','z']].values.astype(float)
    else:
        # si no usamos z, ponemos z=0 para mantener tamaño si DIM fija con z; o usar DIM ajustado
        coords = df_sorted[['x','y']].values.astype(float)
        if USE_Z:
            # si USE_Z true pero no existe z -> add zeros
            coords = np.hstack([coords, np.zeros((coords.shape[0],1))])
    # coords shape = (21,3)

    # inferir mano (si la carpeta no te lo dice)
    side = None
    if infer_side_from_folder is not None:
        side = infer_side_from_folder  # 'Derecha' o 'Izquierda'
    else:
        # heurística simple: comparar x de index finger (8) vs wrist (0)
        # si index.x > wrist.x -> mano derecha (en coordenadas normales 0..1)
        try:
            wrist_x = coords[0,0]
            index_x = coords[8,0]
            side = 'Derecha' if index_x > wrist_x else 'Izquierda'
        except Exception:
            side = 'Derecha'

    # canonizar: si es izquierda y queremos mapear a derecha, reflejar X
    if CANONIZE_TO_RIGHT and side == 'Izquierda':
        # suponer coordenadas normalizadas 0..1: x' = 1 - x
        coords[:,0] = 1.0 - coords[:,0]

    # centrar por wrist (opcional)
    wrist = coords[0].copy()
    coords = coords - wrist  # ahora wrist en 0,0,0 -> invariancia translacional

    # aplanar
    vector = coords.flatten()
    return vector, side

# Recolectar
X = []
y = []
sides = []

for clase in classes:
    clase_dir = os.path.join(landmarks_dir, clase)
    if not os.path.isdir(clase_dir):
        continue
    for archivo in os.listdir(clase_dir):
        if not archivo.endswith(".csv"):
            continue
        path = os.path.join(clase_dir, archivo)
        vec, side = load_and_process_file(path, infer_side_from_folder=clase)
        if vec is None:
            print("Archivo descartado:", path)
            continue
        X.append(vec)
        y.append(clase)
        sides.append(side)

X = np.array(X)
y = np.array(y)
sides = np.array(sides)
print("X shape:", X.shape, "y shape:", y.shape)
# Guardar matrices listas para entrenar
np.save("X_landmarks.npy", X)
np.save("y_labels.npy", y)
np.save("sides.npy", sides)
print("Guardado X_landmarks.npy, y_labels.npy, sides.npy")
