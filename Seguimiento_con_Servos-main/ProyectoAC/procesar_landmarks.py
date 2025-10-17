#!/usr/bin/env python3
"""
procesar_landmarks.py

Lee todos los CSV en dataset/landmarks/<clase> y crea:
 - X_landmarks.npy  (N x D, D = 21*3 si hay Z)
 - y_labels.npy     (N,)
 - sides.npy        (N,) -> 'Unknown' (opcional, aquí se pone 'Unknown' por compatibilidad)

Reglas:
 - Acepta CSV con columnas landmark_index,x,y,z o x,y (si falta z se rellena con 0).
 - Si una CSV tiene menos de 21 filas se descarta (por defecto).
 - Normaliza restando la coordenada de la muñeca (landmark 0).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

BASE = Path(__file__).resolve().parent
LANDMARKS_DIR = BASE / "dataset" / "landmarks"
EXPECTED_LANDMARKS = 21
USE_Z = True  # si tus CSV no tienen z, se rellenará con ceros
PAD_MISSING = False  # si True rellena con ceros, si False descarta archivos con menos landmarks

if not LANDMARKS_DIR.exists():
    print(f"[ERROR] No existe directorio de landmarks: {LANDMARKS_DIR}")
    sys.exit(1)

X_list = []
y_list = []
sides_list = []
skipped = []

for clase_dir in sorted([p for p in LANDMARKS_DIR.iterdir() if p.is_dir()]):
    clase = clase_dir.name
    csv_files = sorted([p for p in clase_dir.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        print(f"[WARN] No hay CSVs en {clase_dir}; saltando.")
        continue
    print(f"[INFO] Procesando clase '{clase}' -> {len(csv_files)} archivos")
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            skipped.append((str(csv_path), "read_error", str(e)))
            continue

        # normalizar columnas (buscar x,y,z)
        cols = [c.lower() for c in df.columns]
        if 'x' not in cols or 'y' not in cols:
            skipped.append((str(csv_path), "missing_xy", "no x/y columns"))
            continue

        # ordenar por landmark_index si existe
        if 'landmark_index' in cols:
            df_sorted = df.sort_values(by='landmark_index').reset_index(drop=True)
        else:
            df_sorted = df.reset_index(drop=True)

        # truncar o pad
        n_rows = len(df_sorted)
        if n_rows < EXPECTED_LANDMARKS:
            if PAD_MISSING:
                pad = EXPECTED_LANDMARKS - n_rows
                pad_df = pd.DataFrame({
                    'landmark_index': list(range(n_rows, EXPECTED_LANDMARKS)),
                    'x': [0.0]*pad,
                    'y': [0.0]*pad
                })
                if USE_Z:
                    pad_df['z'] = [0.0]*pad
                df_sorted = pd.concat([df_sorted, pad_df], ignore_index=True)
            else:
                skipped.append((str(csv_path), "short", n_rows))
                continue
        elif n_rows > EXPECTED_LANDMARKS:
            df_sorted = df_sorted.iloc[:EXPECTED_LANDMARKS]

        # extraer coords
        if USE_Z:
            if 'z' in df_sorted.columns:
                coords = df_sorted[['x','y','z']].values.astype(float)
            else:
                xy = df_sorted[['x','y']].values.astype(float)
                coords = np.hstack([xy, np.zeros((xy.shape[0],1), dtype=float)])
        else:
            coords = df_sorted[['x','y']].values.astype(float)

        # normalizar respecto a muñeca (landmark 0)
        try:
            wrist = coords[0].copy()
            coords = coords - wrist
        except Exception:
            # si algo falla, saltar
            skipped.append((str(csv_path), "normalize_error", "wrist missing"))
            continue

        vec = coords.flatten()
        # verificar dimensión
        expected_dim = EXPECTED_LANDMARKS * (3 if USE_Z else 2)
        if vec.size != expected_dim:
            skipped.append((str(csv_path), "dim_mismatch", vec.size))
            continue

        X_list.append(vec)
        y_list.append(clase)
        sides_list.append("Unknown")

# resumen
print(f"\n[RESULT] recolectados {len(X_list)} ejemplos. descartados: {len(skipped)}")
if skipped:
    print("[INFO] Muestra de descartados (hasta 10):")
    for s in skipped[:10]:
        print("  -", s)

if len(X_list) == 0:
    print("[ERROR] No hay ejemplos válidos para guardar. Revisa dataset/landmarks.")
    sys.exit(1)

X = np.vstack(X_list)
y = np.array(y_list, dtype=object)
sides = np.array(sides_list, dtype=object)

out_X = BASE / "X_landmarks.npy"
out_y = BASE / "y_labels.npy"
out_sides = BASE / "sides.npy"

np.save(out_X, X)
np.save(out_y, y)
np.save(out_sides, sides)

print(f"[OK] Guardado X {out_X} (shape={X.shape}), y {out_y} (N={y.shape[0]}), sides {out_sides}")