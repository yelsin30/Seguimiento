#!/usr/bin/env python3
"""
GenerarLandmarks.py (versión robusta)
Detecta automáticamente la carpeta base si ejecutas el script desde dentro de dataset.
Genera CSVs de landmarks en dataset/landmarks/<clase> a partir de dataset/images/<clase>.
"""
from pathlib import Path
import argparse
import cv2
import mediapipe as mp
import csv
import sys
import unicodedata

def norm(s: str) -> str:
    if s is None:
        return ""
    s2 = unicodedata.normalize("NFKD", str(s))
    s2 = "".join(ch for ch in s2 if not unicodedata.combining(ch))
    return s2.strip().lower()

def find_image_folders(images_dir: Path, requested_classes):
    available = {norm(p.name): p for p in images_dir.iterdir() if p.is_dir()}
    if not requested_classes:
        return [p for _, p in sorted(available.items())]
    resolved = []
    for cls in requested_classes:
        k = norm(cls)
        if k in available:
            resolved.append(available[k])
        else:
            matched = None
            for ak, p in available.items():
                if k in ak or ak in k:
                    matched = p
                    break
            if matched:
                print(f"[INFO] Clase '{cls}' mapeada a carpeta existente: {matched.name}")
                resolved.append(matched)
            else:
                print(f"[WARN] No se encontró carpeta para la clase solicitada '{cls}' dentro de {images_dir}")
    return resolved

def process_folder(folder: Path, out_folder: Path, hands_detector, skip_existing=False, dry_run=False):
    jpgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not jpgs:
        print(f"  - (Vacío) No hay imágenes en {folder}")
        return 0
    out_folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_path in jpgs:
        csv_name = img_path.stem + ".csv"
        csv_path = out_folder / csv_name
        if skip_existing and csv_path.exists():
            continue
        if dry_run:
            print(f"  [DRY] {img_path} -> {csv_path}")
            written += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] No se pudo leer imagen: {img_path.name}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["landmark_index", "x", "y", "z"])
                if results and results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    for i, lm in enumerate(hand_lm.landmark):
                        writer.writerow([i, float(lm.x), float(lm.y), float(lm.z)])
                else:
                    print(f"  [INFO] No se detectó mano en {folder.name}/{img_path.name} -> creado CSV con cabecera")
            written += 1
        except Exception as e:
            print(f"  [ERROR] al escribir {csv_path}: {e}")
    return written

def main():
    parser = argparse.ArgumentParser(description="Generar CSVs de landmarks desde dataset/images")
    parser.add_argument("--base", "-b", default=None, help="Directorio base del proyecto (por defecto detecta automáticamente)")
    parser.add_argument("--classes", "-c", nargs="*", help="Lista de clases (carpetas) a procesar. Si no se especifica procesa todas las subcarpetas")
    parser.add_argument("--max-hands", type=int, default=1, help="Máximo de manos a procesar por imagen")
    parser.add_argument("--skip-existing", action="store_true", help="Omitir CSVs ya existentes")
    parser.add_argument("--dry-run", action="store_true", help="No escribir archivos; solo mostrar acciones")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    # Si el script está dentro de "dataset", usar el padre como base automáticamente
    if args.base:
        base = Path(args.base).resolve()
    else:
        if script_path.parent.name.lower() == "dataset":
            base = script_path.parent.parent.resolve()
            print(f"[INFO] Detectado que el script está dentro de 'dataset'; usando base: {base}")
        else:
            base = script_path.parent.resolve()

    images_dir = base / "dataset" / "images"
    landmarks_dir = base / "dataset" / "landmarks"

    if not images_dir.exists():
        print(f"[ERROR] No existe: {images_dir}")
        print("Ejecuta el script desde la carpeta del proyecto (la que contiene 'dataset'), o pasa --base con la ruta correcta.")
        sys.exit(1)

    images_folders = find_image_folders(images_dir, args.classes or [])
    if not images_folders:
        print(f"[ERROR] No se encontraron carpetas de imágenes en {images_dir}")
        sys.exit(1)

    print(f"[INFO] Carpetas a procesar: {[p.name for p in images_folders]}")
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=args.max_hands,
                        min_detection_confidence=0.5) as hands:
        total = 0
        for folder in images_folders:
            out_folder = landmarks_dir / folder.name
            print(f"Procesando carpeta: {folder} -> salida: {out_folder}")
            written = process_folder(folder, out_folder, hands, skip_existing=args.skip_existing, dry_run=args.dry_run)
            total += written
        print(f"[DONE] Archivos procesados (o planificados en dry-run): {total}")

if __name__ == "__main__":
    main()