import os

def rename_images_in_subfolders():
    # Obtener la ruta actual (donde está el script)
    root_folder = os.getcwd()
    
    # Recorrer todas las subcarpetas
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # Verificar si es una carpeta
        if os.path.isdir(folder_path):
            # Obtener todas las imágenes dentro de la subcarpeta
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
            
            # Ordenar las imágenes por fecha de creación
            images.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

            # Renombrar cada imagen
            for i, image in enumerate(images):
                old_path = os.path.join(folder_path, image)
                # Crear el nuevo nombre con numeración de dos dígitos
                new_name = f"{folder_name}_{i:02d}{os.path.splitext(image)[1]}"
                new_path = os.path.join(folder_path, new_name)

                # Renombrar el archivo
                os.rename(old_path, new_path)
    print(f"Imágenes renombradas")

if __name__ == "__main__":
    rename_images_in_subfolders()
