import os
import cv2
import numpy as np
from tqdm import tqdm

RAW_DIR       = "data/raw"
PROC_DIR      = "data/processed"
IMAGE_SIZE    = (225, 225)

def augment_and_save(src_path, dst_path):
    # Leer y normalizar
    img = cv2.imread(src_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0

    # Guardar original
    cv2.imwrite(dst_path, (img * 255).astype(np.uint8))

    # Flip horizontal
    flipped = cv2.flip(img, 1)
    cv2.imwrite(dst_path.replace(".jpg", "_flip.jpg"),
                (flipped * 255).astype(np.uint8))

    # Rotaciones ±15°
    for angle in (-15, 15):
        M = cv2.getRotationMatrix2D((IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2), angle, 1)
        rot = cv2.warpAffine(img, M, IMAGE_SIZE)
        out_path = dst_path.replace(".jpg", f"_rot{angle}.jpg")
        cv2.imwrite(out_path, (rot * 255).astype(np.uint8))

def run():
    for split in ("train", "val"):
        for cls in os.listdir(os.path.join(RAW_DIR, split)):
            src_folder = os.path.join(RAW_DIR, split, cls)
            dst_folder = os.path.join(PROC_DIR, split, cls)
            os.makedirs(dst_folder, exist_ok=True)

            for fname in tqdm(os.listdir(src_folder), desc=f"{split}/{cls}"):
                src_f = os.path.join(src_folder, fname)
                dst_f = os.path.join(dst_folder, fname)
                augment_and_save(src_f, dst_f)

    print("✅ Preprocesamiento y augmentación completados en data/processed")

if __name__ == "__main__":
    run()
