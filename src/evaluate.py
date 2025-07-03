# src/evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from pathlib import Path

# Paths
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CKPT_DIR      = BASE_DIR / "results"
# Prefiere el modelo fine-tuned si existe
FT_CKPT       = CKPT_DIR / "checkpoint_ft.keras"
HEAD_CKPT     = CKPT_DIR / "checkpoint_head.keras"
MODEL_PATH    = FT_CKPT if FT_CKPT.exists() else HEAD_CKPT

PLOT_CM       = CKPT_DIR / "plots" / "confusion_matrix.png"
REPORT_TXT    = CKPT_DIR / "plots" / "classification_report.txt"
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32

def load_class_names():
    """Obtiene la lista de clases en el mismo orden que el entrenamiento."""
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(PROCESSED_DIR / "train"),
        image_size=IMAGE_SIZE,
        batch_size=1,
        shuffle=False
    )
    return ds.class_names

def load_data(split: str):
    """Carga todas las imágenes y etiquetas de data/processed/{split}."""
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(PROCESSED_DIR / split),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    images, labels = [], []
    for x_batch, y_batch in ds:
        images.append(x_batch.numpy())
        labels.append(y_batch.numpy())
    return np.vstack(images), np.concatenate(labels)

def run():
    # 1) Carga el modelo
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"✅ Modelo cargado: {MODEL_PATH.name}")

    # 2) Carga datos y clases
    class_names = load_class_names()
    x_val, y_val = load_data("val")

    # 3) Predicciones
    preds = model.predict(x_val, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    # 4) Informe de clasificación
    report = classification_report(y_val, y_pred, target_names=class_names)
    print("\n=== Classification Report ===\n")
    print(report)

    # Guarda el reporte en texto
    os.makedirs(REPORT_TXT.parent, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    # 5) Matriz de confusión
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.savefig(str(PLOT_CM))
    print(f"✅ Matriz de confusión guardada en: {PLOT_CM}")
    print(f"✅ Reporte de clasificación guardado en: {REPORT_TXT}")

if __name__ == "__main__":
    run()
