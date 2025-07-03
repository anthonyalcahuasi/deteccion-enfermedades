# src/predict.py

import sys
import tensorflow as tf
import cv2
import numpy as np

# Ruta al modelo guardado (formato Keras)
MODEL_PATH = "results/checkpoint.keras"
# Tamaño de entrada que usó MobileNetV2
IMAGE_SIZE = (224, 224)
# Mapea índice de salida a etiqueta
CLASS_MAP = {
    0: "angular_leaf_spot",
    1: "healthy",
    2: "bean_rust"
}

def load_model(path):
    """Carga el modelo Keras desde disco."""
    return tf.keras.models.load_model(path)

def preprocess(image_path):
    """Lee y normaliza la imagen para inferencia."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: no se pudo leer la imagen en {image_path}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_path, model):
    """Realiza la predicción y devuelve etiqueta y confianza."""
    x = preprocess(image_path)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = CLASS_MAP.get(idx, "desconocido")
    confidence = float(preds[idx])
    return label, confidence

def main():
    if len(sys.argv) != 2:
        print("Uso: python predict.py <ruta_imagen>")
        sys.exit(1)
    image_path = sys.argv[1]
    model = load_model(MODEL_PATH)
    label, conf = predict(image_path, model)
    print(f"Resultado: {label}  |  Confianza: {conf:.2f}")

if __name__ == "__main__":
    main()
