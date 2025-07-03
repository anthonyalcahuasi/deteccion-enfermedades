# demo/app.py

import os
from pathlib import Path
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# Rutas y constantes
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "results" / "checkpoint_ft.keras"
IMAGE_SIZE  = (224, 224)

@st.cache_data(show_spinner=False)
def get_class_map():
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(BASE_DIR / "data" / "processed" / "train"),
        image_size=IMAGE_SIZE, batch_size=1, shuffle=False
    )
    return {i: name for i, name in enumerate(ds.class_names)}

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"❌ No se encontró el modelo en:\n{MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(str(MODEL_PATH))

def load_and_transform(image_bytes: bytes):
    # Decodificar bytes a BGR
    arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Para display: redimensionar y normalizar a [0,1]
    disp = cv2.resize(img_rgb, IMAGE_SIZE) / 255.0
    # Para modelo: aplica preprocess_input (puede producir valores fuera de [0,1])
    inp = preprocess_input(img_rgb.astype(np.float32))
    inp = cv2.resize(inp, IMAGE_SIZE)  # asegúrate mismo tamaño
    inp = np.expand_dims(inp, axis=0)
    return inp, disp

def main():
    st.title("👩‍🌾 Clasificador de Enfermedades en Hojas de Frijol")
    st.write("Sube una imagen JPG/PNG y obtén la predicción optimizada.")

    uploaded = st.file_uploader("Selecciona un archivo", type=["jpg","png"])
    if not uploaded:
        return

    # Lee bytes una sola vez
    data = uploaded.read()
    # Carga el modelo
    model = load_model()
    # Crea batch para predicción y versión para display
    input_batch, display_img = load_and_transform(data)
    # Predice
    preds = model.predict(input_batch, verbose=0)[0]
    idx   = int(np.argmax(preds))
    label = get_class_map()[idx]
    conf  = float(preds[idx])

    # Muestra la imagen correctamente normalizada
    st.image(
        display_img,
        caption=f"**{label}**  (Confianza: {conf:.2f})",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
