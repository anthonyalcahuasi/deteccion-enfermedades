# demo/app.py

import os
from pathlib import Path
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# 1) Calcula automáticamente la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# 2) Apunta al checkpoint fine-tuned
MODEL_PATH = BASE_DIR / "results" / "checkpoint_ft.keras"

IMAGE_SIZE = (224, 224)
CLASS_MAP  = {0: "angular_leaf_spot", 1: "bean_rust", 2: "healthy"}

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"❌ Modelo no encontrado en:\n{MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(str(MODEL_PATH))

def preprocess(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img   = cv2.resize(img, IMAGE_SIZE) / 255.0
    return np.expand_dims(img, 0)

def main():
    st.title("👩‍🌾 Clasificador de Enfermedades en Hojas de Frijol (Fine-Tuned)")
    st.write("Sube una imagen y obtén la predicción optimizada.")

    uploaded = st.file_uploader("Elige un archivo JPG/PNG", type=["jpg","png"])
    if uploaded:
        model = load_model()
        img_array = preprocess(uploaded.read())
        preds     = model.predict(img_array, verbose=0)[0]
        idx       = int(np.argmax(preds))
        label     = CLASS_MAP[idx]
        conf      = preds[idx]

        st.image(img_array[0], caption=f"**{label}** (Confianza: {conf:.2f})", use_column_width=True)

if __name__ == "__main__":
    main()
