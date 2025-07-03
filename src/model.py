# src/model.py

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(num_classes, head_units=256, drop_rate=0.3):
    base = EfficientNetB0(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(head_units, activation='relu')(x)
    x = Dropout(drop_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base.input, outputs=outputs, name="EffNetB0_Classifier")
