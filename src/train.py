# src/train.py

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import preprocess_input
from model import build_model
import matplotlib.pyplot as plt

# Parámetros clave
BATCH_SIZE   = 32
HEAD_EPOCHS  = 10
FT_EPOCHS    = 5
IMAGE_SIZE   = (224, 224)
RAW_DIR      = "data/raw"
PROC_DIR     = "data/processed"
CKPT_DIR     = "results"
HEAD_CKPT    = os.path.join(CKPT_DIR, "checkpoint_head.keras")
FT_CKPT      = os.path.join(CKPT_DIR, "checkpoint_ft.keras")
PLOT_PATH    = os.path.join(CKPT_DIR, "plots", "accuracy_curve.png")

def calculate_class_weights():
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(RAW_DIR, "train"),
        image_size=IMAGE_SIZE,
        batch_size=1,
        shuffle=False
    )
    y = np.concatenate([y.numpy() for _, y in ds])
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): w for c, w in zip(classes, weights)}

def prepare_datasets():
    AUTOTUNE = tf.data.AUTOTUNE

    # 1) Carga original para class_names
    raw_train = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(PROC_DIR, "train"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    class_names = raw_train.class_names

    # 2) Preprocess + augmentation
    def preprocess(image, label):
        # Convierte a float32 y aplica la función de EfficientNet
        return preprocess_input(tf.cast(image, tf.float32)), label

    augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),  # menos rotación
        tf.keras.layers.RandomZoom(0.05),      # menos zoom
    ], name="simple_augment")

    train_ds = (
        raw_train
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .map(lambda x,y: (augmenter(x), y), num_parallel_calls=AUTOTUNE)
        .cache().prefetch(AUTOTUNE)
    )

    val_ds = (
        tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(PROC_DIR, "val"),
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache().prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, class_names

def plot_accuracy(h1, h2=None):
    plt.figure(figsize=(8,5))
    plt.plot(h1.history['accuracy'],    label='head_train_acc')
    plt.plot(h1.history['val_accuracy'],label='head_val_acc')
    if h2:
        plt.plot(h2.history['accuracy'],    label='ft_train_acc')
        plt.plot(h2.history['val_accuracy'],label='ft_val_acc')
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Accuracy: Head vs Fine-Tuning")
    plt.legend()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)

def run():
    # 1) calcula pesos de clase
    class_weight = calculate_class_weights()
    print("🔢 Class weights:", class_weight)

    # 2) datasets
    train_ds, val_ds, class_names = prepare_datasets()
    num_classes = len(class_names)

    # 3) head training
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(HEAD_CKPT, save_best_only=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2)
        ]
    )

    # 4) fine-tuning (backbone entero)
    model = tf.keras.models.load_model(HEAD_CKPT)
    model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FT_EPOCHS,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(patience=2, restore_best_weights=True),
            ModelCheckpoint(FT_CKPT, save_best_only=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=1)
        ]
    )

    # 5) plot
    plot_accuracy(h1, h2)
    print(f"✅ Head model guardado en {HEAD_CKPT}")
    print(f"✅ Fine-tuned model guardado en {FT_CKPT}")

if __name__ == "__main__":
    run()
