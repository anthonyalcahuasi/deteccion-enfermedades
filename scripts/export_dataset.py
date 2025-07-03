import os
import tensorflow_datasets as tfds
from tensorflow.keras.utils import save_img  # <-- Cambiado aquí

def export_beans(base_dir="data/raw"):
    splits = ['train[:80%]', 'train[80%:]']
    (train_ds, val_ds), ds_info = tfds.load(
        'beans',
        split=splits,
        with_info=True,
        as_supervised=True,
    )
    for split_name, ds in zip(['train', 'val'], [train_ds, val_ds]):
        for img, label in tfds.as_numpy(ds):
            cls = ds_info.features['label'].names[label]
            out_dir = os.path.join(base_dir, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            idx = len(os.listdir(out_dir)) + 1
            save_img(os.path.join(out_dir, f"{cls}_{idx}.jpg"), img)
    print("✅ Dataset beans exportado en data/raw")

if __name__ == "__main__":
    export_beans()
