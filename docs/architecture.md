# Arquitectura de Solución

```mermaid
flowchart TD
    TFDS[TensorFlow Datasets] --> Export[export_dataset.py]
    Export --> Preproc[preprocess.py]
    Preproc --> Train[train.py]
    Train --> Evaluate[evaluate.py]
    Train --> Model[checkpoint_ft.keras]
    Model --> Demo[Streamlit App]
