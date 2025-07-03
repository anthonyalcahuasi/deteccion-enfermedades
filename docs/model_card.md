# Model Card

## Descripción  
- Backbone: EfficientNetB0 (Imagenet)  
- Head: Dense(256) + Dropout(0.3)  
- Fine-tuning: 5 épocas en todo el modelo  

## Métricas Principales  
| Clase               | Precision | Recall | F1    | Support |
|---------------------|-----------|--------|-------|---------|
| angular_leaf_spot   | 0.87      | 0.91   | 0.89  | 288     |
| bean_rust           | 0.89      | 0.85   | 0.87  | 276     |
| healthy             | 0.93      | 0.94   | 0.93  | 264     |
| **Overall**         | —         | —      | 0.90  | 828     |

## Limitaciones  
- Solo detecta 3 clases  
- Dataset reducido → potencial sobreajuste  
