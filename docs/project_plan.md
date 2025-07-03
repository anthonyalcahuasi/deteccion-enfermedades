# Plan de Proyecto

## 1. Alcance  
- Detección de enfermedades en hojas de frijol  
- Pipeline: TFDS → Preprocesamiento → Training → Evaluación → Demo

## 2. Hitos  
| Fecha       | Actividad                          | Responsable |
|-------------|------------------------------------|-------------|
| 2025-06-25  | Exportar dataset “beans”           | Hans        |
| 2025-06-28  | Fase 2: Preprocesamiento completo  | Hans        |
| 2025-07-01  | Fase 3: Training + Fine-tuning     | Hans        |
| 2025-07-02  | Fase 4: Evaluación y métricas      | Hans        |
| 2025-07-03  | Fase 5: Demo Streamlit             | Hans        |
| 2025-07-04  | Preparación de entrega y slides    | Hans        |

## 3. Riesgos & Mitigaciones  
- **Sesgo de clase** → oversampling, class-weights  
- **Compatibilidad de versiones** → Python 3.10 + venv  
- **Despliegue fallido** → demo CLI como plan B  
