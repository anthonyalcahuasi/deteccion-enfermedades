# Guía de Despliegue (sin Docker)

## 1. Requisitos Previos

- **Python 3.10**  
- **Git**  
- Un servidor o máquina (local o en la nube) con acceso SSH

## 2. Instalación en Entorno Virtual

```bash
# 1. Clona el repositorio
git clone https://github.com/tu_usuario/deteccion-enfermedades.git
cd deteccion-enfermedades

# 2. Crea y activa un venv
python3.10 -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

# 3. Instala dependencias
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install streamlit
