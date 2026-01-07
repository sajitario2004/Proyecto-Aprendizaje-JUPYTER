# 🧠 Configuración de Entorno para IA en WINDOWS (Miniconda + NVIDIA RTX 50 Series)

Este repositorio documenta la configuración del entorno de desarrollo y los scripts base para proyectos de Deep Learning y Data Science en Windows.

**Estado del Hardware:** Configuración optimizada para trabajar con CPU y preparada para la transición a GPUs de última generación (NVIDIA RTX Serie 50 - Arquitectura Blackwell).

---

## 📋 Requisitos Previos

Antes de ejecutar los comandos, asegúrate de tener instalado:

1.  **[Miniconda (Windows 64-bit)](https://docs.conda.io/en/latest/miniconda.html):** Gestión eficiente de entornos.
2.  **[Visual Studio Code](https://code.visualstudio.com/):** Editor de código recomendado.
3.  **Drivers NVIDIA:** Última versión (Game Ready o Studio) desde GeForce Experience.

---

## 🛠️ Guía de Instalación (Paso a Paso)

Todos los comandos deben ejecutarse en **Anaconda Prompt (Miniconda3)**.

### 1. Creación del Entorno Virtual
Se recomienda utilizar un disco secundario (ej. `D:`) para almacenar las librerías y modelos pesados.

```bash
# 1. Crear el entorno en la carpeta del proyecto (D:\NvidiaIA)
# Confirmar con 'y' cuando se solicite.
conda create --prefix "D:\NvidiaIA" python=3.10

# 2. Activar el entorno (Imprescindible antes de instalar nada)
conda activate "D:\NvidiaIA"
2. Instalación de Librerías (Pip Install)
Instalamos el stack científico básico.

Nota: Este comando instala la versión actual. Si tu GPU es muy nueva (ej. RTX 5080) y PyTorch aún no ha lanzado el soporte oficial estable para Windows, estas librerías funcionarán automáticamente en modo CPU sin errores.

Bash

pip install torch torchvision torchaudio numpy pandas matplotlib jupyterlab notebook
3. Configuración en Visual Studio Code
Abrir VS Code y abrir la carpeta D:\NvidiaIA.

Instalar extensiones: Python y Jupyter.

Crear un archivo nuevo: main.ipynb.

Seleccionar Kernel: Arriba a la derecha, clic en "Select Kernel" -> "Python Environments" -> Seleccionar la ruta D:\NvidiaIA\python.exe.

💻 Código Universal de Inicialización
Copia y pega este bloque al principio de tus notebooks (.ipynb). Este script es híbrido: detecta si la GPU es compatible y funciona; si hay errores de drivers o incompatibilidad (común en lanzamientos recientes como la serie 50), cambia automáticamente a CPU para que puedas seguir trabajando.

Python

import torch
import sys

def get_device_info():
    """
    Configura el dispositivo de cómputo.
    Maneja excepciones específicas para GPUs nuevas (Blackwell/Hopper) 
    que aún no tienen kernel image en la versión estable de PyTorch.
    """
    device_type = "cpu"
    status_msg = "⚠️ MODO CPU (GPU no detectada o drivers incompatibles)"
    
    try:
        # Verificamos si CUDA es visible
        if torch.cuda.is_available():
            # Intentamos una operación real en memoria para confirmar compatibilidad
            # Esto fallará controladamente si la arquitectura (sm_120) no está soportada aún
            dummy = torch.zeros(1).to("cuda")
            
            # Si pasa la prueba anterior, activamos GPU
            device_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            status_msg = f"✅ MODO TURBO ACTIVO: {gpu_name}"
            
    except RuntimeError as e:
        # Captura errores como 'no kernel image is available'
        status_msg = f"⚠️ MODO CPU (GPU detectada pero requiere actualización de PyTorch): {e}"
    except Exception as e:
        status_msg = f"⚠️ MODO CPU (Error general): {e}"
    
    return torch.device(device_type), status_msg

# --- CONFIGURACIÓN GLOBAL ---
DEVICE, MSG = get_device_info()

print("="*60)
print(f"🛠️  Sistema Operativo: {sys.platform}")
print(f"🔥 Versión de PyTorch: {torch.__version__}")
print(f"🎯 Estado del Dispositivo: {MSG}")
print("="*60)

# Ejemplo de prueba (se ejecutará donde diga DEVICE)
x = torch.rand(5, 3).to(DEVICE)
print(f"\nTensor de prueba creado exitosamente en: {x.device}")
⚠️ Nota para Usuarios de RTX Serie 50 (Blackwell)
Si tienes una RTX 5080 / 5090, es normal recibir el error:

RuntimeError: CUDA error: no kernel image is available

Esto ocurre porque la arquitectura de la tarjeta (sm_120) es más nueva que la versión estable de PyTorch en Windows.

Solución:

Usa el modo CPU (el código de arriba lo hace automático).

Espera a la actualización oficial de PyTorch.

Periódicamente, intenta actualizar a la versión "Nightly" (experimental) con este comando en la terminal:

Bash

# Solo ejecutar si se necesita soporte inmediato para GPU nueva
pip install --pre --upgrade torch torchvision --index-url [https://download.pytorch.org/whl/nightly/cu124](https://download.pytorch.org/whl/nightly/cu124)
```
# 🧠 Configuración de Entorno para IA UBUNTU (Miniconda + NVIDIA RTX 50 Series) [Ubuntu]

Este repositorio documenta la configuración del entorno de desarrollo y los scripts base para proyectos de Deep Learning y Data Science en **Ubuntu Linux**.

**Estado del Hardware:** Configuración optimizada para trabajar con CPU y preparada para la transición a GPUs de última generación (NVIDIA RTX Serie 50 - Arquitectura Blackwell).

---

## 📋 Requisitos Previos

Antes de ejecutar los comandos, asegúrate de tener instalado:

1.  **[Miniconda (Linux 64-bit)](https://docs.conda.io/en/latest/miniconda.html#linux-installers):** Gestión eficiente de entornos.
2.  **[Visual Studio Code](https://code.visualstudio.com/):** Editor de código recomendado (`sudo snap install code`).
3.  **Drivers NVIDIA:** Drivers propietarios instalados (vía "Software & Updates" > "Additional Drivers" o línea de comandos).

---

## 🛠️ Guía de Instalación (Paso a Paso)

Todos los comandos deben ejecutarse en la **Terminal**.

### 1. Creación del Entorno Virtual
Crearemos el entorno en una carpeta local (ej. en tu `home`) para tener fácil acceso.

```bash
# 1. Crear el entorno en la carpeta ~/NvidiaIA
# Confirmar con 'y' cuando se solicite.
conda create --prefix ~/NvidiaIA python=3.10

# 2. Activar el entorno (Imprescindible antes de instalar nada)
conda activate ~/NvidiaIA
2. Instalación de Librerías (Pip Install)
Instalamos el stack científico básico.

Nota: Este comando instala la versión actual. Si tu GPU es muy nueva (ej. RTX 5080) y PyTorch aún no ha lanzado el soporte oficial estable, estas librerías funcionarán automáticamente en modo CPU.

Bash

python3 -m pip install torch torchvision torchaudio numpy pandas matplotlib jupyterlab notebook
3. Configuración en Visual Studio Code
Abrir VS Code y abrir la carpeta ~/NvidiaIA (o donde tengas tu código).

Instalar extensiones: Python y Jupyter.

Crear un archivo nuevo: main.ipynb.

Seleccionar Kernel:

Clic en "Select Kernel" (arriba a la derecha).

Seleccionar "Python Environments".

Buscar la ruta: ~/NvidiaIA/bin/python (Importante: en Linux el ejecutable está dentro de la carpeta bin).

💻 Código Universal de Inicialización
Copia y pega este bloque al principio de tus notebooks (.ipynb). Este script es híbrido: detecta si la GPU es compatible y funciona; si hay errores de drivers o incompatibilidad, cambia automáticamente a CPU.

Python

import torch
import sys

def get_device_info():
    """
    Configura el dispositivo de cómputo.
    Maneja excepciones específicas para GPUs nuevas (Blackwell/Hopper) 
    que aún no tienen kernel image en la versión estable de PyTorch.
    """
    device_type = "cpu"
    status_msg = "⚠️ MODO CPU (GPU no detectada o drivers incompatibles)"
    
    try:
        # Verificamos si CUDA es visible
        if torch.cuda.is_available():
            # Intentamos una operación real en memoria para confirmar compatibilidad
            # Esto fallará controladamente si la arquitectura (sm_120) no está soportada aún
            dummy = torch.zeros(1).to("cuda")
            
            # Si pasa la prueba anterior, activamos GPU
            device_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            status_msg = f"✅ MODO TURBO ACTIVO: {gpu_name}"
            
    except RuntimeError as e:
        # Captura errores como 'no kernel image is available'
        status_msg = f"⚠️ MODO CPU (GPU detectada pero requiere actualización de PyTorch): {e}"
    except Exception as e:
        status_msg = f"⚠️ MODO CPU (Error general): {e}"
    
    return torch.device(device_type), status_msg

# --- CONFIGURACIÓN GLOBAL ---
DEVICE, MSG = get_device_info()

print("="*60)
print(f"🛠️  Sistema Operativo: {sys.platform}")
print(f"🔥 Versión de PyTorch: {torch.__version__}")
print(f"🎯 Estado del Dispositivo: {MSG}")
print("="*60)

# Ejemplo de prueba (se ejecutará donde diga DEVICE)
x = torch.rand(5, 3).to(DEVICE)
print(f"\nTensor de prueba creado exitosamente en: {x.device}")
⚠️ Nota para Usuarios de RTX Serie 50 (Blackwell)
Si tienes una RTX 5080 / 5090, es normal recibir el error RuntimeError: CUDA error: no kernel image is available si usas la versión estable de PyTorch.

Solución:

Usa el modo CPU temporalmente (el código de arriba lo gestiona solo).

Si necesitas forzar el uso de GPU antes del soporte oficial, prueba la versión Nightly:

Bash

# Solo ejecutar si se necesita soporte inmediato (Experimental)
pip install --pre --upgrade torch torchvision --index-url [https://download.pytorch.org/whl/nightly/cu124](https://download.pytorch.org/whl/nightly/cu124)
```
---

## 📚 Roadmap de Especialización en IA

Este plan de estudios consta de **36 temas** diseñados para avanzar desde los fundamentos matemáticos hasta la IA Generativa moderna, aprovechando la aceleración por GPU.

### 🐍 MÓDULO 1: Fundamentos de Python Científico
*La base del manejo de datos antes de entrar en redes neuronales.*

- [ ] **01. NumPy I:** Arrays, dimensiones (shapes) y tipos de datos.
- [ ] **02. NumPy II:** Operaciones matemáticas vectorizadas y Broadcasting.
- [ ] **03. Pandas I:** DataFrames, lectura de archivos (CSV/Excel) y exploración.
- [ ] **04. Pandas II:** Limpieza de datos, manejo de nulos y filtrado avanzado.
- [ ] **05. Visualización I:** Gráficos estáticos con Matplotlib (líneas, dispersión).
- [ ] **06. Preprocesamiento:** Normalización, estandarización y One-Hot Encoding.

### 📐 MÓDULO 2: Machine Learning Clásico
*Entendiendo cómo aprenden las máquinas (algoritmos tradicionales).*

- [ ] **07. Conceptos Clave:** Supervisado vs No Supervisado, Overfitting.
- [ ] **08. Regresión Lineal:** Predicción numérica y concepto de "Error".
- [ ] **09. Regresión Logística:** Clasificación binaria y probabilidad.
- [ ] **10. Árboles de Decisión:** Reglas de decisión interpretables.
- [ ] **11. Métricas de Evaluación:** Accuracy, Precision, Recall, Matriz de Confusión.
- [ ] **12. División de Datos:** Train, Validation y Test sets.

### 🔥 MÓDULO 3: Deep Learning & PyTorch
*El núcleo del aprendizaje profundo y uso de GPU.*

- [ ] **13. Tensores en PyTorch:** Diferencias con NumPy y uso de CUDA.
- [ ] **14. El Perceptrón:** La neurona artificial y operaciones matriciales.
- [ ] **15. Redes Densas (MLP):** Capas ocultas y funciones de activación (ReLU).
- [ ] **16. Funciones de Pérdida:** MSE y CrossEntropy.
- [ ] **17. Optimizadores:** Descenso del gradiente, SGD y Adam.
- [ ] **18. El Training Loop:** Epochs, Batches y monitoreo de pérdida.

### 👁️ MÓDULO 4: Visión Artificial (Computer Vision)
*Enseñando a la máquina a "ver" e interpretar imágenes.*

- [ ] **19. Convoluciones:** Filtros, Kernels y Mapas de características.
- [ ] **20. Pooling y Flattening:** Reducción de dimensionalidad.
- [ ] **21. Arquitectura CNN:** Construcción de redes convolucionales completas.
- [ ] **22. Data Augmentation:** Rotaciones y transformaciones para mejorar el dataset.
- [ ] **23. Transfer Learning:** Uso de modelos pre-entrenados (ResNet, VGG).
- [ ] **24. Persistencia:** Guardado (`.pth`) y carga de modelos (Checkpoints).

### 💬 MÓDULO 5: Procesamiento de Lenguaje Natural (NLP)
*Enseñando a la máquina a "leer" y entender texto.*

- [ ] **25. Preprocesamiento NLP:** Tokenización, limpieza y vocabulario.
- [ ] **26. Embeddings:** Representación vectorial de palabras (Word2Vec).
- [ ] **27. Redes Recurrentes (RNNs):** Secuencias y memoria temporal.
- [ ] **28. LSTMs y GRUs:** Memoria a largo plazo y puertas lógicas.
- [ ] **29. Mecanismo de Atención:** La base de los modelos modernos.
- [ ] **30. Transformers:** Arquitectura Encoder-Decoder.

### ✨ MÓDULO 6: IA Generativa y Avanzada
*Estado del arte: LLMs, Difusión y aplicaciones reales.*

- [ ] **31. Modelos BERT:** Entendimiento bidireccional del lenguaje.
- [ ] **32. Modelos GPT:** Generación de texto autorregresiva.
- [ ] **33. Fine-Tuning:** Ajuste de LLMs (LoRA/PEFT) a datos propios.
- [ ] **34. Stable Diffusion:** Generación de imágenes a partir de texto.
- [ ] **35. RAG (Retrieval Augmented Generation):** Chat con documentos privados.
- [ ] **36. Despliegue (Deploy):** Creación de demos web con Gradio/Streamlit.

---
