# SVD Compressor - Documentación del Proyecto

## 📌 Descripción del Proyecto

**SVD Compressor** es una aplicación web Django para experimentar con técnicas de procesamiento de imágenes basadas en álgebra lineal y aprendizaje estadístico, ofreciendo:

- 5 métodos de procesamiento
- Visualización de resultados comparativos
- Métricas cuantitativas de calidad


## 🏗️ Estructura del Proyecto

```bash
svd_project/
├── manage.py
├── svd_project/
│   ├── settings.py
│   ├── urls.py
│   └── ...
└── svdapp/
    ├── forms.py
    ├── utils.py      
    ├── views.py
    ├── templates/
    │   ├── base.html
    │   ├── index.html
    │   └── result.html
    └── static/
        └── css/
            └── styles.css 

```

## 🛠️ Requisitos e Instalación

### Prerrequisitos
- Python 3.8+
- Django 5.2+

### Bibliotecas científicas:
```bash
pip install numpy pillow scikit-image scikit-learn matplotlib
```

### Configuración


## 🛠️ Instalación y Configuración Paso a Paso
### 📦 **1. Clonar el repositorio**
Cloná este proyecto desde GitHub (reemplazá con la URL real del repositorio si aplica):


```bash
git clone 
cd svd_project
```
### 🐍 **2. Crear y activar entorno virtual**
Esto asegura que las dependencias del proyecto no afecten a otras instalaciones de Python en tu sistema.

- En Linux/Mac:
```bash
Copy code
python3 -m venv venv
source venv/bin/activate
```
- En Windows (CMD):

```bash
python -m venv venv
venv\Scripts\activate
```
Si usas PowerShell, ejecuta:
```bash
powershell
venv\Scripts\Activate.ps1
```

### 📚 **3. Instalar dependencias**
Instalar Django y las bibliotecas científicas necesarias para procesamiento de imágenes y visualización:

```bash
pip install -r requirements.txt
```

### **⚙️ 4. Migrar la base de datos**
Aunque este proyecto no usa modelos complejos, se necesita inicializar la base de datos por defecto de Django:

```bash

python manage.py migrate
```

### **🚀 5. Iniciar el servidor local**

Ejecuta el servidor para abrir la app en tu navegador:


```bash
python manage.py runserver
```
Luego accedé a la aplicación desde:

📍 http://127.0.0.1:8000/















## 🔍 Métodos Implementados

# Descripción de Cada Opción

## 1. SVD Convencional (grises)
**Objetivo:** Aproximar/comprimir una imagen en escala de grises usando sus *k* valores singulares más grandes.

**Qué hace:**
- Calcula SVD completa: 𝐴 = 𝑈⋅Σ⋅𝑉ᵀ (𝐴 ∈ ℝ^{m×n}).
- Toma las primeras *k* columnas de 𝑈, *k* valores en Σ y *k* filas de 𝑉ᵀ.
- Reconstruye 𝐴ₖ = 𝑈ₖ⋅Σₖ⋅𝑉ₖᵀ.
- Guarda la imagen reconstruida, grafica los primeros valores singulares y calcula el MSE.

**Uso:** Comprensión teórica de cómo los valores singulares capturan la “energía” de la imagen.

---

## 2. Randomized SVD (grises)
**Objetivo:** Obtener una aproximación rápida de la SVD para imágenes grandes (o *k* intermedio).

**Qué hace:**
- Usa `sklearn.utils.extmath.randomized_svd` para aproximar 𝑈ₖ, Σₖ, 𝑉ₖᵀ.
- Reconstruye Aₖ = 𝑈ₖ⋅Σₖ⋅𝑉ₖᵀ.
- Grafica los *k* valores singulares aproximados.

**Uso:** Situaciones donde la SVD completa sea muy lenta (p. ej., imágenes > 1024×1024).

---

## 3. Compresión SVD a Color (RGB)
**Objetivo:** Comprimir una imagen en color canal por canal con SVD.

**Qué hace:**
- Para cada canal c ∈ {R,G,B}, realiza SVD: 𝐶_c = 𝑈_c⋅Σ_c⋅𝑉_cᵀ.
- Reconstruye cada canal con *k* valores: 𝐶_{c,k}.
- Ensambla la imagen RGB final.
- Calcula MSE total (promedio de los 3 canales) y MSE por canal.
- Grafica los valores singulares de uno de los canales (por defecto, canal Rojo).

**Uso:** Demostrar cómo la compresión afecta cada canal por separado.

---

## 4. Denoising con SVD (grises)
**Objetivo:** Filtrar ruido gaussiano en una imagen en escala de grises.

**Qué hace:**
- Agrega ruido gaussiano (σ ≈ 0.08) a la imagen original.
- Aplica SVD al arreglo ruidoso y reconstruye con *k* valores singulares.
- Calcula MSE y PSNR comparando la reconstrucción con la original limpia.
- Muestra la imagen ruidosa y la imagen “denoised”.

**Uso:** Ejemplificar que los valores singulares pequeños contienen principalmente ruido.

---

## 5. PCA sobre Parches (grises)
**Objetivo:** Aplicar PCA local en parches para comprimir o reconstruir detalles.

**Qué hace:**
- Divide la imagen en parches `patch_size × patch_size`.
- Aplana cada parche y arma una matriz *P* donde cada fila es un parche.
- Centra los datos (resta la media) y aplica PCA reteniendo *k* componentes.
- Reconstruye cada parche y ensambla la imagen final.
- Calcula MSE en la porción recortada y grafica la varianza explicada por cada componente.

**Uso:** Ver cómo PCA local captura texturas repetitivas y detalles finos en bloques.


## 📊 Métricas Calculadas

| Métrica | Descripción |
|--------|-------------|
| MSE (Error Cuadrático Medio) | Diferencia promedio al cuadrado entre imagen original y reconstruida. |
| PSNR (Peak Signal-to-Noise Ratio) | Indica la calidad de una reconstrucción comparada con la original (solo en `denoise`). |
| Varianza Explicada | En PCA, muestra cuánta información conserva cada componente. |

---

## Autores:

- Karen Mosquera

- Juan Camilo García

- Jose David Mayor

- Luciano Barbosa

*Inspirado en ejemplos de Platzi, Tim Baumann, y la documentación de scikit-learn, scikit-image y NumPy.*