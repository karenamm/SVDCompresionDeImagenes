# svdapp/utils.py

import os
import time
import uuid
import numpy as np
from PIL import Image
from skimage import color, util
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
from django.conf import settings
import matplotlib.pyplot as plt  # Se importa aquí para posibles gráficos dentro de utilidades

# Directorios base
UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
OUTPUT_DIR = os.path.join(settings.MEDIA_ROOT, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_uploaded_image(django_img):
    """
    Guarda el archivo subido en MEDIA_ROOT/uploads/ con nombre único basado en timestamp.
    Devuelve la ruta completa al archivo guardado.
    """
    ext = django_img.name.split('.')[-1]
    filename = f"{int(time.time() * 1000)}.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, 'wb+') as dest:
        for chunk in django_img.chunks():
            dest.write(chunk)
    return path

def array_to_pil(img_array):
    """
    Convierte un ndarray (float en [0,1]) a un objeto PIL.Image en modo 'L' (gris) o 'RGB'.
    """
    arr = img_array.copy()
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode='RGB')
    else:
        raise ValueError("Formato de arreglo desconocido para convertir a PIL.")

def pil_to_gray_array(image_path):
    """
    Abre una imagen RGB (o escala de grises) y la convierte a un array 2D float en [0,1].
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img).astype(float) / 255.0
    gray = color.rgb2gray(arr)
    return gray

def pil_to_color_array(image_path):
    """
    Abre una imagen RGB y devuelve un array 3D float en [0,1].
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img).astype(float) / 255.0
    return arr

def svd_truncate_gray(image_arr, k):
    """
    Aplica SVD a image_arr (matriz 2D float en [0,1]) y devuelve:
      - A_k: la reconstrucción en escala de grises (2D float [0,1])
      - error: MSE entre image_arr y A_k
      - tiempo: tiempo de cómputo (float)
      - S: el vector completo de valores singulares (ordenados descendente)
    """
    t0 = time.time()
    U, S, VT = np.linalg.svd(image_arr, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    A_k = U_k @ S_k @ VT_k
    A_k = np.clip(A_k, 0, 1)
    t1 = time.time()
    error = mse(image_arr, A_k)
    return A_k, error, (t1 - t0), S

def randomized_svd_gray(image_arr, k, n_iter=5):
    """
    Aplica Randomized SVD y devuelve:
      - A_k: reconstrucción en 2D float
      - error: MSE entre imagen original y A_k
      - tiempo: tiempo de cómputo
      - S_k: vector de los primeros k valores singulares aproximados
    """
    t0 = time.time()
    U_k, S_k, VT_k = randomized_svd(image_arr, n_components=k, n_iter=n_iter, random_state=42)
    A_k = U_k @ np.diag(S_k) @ VT_k
    A_k = np.clip(A_k, 0, 1)
    t1 = time.time()
    error = mse(image_arr, A_k)
    return A_k, error, (t1 - t0), S_k

def compress_color_svd(image_arr, k):
    """
    Aplica SVD en cada canal (R, G, B) y devuelve:
      - A_k: reconstrucción 3D float ([0,1]) 
      - mse_total: promedio del MSE de los tres canales
      - errores: lista [MSE_R, MSE_G, MSE_B]
      - tiempo: tiempo total de cómputo (float)
      - S_list: lista de vectores de valores singulares para cada canal [S_R, S_G, S_B]
    """
    t0 = time.time()
    channels_rec = []
    errors = []
    S_list = []
    for c in range(3):
        canal = image_arr[:, :, c]
        U, S, VT = np.linalg.svd(canal, full_matrices=False)
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        VT_k = VT[:k, :]
        canal_k = U_k @ S_k @ VT_k
        canal_k = np.clip(canal_k, 0, 1)
        channels_rec.append(canal_k)
        errors.append(mse(canal, canal_k))
        S_list.append(S)
    A_k = np.stack(channels_rec, axis=2)
    t1 = time.time()
    return A_k, float(np.mean(errors)), errors, (t1 - t0), S_list

def denoise_svd_gray(image_arr, k):
    """
    Para denoise: reconstruye con rango k. Devuelve:
      - A_k: imagen denoised (2D float)
      - tiempo: tiempo de cómputo (float)
    """
    t0 = time.time()
    U, S, VT = np.linalg.svd(image_arr, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    A_k = U_k @ S_k @ VT_k
    A_k = np.clip(A_k, 0, 1)
    t1 = time.time()
    return A_k, (t1 - t0)

def pca_patches_gray(image_arr, k, patch_size):
    """
    Aplica PCA a parches de la imagen en escala de grises y devuelve:
      - img_rec: reconstrucción (2D float)
      - error: MSE entre la porción recortada original y la reconstruida
      - tiempo: tiempo de cómputo
      - explained_var: vector de varianzas explicadas por cada componente principal
    """
    t0 = time.time()
    h, w = image_arr.shape
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    A_cropped = image_arr[:new_h, :new_w]

    patches = []
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = A_cropped[i:i+patch_size, j:j+patch_size].reshape(-1)
            patches.append(patch)
    patches = np.array(patches)

    mean_patch = patches.mean(axis=0)
    patches_centered = patches - mean_patch

    pca = PCA(n_components=k)
    scores = pca.fit_transform(patches_centered)
    patches_rec = pca.inverse_transform(scores) + mean_patch
    explained_var = pca.explained_variance_ratio_

    img_rec = np.zeros_like(A_cropped)
    idx = 0
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = patches_rec[idx].reshape(patch_size, patch_size)
            img_rec[i:i+patch_size, j:j+patch_size] = patch
            idx += 1

    img_rec = np.clip(img_rec, 0, 1)
    t1 = time.time()
    error = mse(A_cropped, img_rec)
    return img_rec, float(error), (t1 - t0), explained_var
