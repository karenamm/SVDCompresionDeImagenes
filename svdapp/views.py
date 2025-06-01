# svdapp/views.py

import os
import uuid
import matplotlib.pyplot as plt

from django.shortcuts import render, redirect
from django.conf import settings

from .forms import ImageProcessForm
from .utils import (
    save_uploaded_image,
    pil_to_gray_array,
    pil_to_color_array,
    array_to_pil,
    svd_truncate_gray,
    randomized_svd_gray,
    compress_color_svd,
    denoise_svd_gray,
    pca_patches_gray
)

from skimage import util
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio


def index(request):
    """
    Muestra el formulario para subir una imagen y escoger el tipo de proceso.
    """
    if request.method == 'POST':
        form = ImageProcessForm(request.POST, request.FILES)
        if form.is_valid():
            django_img = form.cleaned_data['image']
            upload_path = save_uploaded_image(django_img)

            process_type = form.cleaned_data['process_type']
            k = form.cleaned_data['k']
            patch_size = form.cleaned_data.get('patch_size') or 0
            filename = os.path.basename(upload_path)

            return redirect(
                'svdapp:process_image',
                filename=filename,
                process_type=process_type,
                k=k,
                patch_size=patch_size
            )
    else:
        form = ImageProcessForm()

    return render(request, 'svdapp/index.html', {'form': form})


def process_image(request, filename, process_type, k, patch_size):
    """
    Procesa la imagen subida según el tipo indicado, genera reconstrucción,
    métricas y (si aplica) gráfica de valores singulares o varianzas.
    """
    upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)

    contexto = {
        'original_url': os.path.join(settings.MEDIA_URL, 'uploads', filename),
        'process_type': process_type,
        'k': k,
        'patch_size': patch_size,
        'metrics': {},
        'outputs': {},
        'plot_url': None,
    }

    session_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join(settings.MEDIA_ROOT, 'outputs', session_id)
    os.makedirs(out_dir, exist_ok=True)

    if process_type in ['svd', 'rand_svd', 'denoise_svd', 'pca_patches']:
        img_arr = pil_to_gray_array(upload_path)
    else:
        img_arr = pil_to_color_array(upload_path)

    # 1) SVD convencional (grises)
    if process_type == 'svd':
        A_k, error, t, S = svd_truncate_gray(img_arr, k)

        contexto['metrics'] = {
            'MSE': f"{error:.6f}",
            'Tiempo (s)': f"{t:.4f}",
        }

        out_img = array_to_pil(A_k)
        out_filename = f"svd_{k}_{filename}"
        out_path = os.path.join(out_dir, out_filename)
        out_img.save(out_path)

        contexto['outputs']['processed'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, out_filename
        )

        n_to_plot = min(len(S), 100)
        plt.figure(figsize=(5, 3))
        plt.plot(range(1, n_to_plot + 1), S[:n_to_plot], marker='o', linestyle='-')
        plt.title(f'Valores singulares (primeros {n_to_plot})')
        plt.xlabel('Índice i')
        plt.ylabel('σᵢ')
        plt.grid(True)

        plot_filename = f"svd_singulars_{k}_{filename.rsplit('.', 1)[0]}.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()

        contexto['plot_url'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, plot_filename
        )

    # 2) Randomized SVD (grises)
    elif process_type == 'rand_svd':
        A_k, error, t, S_k = randomized_svd_gray(img_arr, k, n_iter=10)

        contexto['metrics'] = {
            'MSE': f"{error:.6f}",
            'Tiempo (s)': f"{t:.4f}",
        }

        out_img = array_to_pil(A_k)
        out_filename = f"rand_svd_{k}_{filename}"
        out_path = os.path.join(out_dir, out_filename)
        out_img.save(out_path)

        contexto['outputs']['processed'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, out_filename
        )

        n_to_plot = len(S_k)
        plt.figure(figsize=(5, 3))
        plt.plot(range(1, n_to_plot + 1), S_k, marker='o', linestyle='-')
        plt.title(f'Randomized SVD: primeros {n_to_plot} valores singulares')
        plt.xlabel('Índice i')
        plt.ylabel('σᵢ (aprox.)')
        plt.grid(True)

        plot_filename = f"rand_svd_singulars_{k}_{filename.rsplit('.', 1)[0]}.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()

        contexto['plot_url'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, plot_filename
        )

    # 3) Compresión SVD a color (RGB)
    elif process_type == 'color_svd':
        A_k, mse_total, mse_ch, t, S_list = compress_color_svd(img_arr, k)

        contexto['metrics'] = {
            'MSE total': f"{mse_total:.6f}",
            'MSE R': f"{mse_ch[0]:.6f}",
            'MSE G': f"{mse_ch[1]:.6f}",
            'MSE B': f"{mse_ch[2]:.6f}",
            'Tiempo (s)': f"{t:.4f}",
        }

        out_img = array_to_pil(A_k)
        out_filename = f"color_svd_{k}_{filename}"
        out_path = os.path.join(out_dir, out_filename)
        out_img.save(out_path)

        contexto['outputs']['processed'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, out_filename
        )

        S_R = S_list[0]
        n_to_plot = min(len(S_R), 100)
        plt.figure(figsize=(5, 3))
        plt.plot(range(1, n_to_plot + 1), S_R[:n_to_plot], marker='o', linestyle='-')
        plt.title(f'Valores singulares del canal R (primeros {n_to_plot})')
        plt.xlabel('Índice i')
        plt.ylabel('σᵢ')
        plt.grid(True)

        plot_filename = f"color_svd_singulars_R_{k}_{filename.rsplit('.', 1)[0]}.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()

        contexto['plot_url'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, plot_filename
        )

    # 4) Denoising con SVD (grises)
    elif process_type == 'denoise_svd':
        noise_sigma = 0.08
        img_noisy = util.random_noise(img_arr, mode='gaussian', var=noise_sigma ** 2)
        recon, t = denoise_svd_gray(img_noisy, k)

        psnr_val = peak_signal_noise_ratio(img_arr, recon, data_range=1.0)
        mse_val = mse(img_arr, recon)

        contexto['metrics'] = {
            'MSE tras denoise': f"{mse_val:.6f}",
            'PSNR (dB)': f"{psnr_val:.2f}",
            'Tiempo (s)': f"{t:.4f}",
        }

        noisy_img = array_to_pil(img_noisy)
        noisy_filename = f"noisy_{filename}"
        noisy_path = os.path.join(out_dir, noisy_filename)
        noisy_img.save(noisy_path)
        contexto['outputs']['noisy'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, noisy_filename
        )

        recon_img = array_to_pil(recon)
        out_filename = f"denoise_{k}_{filename}"
        out_path = os.path.join(out_dir, out_filename)
        recon_img.save(out_path)
        contexto['outputs']['processed'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, out_filename
        )

    # 5) PCA sobre parches (grises)
    elif process_type == 'pca_patches':
        patch_size = int(patch_size)
        recon, error, t, explained_var = pca_patches_gray(img_arr, k, patch_size)

        contexto['metrics'] = {
            'MSE': f"{error:.6f}",
            'Tiempo (s)': f"{t:.4f}",
            'Patch size': f"{patch_size}",
        }

        out_img = array_to_pil(recon)
        out_filename = f"pca_{k}k_{patch_size}px_{filename}"
        out_path = os.path.join(out_dir, out_filename)
        out_img.save(out_path)

        contexto['outputs']['processed'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, out_filename
        )

        n_to_plot = min(len(explained_var), 50)
        plt.figure(figsize=(5, 3))
        plt.plot(range(1, n_to_plot + 1), explained_var[:n_to_plot], marker='o', linestyle='-')
        plt.title(f'Varianza explicada PCA (primeros {n_to_plot} componentes)')
        plt.xlabel('Componente principal')
        plt.ylabel('Varianza explicada')
        plt.grid(True)

        plot_filename = f"pca_explained_{k}_{patch_size}px_{filename.rsplit('.', 1)[0]}.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()

        contexto['plot_url'] = os.path.join(
            settings.MEDIA_URL, 'outputs', session_id, plot_filename
        )

    return render(request, 'svdapp/result.html', contexto)
