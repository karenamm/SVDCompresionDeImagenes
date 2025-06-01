# svdapp/forms.py

from django import forms

PROCESS_CHOICES = [
    ('svd', 'SVD Convencional (Grises)'),
    ('rand_svd', 'Randomized SVD (Grises)'),
    ('color_svd', 'Compresión SVD a Color (RGB)'),
    ('denoise_svd', 'Eliminar ruido con SVD (Grises)'),
    ('pca_patches', 'PCA sobre Parches (Grises)'),
]

class ImageProcessForm(forms.Form):
    image = forms.ImageField(label="Selecciona una imagen", required=True)
    process_type = forms.ChoiceField(
        choices=PROCESS_CHOICES,
        label="Tipo de procesamiento",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    k = forms.IntegerField(
        label="Valor de k (nº de valores singulares o componentes)",
        min_value=1, max_value=500,
        initial=50,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 50'})
    )
    patch_size = forms.IntegerField(
        label="Tamaño de parche (sólo para PCA)",
        min_value=2, max_value=256,
        initial=8,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '8, 16, ...'})
    )

    def clean(self):
        cleaned = super().clean()
        proc = cleaned.get('process_type')
        ps = cleaned.get('patch_size')
        if proc == 'pca_patches' and not ps:
            self.add_error('patch_size', 'Para PCA debes indicar el tamaño de parche.')
