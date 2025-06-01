# svdapp/urls.py

from django.urls import path
from . import views

app_name = 'svdapp'

urlpatterns = [
    path('', views.index, name='index'),
    path(
        'process/<str:filename>/<str:process_type>/<int:k>/<int:patch_size>/',
        views.process_image,
        name='process_image'
    ),
]
