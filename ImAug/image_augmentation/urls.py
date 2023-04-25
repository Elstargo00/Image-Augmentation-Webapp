from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("process_augmentation/", views.process_augmentation, name="process_augmentation")
]