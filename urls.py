# code
from django.contrib import admin
from django.urls import path
from projectapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.index),
    path("predict/", views.predict),
    path("predict/result", views.result),
]