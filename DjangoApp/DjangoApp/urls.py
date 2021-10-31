"""DjangoApp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from MLSongs import views

urlpatterns = [
    path('', views.get_random_song),
    path('admin/', admin.site.urls),
    path('song', views.get_random_song),
    path('debug', views.debug),
    path('about', views.about),
    path('help', views.help),
    path('song/<slug:model>/<slug:instrument>', views.model_song),
    path('generate/<slug:model>/<slug:instrument>', views.execute_model_once),
    path('generate/<slug:model>/<slug:instrument>/<int:count>', views.execute_model)
]
