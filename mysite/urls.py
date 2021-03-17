"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from django.conf import settings
from django.contrib import admin
from articles import views
from django.urls import include, path
from django.conf.urls import url
from django.conf.urls.static import static # new

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),
    path('login', views.index, name='login'),
    path('predict', views.predict_number, name='predict'),

    url(r'^article_detail/(?P<pk>\d+)$', views.ArticleDetail.as_view(), name='article_detail'),
]

if settings.DEBUG: # new
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)