from django.urls import path, include

from . import views

urlpatterns = [
    path("", views.home, name = "home"),      # 0 = percorso, 1 = view
    path("calolo_limiti", views.calcolo_limiti, name="calcolo_limiti"),      # 0 = percorso, 1 = view
    path('come_funziona', views.come_funziona),

    path('test', views.test, name='test'),
    path('ajax-text', views.ajax_text2math, name='ajax-text'),

    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]

