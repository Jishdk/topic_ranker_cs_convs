from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('results/<int:topic_id>/<str:topic_name>/', views.results, name='results'),
]
