from django.urls import path
from .views import HealthCheckView, PredictView, BatchPredictView

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('batch-predict/', BatchPredictView.as_view(), name='batch-predict'),
]