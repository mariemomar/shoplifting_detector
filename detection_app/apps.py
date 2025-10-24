from django.apps import AppConfig

class DetectionAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection_app'
    
    def ready(self):
        # Import and initialize model when app is ready
        try:
            from .ml_model.model_loader import detector
            print("Shoplifting detection model initialized")
        except Exception as e:
            print(f"Error initializing model: {e}")