from rest_framework import serializers

class VideoUploadSerializer(serializers.Serializer):
    video = serializers.FileField(
        max_length=100,
        allow_empty_file=False,
        help_text="Upload a video file for shoplifting detection"
    )
    
    def validate_video(self, value):
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        extension = value.name.lower()
        
        if not any(extension.endswith(ext) for ext in valid_extensions):
            raise serializers.ValidationError(
                f"Unsupported file format. Supported formats: {', '.join(valid_extensions)}"
            )
        
        # Limit file size (50MB)
        max_size = 50 * 1024 * 1024
        if value.size > max_size:
            raise serializers.ValidationError("File size too large. Maximum size is 50MB.")
        
        return value

class PredictionResultSerializer(serializers.Serializer):
    is_shoplifting = serializers.BooleanField()
    confidence = serializers.FloatField(min_value=0, max_value=1)
    raw_score = serializers.FloatField()
    status = serializers.CharField()
    error = serializers.CharField(required=False, allow_blank=True)

class HealthCheckSerializer(serializers.Serializer):
    status = serializers.CharField()
    model_loaded = serializers.BooleanField()
    device = serializers.CharField()