import os
import uuid
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

from .serializers import (
    VideoUploadSerializer, 
    PredictionResultSerializer, 
    HealthCheckSerializer
)
from .utils import predict_shoplifting
from .ml_model.model_loader import detector

class HealthCheckView(APIView):
    def get(self, request):
        """Check if the model and API are working"""
        health_status = {
            'status': 'healthy',
            'model_loaded': detector.model is not None,
            'device': str(detector.device),
            'message': 'Shoplifting Detection API is running'
        }
        serializer = HealthCheckSerializer(health_status)
        return Response(serializer.data)

class PredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        """Handle video upload and prediction"""
        # Validate uploaded file
        serializer = VideoUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                serializer.errors, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        video_file = serializer.validated_data['video']
        
        try:
            # Save uploaded file temporarily
            filename = f"{uuid.uuid4()}_{video_file.name}"
            temp_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_videos', filename)
            
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
            
            # Make prediction
            result = predict_shoplifting(temp_path)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Ignore cleanup errors
            
            # Return prediction result
            if result['status'] == 'error':
                return Response(
                    {'error': result['error']}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            output_serializer = PredictionResultSerializer(result)
            return Response(output_serializer.data)
            
        except Exception as e:
            # Clean up on error
            try:
                if 'temp_path' in locals():
                    os.remove(temp_path)
            except OSError:
                pass
            
            return Response(
                {'error': f'Prediction failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class BatchPredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        """Handle multiple video predictions"""
        videos = request.FILES.getlist('videos')
        results = []
        
        for video_file in videos:
            # Validate each file
            serializer = VideoUploadSerializer(data={'video': video_file})
            if not serializer.is_valid():
                results.append({
                    'filename': video_file.name,
                    'error': serializer.errors.get('video', ['Invalid file'])[0],
                    'status': 'error'
                })
                continue
            
            try:
                # Save and process each video
                filename = f"{uuid.uuid4()}_{video_file.name}"
                temp_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_videos', filename)
                
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                with open(temp_path, 'wb+') as destination:
                    for chunk in video_file.chunks():
                        destination.write(chunk)
                
                # Make prediction
                result = predict_shoplifting(temp_path)
                result['filename'] = video_file.name
                
                # Clean up
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'filename': video_file.name,
                    'error': str(e),
                    'status': 'error'
                })
        
        return Response({'results': results})