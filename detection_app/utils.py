import cv2
import numpy as np
import torch
from django.conf import settings
import os

def process_video(video_path, num_frames=16, size=(224, 224)):
    """
    Process video and extract frames for model prediction
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        frames.append(frame)
    
    cap.release()
    
    # Handle frame extraction
    total = len(frames)
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        frames += [frames[-1]] * (num_frames - total)
    
    # Convert to tensor
    frames = np.array(frames, dtype=np.float32) / 255.0
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
    frames = frames.unsqueeze(0)  # Add batch dimension
    
    return frames

def predict_shoplifting(video_path):
    """
    Make prediction on a video
    """
    try:
        from .ml_model.model_loader import detector
        
        # Process video
        frames = process_video(video_path)
        frames = frames.to(detector.device)
        
        # Make prediction
        with torch.no_grad():
            output = detector.model(frames)
            prediction = output.squeeze().item()
            
        # Interpret results
        is_shoplifting = prediction > 0.5
        confidence = prediction if is_shoplifting else 1 - prediction
        
        return {
            'is_shoplifting': bool(is_shoplifting),
            'confidence': float(confidence),
            'raw_score': float(prediction),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }