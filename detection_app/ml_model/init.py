import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import os
from django.conf import settings

class EfficientNetB0_LSTM(nn.Module):
    def __init__(self, hidden_size=128, num_classes=1):
        super().__init__()
        
        backbone = efficientnet_b0(pretrained=False)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        
        feats = self.backbone(x)
        feats = feats.view(b, t, -1)
        
        _, (h_n, _) = self.lstm(feats)
        h_n = h_n.squeeze(0)
        
        out = self.dropout(h_n)
        out = self.fc(out)
        
        return torch.sigmoid(out)

class ShopliftingDetector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = EfficientNetB0_LSTM()
            
            # Load state dict
            checkpoint = torch.load(settings.MODEL_PATH, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def get_model(self):
        return self.model

# Global instance
detector = ShopliftingDetector()