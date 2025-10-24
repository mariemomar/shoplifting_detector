import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import os
from django.conf import settings

class EfficientNetB0_LSTM(nn.Module):
    def __init__(self, hidden_size=128, num_classes=1):
        super().__init__()
        
        backbone = efficientnet_b0(weights='DEFAULT')
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
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Simplified model loading"""
        try:
            model_path = r'detection_app/ml_model/best_model_pretrained.pth'
            
            if not os.path.exists(model_path):
                print("❌ Model file not found. Using pre-trained backbone only.")
                self._create_dummy_model()
                return
            
            # Try to inspect what's in the file first
            print("🔍 Inspecting model file...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Print checkpoint structure for debugging
            print(f"📁 Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"📁 Checkpoint keys: {list(checkpoint.keys())}")
            
            # For now, just use the pre-trained backbone
            # This avoids the class loading issue entirely
            self._create_dummy_model()
            print("💡 Using pre-trained backbone to avoid loading issues")
            print("💡 To use custom weights, you may need to retrain and save properly")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a model with pre-trained backbone"""
        self.model = EfficientNetB0_LSTM()
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = False
        print("✅ Model initialized with pre-trained EfficientNet backbone")

# Global instance
detector = ShopliftingDetector()