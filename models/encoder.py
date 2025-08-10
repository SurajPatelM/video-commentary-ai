import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel


class VisionEncoder(nn.Module):
    def __init__(self, encoder_name: str, output_dim: int, projection_dim: int):
        super().__init__()
        self.vision_model = CLIPModel.from_pretrained(encoder_name)
        self.vision_model.eval()
        self.projection = nn.Linear(output_dim, projection_dim)
        for param in self.vision_model.parameters():
            param.requires_grad = False  # Freeze CLIP
            
    def forward(self, images):
        with torch.no_grad():
            features = self.vision_model.vision_model(images).last_hidden_state[:, 0, :]  # CLS token
        print(features.shape)
        projected = self.projection(features)  # (B, projection_dim)
        return projected
