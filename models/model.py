import torch
import torch.nn as nn
from models.encoder import VisionEncoder
from models.decoder import TextDecoder

class SwinBart(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Init encoder
        self.encoder = VisionEncoder(
            encoder_name=cfg.vision_encoder_cfg.encoder,
            output_dim=cfg.vision_encoder_cfg.output_dim,
            projection_dim=cfg.decoder_cfg.hidden_dim
        )

        # Init decoder
        self.decoder = TextDecoder(
            decoder_name=cfg.decoder_cfg.decoder
        )

    def forward(self, images, captions):
        image_features = self.encoder(images)
        outputs = self.decoder.forward(image_features, captions)
        return outputs
    def generate(self, images, max_length=128, num_beams=4):
        image_features = self.encoder(images)
        generated_ids = self.decoder.generate(
            image_embeddings=image_features,
            max_length=max_length,
            num_beams=num_beams
        )
        return generated_ids
    
class CLIPGPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = VisionEncoder(
            encoder_name=cfg.vision_encoder_cfg.encoder,
            output_dim=cfg.vision_encoder_cfg.output_dim,      # 1024 for CLIP ViT-B/32
            projection_dim=cfg.decoder_cfg.hidden_dim          # 768 for GPT2
        )
        self.decoder = TextDecoder(
            decoder_name=cfg.decoder_cfg.decoder               # e.g. "distilgpt2"
        )

    def forward(self, images, captions):
        image_features = self.encoder(images)
        return self.decoder(image_features, captions)

    def generate(self, images, max_length=128, num_beams=4):
        image_features = self.encoder(images)
        return self.decoder.generate(
            image_embeddings=image_features,
            max_length=max_length,
            num_beams=num_beams
        )