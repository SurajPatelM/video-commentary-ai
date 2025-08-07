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