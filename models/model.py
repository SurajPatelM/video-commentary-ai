import torch
import torch.nn as nn
from models.encoder import VisionEncoder
from models.decoder import TextDecoder
from models.audio_encoder import AudioEncoder

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
        self.audio_encoder = AudioEncoder(
            encoder_name=cfg.audio_encoder_cfg.encoder,
            n_mfcc=cfg.audio_encoder_cfg.n_mfcc,
            projection_dim=cfg.decoder_cfg.hidden_dim
        )

    def forward(self, images, captions, audio):
        image_features = self.encoder(images)
        audio_features = self.audio_encoder(audio)
        concat_features = torch.cat((image_features, audio_features), dim=1)
        normalized_features = nn.functional.normalize(concat_features, dim=1)
        outputs = self.decoder.forward(normalized_features, captions)
        return outputs
    def generate(self, images, audio, max_length=128, num_beams=4):
        image_features = self.encoder(images)
        audio_features = self.audio_encoder(audio)
        concat_features = torch.cat((image_features, audio_features), dim=1)
        normalized_features = nn.functional.normalize(concat_features, dim=1)
        generated_ids = self.decoder.generate(
            image_audio_embeddings=normalized_features,
            max_length=max_length,
            num_beams=num_beams
        )
        return generated_ids