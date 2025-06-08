import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.fusion import Fusion

class MultiModalModel(nn.Module):
    def __init__(self,
                 text_encoder_cfg: dict,
                 vision_encoder_cfg: dict,
                 fusion_cfg: dict,
                 decoder_cfg: dict):
        super().__init__()

        # Build encoders
        self.text_encoder = Encoder(**text_encoder_cfg)
        self.vision_encoder = Encoder(**vision_encoder_cfg)

        # Build fusion layer
        self.fusion = Fusion(
            input_dim_1=self.text_encoder.output_dim,
            input_dim_2=self.vision_encoder.output_dim,
            output_dim=fusion_cfg["output_dim"],
            fusion_type=fusion_cfg.get("fusion_type", "concat")
        )

        # Build decoder
        self.decoder = Decoder(
            input_dim=fusion_cfg["output_dim"],
            decoder=decoder_cfg["decoder"],
            lora=decoder_cfg["lora"],
            output_dim=decoder_cfg["output_dim"],
            freeze=decoder_cfg["freeze"]
        )

    def forward(self, text_inputs, image_inputs, decoder_attention_mask=None, labels=None):
        text_embed = self.text_encoder(text_inputs)
        print(text_embed.shape)
        vision_embed = self.vision_encoder(image_inputs)
        print(vision_embed.shape)

        fused = self.fusion(text_embed, vision_embed)
        print(fused.shape)

        text_attention = text_inputs['attention_mask']
        vision_attention = torch.ones(vision_embed.shape[:2], dtype=torch.long).to(vision_embed.device)
        fused_attention = torch.cat([text_attention, vision_attention], dim=1)

        # Pass labels into decoder
        loss, logits = self.decoder(fused, attention_mask=fused_attention, labels=labels)
        print(logits.shape)
        return loss, logits