import torch
import torch.nn as nn
import lightning as L
import timm
from transformers import BartForConditionalGeneration, BartTokenizer

class SwinBartLightning(L.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        # Vision model
        
        self.device_name = torch.device(cfg.trainer.device)
        self.vision_model_name = cfg.vision_encoder_cfg.encoder
        self.vision_model_output_dim = cfg.vision_encoder_cfg.output_dim
        self.decoder_name = cfg.decoder_cfg.decoder
        
        # model defs
        self.vision_model = timm.create_model(self.vision_model_name, pretrained=True)
        self.vision_model.eval()
        
        # BART model
        self.bart_model = BartForConditionalGeneration.from_pretrained(self.decoder_name)
        self.bart_tokenizer = BartTokenizer.from_pretrained(self.decoder_name)

        # Image projection layer
        self.image_projection = nn.Linear(self.vision_model_output_dim, self.bart_model.config.d_model)
        
            
    
    def forward(self, inputs):
        # Move images to the same device as the model
        images, captions = inputs["images"].to(self.device_name), inputs["captions"]
        
        # Extract features from images using vision model
        image_features = self.vision_model(images).to(self.device_name)  # Now images are on the correct device
        
        # Project image features to match the dimension of BART's hidden layer
        projected_image_features = self.image_projection(image_features).to(self.device_name)
        
        # Tokenize captions
        tokenized_captions_dict = self.bart_tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device_name)
        
        # Get sequence length of captions
        seq_len = tokenized_captions_dict["input_ids"].shape[1]
        
        # Repeat image features to match the sequence length of captions
        repeated_image_features = projected_image_features.unsqueeze(1).repeat(1, seq_len, 1).to(self.device_name)
        
        # Forward pass through the BART model
        outputs = self.bart_model(
            input_ids=tokenized_captions_dict["input_ids"],
            attention_mask=tokenized_captions_dict["attention_mask"],
            labels=tokenized_captions_dict["input_ids"],
            encoder_outputs=(repeated_image_features, ),
        )
        return outputs


    def training_step(self, batch, batch_idx):
        images, captions = batch["images"].to(self.device_name), batch["captions"]
        
        # Get the model output
        outputs = self.forward({"images": images, "captions": captions})
        
        # Compute loss (cross entropy)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch["images"].to(self.device_name), batch["captions"]
        
        # Get the model output
        outputs = self.forward(inputs={"images": images, "captions": captions})
        
        # Compute loss (cross entropy)
        loss = outputs.loss
        
        return loss

    def configure_optimizers(self):
        # Use AdamW optimizer, which works well for transformers
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        
        return optimizer