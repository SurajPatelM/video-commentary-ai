import torch
import lightning as L
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer

from models.MultiModal import MultiModalModel
from models.encoder import Encoder
from models.decoder import Decoder
from models.fusion import Fusion

class LitMultiModalModule(L.LightningModule):
    def __init__(self, text_encoder_cfg, vision_encoder_cfg, fusion_cfg, decoder_cfg, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(decoder_cfg.decoder)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = MultiModalModel(
            text_encoder_cfg=text_encoder_cfg,
            vision_encoder_cfg=vision_encoder_cfg,
            fusion_cfg=fusion_cfg,
            decoder_cfg=decoder_cfg
        )

        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        self.lr = lr

    def forward(self, image_inputs, text_inputs, attention_mask=None, labels=None):
        return self.model(
            text_inputs=text_inputs,
            image_inputs=image_inputs,
            decoder_attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        images, captions = batch
        tokenized = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(images.device)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        _, logits = self.forward(images, tokenized, attention_mask)
        print(logits.shape)
        target_len = input_ids.shape[1]
        logits = logits[:, :96, :]
        print(logits.shape)
        # print(logits.view(-1, logits.size(-1)).shape)
        print(input_ids.contiguous().view(-1).shape)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), input_ids.contiguous().view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        images, captions = batch
        tokenized = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(images.device)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        loss, _ = self.forward(images, tokenized, attention_mask=attention_mask, labels=input_ids)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)
