import os
import torch
import lightning as L
from models.model import SwinBart
import os

class TrainerModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = SwinBart(cfg)
        self.best_val_loss = float("inf")  # initialize best loss
        self.save_path = os.path.join(cfg.trainer.output_dir, "best_model.pth")

    def forward(self, images, captions):
        return self.model(images, captions)

    def training_step(self, batch, batch_idx):
        images, captions = batch["images"], batch["captions"]
        outputs = self(images, captions)
        loss = outputs.loss
        self.log("train_loss", loss)
        self._save_best_model()
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch["images"], batch["captions"]
        outputs = self(images, captions)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.trainer.lr)

    def on_validation_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss = val_loss.item()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_best_model()

    def _save_best_model(self):
        print(f"🔍 New best model found with val_loss = {self.best_val_loss:.4f}. Saving to {self.save_path}")
        torch.save(self.model.state_dict(), self.save_path)