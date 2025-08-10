import os
import torch
import lightning as L
from models.model import SwinBart

class TrainerModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = SwinBart(cfg)
        self.best_val_loss = float("inf")
        self.save_path = os.path.join(cfg.trainer.output_dir, "best_model.pth")
        os.makedirs(cfg.trainer.output_dir, exist_ok=True)  # ensure dir exists

    def forward(self, images, captions):
        return self.model(images, captions)

    def training_step(self, batch, batch_idx):
        images, captions = batch["images"], batch["captions"]
        outputs = self(images, captions)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch["images"], batch["captions"]
        outputs = self(images, captions)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # aggregated/most-recent val_loss is available via callback_metrics
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_value = float(val_loss.item())
            if val_value < self.best_val_loss:
                self.best_val_loss = val_value
                self._save_best_model()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.trainer.lr)

    def _save_best_model(self):
        print(f"🔍 New best model found with val_loss = {self.best_val_loss:.4f}. Saving to {self.save_path}")
        torch.save(self.model.state_dict(), self.save_path)
