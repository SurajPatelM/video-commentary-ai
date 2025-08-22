import os
import torch
import lightning as L
from models.model import SwinBart

class TrainerModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = SwinBart(cfg)
        self.best_val_loss = float("inf")  # initialize best loss
        self.save_dir = cfg.trainer.output_dir
        self.save_path = os.path.join(self.save_dir, "best_model.pth")
        os.makedirs(self.save_dir, exist_ok=True)
        

    def forward(self, images, captions, audio):
        return self.model(images, captions, audio)

    def training_step(self, batch, batch_idx):
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        outputs = self(images, captions, audio)
        loss = outputs.loss
        # log per-step, aggregate separately
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        outputs = self(images, captions, audio)
        loss = outputs.loss
        # In val loop Lightning aggregates by default on_epoch=True, but make it explicit.
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.trainer.lr)

    def on_validation_epoch_end(self):
        """Called once after all validation batches are processed and metrics are aggregated."""
        # Get the aggregated val_loss for the epoch from callback_metrics
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return  # nothing to compare
        # convert to Python float
        try:
            current = float(val_loss.detach().cpu().item())
        except Exception:
            current = float(val_loss)

        if current < self.best_val_loss and self.trainer.is_global_zero:
            self.best_val_loss = current
            self._save_best_model(current)

    def _save_best_model(self, current_val_loss: float):
        # latest best (overwrites)
        torch.save(self.model.state_dict(), self.save_path)
        # optional: archive each new best with step+epoch
        archive_path = os.path.join(
            self.save_dir, f"best_step{self.global_step}_epoch{self.current_epoch}_loss{current_val_loss:.4f}.pth"
        )
        # torch.save(self.model.state_dict(), archive_path)
        print(
            f"✅ New best model (val_loss={current_val_loss:.4f}) saved to:\n"
            f"  - {self.save_path} (latest best)\n"
            f"  - {archive_path} (archive)"
        )
