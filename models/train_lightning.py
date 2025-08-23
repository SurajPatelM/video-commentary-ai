import os
import torch
import lightning as L
from models.model import SwinBart
import torchvision

class TrainerModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = SwinBart(cfg)
        self.best_val_loss = float("inf")
        self.save_dir = cfg.trainer.output_dir
        self.save_path = os.path.join(self.save_dir, "best_model.pth")
        os.makedirs(self.save_dir, exist_ok=True)
        # Save hyperparameters to logger
        self.save_hyperparameters(cfg)

    def forward(self, images, captions, audio):
        return self.model(images, captions, audio)

    def training_step(self, batch, batch_idx):
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        outputs = self(images, captions, audio)
        loss = outputs.loss

        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Optional: log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, logger=True)
        if batch_idx == 0 and isinstance(self.logger, L.pytorch.loggers.TensorBoardLogger):
            # Log first 4 images
            grid = images[:4]
            self.logger.experiment.add_images("train/images", grid, self.current_epoch)

            # Log MFCC as image (assuming shape [B, C, H, W] or [B, H, W])
            mfccs = audio[:4]  # Shape: [B, C, H, W] or [B, H, W]
            if mfccs.dim() == 3:  # If [B, H, W], add channel dimension
                mfccs = mfccs.unsqueeze(1)
            mfcc_grid = torchvision.utils.make_grid(mfccs, normalize=True, scale_each=True)
            self.logger.experiment.add_image("train/audio_mfcc", mfcc_grid, self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        outputs = self(images, captions, audio)
        loss = outputs.loss

        self.log("val/loss_step", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("val/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log sample inputs to TensorBoard
        if batch_idx == 0 and isinstance(self.logger, L.pytorch.loggers.TensorBoardLogger):
            # Log first 4 images
            grid = images[:4]
            self.logger.experiment.add_images("val/images", grid, self.current_epoch)

            # Log MFCC as image (assuming shape [B, C, H, W] or [B, H, W])
            mfccs = audio[:4]  # Shape: [B, C, H, W] or [B, H, W]
            if mfccs.dim() == 3:  # If [B, H, W], add channel dimension
                mfccs = mfccs.unsqueeze(1)
            mfcc_grid = torchvision.utils.make_grid(mfccs, normalize=True, scale_each=True)
            self.logger.experiment.add_image("val/audio_mfcc", mfcc_grid, self.current_epoch)

            # Optional: log raw audio (if you have it and want to hear it)
            # self.logger.experiment.add_audio("val/audio_waveform", raw_audio[0], self.current_epoch, sample_rate=16000)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.trainer.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val/loss_epoch")
        if val_loss is None:
            return

        try:
            current = float(val_loss.detach().cpu().item())
        except Exception:
            current = float(val_loss)

        if current < self.best_val_loss and self.trainer.is_global_zero:
            self.best_val_loss = current
            self._save_best_model(current)

    def _save_best_model(self, current_val_loss: float):
        torch.save(self.model.state_dict(), self.save_path)
        archive_path = os.path.join(
            self.save_dir,
            f"best_step{self.global_step}_epoch{self.current_epoch}_loss{current_val_loss:.4f}.pth"
        )

        # Save archive as well
        # torch.save(self.model.state_dict(), archive_path)

        print(
            f"✅ New best model (val_loss={current_val_loss:.4f}) saved to:\n"
            f"  - {self.save_path} (latest best)\n"
            f"  - {archive_path} (archive)"
        )
