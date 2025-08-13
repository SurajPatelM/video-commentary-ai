import torch
import hydra
from omegaconf import DictConfig
from lightning import Trainer as LTrainer
from data.datamodule import VideoCaptionDataModule
from models.train_lightning import TrainerModule

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # DataModule (uses  dataset + collate_fn under the hood)
    dm = VideoCaptionDataModule(cfg)

    # Model
    model = TrainerModule(cfg)

    # Lightning Trainer — per-batch validation enabled
    trainer = LTrainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,
        max_epochs=cfg.trainer.epochs,  
        precision=32,
        val_check_interval=1,      # validate after EVERY train batch
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,    
    )

    # Train
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
