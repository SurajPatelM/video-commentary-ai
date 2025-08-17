import torch
import hydra
from omegaconf import DictConfig
from lightning import Trainer as LTrainer

from data.datamodule import VideoDataModule
from models.train_lightning import TrainerModule


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Clear MPS cache if available
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # DataModule (handles dataset + dataloaders)
    datamodule = VideoDataModule(cfg)
    datamodule.setup(stage="train")

    # Model
    model = TrainerModule(cfg)

    # Lightning Trainer
    trainer = LTrainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.epochs,
        precision=cfg.trainer.precision,
        default_root_dir=cfg.trainer.output_dir,
    )

    # Training
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
