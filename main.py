import torch
import hydra
from omegaconf import DictConfig
from lightning import Trainer as LTrainer
from torch.utils.data import DataLoader
from data.dataset import VideoCaptionDatasetCSV
from models.train_lightning import TrainerModule

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Dataset and Dataloader
    dataset = VideoCaptionDatasetCSV(
        captions_dir=cfg.data.captions_dir,
        frames_dir=cfg.data.frames_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        pin_memory=True
    )

    # Model
    model = TrainerModule(cfg)

    # Lightning Trainer
    trainer = LTrainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,
        max_epochs=cfg.trainer.epochs,
        precision=32
    )

    # Training
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
