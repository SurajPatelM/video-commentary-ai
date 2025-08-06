import torch
import hydra
from omegaconf import DictConfig
from lightning import Trainer
from torch.utils.data import DataLoader
from data.dataset import VideoCaptionDataset, VideoCaptionDatasetCSV
# from train.train import LitMultiModalModule
from models.baseline_swin_bart import SwinBartLightning

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    dataset = VideoCaptionDatasetCSV(captions_dir=cfg.data.captions_dir, frames_dir=cfg.data.frames_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, shuffle= False, pin_memory=True)
    model = SwinBartLightning(cfg)
    trainer = Trainer(
        accelerator=cfg.trainer.device,
        devices=1,
        max_epochs=cfg.trainer.epochs,
        precision=32 
    )
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
