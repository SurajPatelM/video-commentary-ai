# datamodule_basic.py
import lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from data.dataset import VideoCaptionDatasetCSV, VideoCaptionDataset


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset = None

    def setup(self, stage: str = None):
        d = self.cfg.data
        if self.cfg.data.use_csv:
            self.dataset = VideoCaptionDatasetCSV(
                captions_dir=d.captions_dir,
                frames_dir=d.frames_dir
            )
        else:
            self.dataset = VideoCaptionDataset(
                frames_dir=d.frames_dir,
                captions_dir=d.captions_dir,
            )

    def train_dataloader(self):
        d = self.cfg.data
        return DataLoader(
            self.dataset,
            batch_size=d.batch_size,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        d = self.cfg.data
        return DataLoader(
            self.dataset,
            batch_size=d.batch_size,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        d = self.cfg.data
        return DataLoader(
            self.dataset,
            batch_size=d.batch_size,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            shuffle=False,
        )
