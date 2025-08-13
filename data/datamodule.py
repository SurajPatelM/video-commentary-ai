# data/datamodule.py
from typing import Optional
import lightning as L
from torch.utils.data import DataLoader
from .dataset import VideoCaptionDatasetCSV, VideoCaptionDataset

class VideoCaptionDataModule(L.LightningDataModule):
    """
    DataModule that uses your existing datasets and Hydra cfg.
    Expects:
      cfg.data.captions_dir
      cfg.data.frames_dir
      (optional) cfg.data.val_captions_dir
      (optional) cfg.data.test_captions_dir
      cfg.trainer.batch_size
      (optional) cfg.training.num_workers
      (optional) cfg.data.use_csv (default True)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _build_dataset(self, split_dir: str, *, use_csv: Optional[bool] = None):
        use_csv = getattr(self.cfg.data, "use_csv", True) if use_csv is None else use_csv
        if use_csv:
            return VideoCaptionDatasetCSV(
                captions_dir=split_dir,
                frames_dir=self.cfg.data.frames_dir
            )
        return VideoCaptionDataset(
            captions_dir=split_dir,
            frames_dir=self.cfg.data.frames_dir
        )

    def setup(self, stage: Optional[str] = None):
        self.train_ds = self._build_dataset(self.cfg.data.captions_dir)
        val_dir = getattr(self.cfg.data, "val_captions_dir", None)
        self.val_ds = self._build_dataset(val_dir or self.cfg.data.captions_dir)
        test_dir = getattr(self.cfg.data, "test_captions_dir", None)
        if test_dir:
            self.test_ds = self._build_dataset(test_dir)

    def _num_workers(self) -> int:
        return getattr(self.cfg.training, "num_workers", 2)

    def _batch_size(self) -> int:
        return self.cfg.trainer.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self._batch_size(),
            shuffle=True,
            num_workers=self._num_workers(),
            collate_fn=self.train_ds.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self._batch_size(),
            shuffle=False,
            num_workers=self._num_workers(),
            collate_fn=self.val_ds.collate_fn,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self._batch_size(),
            shuffle=False,
            num_workers=self._num_workers(),
            collate_fn=self.test_ds.collate_fn,
        )
