# data/datamodule.py
import os
import random
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
        train_dir = self.cfg.data.captions_dir
        val_dir = getattr(self.cfg.data, "val_captions_dir", None)
        test_dir = getattr(self.cfg.data, "test_captions_dir", None)

        use_csv = getattr(self.cfg.data, "use_csv", True)

        # If explicit val/test dirs exist, keep your existing behavior
        if (val_dir is not None or test_dir is not None) or not use_csv:
            self.train_ds = self._build_dataset(train_dir)
            self.val_ds = self._build_dataset(val_dir or train_dir)
            if test_dir:
                self.test_ds = self._build_dataset(test_dir)
            return

        # --- AUTO SPLIT BY VIDEO (CSV file) when no val/test dirs provided ---
        all_csv = sorted([f for f in os.listdir(train_dir) if f.endswith(".csv")])
        if not all_csv:
            raise RuntimeError(f"No CSV files found in {train_dir}")

        # knobs (defaults work even if not in YAML)
        val_pct  = float(getattr(self.cfg.data, "val_split_pct", 0.1))   # 10% val by default
        test_pct = float(getattr(self.cfg.data, "test_split_pct", 0.0))  # 0% test by default
        seed     = int(getattr(self.cfg.data, "seed", 42))

        rnd = random.Random(seed)
        rnd.shuffle(all_csv)

        n = len(all_csv)
        n_val  = max(1, int(n * val_pct)) if val_pct > 0 else 0
        n_test = max(1, int(n * test_pct)) if test_pct > 0 else 0

        # ensure at least one train video
        if n_val + n_test >= n:
            # reduce test first, then val if needed
            reduce = (n_val + n_test) - (n - 1)
            take_from_test = min(n_test, reduce)
            n_test -= take_from_test
            reduce -= take_from_test
            if reduce > 0:
                n_val = max(0, n_val - reduce)

        val_files  = all_csv[:n_val]
        test_files = all_csv[n_val:n_val + n_test]
        train_files = all_csv[n_val + n_test:]

        # build datasets with explicit file lists (prevents frame leakage across videos)
        self.train_ds = VideoCaptionDatasetCSV(
            captions_dir=train_dir,
            frames_dir=self.cfg.data.frames_dir,
            csv_files=train_files
        )
        self.val_ds = VideoCaptionDatasetCSV(
            captions_dir=train_dir,
            frames_dir=self.cfg.data.frames_dir,
            csv_files=val_files if n_val > 0 else train_files  # guarantee val exists
        )
        self.test_ds = VideoCaptionDatasetCSV(
            captions_dir=train_dir,
            frames_dir=self.cfg.data.frames_dir,
            csv_files=test_files
        ) if n_test > 0 else None

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
