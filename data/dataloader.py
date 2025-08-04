from torch.utils.data import DataLoader
from data.dataset import VideoCaptionDataset, VideoCaptionDatasetCSV

def build_dataloader(cfg, split="train"):
    """
    Args:
        cfg: Hydra config object
        split: which split to load ("train", "val", etc.) — currently only "train"
    
    Returns:
        PyTorch DataLoader instance
    """
    dataset = VideoCaptionDatasetCSV(
        captions_dir=cfg.data.captions_dir,
        frames_dir=cfg.data.frames_dir
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.training.num_workers if "num_workers" in cfg.training else 2,
        pin_memory=True
    )
    return dataloader