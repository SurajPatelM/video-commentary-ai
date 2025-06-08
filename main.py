import hydra
from omegaconf import DictConfig
from lightning import Trainer
from torch.utils.data import DataLoader
from data.dataset import VideoCaptionDataset
from train.train import LitMultiModalModule

@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    dataset = VideoCaptionDataset(captions_dir=cfg.data.captions_dir, frames_dir=cfg.data.frames_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, shuffle=True)

    model = LitMultiModalModule(
        text_encoder_cfg = cfg.text_encoder_cfg,
        vision_encoder_cfg = cfg.vision_encoder_cfg,
        fusion_cfg=cfg.fusion_cfg,
        decoder_cfg=cfg.decoder_cfg,
        lr=cfg.trainer.lr
    )

    trainer = Trainer(
        accelerator=cfg.trainer.device,
        devices=1,
        max_epochs=cfg.trainer.epochs,
        precision=32 
    )
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
